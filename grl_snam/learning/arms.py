"""Arms: the policies a learning claim is tested against, and the table that compares them.

An arm is ``NAME=SPEC``:

- ``NAME=path/to/policy.cvcnav``: a ``.cvcnav`` blob, e.g. a training run's step-0 checkpoint
  (``<prefix>.step000000.cvcnav``, the untrained arm) or its held-out pick (the trained arm);
- ``NAME=path/to/checkpoint.pt``: a ``grl-snam train`` checkpoint or a bare state dict;
- ``NAME=const:...``: a constant policy, the same coefficients everywhere (:func:`parse_constant`).

A constant policy is a ``CoefMLP`` that ignores its input: every weight is zero, and the output's
bias is the inverse of the output activation at each constant (``softplus^-1(c)`` through the
``log(expm1)`` fold for alpha, beta and gamma, ``logit(c / max)`` for a sigmoid lam head). It runs
on every path a trained net runs on, torch and libcvc's ``coef_mlp`` alike
(``coef_export --constant``). It answers whether the learned state dependence earns anything over
the best fixed setting.

:func:`arms_table` turns per-unit values into ``grl_snam.arms/v1``: per case, per metric, per arm,
the per-seed values, their mean and 95% bootstrap CI, and the paired difference to the reference
arm. It has the shape of the downstream A/B harness's JSON (``arms``; ``results[].case`` /
``metrics[metric][arm] = {mean, ci95, n, paired_diff}``, the difference being arm minus
reference, the reference being the first arm unless named), so one table reader reads both; the
per-seed ``seeds``/``values``, ``direction`` and ``primary`` are additions.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .stats import N_BOOT, as_rng, paired_diff, summarize

ARMS_SCHEMA = "grl_snam.arms/v1"

#: A CoefMLP's output columns, in order (the same names as ``grl_snam.learning_history``).
HEADS: tuple[str, ...] = ("alpha", "beta", "gamma", "lam_soft", "lam_hard")
#: The constant arm's default alpha, beta and gamma: CoefMLP's own output bias, the regime its
#: untrained net is built around.
DEFAULT_ABG: tuple[float, float, float] = (1.0, 3.0, 4.0)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    spec: str

    @property
    def kind(self) -> str:
        """``"const"``, ``"cvcnav"`` or ``"pt"``."""
        if self.spec.startswith("const:"):
            return "const"
        return "cvcnav" if self.spec.endswith(".cvcnav") else "pt"

    def describe(self) -> dict:
        d = {"name": self.name, "flags": [self.spec], "kind": self.kind}
        if self.kind != "const" and Path(self.spec).is_file():
            d["sha256"] = hashlib.sha256(Path(self.spec).read_bytes()).hexdigest()
        return d


def parse_arm(text: str) -> ArmSpec:
    """``"NAME=SPEC"`` -> :class:`ArmSpec`."""
    name, sep, spec = text.partition("=")
    if not sep or not name.strip() or not spec.strip():
        raise ValueError(f"an arm is NAME=PATH or NAME=const:..., not {text!r}")
    return ArmSpec(name.strip(), spec.strip())


def parse_constant(spec: str) -> dict[str, float]:
    """The heads' constants from ``const:...`` (the prefix is optional):

    - ``const:`` or ``const:bias``: CoefMLP's bias, alpha=1, beta=3, gamma=4;
    - ``const:1.2,3,4``: alpha, beta, gamma (then lam_soft, lam_hard), in order;
    - ``const:alpha=1.2,gamma=5``: by name; alpha, beta and gamma not named keep the bias;
    - ``const:hist=H.jsonl``: each head's mean at the run's held-out pick (else its last
      checkpoint), from a learning history: the trained policy with its state dependence removed.
    """
    body = spec[len("const:") :] if spec.startswith("const:") else spec
    body = body.strip()
    if body in ("", "bias"):
        return dict(zip(HEADS, DEFAULT_ABG, strict=False))
    if body.startswith("hist="):
        return constants_from_history(body[len("hist=") :])
    items = [t.strip() for t in body.split(",") if t.strip()]
    if all("=" not in t for t in items):
        if len(items) > len(HEADS):
            raise ValueError(f"at most {len(HEADS)} constants ({', '.join(HEADS)}): {spec!r}")
        return dict(zip(HEADS, (float(t) for t in items), strict=False))
    out = dict(zip(HEADS, DEFAULT_ABG, strict=False))
    for t in items:
        k, sep, v = t.partition("=")
        if not sep or k.strip() not in HEADS:
            raise ValueError(f"const: names {', '.join(HEADS)}; got {t!r} in {spec!r}")
        out[k.strip()] = float(v)
    return out


def constants_from_history(path: str) -> dict[str, float]:
    """Each head's ``mean`` at the held-out pick of the (last) run in ``path``, else at its last
    checkpoint with head means."""
    from .history import load

    runs = load(path)
    if not runs:
        raise ValueError(f"{path}: no run")
    rows = [r for r in runs[-1].checkpoints() if any("mean" in d for d in r.heads.values())]
    if not rows:
        raise ValueError(f"{path}: no checkpoint records a head mean")
    row = next((r for r in rows if r.selected), rows[-1])
    out = {h: row.heads[h]["mean"] for h in HEADS if "mean" in row.heads.get(h, {})}
    if not {"alpha", "beta", "gamma"} <= set(out):
        raise ValueError(f"{path}: the checkpoint at {row.x:g} lacks alpha/beta/gamma means")
    return out


def _softplus_inv(c: float) -> float:
    return math.log(math.expm1(c))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def constant_coef_mlp(values, *, like=None):
    """A ``CoefMLP`` whose output is ``values`` (a mapping by head name or a sequence in head
    order) for every input. ``like`` (a CoefMLP) gives the input layout (in_dim, use_mu,
    use_risk, hidden) and the lam ceilings; the lam heads come from ``values``."""
    import torch

    import grl_snam.sdf_nav as sdf_nav

    vals = dict(values) if isinstance(values, Mapping) else dict(zip(HEADS, values, strict=False))
    unknown = set(vals) - set(HEADS)
    if unknown:
        raise ValueError(f"unknown heads {sorted(unknown)}; heads are {', '.join(HEADS)}")
    if "lam_hard" in vals and "lam_soft" not in vals:
        raise ValueError("a lam_hard constant needs a lam_soft one (lam_hard is the 5th output)")
    abg = [float(vals.get(h, d)) for h, d in zip(HEADS[:3], DEFAULT_ABG, strict=True)]
    for h, c in zip(HEADS, abg, strict=False):
        if not c > 0.0:
            raise ValueError(f"{h} = {c!r}: a softplus head is positive, so its constant must be")
    use_lam = "lam_soft" in vals
    use_lam_hard = "lam_hard" in vals
    kw = dict(hidden=64, in_dim=5, use_mu=False, use_risk=False, lam_sigmoid=False)
    if like is not None:
        kw = dict(
            hidden=like.net[0].out_features,
            in_dim=like.in_dim,
            use_mu=like.use_mu,
            use_risk=like.use_risk,
            lam_sigmoid=bool(getattr(like, "lam_sigmoid", False)) and use_lam,
            lam_soft_max=getattr(like, "lam_soft_max", 5.0),
            lam_hard_max=getattr(like, "lam_hard_max", 10.0),
        )
    sigmoid = kw["lam_sigmoid"] or use_lam_hard
    lam_soft_max = kw.get("lam_soft_max", 5.0)
    lam_hard_max = kw.get("lam_hard_max", 10.0)
    if use_lam:
        c = float(vals["lam_soft"])
        if sigmoid and not 0.0 < c < lam_soft_max:
            raise ValueError(f"lam_soft = {c!r}: a sigmoid head needs 0 < c < {lam_soft_max}")
        if not sigmoid and not c > 0.0:
            raise ValueError(f"lam_soft = {c!r}: a softplus head needs c > 0")
    if use_lam_hard and not 0.0 < float(vals["lam_hard"]) < lam_hard_max:
        raise ValueError(f"lam_hard = {vals['lam_hard']!r}: needs 0 < c < {lam_hard_max}")
    model = sdf_nav.CoefMLP(
        bias=tuple(abg),
        use_lam=use_lam,
        lam_init=float(vals.get("lam_soft", 0.4)),
        use_lam_hard=use_lam_hard,
        lam_hard_init=float(vals.get("lam_hard", 1.0)),
        **kw,
    )
    with torch.no_grad():
        for m in model.net:
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                m.bias.zero_()
        # alpha/beta/gamma (and a v1 softplus lam_soft): the raw bias buffer is the constant,
        # which the forward (and the C++ loader) folds through log(expm1) = softplus^-1.
        model.bias.copy_(torch.tensor(abg))
        if use_lam and not model.lam_sigmoid:
            model.lam_raw_bias.fill_(float(vals["lam_soft"]))
        if model.lam_sigmoid:
            model.net[-1].bias[3] = _logit(float(vals["lam_soft"]) / model.lam_soft_max)
            if use_lam_hard:
                model.net[-1].bias[4] = _logit(float(vals["lam_hard"]) / model.lam_hard_max)
    return model.eval()


def load_arm_model(arm: ArmSpec, *, like=None):
    """The arm's policy as a torch ``CoefMLP`` (eval mode)."""
    if arm.kind == "const":
        return constant_coef_mlp(parse_constant(arm.spec), like=like)
    if arm.kind == "cvcnav":
        from ..tools.coef_export import read_coef_mlp

        return read_coef_mlp(arm.spec)
    from ..tools.scorecard_eval import load_model

    return load_model(arm.spec)


# --------------------------------------------------------------------------- the table


def arms_table(
    values: Mapping[str, Mapping[str, Mapping[str, Mapping]]],
    arms: Sequence[ArmSpec | str],
    *,
    reference: str | None = None,
    direction: Mapping[str, str] | None = None,
    primary: Mapping[str, Sequence[str]] | None = None,
    case_flags: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = N_BOOT,
    seed: int = 1,
    meta: Mapping | None = None,
) -> dict:
    """``grl_snam.arms/v1`` from ``values[case][metric][arm] = {unit: value}``.

    ``arms`` fixes the column order, ``reference`` the arm the paired differences are taken to
    (default the first). ``direction[metric]`` is ``"up"`` (higher is better) or ``"down"``;
    ``primary[case]`` names the metrics a claim is made on. The bootstrap draws come from one
    generator seeded ``seed``, consumed case by case, metric by metric, arm by arm (the arm's CI,
    then its paired difference's), as the A/B harness does.
    """
    specs = [a if isinstance(a, ArmSpec) else ArmSpec(str(a), "") for a in arms]
    names = [a.name for a in specs]
    ref = names[0] if reference is None else reference
    if ref not in names:
        raise ValueError(f"reference arm {ref!r} is not one of {names}")
    rng = as_rng(seed)
    results = []
    for case, metrics in values.items():
        units: list = []
        out_metrics: dict = {}
        for metric, per_arm in metrics.items():
            base = per_arm.get(ref, {})
            mo: dict = {}
            for a in names:
                got = per_arm.get(a)
                if not got:
                    continue
                for u in got:
                    if u not in units:
                        units.append(u)
                entry = summarize(got.values(), n_boot, rng)
                entry["seeds"] = list(got.keys())
                entry["values"] = [_jsonable(v) for v in got.values()]
                if a != ref:
                    d = paired_diff(base, got, n_boot, rng)
                    if d is not None:
                        entry["paired_diff"] = d
                mo[a] = entry
            if mo:
                out_metrics[metric] = mo
        res = {
            "case": case,
            "flags": list((case_flags or {}).get(case, [])),
            "episodes": len(units),
            "seeds": units,
            "metrics": out_metrics,
        }
        if primary and case in primary:
            res["primary"] = list(primary[case])
        results.append(res)
    return {
        "schema": ARMS_SCHEMA,
        "arms": [a.describe() if a.spec else {"name": a.name, "flags": []} for a in specs],
        "reference": ref,
        "boot": n_boot,
        "seed": seed,
        "direction": dict(direction or {}),
        "meta": dict(meta or {}),
        "results": results,
    }


def _jsonable(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def arm_values(table: Mapping, case: str, metric: str, arm: str) -> dict | None:
    """``{seed: value}`` of one arm's metric, from a table with per-seed values; ``None``
    without them (a harness table without ``seeds``/``values``)."""
    for res in table.get("results", []):
        if res.get("case") != case:
            continue
        e = res.get("metrics", {}).get(metric, {}).get(arm)
        if not e or "values" not in e:
            return None
        seeds = e.get("seeds") or list(range(len(e["values"])))
        return dict(zip((str(s) for s in seeds), e["values"], strict=True))
    return None
