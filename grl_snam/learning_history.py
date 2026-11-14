"""Learning instruments for the coefficient trainers: is training learning, or fitting noise?

A falling training loss does not show that a policy learned anything: it can fall while the
deployed behaviour stays where it was, and a checkpoint picked on the training loss is picked on
the very numbers it was fitted to. These instruments record what that judgement needs. All of
them are OPT-IN observers; with none on, the trainers run exactly as before.

- **The history** (a JSON-lines file). A header, then one record per outer step:
  - the loss terms (window means);
  - the global gradient norm ``clip_grad_norm_`` returns, BEFORE the clip (mean and max over the
    step's optimizer steps, and the share that clipped), and, when the trainer measures them, each
    loss term's own gradient norm (``grad_norm.terms.<term>``);
  - the learning rate.

  Then one record per checkpoint:
  - per head (``alpha``, ``beta``, ``gamma``, and ``lam_soft`` / ``lam_hard`` on a net that learns
    them): its mean and spread (standard deviation) over a fixed probe batch of agents, i.e. how
    much it varies with the situation an agent is in;
  - per head, its relative parameter change ``||dtheta|| / ||theta||`` since the last checkpoint
    and since the start, with the trunk (the shared layers) alongside;
  - the held-out fitness, when one is given, and the held-out terms behind it when the fitness
    returns them (a dict with a ``"fitness"`` key: the rest is recorded as ``heldout``).
- **Checkpoints**: the policy at each checkpoint saved as ``<prefix>.stepNNNNNN.cvcnav`` (or the
  extension the trainer's ``save`` writes, ``checkpoint_ext``).
- **The held-out pick** (``best_path``): the checkpoint with the highest held-out fitness, the
  untrained policy (step 0) included, saved as a ``.cvcnav``. A best at step 0 means no step
  improved on the start policy on held-out data.

The probe and the fitness run with the torch RNG forked, and the model's train/eval mode is
restored after them, so with the instruments on the trained weights are the same, bit for bit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

HISTORY_SCHEMA = "grl_snam.coef_train_history/v1"

#: The CoefMLP's output columns, in order (``sdf_nav.CoefMLP``: 3, 4 or 5 outputs).
COEF_HEADS: tuple[str, ...] = ("alpha", "beta", "gamma", "lam_soft", "lam_hard")


def checkpoint_path(prefix: str, step: int, ext: str = ".cvcnav") -> str:
    """``"<prefix>.step000040.cvcnav"``: an intermediate checkpoint's name."""
    return f"{prefix}.step{step:06d}{ext}"


def relative_change(now: torch.Tensor, before: torch.Tensor) -> float:
    """``||now - before|| / ||before||`` (0 for an all-zero ``before`` that did not move)."""
    num = float(torch.linalg.vector_norm((now - before).double()))
    den = float(torch.linalg.vector_norm(before.double()))
    if den <= 0.0:
        return 0.0 if num <= 0.0 else math.inf
    return num / den


def coef_heads(model, feat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every head a ``sdf_nav.CoefMLP`` predicts for ``feat``, by name, ``[n]`` each."""
    lam = getattr(model, "use_lam", False) or getattr(model, "use_lam_hard", False)
    out = model.coeffs_and_lam(feat) if lam else model(feat)
    return {name: t.reshape(-1) for name, t in zip(COEF_HEADS, out, strict=False)}


def coef_param_groups(model) -> dict[str, list[torch.Tensor]]:
    """A ``sdf_nav.CoefMLP``'s parameters by head (each head is one row of the output layer, its
    weight row and bias) and ``trunk`` (the hidden layers)."""
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    out_layer, hidden = layers[-1], layers[:-1]
    groups: dict[str, list[torch.Tensor]] = {}
    for k in range(out_layer.out_features):
        name = COEF_HEADS[k] if k < len(COEF_HEADS) else f"out{k}"
        groups[name] = [out_layer.weight[k], out_layer.bias[k : k + 1]]
    groups["trunk"] = [p for lin in hidden for p in (lin.weight, lin.bias)]
    return groups


def _flat(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).to("cpu", torch.float64) for p in params])


@dataclass
class CoefTrainMonitorCfg:
    """What a coefficient trainer records. The defaults turn everything off."""

    #: The history file; ``None``: none.
    history: str | None = None
    #: Outer steps between checkpoints (0: only the start and the end, when anything is on).
    checkpoint_every: int = 0
    #: Save each checkpoint's policy as ``<prefix>.stepNNNNNN.cvcnav``; ``None``: no blobs.
    checkpoint_prefix: str | None = None
    #: The checkpoints' extension: what the trainer's ``save`` writes (``.pt`` for a torch one).
    checkpoint_ext: str = ".cvcnav"
    #: Save the checkpoint with the highest held-out fitness here; ``None``: no pick.
    best_path: str | None = None
    #: Agents in the probe batch the head numbers are read on, and its seed.
    probe_agents: int = 256
    probe_seed: int = 0x5EED

    @property
    def enabled(self) -> bool:
        return bool(self.history or self.checkpoint_prefix or self.best_path)


class CoefTrainMonitor:
    """The trainer-side state of the instruments (see the module docstring).

    ``probe(model) -> {head: [n]}`` reads the heads on the fixed probe batch; ``fitness(model) ->
    float`` (higher is better) scores held-out data, ``None`` without one; it may instead return
    a dict of held-out terms with the score under ``"fitness"``, and the checkpoint record then
    carries the rest as ``heldout``. ``save(model, path)`` writes a ``.cvcnav`` (or whatever
    ``checkpoint_ext`` names). ``best_path`` needs a fitness.
    """

    def __init__(
        self,
        cfg: CoefTrainMonitorCfg,
        *,
        save: Callable,
        fitness: Callable | None = None,
        meta: dict | None = None,
    ) -> None:
        if cfg.best_path and fitness is None:
            raise ValueError("best_path picks on a held-out fitness: pass fitness=")
        if cfg.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0")
        self.cfg = cfg
        self.save = save
        self.fitness = fitness
        self.meta = dict(meta or {})
        self.best = -math.inf
        self.best_step: int | None = None
        self.checkpoints: list[dict] = []
        self._fh = None
        self._probe = None
        self._groups: dict[str, list[torch.Tensor]] = {}
        self._init: dict[str, torch.Tensor] = {}
        self._prev: dict[str, torch.Tensor] = {}
        self._last_ckpt = -1
        self._reset_step()

    # ---- lifecycle
    def begin(self, model, probe: Callable) -> None:
        """Before the first step: the header and the step-0 checkpoint (the untrained policy)."""
        self._probe = probe
        self._groups = coef_param_groups(model)
        self._init = {g: _flat(ps) for g, ps in self._groups.items()}
        self._prev = dict(self._init)
        if self.cfg.history:
            path = Path(self.cfg.history)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
            self._write(
                {
                    "kind": "header",
                    "schema": HISTORY_SCHEMA,
                    "checkpoint_every": self.cfg.checkpoint_every,
                    "probe_agents": self.cfg.probe_agents,
                    "probe_seed": self.cfg.probe_seed,
                    "fitness": self.fitness is not None,
                    "heads": list(self._groups),
                    **self.meta,
                }
            )
        self.checkpoint(0, model)

    def window(
        self,
        terms: dict[str, float],
        grad_norm,
        lr: float,
        clip: float,
        term_grads: dict[str, float] | None = None,
    ) -> None:
        """After each optimizer step: its loss terms and its pre-clip gradient norm, and, when the
        trainer measured them, each loss term's own gradient norm (``term_grads``)."""
        for k, v in terms.items():
            self._terms[k] = self._terms.get(k, 0.0) + float(v)
        for k, v in (term_grads or {}).items():
            self._term_gn.setdefault(k, []).append(float(v))
        g = float(grad_norm)
        self._gn.append(g)
        self._clipped += int(clip > 0.0 and g > clip)
        self._lr = float(lr)

    def end_step(self, step: int, model) -> None:
        """After outer step ``step`` (0-based): its record, and a checkpoint when one is due."""
        n = max(len(self._gn), 1)
        grad_norm = {
            "mean": sum(self._gn) / n if self._gn else 0.0,
            "max": max(self._gn) if self._gn else 0.0,
            "clipped_frac": self._clipped / n,
        }
        if self._term_gn:
            grad_norm["terms"] = {k: sum(v) / len(v) for k, v in self._term_gn.items()}
        self._write(
            {
                "kind": "step",
                "step": step + 1,
                "windows": len(self._gn),
                "terms": {k: v / n for k, v in self._terms.items()},
                "lr": self._lr,
                "grad_norm": grad_norm,
            }
        )
        self._reset_step()
        every = self.cfg.checkpoint_every
        if every > 0 and (step + 1) % every == 0:
            self.checkpoint(step + 1, model)

    def finish(self, steps: int, model) -> None:
        """After the last step: its checkpoint (unless one was just taken) and the end record."""
        if self._last_ckpt != steps:
            self.checkpoint(steps, model)
        self._write({"kind": "end", "best_step": self.best_step, "best_fitness": self.best_or_none})
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def best_or_none(self) -> float | None:
        return None if self.best_step is None else self.best

    # ---- one checkpoint
    def checkpoint(self, step: int, model) -> dict:
        was_training = model.training
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            heads = {h: v.detach().to("cpu", torch.float64) for h, v in self._probe(model).items()}
            got = None if self.fitness is None else self.fitness(model)
        heldout = None
        if isinstance(got, dict):
            heldout = {k: v for k, v in got.items() if k != "fitness"}
            got = got["fitness"]
        fit = None if got is None else float(got)
        rec_heads: dict[str, dict] = {}
        for g, ps in self._groups.items():
            now = _flat(ps)
            r: dict = {}
            if g in heads:
                v = heads[g]
                r["mean"] = float(v.mean())
                r["spread"] = float(v.std(unbiased=False))
            r["dtheta_rel"] = relative_change(now, self._prev[g])
            r["dtheta_rel_init"] = relative_change(now, self._init[g])
            rec_heads[g] = r
            self._prev[g] = now
        blob = None
        if self.cfg.checkpoint_prefix:
            blob = checkpoint_path(self.cfg.checkpoint_prefix, step, self.cfg.checkpoint_ext)
            self.save(model, blob)
        best = False
        if fit is not None and fit > self.best:
            self.best, self.best_step, best = fit, step, True
            if self.cfg.best_path:
                self.save(model, self.cfg.best_path)
        model.train(was_training)
        rec = {
            "kind": "checkpoint",
            "step": step,
            "fitness": fit,
            "best": best,
            "heads": rec_heads,
            "blob": blob,
        }
        if heldout is not None:
            rec["heldout"] = heldout
        self.checkpoints.append(rec)
        self._last_ckpt = step
        self._write(rec)
        return rec

    # ---- internals
    def _reset_step(self) -> None:
        self._terms: dict[str, float] = {}
        self._term_gn: dict[str, list[float]] = {}
        self._gn: list[float] = []
        self._clipped = 0
        self._lr = 0.0

    def _write(self, rec: dict) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(rec) + "\n")


def read_history(path: str | Path) -> list[dict]:
    """Every record of a history file, in order."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- per-epoch helpers


def module_param_groups(model, heads: dict[str, torch.nn.Module]) -> dict[str, list[torch.Tensor]]:
    """``heads`` (name -> head module) and ``trunk`` (every other parameter of ``model``)."""
    owned = {id(p) for m in heads.values() for p in m.parameters()}
    groups = {name: list(m.parameters()) for name, m in heads.items()}
    groups["trunk"] = [p for p in model.parameters() if id(p) not in owned]
    return groups


def snapshot(groups: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Each group's parameters, flat and detached: what :func:`relative_change` compares."""
    return {g: _flat(ps) for g, ps in groups.items()}


def scene_summary(values: torch.Tensor, scene_ids: Sequence) -> dict:
    """``mean`` over the samples, ``spread`` = the standard deviation of the per-scene means,
    ``within`` = the mean within-scene standard deviation, and ``n_scenes``."""
    v = values.detach().to("cpu", torch.float64).reshape(-1)
    by: dict = {}
    for i, s in enumerate(scene_ids):
        by.setdefault(s, []).append(i)
    means = torch.stack([v[idx].mean() for idx in by.values()])
    withins = torch.stack([v[idx].std(unbiased=False) for idx in by.values()])
    return {
        "mean": float(v.mean()),
        "spread": float(means.std(unbiased=False)),
        "within": float(withins.mean()),
        "n_scenes": len(by),
    }
