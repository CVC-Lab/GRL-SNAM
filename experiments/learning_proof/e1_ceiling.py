"""E1's ceiling: can the best scalar damping per region, fitted directly on the generator's own
windows, recover the planted gamma?

``train_coef_energy`` fits ``(alpha_j, beta, gamma)`` through a rigid-point surrogate
(``integrate_surrogate_v2``: barrier + goal spring - gamma * v, unit mass) to short windows of the
ring's centre. The generator's centre obeys ``M_o dv/dt = F_barrier + F_stage - gamma_o(x) v``, so
the gamma the surrogate should recover is ``gamma_o / M_o``. If even free scalars per region,
fitted straight to the windows with the trainer's own integrator and loss (traj + vel), miss that,
no net trained through the same surrogate can find it, and E1 stops there.

Three fits per dataset, by L-BFGS on log-parameters over every window that starts and ends inside
a region (by the centre's x):

- ``shared``: gamma per region; one alpha (every obstacle) and one beta shared by both regions;
- ``free``: alpha, beta and gamma all per region;
- ``pinned``: gamma fixed at the truth, alpha and beta fitted as in ``shared``. Its loss against
  ``shared``'s says whether the surrogate's loss prefers the true gamma at all.

Each is reported on all such windows and on the moving ones (``|v0| > --moving``).

    python experiments/learning_proof/e1_ceiling.py runs/e1/data/planted runs/e1/data/control \\
        --out ceiling.json
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from grl_snam.surrogate_robust import integrate_surrogate_v2
from grl_snam.train_coef_energy import ShortRollouts, collate_short


def windows(root, n, seed):
    ds = ShortRollouts(root, rng=random.Random(seed))
    r = random.Random(seed)
    return [ds.draw(r)[0] for _ in range(n)]


def region_of(items, spec):
    """Each window's region index (by the centre's x at both ends), -1 when it straddles."""
    out = []
    for it in items:
        x0, x1 = float(it["o0"][0]), float(it["o_tgt"][0])
        k = -1
        for j, r in enumerate(spec["regions"]):
            lo, hi = float(r["x_lo"]), float(r["x_hi"])
            if lo <= x0 < hi and lo <= x1 < hi:
                k = j
        out.append(k)
    return torch.tensor(out)


def fit(batch, reg, n_regions, *, free, iters, lr, device, pin_gamma=None):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    reg = reg.to(device)
    n_ab = n_regions if free else 1
    p = {
        "log_a": torch.zeros(n_ab, device=device, requires_grad=True),
        "log_b": torch.zeros(n_ab, device=device, requires_grad=True),
    }
    if pin_gamma is None:
        p["log_g"] = torch.zeros(n_regions, device=device, requires_grad=True)
    else:
        p["log_g"] = torch.tensor(pin_gamma, device=device).log()
    params = [v for v in p.values() if v.requires_grad]
    opt = torch.optim.LBFGS(params, lr=lr, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)  # fmt: skip
    idx_ab = reg if free else torch.zeros_like(reg)
    B = b["o0"].shape[0]

    def loss_fn():
        a = p["log_a"].exp()[idx_ab]
        alphas = a.unsqueeze(-1).expand(B, b["C"].shape[1]) * b["obs_mask"].float()
        beta = p["log_b"].exp()[idx_ab]
        gamma = p["log_g"].exp()[reg]
        oT, vT, _ = integrate_surrogate_v2(
            b["o0"], b["v0"], b["goal"], b["C"], b["R"], b["obs_mask"], alphas, beta, gamma,
            b["d_hat"], b["dt_prime"], b["H"], robot_radius=torch.zeros(B, device=device),
            margin_factor=0.5,
        )  # fmt: skip
        return ((oT - b["o_tgt"]) ** 2).mean() + ((vT - b["v_tgt"]) ** 2).mean()

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for _ in range(4):  # restarts of the L-BFGS history
        opt.step(closure)
    with torch.no_grad():
        final = float(loss_fn())
    return {
        "gamma": p["log_g"].exp().tolist(),
        "alpha": p["log_a"].exp().tolist(),
        "beta": p["log_b"].exp().tolist(),
        "loss": final,
    }


def summarize(got, spec):
    m_o = float(spec["M_o"])
    names = [r["name"] for r in spec["regions"]]
    truths = [float(r["gamma"]) / m_o for r in spec["regions"]]
    out = {"loss": got["loss"], "alpha": got["alpha"], "beta": got["beta"], "regions": {}}
    for name, t, g in zip(names, truths, got["gamma"], strict=True):
        out["regions"][name] = {"truth": t, "fit": g, "rel_err": abs(g - t) / t}
    ratio_t = truths[-1] / truths[0]
    ratio = got["gamma"][-1] / got["gamma"][0]
    out["ratio"] = {"truth": ratio_t, "fit": ratio, "rel_err": abs(ratio - ratio_t) / ratio_t}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="+", help="dataset roots (manifest.json + planted.json)")
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200, help="L-BFGS iterations per restart")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--moving", type=float, default=0.05, help="|v0| above which a window moves")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = {"args": vars(args), "datasets": {}}
    for root in args.roots:
        with open(os.path.join(root, "planted.json")) as f:
            spec = json.load(f)
        items = windows(root, args.windows, args.seed)
        reg = region_of(items, spec)
        keep = [i for i in range(len(items)) if int(reg[i]) >= 0]
        moving = [i for i in keep if float(items[i]["v0"].norm()) > args.moving]
        res = {"n_windows": len(items), "n_in_region": len(keep), "n_moving": len(moving)}
        for label, idx in (("all", keep), ("moving", moving)):
            batch = collate_short([items[i] for i in idx])
            r = reg[idx]
            nreg = len(spec["regions"])
            kw = dict(iters=args.iters, lr=args.lr, device=args.device)
            truth = [float(x["gamma"]) / float(spec["M_o"]) for x in spec["regions"]]
            res[label] = {
                "n_per_region": [int((r == j).sum()) for j in range(nreg)],
                "shared": summarize(fit(batch, r, nreg, free=False, **kw), spec),
                "free": summarize(fit(batch, r, nreg, free=True, **kw), spec),
                "pinned": summarize(fit(batch, r, nreg, free=False, pin_gamma=truth, **kw), spec),
            }
        result["datasets"][os.path.basename(os.path.normpath(root))] = res
        print(os.path.basename(os.path.normpath(root)), json.dumps(res, indent=1))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
