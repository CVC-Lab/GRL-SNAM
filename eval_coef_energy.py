#!/usr/bin/env python3
"""
Visual evaluation of the new coefficient-based surrogate (CoefEnergyNet).
- Loads a trained checkpoint from train_coef_energy.py
- At each planner step, predicts {alphas, beta, gamma} from local context
- Injects them into the *modified* stagewise integrator (IPCBarrier uses obs.W)
- Captures PNG frames and writes GIF/MP4 for visual validation

Usage
-----
python -m eval_coef_energy \
  --ckpt checkpoints/coef_energy/epoch_049.pt \
  --case case1-tight --steps 800 --fps 20 --alpha_mode weight

alpha_mode ∈ {weight, radius, both, none} selects how to map α_j:
  weight: W_j ← W_j * α_j^+
  radius: R_j ← R_j + k_rad * α_j^+
  both:   apply both of the above
  none:   ignore α (for ablation)

Assumptions
-----------
• You applied the `IPCBarrier` patch so per-obstacle weights W scale the barrier.
• Your generator exposes: GenCfg, set_all_seeds, sample_obstacles_case1_tight, planner_from_cfg,
  save_episode_snapshot (same API as your old eval file).
• Your stagewise module is available as scripts.spline_stagewise6 (or adapt imports below).
"""
from __future__ import annotations
import os, glob, math, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

# --- project imports (adjust if paths differ)
from train_coef_energy import CoefEnergyNet  # new model
import scripts.ring_dataset_maxmin as gen
import scripts.spline_stagewise6 as ssi
from surrogate_robust import integrate_surrogate_v2

# ---- paste near top of your eval file, after imports ----
class HistSecantController:
    """
    One-step history-based (secant) sensitivity controller.
    Adjusts: beta, gamma, and top-K alphas (nearest obstacles) without extra sims.
    """
    def __init__(self, k_alpha=2, lr_beta=0.15, lr_gamma=0.10, lr_alpha=0.4,
                 safe_margin=0.08, v_min=0.25, v_max=0.5, prog_eps=0.01, ema=0.9):
        self.k_alpha = k_alpha
        self.lr_b, self.lr_g, self.lr_a = lr_beta, lr_gamma, lr_alpha
        self.safe_margin = safe_margin
        self.v_min = v_min
        self.v_max = v_max
        self.prog_eps = prog_eps
        self.ema = ema
        # history
        self.prev = None  # dict with {theta, y}
        # running Jacobian (optional EMA)
        self.J = None     # shape (3, D) where D = 2 + k_alpha

    def _select_alpha_indices(self, o_w, Cw, Rw, Ww):
        if Cw.shape[0] == 0: return np.array([], dtype=int)
        d = np.linalg.norm(o_w[None,:] - Cw, axis=1) - Rw
        j = np.argsort(d)[:self.k_alpha]
        return j

    def update(self, alphas, beta, gamma,  # current params (torch, shapes [1,N], [1], [1])
               o_w, v_w, goal, Cw, Rw, Ww,  # numpy (world slice you already have)
               clearance_now, dist_now, speed_now):
        # choose alpha subset
        idx = self._select_alpha_indices(o_w, Cw, Rw, Ww)
        a_sub = (alphas.squeeze(0).detach().cpu().numpy()[idx] if idx.size else np.zeros((0,), dtype=np.float32))
        b = float(beta.squeeze(0).detach().cpu().item())
        g = float(gamma.squeeze(0).detach().cpu().item())

        # current observable vector (we use signs so 'lower is worse')
        y = np.array([-float(clearance_now), float(dist_now),# float(speed_now), 
                      -float(speed_now)], dtype=np.float32)
        theta = np.concatenate([np.array([b, g], dtype=np.float32), a_sub], axis=0)  # D = 2+k_alpha

        # targets: keep clearance >= safe_margin, decrease dist, keep speed >= v_min if safe
        y_tgt = np.array([-self.safe_margin, dist_now - self.prog_eps,
                         # min(speed_now, self.v_max),
                          -max(speed_now, self.v_min if clearance_now >= self.safe_margin else 0.0)], dtype=np.float32)

        # if first frame: store and return
        if self.prev is None:
            self.prev = {"theta": theta.copy(), "y": y.copy(), "idx": idx}
            return alphas, beta, gamma  # unchanged

        # form Δ
        dtheta = theta - self.prev["theta"]
        dy = y - self.prev["y"]
        self.prev = {"theta": theta.copy(), "y": y.copy(), "idx": idx}

        # if tiny change, skip
        if np.linalg.norm(dtheta) < 1e-6:
            return alphas, beta, gamma

        # rank-1 secant J update: J ≈ dy dtheta^T / (||dtheta||^2 + eps)
        denom = float(np.dot(dtheta, dtheta) + 1e-8)
        J_new = np.outer(dy, dtheta) / denom  # (3, D)
        # keep a running EMA to smooth noise
        self.J = (self.J * self.ema + J_new * (1.0 - self.ema)) if self.J is not None else J_new

        # desired Δy
        dy_des = (y_tgt - y)  # move current y toward target y

        # small LS step on parameters: minimize ||J Δθ - dy_des||_2
        # Δθ* = argmin ||J Δθ - dy_des|| -> normal eq: (J^T J) Δθ = J^T dy_des
        JTJ = self.J.T @ self.J + 1e-6 * np.eye(self.J.shape[1], dtype=np.float32)
        rhs = self.J.T @ dy_des
        try:
            dtheta_star = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            dtheta_star = np.linalg.lstsq(JTJ, rhs, rcond=None)[0]

        # apply per-head learning rates and nonnegativity
        dtheta_star[0] *= self.lr_b  # beta
        dtheta_star[1] *= self.lr_g  # gamma
        if dtheta_star.shape[0] > 2:
            dtheta_star[2:] *= self.lr_a  # alphas

        theta_new = np.clip(theta + dtheta_star, 0.0, np.inf)

        # map back to tensors (blend with model outputs for stability)
        blend = 0.0
        b_new = blend * b + (1.0 - blend) * theta_new[0]
        g_new = blend * g + (1.0 - blend) * theta_new[1]
        if idx.size:
            a_full = alphas.squeeze(0).detach().cpu().numpy()
            a_full[idx] = blend * a_full[idx] + (1.0 - blend) * theta_new[2:]
        else:
            a_full = alphas.squeeze(0).detach().cpu().numpy()

        # back to torch on the right device/dtype
        dev = alphas.device; dtype = alphas.dtype
        al_out = torch.as_tensor(a_full, device=dev, dtype=dtype).unsqueeze(0)
        b_out  = torch.as_tensor([b_new], device=dev, dtype=beta.dtype)
        g_out  = torch.as_tensor([g_new], device=dev, dtype=gamma.dtype)
        return al_out, b_out, g_out


##### Test-Time Finetuner (TTT)

class OnlineFinetuner:
    def __init__(self, model, lr=1e-4, max_steps=2, weight_decay=0.0, prox_lambda=1e-3, head_name_filters=("head","out","proj")):
        self.model = model
        self.max_steps = max_steps
        self.prox_lambda = torch.tensor(prox_lambda, dtype=torch.float64)
        # select a small, stable subset of parameters (final/near-final layers)
        trainable = []
        for n,p in model.named_parameters():
            if any(tag in n for tag in head_name_filters):
                p.requires_grad_(True)
                trainable.append((n,p))
            else:
                p.requires_grad_(False)
        self.params = [p for _,p in trainable]
        self.opt = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)
        # store anchors for proximal regularization
        self._anchors = {id(p): p.detach().clone() for p in self.params}

    def step(self, obs_feats, obs_mask, goal_feats, targets):
        """
        targets = dict(alphas: [1,N], beta: [1], gamma: [1])
        """
        self.model.train()
        losses = {}
        for _ in range(self.max_steps):
            self.opt.zero_grad()
            a_pred, b_pred, g_pred = self.model(obs_feats, obs_mask, goal_feats)
            L_a = torch.nn.functional.mse_loss(a_pred, targets["alphas"]) if a_pred.numel() else a_pred.sum()*0
            L_b = torch.nn.functional.mse_loss(b_pred, targets["beta"])
            L_g = torch.nn.functional.mse_loss(g_pred, targets["gamma"])
            # proximal anchor: keep heads near checkpoint
            L_prox = 0.0
            for p in self.params:
                L_prox = L_prox + (p - self._anchors[id(p)]).pow(2).mean()
            L = L_a + L_b + L_g + self.prox_lambda * L_prox
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            losses = {"L": float(L.item()), "La": float(L_a.item() if a_pred.numel() else 0.0),
                      "Lb": float(L_b.item()), "Lg": float(L_g.item()), "Lprox": float((self.prox_lambda*L_prox).item())}
        self.model.eval()
        return losses

# -----------------------------
# Helpers for local features (match training features)
# -----------------------------

def ttt_rollout_loss(model, obs_feats_t, obs_mask_t, goal_feats_t,
                     o_t, v_t, goal_t, C_t, R_t, mask_t, d_hat_t, dt_t,
                     o_tp1_obs, v_tp1_obs,
                     integrate_surrogate, robot_radius=None, margin_factor=0.5):
    """
    Builds loss: ||ô - o_{t+1}||^2 + 0.25||v̂ - v_{t+1}||^2 using a single surrogate step.
    All tensors are shaped for B=1 (batch size 1).
    """
    model.train()
    a, b, g = model(obs_feats_t, obs_mask_t, goal_feats_t)
    o_hat, v_hat, _ = integrate_surrogate(
        o_t, v_t, goal_t, C_t, R_t, mask_t, a, b, g, d_hat_t, dt_t, H=torch.tensor([1], device=o_t.device),
        robot_radius=(robot_radius if robot_radius is not None else 0.0),
        margin_factor=margin_factor
    )
    L = torch.nn.functional.mse_loss(o_hat, o_tp1_obs) + 0.25 * torch.nn.functional.mse_loss(v_hat, v_tp1_obs)
    return L, {"La": float(a.mean().item() if a.numel() else 0.0)}

def ttt_constraint_loss(model, obs_feats_t, obs_mask_t, goal_feats_t,
                     o_t, v_t, goal_t, C_t, R_t, mask_t, d_hat_t, dt_t,
                     integrate_surrogate, robot_radius=None, margin_factor=0.5, 
                     v_max=1.0, lambda_v=None,):
    model.train()
    a, b, g = model(obs_feats_t, obs_mask_t, goal_feats_t)
    o_hat, v_hat, clr = integrate_surrogate(
        o_t, v_t, goal_t, C_t, R_t, mask_t, a, b, g, d_hat_t, dt_t, H=torch.tensor([1], device=o_t.device),
        robot_radius=(robot_radius if robot_radius is not None else 0.0),
        margin_factor=margin_factor
    )
    speed_next = torch.linalg.norm(v_hat)
    g = torch.nn.functional.relu(speed_next - v_max)
    safe_margin = margin_factor * robot_radius if robot_radius is not None else 1.0
    # optional: only enforce when safely far from obstacles
    safe_gate = (clr <= safe_margin).float()
    g = safe_gate * g

    # augmented Lagrangian term
    L_speed = 0.5 * g.pow(2)
    # proximal-to-checkpoint on head weights
    return L_speed 

def build_local_feats(o_w: np.ndarray, goal_w: np.ndarray, C_w: np.ndarray, R_w: np.ndarray, W_w: np.ndarray):
    """Return (obs_feats[B=1,N,6], goal_feats[B=1,4]) with training-compatible semantics:
    obs_feats: [cx, cy, r, w, dx_goal, dy_goal]
    goal_feats: [dgx, dgy, ||dg||, 1]
    """
    o = torch.as_tensor(o_w, dtype=torch.float32)
    g = torch.as_tensor(goal_w, dtype=torch.float32)
    C = torch.as_tensor(C_w, dtype=torch.float32) if C_w.size else torch.zeros(0,2)
    R = torch.as_tensor(R_w, dtype=torch.float32) if R_w.size else torch.zeros(0)
    W = torch.as_tensor(W_w, dtype=torch.float32) if W_w.size else torch.zeros(0)
    if C.ndim == 1:
        C = C.reshape(0,2)
    dg = (g - o)
    gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)  # [1,4]
    if C.shape[0] == 0:
        obs_feats = torch.zeros(1,0,6)
    else:
        dxdy = (g.unsqueeze(0) - C)  # (N,2)
        obs_feats = torch.cat([C, R.unsqueeze(-1), W.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)  # [1,N,6]
    return obs_feats, goal_feats

# -----------------------------
# α-mapping utilities
# -----------------------------

def map_alpha_to_world(W_in: np.ndarray, R_in: np.ndarray, alphas: np.ndarray, mode: str, k_rad: float = 0.05):
    al = np.maximum(alphas, 0.0)
    W_out = W_in.copy()
    R_out = R_in.copy()
    if mode in ("weight", "both"):
        W_out = W_out * (al if al.size else 1.0)
    if mode in ("radius", "both"):
        R_out = R_out + k_rad * (al if al.size else 0.0)
    return W_out, R_out

# -----------------------------
# Main rollout
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Visual evaluation for CoefEnergyNet")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--case", type=str, default="case1-tight")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--alpha_mode", type=str, default="weight", choices=["weight","radius","both","none"]) 
    ap.add_argument("--seed", type=int, default=2312)
    ap.add_argument("--correction", action='store_true')
    ap.add_argument("--online_finetune", action='store_true')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = CoefEnergyNet().to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    gen.set_all_seeds(args.seed)
    # World & planner
    cfg = gen.GenCfg(); cfg.seed = args.seed; 

    cfg.start = np.array([-1.0, 2.0], float)
    cfg.goal  = np.array([ 9.0,  -0.9], float)

    cfg.d_hat = getattr(cfg, "d_hat", 0.5)
    if args.case.startswith("case1"):
        C,R,W = gen.sample_obstacles_case1_tight(cfg)
    else:
        # Fallback to a default sampler
        C,R,W = gen.sample_obstacles_case2_harder(cfg)
    world = ssi.WorldObstacles(C, R, W, d_hat=cfg.d_hat)
    # planner = gen.planner_from_cfg(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius)
    planner = gen.planner_from_cfg(cfg, world, cfg.k_bulk, cfg.gamma_s, cfg.d_hat, cfg.radius)

    # Snapshots folder
    snap_parent_dir = f"snaps_coef/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(snap_parent_dir, exist_ok=True)
    snap_dir = os.path.join(snap_parent_dir, os.path.basename(args.ckpt).split('.')[0])
    os.makedirs(snap_dir, exist_ok=True)

    # --- rollout
    dt = cfg.dt; T = int(args.steps)
    frames_png = []
    frames_meta = []

    # capture initial frame
    def capture_frame(sys):
        entry = {}
        entry["center"] = sys.o.detach().cpu().to(torch.float32)
        entry["theta"]  = (sys.theta.detach().cpu().to(torch.float32)
                            if hasattr(sys, "theta") else torch.tensor(0.0))
        if hasattr(sys, "Pw") and sys.Pw is not None:
            entry["Pw"] = sys.Pw.detach().cpu().to(torch.float32)
        elif hasattr(sys, "Ploc") and sys.Ploc is not None:
            o = entry["center"]; theta = float(entry["theta"]) if torch.is_tensor(entry["theta"]) else entry["theta"]
            c,s = math.cos(theta), math.sin(theta)
            Rm = torch.tensor([[c,-s],[s,c]], dtype=torch.float32)
            entry["Pw"] = (sys.Ploc.detach().cpu().to(torch.float32) @ Rm.T) + o
        else:
            entry["Pw"] = sys.world_points()
        return entry

    frames_meta.append(capture_frame(planner.sys))
    reinitialize = True
    new_stage_timer = 0
    # if args.correction:
    #     controller = HistSecantController(k_alpha=1, lr_beta=1.0, lr_gamma=1.0, lr_alpha=1.0, safe_margin=0.5*getattr(cfg,"radius",0.16), prog_eps=0.1, v_min=0.05, v_max=5.0, ema=0.9)
    if args.online_finetune:
        finetuner = OnlineFinetuner(model, lr=1e-4, max_steps=1, prox_lambda=1e-3)

    for t in range(T):
        sys = planner.sys
        o_w = sys.o.detach().cpu().numpy()
        Cw, Rw, Ww = planner.stage_slice(world.C_np, world.R_np, world.W_np)
        # build features
        obs_feats, goal_feats = build_local_feats(o_w, cfg.goal, Cw, Rw, Ww)
        obs_mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device) if obs_feats.shape[1] else torch.zeros(1,0, dtype=torch.bool, device=device)


        # predict coefficients
        with torch.no_grad():
            alphas, beta, gamma = model(obs_feats.to(device), obs_mask, goal_feats.to(device))
        # numpy for world update
        print(t, alphas, beta, gamma)
        # correction
        # compute current metrics (cheap)
        
        if args.correction and new_stage_timer >= 5:
            if reinitialize:
                controller = HistSecantController(k_alpha=1, lr_beta=0.25, lr_gamma=0.05, lr_alpha=0.4, safe_margin=0.3*getattr(cfg,"radius",0.16), prog_eps=0.02, v_min=0.05, v_max=1.0, ema=0.99)
                reinitialize = False
                
            dist_now = np.linalg.norm(o_w - cfg.goal)
            speed_now = float(np.linalg.norm(planner.sys.v_o.detach().cpu().numpy())) if hasattr(planner.sys, "v_o") else 0.0
            # quick min clearance from current slice
            clr_now = np.inf
            if Cw.shape[0] > 0:
                clr_now = float(np.min(np.linalg.norm(o_w[None,:] - Cw, axis=1) - Rw))

            # create once (outside loop)
            # controller = HistSecantController(k_alpha=2, safe_margin=0.08, v_min=0.25)

            # update params without extra sims
            alphas_use, beta_use, gamma_use = controller.update(
                alphas, beta, gamma, o_w, speed_now, cfg.goal, Cw, Rw, Ww, clr_now, dist_now, speed_now
            )
            print(t, alphas_use, beta_use, gamma_use)
            al_np = alphas_use.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
            beta_f = float(beta_use.squeeze(0).item())
            gamma_f = float(gamma_use.squeeze(0).item())
        else:
            al_np = alphas.squeeze(0).detach().cpu().numpy() if obs_feats.shape[1] else np.zeros_like(Rw)
            beta_f = float(beta.squeeze(0).item())
            gamma_f = float(gamma.squeeze(0).item())
        # map α
        W_adj, R_adj = Ww.copy(), Rw.copy()
        if args.alpha_mode != "none":
            W_adj, R_adj = map_alpha_to_world(Ww, Rw, al_np, args.alpha_mode)
            # inject goal weight and damping
            planner.stage_field.w_goal = max(0.0, beta_f)
            planner.sys.gamma_o = max(0.0, gamma_f)
        # rebuild world slice with adjusted W/R
        world_step = ssi.WorldObstacles(Cw, R_adj, W_adj, d_hat=cfg.d_hat)
        
        # advance
        stage_idx = planner.sm.current_stage_idx
        new_stage_timer += 1
        o_t      = torch.as_tensor(frames_meta[-1]["center"], device=device).unsqueeze(0)
        v_t      = (planner.sys.v_o.detach().to(device).unsqueeze(0)
                        if hasattr(planner.sys, "v_o") else torch.zeros(1,2, device=device))
        info = planner.step(dt, world_step)
        if args.online_finetune:
                        
            # Build single-frame tensors from your recorded previous frame (frames_meta[-1]) and the current observed next state
            # o_prev, v_prev from planner.sys (stored in frames_meta[-1]); o_next from current info["center"]; v_next from planner.sys.v_o
            param_dtype = next(model.parameters()).dtype  
            # AFTER the step
            #o_tp1 = torch.as_tensor(info["center"], device=device, dtype=param_dtype).unsqueeze(0)
            #v_tp1 = planner.sys.v_o.detach().to(device, dtype=param_dtype).unsqueeze(0) if hasattr(planner.sys, "v_o") else torch.zeros(1,2, device=device, dtype=param_dtype)

            # Pack world slice as 1xN tensors
            C_t  = torch.as_tensor(Cw, device=device, dtype=param_dtype).unsqueeze(0)
            R_t  = torch.as_tensor(Rw, device=device, dtype=param_dtype).unsqueeze(0)
            mask_t = torch.ones(1, Cw.shape[0], dtype=torch.bool, device=device) if Cw.shape[0] else torch.zeros(1,0,dtype=torch.bool,device=device)
            goal_t  = torch.as_tensor(cfg.goal, device=device, dtype=param_dtype).unsqueeze(0)
            d_hat_t = torch.as_tensor([cfg.d_hat], device=device, dtype=param_dtype)
            dt_t    = torch.as_tensor([cfg.dt], device=device, dtype=param_dtype)

            # One tiny update step (prox-regularized)
            finetuner.model.train()
            finetuner.opt.zero_grad()
            L_speed = ttt_constraint_loss(finetuner.model, obs_feats.to(device), obs_mask, goal_feats.to(device),
                                        o_t, v_t, goal_t, C_t, R_t, mask_t, d_hat_t, dt_t,
                                        integrate_surrogate=integrate_surrogate_v2,  # from surrogate_robust
                                        robot_radius=torch.as_tensor([getattr(cfg,"radius",0.0)],device=device),
                                        margin_factor=0.5)
            # add proximal anchor
            L_proxW = torch.zeros((), device=device, dtype=param_dtype)
            for p in finetuner.params:
                L_proxW = L_proxW + (p - finetuner._anchors[id(p)].to(p.device, dtype=p.dtype)).pow(2).mean()

            # total loss: ONLY speed constraint + prox
            L_total = L_speed + finetuner.prox_lambda * L_proxW
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(finetuner.params, 1.0)
            finetuner.opt.step()
            finetuner.model.eval()
                    
        if stage_idx != planner.sm.current_stage_idx:
            reinitialize = True
            new_stage_timer = 0

        
        # snapshot every k steps
        if t % 3 == 0 or t == T-1:
            png_path = os.path.join(snap_dir, f"frame_{t:04d}.png")
            gen.save_episode_snapshot(png_path, planner, frames_meta, world_step, cfg.start, cfg.goal, cfg)
            frames_png.append(png_path)
        frames_meta.append(capture_frame(planner.sys))

        if np.linalg.norm(np.asarray(info["center"], float) - cfg.goal) <= cfg.goal_tol:
            break

    # encode GIF & MP4
    pngs = sorted(glob.glob(os.path.join(snap_dir, "frame_*.png")))
    if pngs:
        gif_path = os.path.join(snap_dir, "rollout.gif")
        mp4_path = os.path.join(snap_dir, "rollout.mp4")
        frames = [iio.imread(p) for p in pngs]
        iio.imwrite(gif_path, frames, loop=0, fps=args.fps)
        try:
            import imageio
            with imageio.get_writer(mp4_path, format="FFMPEG", mode="I", fps=args.fps, codec="libx264", quality=7) as w:
                for fr in frames: w.append_data(fr)
        except Exception as e:
            print("MP4 writer not available:", e)
        print({"gif": gif_path, "mp4": mp4_path})
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()
