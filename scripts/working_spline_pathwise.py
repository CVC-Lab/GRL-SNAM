# stagewise_spline_icp_with_stage_field.py
# Deformable periodic B-spline with ICP barrier + stage force field (goal+radial+tangent+flow+boundary)
# Produces a snapshot (and optional GIF) showing start/mid/end.

from __future__ import annotations
import os, math, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- utils ---------------------------

def rot2(theta: torch.Tensor):
    c = torch.cos(theta); s = torch.sin(theta)
    return torch.stack([torch.stack([c,-s],-1), torch.stack([s,c],-1)], -2)

# --------------------------- stages (numpy) ---------------------------

@dataclass
class Stage:
    center: np.ndarray
    size: Tuple[float,float]
    entry_point: np.ndarray
    exit_point: np.ndarray
    stage_id: int
    def __post_init__(self):
        W,H = self.size; cx,cy = self.center
        self.bounds = (cx-W/2, cx+W/2, cy-H/2, cy+H/2)

class StageManager:
    def __init__(self, start_xy: np.ndarray, goal_xy: np.ndarray, stage_size=(2.6,2.0), overlap_ratio=0.3):
        self.start = np.array(start_xy,float); self.goal = np.array(goal_xy,float)
        self.stage_size = tuple(stage_size); self.overlap_ratio = float(overlap_ratio)
        self.stages: List[Stage] = []; self.current_stage_idx = 0; self._build()
    def _build(self):
        d = self.goal - self.start; L = np.linalg.norm(d)
        if L < 1e-6:
            self.stages = [Stage((self.start+self.goal)/2,self.stage_size,self.start.copy(),self.goal.copy(),0)]
            return
        u = d / L; step = self.stage_size[0]*(1-self.overlap_ratio); n=max(1,int(np.ceil(L/step)))
        for i in range(n):
            t = 0.0 if n==1 else i/(n-1)
            c = self.start + t*d
            entry = self.start if i==0 else self.stages[-1].exit_point
            exitp = self.goal if i==n-1 else (self.start + (i+1)/n*d)
            self.stages.append(Stage(c,self.stage_size,entry,exitp,i))
    def current(self)->Stage: return self.stages[self.current_stage_idx]
    def advance_if_needed(self, robot_xy: np.ndarray, thresh=0.45)->bool:
        st = self.current()
        if np.linalg.norm(robot_xy - st.exit_point) < thresh and self.current_stage_idx < len(self.stages)-1:
            self.current_stage_idx += 1; return True
        if self.current_stage_idx < len(self.stages)-1:
            nxt = self.stages[self.current_stage_idx+1]; path_dir = self.goal - self.start
            if np.dot(robot_xy - st.center, path_dir) > np.dot(nxt.center - st.center, path_dir):
                self.current_stage_idx += 1; return True
        return False

# --------------------------- obstacles & ICP (torch) ---------------------------

class ObstacleProviderTorch:
    def __init__(self, centers, radii, weights=None, device="cpu", dtype=torch.float32, eps=1e-12):
        self.C = torch.as_tensor(centers, dtype=dtype, device=device)  # (C,2)
        self.R = torch.as_tensor(radii,   dtype=dtype, device=device)  # (C,)
        if weights is None: weights = torch.ones_like(self.R)
        self.W = torch.as_tensor(weights, dtype=dtype, device=device)
        self.eps = float(eps)
    def compute_all(self, q: torch.Tensor):
        diff = q.unsqueeze(-2) - self.C              # (...,C,2)
        r = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-12)
        d = r.squeeze(-1) - self.R                   # (...,C)
        g = diff / r                                  # (...,C,2)
        return d, g

class IPCBarrier:
    def __init__(self, obstacles: ObstacleProviderTorch, d_hat=1.0, violation_penalty=5e2, eps=1e-12):
        self.obs = obstacles; self.d_hat=float(d_hat); self.vp=float(violation_penalty); self.eps=float(eps)
    def _piecewise(self, d: torch.Tensor):
        dh = d.new_tensor(self.d_hat); vp = d.new_tensor(self.vp); safe = torch.clamp(d, min=self.eps)
        b_in = -(d-dh)**2 * torch.log(safe/dh)
        F_in = (dh-d)*(2.0*torch.log(safe/dh) - dh/safe) + 1.0
        b = torch.where(d<=self.eps, vp, torch.where(d<dh, b_in, torch.zeros_like(d)))
        F = torch.where(d<=self.eps, vp, torch.where(d<dh, F_in, torch.zeros_like(d)))
        return b,F
    def barrier_and_grad(self, q: torch.Tensor):
        d,g = self.obs.compute_all(q); b,F = self._piecewise(d)
        return b.sum(-1), (F.unsqueeze(-1)*g).sum(-2)

# --------------------------- periodic cubic B-spline ---------------------------

def periodic_cubic_uniform_matrices(n_ctrl:int, n_samples:int, device="cpu", dtype=torch.float32):
    K=n_samples
    t_vals = torch.linspace(0.0, float(n_ctrl), K+1, device=device, dtype=dtype)[:-1]
    k0 = torch.floor(t_vals).long(); u=(t_vals-k0.to(dtype)); u2=u*u; u3=u2*u
    w_m1=(1-3*u+3*u2-u3)/6.0; w_0=(4-6*u2+3*u3)/6.0; w_p1=(1+3*u+3*u2-3*u3)/6.0; w_p2=u3/6.0
    dw_m1=(-3+6*u-3*u2)/6.0; dw_0=(-12*u+9*u2)/6.0; dw_p1=(3+6*u-9*u2)/6.0; dw_p2=(3*u2)/6.0
    B=torch.zeros(K,n_ctrl,device=device,dtype=dtype); D=torch.zeros_like(B)
    i_m1=(k0-1)%n_ctrl; i_0=k0%n_ctrl; i_p1=(k0+1)%n_ctrl; i_p2=(k0+2)%n_ctrl; ar=torch.arange(K,device=device)
    for idx,wt,dw in [(i_m1,w_m1,dw_m1),(i_0,w_0,dw_0),(i_p1,w_p1,dw_p1),(i_p2,w_p2,dw_p2)]:
        B[ar,idx]+=wt; D[ar,idx]+=dw
    return B,D

# ---- polygon area and gradient wrt samples ----
def polygon_area_and_grad(X: torch.Tensor):
    x,y = X[:,0], X[:,1]
    x_next = torch.roll(x,-1,0); y_next = torch.roll(y,-1,0)
    A = 0.5*torch.sum(x*y_next - y*x_next)
    y_prev = torch.roll(y,1,0); x_prev = torch.roll(x,1,0)
    dA_dx = 0.5*(y_next - y_prev); dA_dy = 0.5*(x_prev - x_next)
    return A, torch.stack([dA_dx,dA_dy],-1)

# --------------------------- Stage force field (torch) ---------------------------

class StageForceFieldTorch:
    """
    Torch reimplementation of the SE(2) stage force field.
    Returns an (K,2) vector field evaluated at the curve samples X.
    """
    def __init__(self, r_safe=0.4, r_contact=0.2,
                 w_goal=1.0, w_radial=2.0, w_tangential=0.5,
                 w_flow=1.0, w_boundary=0.5):
        self.r_safe=float(r_safe); self.r_contact=float(r_contact)
        self.w_goal=float(w_goal); self.w_radial=float(w_radial)
        self.w_tangential=float(w_tangential); self.w_flow=float(w_flow)
        self.w_boundary=float(w_boundary)
        self.J = torch.tensor([[0.,-1.],[1.,0.]], dtype=torch.float32)

    # ---- individual fields ----
    def _goal(self, X, target):
        # (K,2)
        v = target.unsqueeze(0) - X
        n = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
        return v / n

    def _radial(self, X, C, R):
        # Summed over obstacles -> (K,2)
        if C.numel() == 0: return torch.zeros_like(X)
        K, Cn = X.shape[0], C.shape[0]
        diff = X.unsqueeze(1) - C.unsqueeze(0)                          # (K,C,2)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)         # (K,C)
        d = dist - R.unsqueeze(0)                                       # (K,C)
        # very strong contact push
        contact = (d <= self.r_contact).unsqueeze(-1)                   # (K,C,1)
        strong = (diff / dist.unsqueeze(-1)) * contact * 1e4            # (K,C,2)
        # ICP-like gradient in [r_contact, r_safe]
        d_ref = X.new_tensor(self.r_safe)
        in_band = ((d > self.r_contact) & (d < d_ref)).unsqueeze(-1)    # (K,C,1)
        db_dd = -(2*(d - d_ref)*torch.log(d.clamp_min(1e-8)/d_ref)
                 + (d - d_ref)**2 / d.clamp_min(1e-8))                  # (K,C)
        grad = db_dd.unsqueeze(-1) * (diff / dist.unsqueeze(-1))        # (K,C,2)
        grad = torch.where(in_band, grad, torch.zeros_like(grad))       # (K,C,2)
        return (strong + grad).sum(dim=1)                               # (K,2)

    def _tangent(self, X, target, C, R):
        if C.numel() == 0: return torch.zeros_like(X)
        K, Cn = X.shape[0], C.shape[0]
        diff = X.unsqueeze(1) - C.unsqueeze(0)                          # (K,C,2)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)         # (K,C)
        d = dist - R.unsqueeze(0)                                       # (K,C)
        n_hat = diff / dist.unsqueeze(-1)                                # (K,C,2)
        # rotate by +90°
        J = self.J.to(device=X.device, dtype=X.dtype)
        t_hat = (J @ n_hat.transpose(-1, -2)).transpose(-1, -2)         # (K,C,2)

        # goal alignment sign
        gdir = (target.unsqueeze(0) - X)                                 # (K,2)
        gdir = gdir / torch.linalg.norm(gdir, dim=1, keepdim=True).clamp_min(1e-8)
        gdir = gdir.unsqueeze(1).expand(-1, Cn, -1)                      # (K,C,2)
        sign = torch.sign((t_hat * gdir).sum(dim=-1, keepdim=True))      # (K,C,1)

        # strength
        inner = torch.clamp(self.r_safe - d, min=0.0) / self.r_safe      # (K,C)
        strength = torch.where(d < self.r_safe, inner, X.new_tensor(0.1))# (K,C)
        strength = strength.unsqueeze(-1)                                 # (K,C,1)
        influence = (d < (self.r_safe + 0.5)).unsqueeze(-1).float()      # (K,C,1)

        F = sign * strength * t_hat * influence                           # (K,C,2)
        return F.sum(dim=1)                                               # (K,2)

    def _flow(self, X, target, C, R):
        if C.numel() == 0: return torch.zeros_like(X)
        K, Cn = X.shape[0], C.shape[0]
        diff = X.unsqueeze(1) - C.unsqueeze(0)                           # (K,C,2)
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)          # (K,C)
        d = dist - R.unsqueeze(0)                                        # (K,C)
        n_hat = diff / dist.unsqueeze(-1)                                 # (K,C,2)
        J = self.J.to(device=X.device, dtype=X.dtype)
        t_hat = (J @ n_hat.transpose(-1, -2)).transpose(-1, -2)          # (K,C,2)

        obs_to_goal = (target - C)                                       # (C,2)
        goal_unit = obs_to_goal / torch.linalg.norm(obs_to_goal, dim=-1, keepdim=True).clamp_min(1e-8)  # (C,2)
        goal_unit = goal_unit.unsqueeze(0).expand(K, -1, -1)             # (K,C,2)
        sign = torch.sign((t_hat * goal_unit).sum(dim=-1, keepdim=True)) # (K,C,1)
        flow_dir = sign * t_hat                                          # (K,C,2)

        # blend a small radial component when far
        far = (d > self.r_safe).float().unsqueeze(-1)                    # (K,C,1)
        radial_comp = goal_unit                                          # (K,C,2)
        flow_dir = 0.7*flow_dir + 0.3*far*radial_comp                    # (K,C,2)

        flow_strength = torch.where(d < self.r_contact, X.new_tensor(1.0),
                             torch.where(d < self.r_safe, X.new_tensor(0.8),
                                         torch.clamp(0.3*(1.0-(d-self.r_safe)), min=0.0)))  # (K,C)
        F = flow_dir * flow_strength.unsqueeze(-1)                       # (K,C,2)
        return F.sum(dim=1)                                              # (K,2)

    def _boundary(self, X, bounds):
        xmin,xmax,ymin,ymax = [X.new_tensor(v) for v in bounds]
        m = X.new_tensor(0.5)
        F = torch.zeros_like(X)                                          # (K,2)
        left  = X[:,0] < (xmin+m); right = X[:,0] > (xmax-m)
        bot   = X[:,1] < (ymin+m); top   = X[:,1] > (ymax-m)
        F[left,0]  += ((xmin+m) - X[left,0]) / m * 2.0
        F[right,0] += (-(X[right,0] - (xmax-m)) / m) * 2.0
        F[bot,1]   += ((ymin+m) - X[bot,1]) / m * 2.0
        F[top,1]   += (-(X[top,1] - (ymax-m)) / m) * 2.0
        return F

    # ---- full field ----
    def field(self, X: torch.Tensor, target: torch.Tensor, bounds, C: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # Everything reduced to (K,2) before the sum
        Fg = self._goal(X, target)                      # (K,2)
        Fr = self._radial(X, C, R)                      # (K,2)
        Ft = self._tangent(X, target, C, R)             # (K,2)
        Ff = self._flow(X, target, C, R)                # (K,2)
        Fb = self._boundary(X, bounds)                  # (K,2)
        return Fg
        # return (self.w_goal*Fg + self.w_radial*Fr + self.w_tangential*Ft
        #         + self.w_flow*Ff + self.w_boundary*Fb)  # (K,2)

# --------------------------- deformable system ---------------------------

class DeformableSplineSystem:
    def __init__(self, n_ctrl=20, K=220, device="cpu", dtype=torch.float32,
                 lam_reg=20.0, mu=0.25, d_hat=1.0,
                 M=0.1, gamma=4.0, I=0.6, gamma_th=1.6, M_o=1.5, gamma_o=4.0,
                 seed=7, radius=0.25):
        torch.manual_seed(seed); self.device=device; self.dtype=dtype
        self.B,self.D = periodic_cubic_uniform_matrices(n_ctrl,K,device,dtype)
        self.w = torch.full((K,), 1.0/float(K), dtype=dtype, device=device)
        ang = torch.linspace(0,2*math.pi,n_ctrl+1,device=device,dtype=dtype)[:-1]
        self.P0_loc = radius * torch.stack([torch.cos(ang), torch.sin(ang)],-1)
        self.Ploc = self.P0_loc.clone(); self.Vloc = torch.zeros_like(self.Ploc)
        self.theta = torch.tensor(0.0,dtype=dtype,device=device); self.omega = torch.tensor(0.0,dtype=dtype,device=device)
        self.o = torch.tensor([0.0,0.0],dtype=dtype,device=device); self.v_o = torch.tensor([0.0,0.0],dtype=dtype,device=device)
        self.lam_reg=float(lam_reg); self.mu=float(mu); self.d_hat=float(d_hat)
        self.M=float(M); self.gamma=float(gamma); self.I=float(I); self.gamma_th=float(gamma_th)
        self.M_o=float(M_o); self.gamma_o=float(gamma_o)
        self.J = torch.tensor([[0.,-1.],[1.,0.]], dtype=dtype, device=device)
        self._prev_o = self.o.clone(); self._no_prog = 0; self._stuck=False; self.A0=None

    def world_points(self):
        R = rot2(self.theta); return (self.Ploc @ R.T) + self.o

    def sample_curve(self):
        Pw = self.world_points(); X = self.B @ Pw; Xp = self.D @ Pw
        ell = Xp.norm(dim=1).clamp_min(1e-8); T_hat = Xp/ell.unsqueeze(-1)
        return Pw,X,Xp,ell,T_hat

    def step(
        self,
        dt: float,
        obstacle_provider: ObstacleProviderTorch,
        barrier: IPCBarrier,
        stage_force,                # StageForceFieldTorch
        stage_bounds,               # (xmin, xmax, ymin, ymax)
        target_xy: Tuple[float, float]
    ):
        # --- samples and tangents ---
        Pw, X, Xp, ell, T_hat = self.sample_curve()
        w = self.w
        R = rot2(self.theta)
        J = self.J.to(device=X.device, dtype=X.dtype)
        target = torch.as_tensor(target_xy, dtype=self.dtype, device=self.device)

        # --- ICP barrier (conservative) ---
        bsum, g_sum = barrier.barrier_and_grad(X)                  # (K,), (K,2)
        term1 = (w * ell).unsqueeze(-1) * g_sum                    # (K,2)
        term2 = (w * bsum).unsqueeze(-1) * T_hat                   # (K,2)
        gradP_world = self.B.T @ term1 + self.D.T @ term2          # (n_ctrl,2)

        # --- Adaptive area control (conservative) ---
        A, dA_dX = polygon_area_and_grad(X)                        # scalar, (K,2)
        d_all, _ = obstacle_provider.compute_all(X)                # (K,C)
        min_d_each = d_all.min(dim=1).values
        min_clear = min_d_each.min()

        # Robust A0 init (handles missing or None or non-finite)
        A0_val = getattr(self, "A0", None)
        if (A0_val is None) or (not np.isfinite(A0_val)):
            self.A0 = float(torch.abs(A.detach())) + 1e-6
        A0_t = X.new_tensor(self.A0)

        alpha, beta = 0.25, 2.5
        squeeze_factor = alpha + (1 - alpha) * torch.tanh(beta * torch.clamp(min_clear, min=0.0))
        A_target = squeeze_factor * A0_t

        k_area = 1.5
        gradX_area = 2.0 * k_area * (A - A_target) * dA_dX / X.shape[0]     # (K,2)
        gradP_world = gradP_world + self.B.T @ (w.unsqueeze(-1) * gradX_area)

        # --- chain to local + shape regularizer ---
        gradP_loc = (gradP_world @ R) + self.lam_reg * (self.Ploc - self.P0_loc)
        F_loc_cons = -gradP_loc

        # --- conservative rigid forces (barrier + area) ---
        X_minus_o = X - self.o.unsqueeze(0)
        dU_do = torch.sum(term1, dim=0) + torch.sum(w.unsqueeze(-1) * gradX_area, dim=0)   # (2,)
        F_o_cons = -dU_do
        gJx = torch.einsum("kd,dd,kd->k", g_sum, J, X_minus_o)
        dU_dtheta = torch.sum(w * ell * gJx)
        g_area_J = torch.einsum("kd,dd,kd->k", gradX_area, J, X_minus_o)
        dU_dtheta = dU_dtheta + torch.sum(w * g_area_J)
        tau_cons = -dU_dtheta

        # --- contact friction (dissipative) ---
        dists, _ = obstacle_provider.compute_all(X)                # (K,C)
        min_d = dists.min(dim=1).values                            # (K,)
        contact_mask = (min_d < barrier.d_hat).to(X.dtype)
        Pdot_world = (self.Vloc @ R.T) + self.omega * ((self.Ploc @ R.T) @ J.T) + self.v_o
        Xdot = self.B @ Pdot_world                                  # (K,2)
        v_t = torch.sum(Xdot * T_hat, dim=1)                        # (K,)
        p_k = g_sum.norm(dim=1)                                     # (K,)
        f_fric = -self.mu * (p_k * v_t * contact_mask).unsqueeze(-1) * T_hat  # (K,2)
        F_world_from_fric = self.B.T @ (w.unsqueeze(-1) * f_fric)   # (n_ctrl,2)
        F_loc_from_fric   = F_world_from_fric @ R.T                 # (n_ctrl,2)
        F_o_from_fric     = torch.sum(w.unsqueeze(-1) * f_fric, dim=0)        # (2,)
        tau_fric          = torch.sum(w * torch.sum(f_fric * (J @ X_minus_o.T).T, dim=1))

        # --- stage force field (non-conservative) in sample space ---
        F_stage = stage_force.field(X, target, stage_bounds, obstacle_provider.C, obstacle_provider.R)  # (K,2)

        F_stage = 1.5 * F_stage

        # Map stage forces to generalized coords
        F_world_from_stage = self.B.T @ (w.unsqueeze(-1) * F_stage)  # (n_ctrl,2)
        F_loc_from_stage   = F_world_from_stage @ R.T                # (n_ctrl,2)
        F_o_from_stage     = torch.sum(w.unsqueeze(-1) * F_stage, dim=0)      # (2,)
        tau_stage          = torch.sum(w * torch.sum(F_stage * (J @ X_minus_o.T).T, dim=1))

        # --- integrate dynamics ---
        F_loc_total = F_loc_cons + F_loc_from_fric + F_loc_from_stage
        self.Vloc = self.Vloc + dt * (F_loc_total / self.M - self.gamma * self.Vloc)
        self.Ploc = self.Ploc + dt * self.Vloc

        tau_total = tau_cons + tau_fric + tau_stage - self.gamma_th * self.omega
        self.omega = self.omega + dt * (tau_total / self.I)
        self.theta = self.theta + dt * self.omega

        F_o_total = F_o_cons + F_o_from_fric + F_o_from_stage - self.gamma_o * self.v_o
        self.v_o = self.v_o + dt * (F_o_total / self.M_o)
        self.o = self.o + dt * self.v_o
        # --- diagnostics ---

        U_barrier = torch.sum(w * bsum * ell)
        U_reg = 0.5 * self.lam_reg * ((self.Ploc - self.P0_loc) ** 2).sum()
        U_area = float((k_area * (A - A_target) ** 2).item())

        return {
            "U_barrier": float(U_barrier.item()),
            "U_reg": float(U_reg.item()),
            "U_area": U_area,
            "center": self.o.detach().cpu().numpy().tolist(),
            "theta": float(self.theta.item()),
            "min_d": float(min_d.min().item()),
        }


# --------------------------- world obstacles holder ---------------------------

class WorldObstacles:
    def __init__(self, centers, radii, weights=None, d_hat=0.28):
        self.C_np=np.asarray(centers,float); self.R_np=np.asarray(radii,float)
        self.W_np=np.asarray(weights if weights is not None else np.ones_like(self.R_np),float)
        self.d_hat=float(d_hat)

# --------------------------- viz helpers ---------------------------

def draw_obstacles(ax, C,R, **kw):
    C=np.asarray(C); R=np.asarray(R)
    for (cx,cy),rr in zip(C,R):
        ax.add_patch(plt.Circle((cx,cy), rr, fill=False, lw=2, **kw))

def render_curve(ax, B, Pw, label=None, color=None):
    Xv=(B @ torch.tensor(Pw)).numpy(); ax.plot(Xv[:,0],Xv[:,1],lw=2,label=label,color=color)

# --------------------------- planner wrapper ---------------------------

class StagewiseSplinePlanner:
    def __init__(self, start_xy, goal_xy, stage_size=(2.5,2.0), overlap=0.3,
                 n_ctrl=20, K=240, d_hat=1.0, lam_reg=20.0, mu=0.25, seed=7):
        self.sm = StageManager(np.array(start_xy), np.array(goal_xy), stage_size, overlap)
        self.sys = DeformableSplineSystem(n_ctrl=n_ctrl, K=K, d_hat=d_hat, lam_reg=lam_reg, mu=mu, seed=seed)
        # initialize pose
        self.sys.o = torch.tensor(start_xy, dtype=self.sys.dtype)
        d = np.array(goal_xy)-np.array(start_xy); self.sys.theta = torch.tensor(math.atan2(d[1],d[0]), dtype=self.sys.dtype)
        self.stage_field = StageForceFieldTorch(r_safe=0.35, r_contact=0.18,
                                                w_goal=1.0, w_radial=2.0, w_tangential=1.4, w_flow=0.9, w_boundary=0.5)

    def _stage_slice(self, C,R,W) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        st = self.sm.current(); xmin,xmax,ymin,ymax = st.bounds
        m = (C[:,0] >= xmin-0.5) & (C[:,0] <= xmax+0.5) & (C[:,1] >= ymin-0.5) & (C[:,1] <= ymax+0.5)
        if np.any(m): return C[m],R[m],W[m]
        return C,R,W

    def step(self, dt, world_obs: WorldObstacles):
        st = self.sm.current(); C,R,W = self._stage_slice(world_obs.C_np, world_obs.R_np, world_obs.W_np)
        obs_t = ObstacleProviderTorch(C,R,W, dtype=self.sys.dtype)
        barrier = IPCBarrier(obs_t, d_hat=world_obs.d_hat)
        info = self.sys.step(dt, obs_t, barrier, self.stage_field, st.bounds, tuple(st.exit_point))
        self.sm.advance_if_needed(np.array(info["center"],float))
        return info

# --------------------------- demo ---------------------------

def run_demo(out_dir="./out_stage_spline", total_steps=900, dt=0.015,
             stage_size=(2.6,2.0), overlap=0.3, snapshot_every=5):
    os.makedirs(out_dir,exist_ok=True)
    # centers=np.array([[1.0,0.5],[2.5,-0.8],[3.2,0.3],[4.0,1.0],[4.8,-0.5],
    #                   [5.5,0.8],[6.2,-0.2],[6.8,0.9],[7.5,-0.7],[8.2,0.4],
    #                   [2.0,1.2],[3.5,-1.3],[5.0,1.5],[6.5,-1.1],[7.8,1.2]],float)
    # radii=np.array([0.40,0.50,0.30,0.60,0.40,0.50,0.30,0.40,0.50,0.30,0.30,0.40,0.30,0.40,0.30],float)
    # weights=np.array([1.0,1.2,0.8,1.5,0.9,1.1,0.7,1.0,1.3,0.8,0.6,1.0,0.7,1.1,0.6],float)
    centers=np.array([[2.5,-0.8],[3.2,0.3],[4.0,1.0],[4.8,-0.5],
                      [5.5,0.8],[6.2,-0.2],[6.8,0.9],[7.5,-0.7],[8.2,0.4],
                      [2.0,1.2],[3.5,-1.3],[5.0,1.5],[6.5,-1.1],[7.8,1.2]],float)
    radii=np.array([0.50,0.30,0.60,0.40,0.50,0.30,0.40,0.50,0.30,0.30,0.40,0.30,0.40,0.30],float)
    weights=np.array([1.2,0.8,1.5,0.9,1.1,0.7,1.0,1.3,0.8,0.6,1.0,0.7,1.1,0.6],float)
    world_obs=WorldObstacles(centers,radii,weights,d_hat=1.0)

    start=np.array([-1.0,-0.5]); goal=np.array([9.0,0.2])
    planner=StagewiseSplinePlanner(start, goal, stage_size=stage_size, overlap=overlap,
                                   n_ctrl=20, K=240, d_hat=1.0, lam_reg=20.0, mu=0.25, seed=7)

    frames=[]; mid=None
    for t in range(total_steps):
        planner.step(dt, world_obs)
        if t%snapshot_every==0 or t in (0,total_steps//2,total_steps-1):
            Pw=planner.sys.world_points().detach().cpu().numpy()
            frames.append(Pw)
            if t==total_steps//2: mid=Pw.copy()

    # snapshot
    fig=plt.figure(figsize=(7.6,3.8),dpi=140); ax=fig.add_subplot(111); ax.set_aspect("equal","box")
    draw_obstacles(ax,centers,radii, ec="k")
    if frames: render_curve(ax, planner.sys.B, frames[0], "start", "#1f77b4")
    if mid is not None: render_curve(ax, planner.sys.B, mid, "mid", "#ff7f0e")
    render_curve(ax, planner.sys.B, frames[-1], "end", "#2ca02c")
    for i,st in enumerate(planner.sm.stages):
        xmin,xmax,ymin,ymax=st.bounds
        ax.add_patch(plt.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, fill=False,
                                   lw=1.6, ec="#00cfdc" if i==planner.sm.current_stage_idx else (0.4,0.6,0.6,0.6)))
        ax.plot([st.entry_point[0]],[st.entry_point[1]],"o",ms=4,color="#85ff85")
        ax.plot([st.exit_point[0]],[st.exit_point[1]],"o",ms=4,color="#ff8585")
    ax.plot([start[0]],[start[1]],"o",color="green",ms=6); ax.plot([goal[0]],[goal[1]],"*",color="gold",ms=10)
    ax.legend(loc="upper left",fontsize=8)
    ax.set_title("Stagewise deformable spline planning: ICP + stage field (goal+radial+tangent+flow) + area")
    ax.set_xlim(-2.5,10.0); ax.set_ylim(-2.4,2.4)
    snap=os.path.join(out_dir,"stagewise_spline_snapshot.png"); fig.savefig(snap,bbox_inches="tight"); plt.close(fig)

    # gif (optional)
    gif=os.path.join(out_dir,"stagewise_spline_motion.gif")
    try:
        import imageio
        imgs=[]
        for Pw in frames:
            fig=plt.figure(figsize=(6.4,3.6),dpi=110); ax=fig.add_subplot(111); ax.set_aspect("equal","box")
            draw_obstacles(ax,centers,radii, ec="k"); render_curve(ax, planner.sys.B, Pw)
            ax.set_xlim(-2.5,10.0); ax.set_ylim(-2.4,2.4); ax.set_xticks([]); ax.set_yticks([])
            fig.canvas.draw()
            buf=np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
            arr=buf.reshape(fig.canvas.get_width_height()[::-1]+(4,))[:,:,1:]; imgs.append(arr); plt.close(fig)
        imageio.mimsave(gif, imgs, fps=max(6,int(1.0/dt)))
    except Exception:
        gif=None

    return {"snapshot":snap, "gif":gif, "stages":len(planner.sm.stages),
            "final_center":planner.sys.o.detach().cpu().numpy().tolist(),
            "final_theta":float(planner.sys.theta.item())}

# --------------------------- CLI ---------------------------

def parse_args():
    ap=argparse.ArgumentParser("Stagewise deformable spline with stage field")
    ap.add_argument("--out",type=str,default="./out_stage_spline")
    ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--dt",type=float,default=0.02)
    ap.add_argument("--stage_w",type=float,default=2.6)
    ap.add_argument("--stage_h",type=float,default=2.0)
    ap.add_argument("--overlap",type=float,default=0.3)
    return ap.parse_args()

if __name__=="__main__":
    args=parse_args()
    out=run_demo(out_dir=args.out, total_steps=args.steps, dt=args.dt,
                 stage_size=(args.stage_w,args.stage_h), overlap=args.overlap)
    print("Outputs:"); print(out)
