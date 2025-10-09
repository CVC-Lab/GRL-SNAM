# energy_data_navmax.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------- small utils

def _as_tensor(x, dtype=torch.float32):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype)

def rot2d(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

def world_to_local(x: torch.Tensor, o: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", R.transpose(-1, -2), x - o)

def make_coord_grid(H: int, W: int, half: float, device="cpu", dtype=torch.float32):
    ys = torch.linspace(-half, half, H, device=device, dtype=dtype)
    xs = torch.linspace(-half, half, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([X, Y], dim=-1)  # [H,W,2] (x,y) in local coords

def signed_distance_to_circle(xy: torch.Tensor, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(xy - c, dim=-1) - r

def barrier_map_from_sd(sd: torch.Tensor, d_hat: float, kind="icp"):
    # Smooth barrier: ~0 when sd >= d_hat; grows as sd ↓ 0 (or <0)
    d_pos = torch.clamp(sd, min=0.0)
    outside = torch.clamp(d_hat - d_pos, min=0.0)
    inside = torch.clamp(-sd, min=0.0)
    if kind == "exp":
        return torch.exp(torch.clamp(d_hat - sd, min=0.0)) - 1.0
    return outside**2 + 5.0 * inside**2

def quadratic_goal(grid_xy: torch.Tensor, g_local: torch.Tensor):
    d = grid_xy - g_local
    return 0.5 * (d[..., 0]**2 + d[..., 1]**2)

@dataclass
class EpisodeRef:
    path: str
    case: str
    idx: int
    success: bool

# ---------- dataset

class NavMaximinLocalPatches(Dataset):
    """
    Schema-aware dataset for nav_dataset_maximin/* generated episodes.

    Each item corresponds to a *snapshot* (an entry in `frames` aligned with `frame_state`).
    Output contains:
        - grid_xy        [H,W,2]   local coords
        - goal_map       [1,H,W]
        - coord_map      [2,H,W]   normalized [-1,1]
        - barrier_stack  [n_local,H,W]  per-obstacle
        - obs_feats      [n_local,6]    (cx,cy,r,||c||,angle,proximity)
        - obs_weights    [n_local]      (for α supervision)
        - goal_feats     [4]            (gx,gy,||g||,angle)
        - pos_xy         [1,2]          agent center in local (for sampling/vis)
        - dir_xy         [1,2]          desired direction (from v_o or FD)
        - meta: episode index, snapshot index, stage idx, bounds
    """
    def __init__(
        self,
        root_or_manifest: str,
        hat_d: Optional[float] = None,
        H: int = 64,
        W: int = 64,
        roi_pad: float = 0.50,          # extra margin on stage bounds when selecting obstacles
        barrier_kind: str = "icp",
        device: str = "cpu",
        dtype = torch.float32,
    ):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.roi_pad = float(roi_pad)
        self.barrier_kind = barrier_kind
        self.device, self.dtype = device, dtype

        # Load manifest
        if os.path.isdir(root_or_manifest):
            man = os.path.join(root_or_manifest, "manifest.json")
            with open(man, "r") as f:
                records = json.load(f)
            self.root = root_or_manifest
        else:
            with open(root_or_manifest, "r") as f:
                records = json.load(f)
            self.root = os.path.dirname(root_or_manifest)

        # collect episode refs
        self.episodes: List[EpisodeRef] = []
        for r in records:
            self.episodes.append(EpisodeRef(path=r["path"], case=r["case"], idx=int(r["idx"]), success=bool(r["success"])))

        # build flat index: (ep_i, k) where k indexes snapshots in frames/frame_state
        self.items: List[Tuple[int, int]] = []
        self.cache: Dict[int, Dict[str, Any]] = {}  # optional lazy cache per-episode
        for ei, ref in enumerate(self.episodes):
            ep = torch.load(ref.path, weights_only=False, map_location="cpu")
            n_k = len(ep.get("frames", []))
            # ensure alignment with frame_state if present
            if "frame_state" in ep:
                n_k = min(n_k, len(ep["frame_state"]))
            for k in range(n_k):
                self.items.append((ei, k))

        # global default d_hat (can be overridden per-episode)
        self.default_hat_d = float(hat_d) if hat_d is not None else 1.0

    def __len__(self): return len(self.items)

    # ----- helpers

    def _get_episode(self, ei: int) -> Dict[str, Any]:
        if ei not in self.cache:
            self.cache[ei] = torch.load(self.episodes[ei].path, weights_only=False, map_location="cpu")
        return self.cache[ei]

    def _stage_bounds_with_pad(self, bounds: List[float]) -> Tuple[float,float,float,float]:
        xmin,xmax,ymin,ymax = bounds
        dx = self.roi_pad * (xmax - xmin)
        dy = self.roi_pad * (ymax - ymin)
        return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

    # ----- dataset API

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | bool]:
        ei, k = self.items[idx]
        ep = self._get_episode(ei)
        obs = ep["obstacles"]; params = ep.get("params", {}); meta = ep.get("meta", {})
        frames = ep.get("frames", [])
        fstate = ep.get("frame_state", [])

        # pose & velocity at snapshot k
        fr = frames[k]
        o_w = _as_tensor(fr.get("o"), self.dtype).flatten()[:2]             # [2]
        theta = float(fr.get("theta", 0.0))
        R_t = rot2d(torch.tensor(theta, dtype=self.dtype))                   # [2,2]
        v_o = fr.get("v_o", None)
        if v_o is not None:
            v_w = _as_tensor(v_o, self.dtype).flatten()[:2]
        else:
            # FD fallback using neighbor frame center
            if k+1 < len(frames):
                c_next = _as_tensor(frames[k+1]["center"], self.dtype).flatten()[:2]
                c_now  = _as_tensor(fr["center"], self.dtype).flatten()[:2]
            else:
                c_next = _as_tensor(fr["center"], self.dtype).flatten()[:2]
                c_now  = _as_tensor(frames[max(0,k-1)]["center"], self.dtype).flatten()[:2]
            v_w = (c_next - c_now)

        # goal from meta; fallback to params or final_center
        goal = None
        for cont, key in [(meta, "goal"), (params, "goal")]:
            if isinstance(cont.get(key, None), (list, tuple)) or isinstance(cont.get(key, None), torch.Tensor):
                goal = _as_tensor(cont[key], self.dtype).flatten()[:2]; break
        if goal is None:
            goal = _as_tensor(ep.get("final_center", [0.0,0.0]), self.dtype).flatten()[:2]

        # obstacle arrays (world, constant per episode)
        C_w = _as_tensor(obs["centers"], self.dtype)              # [N,2]
        R_w = _as_tensor(obs["radii"],   self.dtype).flatten()    # [N]
        W_w = _as_tensor(obs.get("weights", torch.zeros_like(R_w)), self.dtype).flatten()
        d_hat_env = float(obs.get("d_hat_env", params.get("d_hat", self.default_hat_d)))
        hat_d = float(params.get("d_hat", d_hat_env if d_hat_env > 0 else self.default_hat_d))

        # stage ROI (filter obstacles by current stage bounds +/- pad)
        bounds = fstate[k]["bounds"] if (k < len(fstate)) else None
        if bounds is not None:
            xmin,xmax,ymin,ymax = self._stage_bounds_with_pad(bounds)
            sel = (C_w[:,0] >= xmin) & (C_w[:,0] <= xmax) & (C_w[:,1] >= ymin) & (C_w[:,1] <= ymax)
            C_w, R_w, W_w = C_w[sel], R_w[sel], W_w[sel]

        # map to local frame at (o_w, theta)
        C_l = world_to_local(C_w, o_w, R_t) if C_w.numel() else C_w
        g_l = world_to_local(goal, o_w, R_t)
        v_l = world_to_local(v_w, torch.zeros(2, dtype=self.dtype), R_t)
        v_l = v_l / (torch.linalg.norm(v_l) + 1e-9)

        if v_w is not None:
            v_l = world_to_local(v_w, torch.zeros(2, dtype=self.dtype), R_t)
        else:
            v_l = torch.zeros(2, dtype=self.dtype)

        # window & grid
        grid_xy = make_coord_grid(self.H, self.W, half=hat_d, dtype=self.dtype)
        # keep only obstacles whose centers are within window + their radii
        if C_l.numel():
            keep = (torch.abs(C_l).amax(dim=-1) <= (hat_d + R_w))
            C_l, R_l, W_l = C_l[keep], R_w[keep], W_w[keep]
        else:
            R_l = C_l.new_zeros((0,)); W_l = C_l.new_zeros((0,))

        # rasterize barriers
        if C_l.numel():
            sds = [signed_distance_to_circle(grid_xy, C_l[j], R_l[j]) for j in range(C_l.shape[0])]
            barrier_stack = torch.stack([barrier_map_from_sd(sd, hat_d, kind=self.barrier_kind) for sd in sds], dim=0)
            norms = torch.linalg.norm(C_l, dim=-1)
            ang = torch.atan2(C_l[:,1], C_l[:,0])
            prox = (hat_d - torch.clamp(norms - R_l, min=0.0)) / hat_d
            obs_feats = torch.stack([C_l[:,0], C_l[:,1], R_l, norms, ang, prox], dim=-1)
        else:
            barrier_stack = torch.zeros(0, self.H, self.W, dtype=self.dtype)
            obs_feats = torch.zeros(0, 6, dtype=self.dtype)
            W_l = torch.zeros(0, dtype=self.dtype)

        goal_map = quadratic_goal(grid_xy, g_l).unsqueeze(0)      # [1,H,W]
        coord_map = torch.stack([grid_xy[...,0]/hat_d, grid_xy[...,1]/hat_d], dim=0)

        gl_norm = torch.linalg.norm(g_l)
        gl_ang = torch.atan2(g_l[1], g_l[0])
        goal_feats = torch.tensor([g_l[0], g_l[1], gl_norm, gl_ang], dtype=self.dtype)

        # agent center in local
        pos_xy = torch.zeros(1,2, dtype=self.dtype)               # agent is origin of local chart
        dir_xy = v_l.view(1,2)

        stage_idx = int(fstate[k]["stage_idx"]) if (k < len(fstate)) else -1

        return dict(
            grid_xy=grid_xy,                # [H,W,2]
            goal_map=goal_map,              # [1,H,W]
            coord_map=coord_map,            # [2,H,W]
            barrier_stack=barrier_stack,    # [n,H,W]
            obs_feats=obs_feats,            # [n,6]
            obs_weights=W_l,                # [n]
            goal_feats=goal_feats,          # [4]
            pos_xy=pos_xy,                  # [1,2]
            dir_xy=dir_xy,                  # [1,2]
            meta=dict(
                episode=ei, snap=k, stage_idx=stage_idx,
                hat_d=hat_d, success=bool(self.episodes[ei].success),
                bounds=bounds if bounds is not None else None
            )
        )


# ---------- collate (variable #obstacles)

def _pad_stack_NHW(items: List[torch.Tensor]):
    Nmax = max([x.shape[0] if x.numel() else 0 for x in items]) if items else 0
    if Nmax == 0:
        B,H,W = items[0].shape[1], items[0].shape[2], items[0].shape[3]
    outs, mask = [], []
    for x in items:
        if x.numel() == 0:
            H,W = items[0].shape[-2:]
            outs.append(torch.zeros(Nmax, H, W, dtype=items[0].dtype))
            mask.append(torch.zeros(Nmax, dtype=torch.bool))
        else:
            n,H,W = x.shape
            if n < Nmax:
                pad = torch.zeros(Nmax - n, H, W, dtype=x.dtype)
                outs.append(torch.cat([x, pad], dim=0))
                m = torch.zeros(Nmax, dtype=torch.bool); m[:n] = True; mask.append(m)
            else:
                outs.append(x); m = torch.ones(n, dtype=torch.bool); mask.append(m)
    return torch.stack(outs, 0), torch.stack(mask, 0)

def _pad_stack_ND(items: List[torch.Tensor], D: int):
    Nmax = max([x.shape[0] if x.numel() else 0 for x in items]) if items else 0
    outs, mask = [], []
    for x in items:
        if x.numel() == 0:
            outs.append(torch.zeros(Nmax, D, dtype=items[0].dtype))
            mask.append(torch.zeros(Nmax, dtype=torch.bool))
        else:
            n = x.shape[0]
            if n < Nmax:
                pad = torch.zeros(Nmax - n, D, dtype=x.dtype)
                outs.append(torch.cat([x, pad], dim=0))
                m = torch.zeros(Nmax, dtype=torch.bool); m[:n] = True; mask.append(m)
            else:
                outs.append(x); m = torch.ones(n, dtype=torch.bool); mask.append(m)
    return torch.stack(outs, 0), torch.stack(mask, 0)

def collate_navmax(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    H,W = batch[0]["goal_map"].shape[-2:]
    grid_xy   = torch.stack([b["grid_xy"]   for b in batch], 0)      # [B,H,W,2]
    goal_map  = torch.stack([b["goal_map"]  for b in batch], 0)      # [B,1,H,W]
    coord_map = torch.stack([b["coord_map"] for b in batch], 0)      # [B,2,H,W]
    barrier_stack, obs_mask = _pad_stack_NHW([b["barrier_stack"] for b in batch])  # [B,N,H,W]
    obs_feats, _            = _pad_stack_ND([b["obs_feats"] for b in batch], D=6)  # [B,N,6]
    obs_weights, _          = _pad_stack_ND([b["obs_weights"].unsqueeze(-1) for b in batch], D=1)  # [B,N,1]
    obs_weights = obs_weights.squeeze(-1)  # [B,N]
    goal_feats = torch.stack([b["goal_feats"] for b in batch], 0)    # [B,4]
    pos_xy = torch.stack([b["pos_xy"].squeeze(0) for b in batch], 0) # [B,2]
    dir_xy = torch.stack([b["dir_xy"].squeeze(0) for b in batch], 0) # [B,2]

    meta = dict(
        episode = torch.tensor([b["meta"]["episode"] for b in batch], dtype=torch.long),
        snap    = torch.tensor([b["meta"]["snap"]    for b in batch], dtype=torch.long),
        stage   = torch.tensor([b["meta"]["stage_idx"] for b in batch], dtype=torch.long),
        hat_d   = torch.tensor([b["meta"]["hat_d"]  for b in batch], dtype=torch.float32),
        success = torch.tensor([1 if b["meta"]["success"] else 0 for b in batch], dtype=torch.long),
    )

    return dict(
        grid_xy=grid_xy, goal_map=goal_map, coord_map=coord_map,
        barrier_stack=barrier_stack, obs_mask=obs_mask,
        obs_feats=obs_feats, obs_weights=obs_weights,
        goal_feats=goal_feats, pos_xy=pos_xy, dir_xy=dir_xy,
        meta=meta
    )
