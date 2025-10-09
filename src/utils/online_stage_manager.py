# --- in spline_stagewise6.py ---

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from src.utils.dijkstra import exit_to_boundary_dijkstra, exit_to_boundary_reachable, exit_to_boundary_exhaustive

# your existing Stage dataclass (kept here for reference / import)
@dataclass
class Stage:
    center: np.ndarray
    size: Tuple[float, float]
    entry_point: np.ndarray
    exit_point: np.ndarray
    stage_id: int
    def __post_init__(self):
        W, H = self.size; cx, cy = self.center
        self.bounds = (cx - W/2, cx + W/2, cy - H/2, cy + H/2)

@dataclass
class RecoveryCfg:
    stuck_window: int = 20          # steps to judge progress
    min_progress: float = 1e-3     # meters improvement required
    max_relax: int = 2              # how many relax rounds (inflate/clearance)
    detour_fracs: tuple = (0.4,0.8,1.2)
    rand_radii: tuple = (0.7, 0.9)  # × W (ring radii for random relocate)
    rand_tries_per_radius: int = 3
    use_global_for_relocate: bool = False  # set False to be strict-local



# ---------- small geometry helpers ----------
def _rect_contains(b, p):
    return (b[0] <= p[0] <= b[1]) and (b[2] <= p[1] <= b[3])

def _dist_to_rect(b, p):
    x0,x1,y0,y1 = b
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    if dx == 0.0 and dy == 0.0:
        dleft  = p[0] - x0; dright = x1 - p[0]
        dbot   = p[1] - y0; dtop   = y1 - p[1]
        return min(dleft, dright, dbot, dtop)
    return (dx*dx + dy*dy) ** 0.5

def _bounds_from_center(cx, cy, W, H):
    return (cx - W/2, cx + W/2, cy - H/2, cy + H/2)

def _center_of(bounds):
    x0,x1,y0,y1 = bounds
    return np.array([(x0+x1)/2.0, (y0+y1)/2.0], dtype=float)

# ---------- the online manager ----------
class StageManagerOnline:
    """
    Online stage builder: maintains a single live box, but exposes a Stage list
    with a valid exit_point for the current stage at all times.
    """
    def __init__(
        self,
        stage_size=(3.0, 2.0),
        inflate=0.0,
        near_eps=0.15,
        advance_frac=0.9,
        detour_fracs=(0.25, 0.50, 0.75),
        exit_solver=None,    # optional: your width-aware overlap solver
        exit_kwargs=None,
    ):
        self.stage_size   = tuple(map(float, stage_size))
        self.inflate      = float(inflate)
        self.near_eps     = float(near_eps)
        self.advance_frac = float(advance_frac)
        self.detour_fracs = tuple(detour_fracs)
        self.exit_solver  = exit_solver
        self.exit_kwargs  = exit_kwargs or {}

        # public fields expected elsewhere
        self.stages: list[Stage] = []
        self.current_stage_idx: int = 0
        self.boundary_inflation = 0.001

        # internal live state
        self._live_bounds: Optional[Tuple[float,float,float,float]] = None
        self._planned_next_bounds: Optional[Tuple[float,float,float,float]] = None
        self._goal_xy = None
        self._world   = None
        self._next_id = 0
        self._cur_pos = None
        self._vel = None
        
        self.recover = RecoveryCfg()
        self._d_hist = []    # goal distance history

    def _upd_progress(self, pos, goal):
        self._vel = float(np.linalg.norm(np.asarray(pos,float) - np.asarray(self._cur_pos,float)))
        self._d_hist.append(self._vel)
        if len(self._d_hist) > self.recover.stuck_window:
            self._d_hist.pop(0)
        # record for velocity direction
        self._cur_pos = pos

    def _is_stuck(self):
        if len(self._d_hist) < self.recover.stuck_window:
            return False
        return self._d_hist[0] < self.recover.min_progress and self._d_hist[-1] < self.recover.min_progress
    
    def _slice_CR(self, C, R, bounds):
        if len(C)==0: return C, R
        
        C = np.asarray(C)
        R = np.asarray(R).reshape(-1)
        x0,x1,y0,y1 = bounds
        # optional padding around the box
        infl = getattr(self, "boundary_inflation", 0.0)
        x0 -= infl; x1 += infl; y0 -= infl; y1 += infl

        # circle-vs-AABB test (branchless, vectorized)
        # distance from center to box along each axis (0 if inside slab)
        zero = np.zeros_like(R, dtype=C.dtype)
        dx = np.maximum.reduce([x0 - C[:, 0], zero, C[:, 0] - x1])
        dy = np.maximum.reduce([y0 - C[:, 1], zero, C[:, 1] - y1])

        # intersect iff dx^2 + dy^2 <= r^2  (use < if you don't want "touching" to count)
        m = (dx*dx + dy*dy) <= (R*R)

        return C[m], R[m]
        
    def _score_candidate(self, entry, nb, Cw_nb, Rw_nb, goal):
        # exit using local-only solver
        ex = self._solve_exit_local(entry, nb, nb, Cw_nb, Rw_nb)
        if ex is None: return None, -1e9
        # score: prefer goal-ward & clearance; penalize tiny steps
        step = float(np.linalg.norm(np.asarray(ex)-np.asarray(entry)))
        if step < 0.1*min(self.stage_size): return None, -1e9
        # distance-to-goal gain
        g0 = float(np.linalg.norm(np.asarray(goal)-np.asarray(entry)))
        g1 = float(np.linalg.norm(np.asarray(goal)-np.asarray(ex)))
        gain = g0 - g1
        # cheap clearance proxy: min dist to obstacles in candidate crop
        if len(Cw_nb):
            cl = np.min(np.linalg.norm(np.asarray(ex)-Cw_nb, axis=1) - (Rw_nb + self.inflate))
        else:
            cl = 1.0
        score = 1.0*gain + 0.6*cl
        return ex, score

    # ----------- public API -----------
    
    def reset(self, start_xy, goal_xy, world):
        """
        Initialize the first stage at start, and PLAN its exit immediately
        (so exit_point is never None).
        """
        self._cur_pos = start_xy
        self._goal_xy = np.asarray(goal_xy, float)
        self._world   = world
        self.stages.clear()
        self.current_stage_idx = 0
        self._next_id = 0

        W, H = self.stage_size
        start_xy = np.asarray(start_xy, float)
        b0 = _bounds_from_center(start_xy[0], start_xy[1], W, H)
        # Plan an exit (and a candidate next bounds) right away:
        # build local slice for the very first box
        Cw, Rw = self._slice_CR(world.C_np, world.R_np, b0)  

        ex0, nb0 = self._plan_exit_for_local(b0, start_xy, self._goal_xy, Cw, Rw)

        # Guarantee non-None exit for API: if planning failed, use goal as exit
        if ex0 is None:  # safe fallback
            ex0 = self._fallback_exit(start_xy, self._goal_xy)

        st0 = Stage(center=_center_of(b0), size=(W,H),
                    entry_point=start_xy.copy(), exit_point=np.asarray(ex0, float),
                    stage_id=self._next_id)
        self._next_id += 1
        self.stages.append(st0)
        self._live_bounds = b0
        self._planned_next_bounds = nb0  # may be None if planning failed

    def stage_slice(self, C, R, Wmask):
        """Crop obstacles to the current live bounds (keeps old signature)."""
        b = self._live_bounds
        if b is None or len(C) == 0:
            return C, R, Wmask
        infl = getattr(self, "boundary_inflation", 0.0)
        x0,x1,y0,y1 = b
        x0 -= infl; x1 += infl; y0 -= infl; y1 += infl

        # circle-vs-AABB test (branchless, vectorized)
        # distance from center to box along each axis (0 if inside slab)
        zero = np.zeros_like(R, dtype=C.dtype)
        dx = np.maximum.reduce([x0 - C[:, 0], zero, C[:, 0] - x1])
        dy = np.maximum.reduce([y0 - C[:, 1], zero, C[:, 1] - y1])

        # intersect iff dx^2 + dy^2 <= r^2  (use < if you don't want "touching" to count)
        m = (dx*dx + dy*dy) <= (R*R)
        return C[m], R[m], Wmask[m] if hasattr(Wmask, "__len__") and len(Wmask)==len(C) else Wmask

    
    def current(self) -> Stage:
        return self.stages[self.current_stage_idx]

    def advance_if_needed_local(self, pos_xy, goal_xy, Cw, Rw) -> bool:
        """
        Same logic as advance_if_needed, but uses *only* the sliced obstacles (Cw,Rw).
        """
        if self._live_bounds is None:
            return False

        pos  = np.asarray(pos_xy, float)
        goal = np.asarray(goal_xy, float)

        # trigger only when truly near/at boundary
        if _dist_to_rect(self._live_bounds, pos) > self.near_eps and not self._is_stuck():
            return False

        # current stage: must already have a valid exit_point (look-ahead on reset/previous step)
        cur = self.stages[-1]

        if cur.exit_point is None:
            raise AssertionError("Frame needs to have an exit point!")
            # # emergency: plan *within* the local view
            # ex_now, _ = self._plan_exit_for_local(self._live_bounds, pos, goal, Cw, Rw)
            # if ex_now is None:  # give a deterministic, safe fallback
            #     ex_now = self._fallback_exit(pos, goal)
            # cur.exit_point = tuple(ex_now)

        # propose next stage using local view
        # We are at/near the boundary; open the next stage using the current exit as entry.
        W, H = self.stage_size
        
        # 1) pick a next-box *center* by pushing forward from the current box center
        x0,x1,y0,y1 = self._live_bounds
        cur_c = np.array([(x0+x1)/2.0, (y0+y1)/2.0], dtype=float)
        gdir  = np.asarray(goal, float) - pos
        gdir /= (np.linalg.norm(gdir) + 1e-12)

        vdir =  pos - np.asarray(self._cur_pos, float)
        vdir_norm = np.linalg.norm(vdir)
        if vdir_norm <= 1e-8: # corner case
            nxt_center = cur_c + gdir * (self.advance_frac * (W + H) / 2)
        else:
            vdir /= (np.linalg.norm(vdir) + 1e-12)
            nxt_center = cur_c + (vdir + gdir) / 2 * (self.advance_frac * (W + H) / 2)

        next_b = _bounds_from_center(nxt_center[0], nxt_center[1], W, H)
        Cw, Rw =  self._slice_CR(self._world.C_np, self._world.R_np, next_b)

        # look-ahead: plan the next stage’s exit using the *new* box’s local view
        # NOTE: we don’t have its local crop yet; approximate with the same Cw,Rw (OK for look-ahead).
        if _rect_contains(next_b, goal):
            ex_next = goal.copy()
        ex_next, _ = self._plan_exit_for_local(next_b, pos, goal, Cw, Rw)
        if ex_next is None or next_b is None or self._is_stuck():
            # try recovery (relax/sweep/random relocate)
            #print(self._d_hist)
            ex_next, next_b = self._recover_next_box(pos, goal, self._world, Cw, Rw, cur_c)


        st_next = Stage(
            center=np.array([(next_b[0]+next_b[1])/2.0, (next_b[2]+next_b[3])/2.0], dtype=float),
            size=(W, H),
            entry_point=pos.copy(),
            exit_point=np.asarray(ex_next, float),
            stage_id=self._next_id
        )
        self._next_id += 1
        self.stages.append(st_next)

        # 5) commit book-keeping
        self._live_bounds = next_b
        self.current_stage_idx = len(self.stages) - 1
        return True
    
    def _recover_next_box(self, entry_next, goal, world, Cw, Rw, cur_c):
        W,H = self.stage_size
        gdir = (np.asarray(goal,float) - np.asarray(entry_next,float))
        gdir /= (np.linalg.norm(gdir)+1e-12)

        base_center = cur_c #+ gdir * self.advance_frac * (W + H) / 2
        best = (None, None, -1e9)
        # for s in self.recover.detour_fracs:
        #     for sign in (+1,-1):
        #         cc = base_center + sign*s*H*np.array([-gdir[1], gdir[0]])
        #         nb = (cc[0]-W/2, cc[0]+W/2, cc[1]-H/2, cc[1]+H/2)
        #         Cw_nb, Rw_nb = (
        #             self._slice_CR_bounds(world.C_np, world.R_np, nb)
        #             if self.recover.use_global_for_relocate else (Cw, Rw)
        #         )
        #         ex, sc = self._score_candidate(entry_next, nb, Cw_nb, Rw_nb, goal)
        #         if ex is not None and sc > best[2]:
        #             best = (ex, nb, sc)
        # if best[0] is not None:
        #     return best[0], best[1]
        # random relocate on rings around base_center
        rng = np.random.default_rng()

        for r in self.recover.rand_radii:
            for _ in range(self.recover.rand_tries_per_radius):
                ang = float(rng.uniform(-np.pi / 2, np.pi / 2))
                cc = base_center + gdir * r*(W+H)/2*np.array([np.cos(ang), np.sin(ang)])
                nb = (cc[0]-W/2, cc[0]+W/2, cc[1]-H/2, cc[1]+H/2)
                if not _rect_contains(entry_next, nb):
                    continue
                Cw_nb, Rw_nb = (
                    self._slice_CR(world.C_np, world.R_np, nb)
                    if self.recover.use_global_for_relocate else (Cw, Rw)
                )
                ex, sc = self._score_candidate(entry_next, nb, Cw_nb, Rw_nb, goal)
                if ex is not None and sc > best[2]:
                    best = (ex, nb, sc)
        if best[0] is not None:
            return best[0], best[1]

        # last resort: ray to boundary of a forward box (guarantees progress)
        cc = base_center
        nb = (cc[0]-W/2, cc[0]+W/2, cc[1]-H/2, cc[1]+H/2)
        ex = self._fallback_exit(entry_next, goal)  # ray to boundary inside nb
        return ex, nb
    
    def _plan_exit_for_local(self, cur_bounds, entry, goal, Cw, Rw):
        """
        Returns (exit_point, next_bounds) computed using only the local obstacles (Cw,Rw).
        """
        entry = np.asarray(entry, float)
        goal  = np.asarray(goal,  float)
        x0,x1,y0,y1 = cur_bounds

        # Forward proposal: next box center along entry->goal
        W, H = self.stage_size
        cur_c = np.array([(x0+x1)/2.0, (y0+y1)/2.0], dtype=float)
        path  = goal - entry
        n = np.linalg.norm(path) + 1e-12
        path = path / n
        nxt_center = cur_c + path * (self.advance_frac * W)
        cand_nb    = _bounds_from_center(nxt_center[0], nxt_center[1], W, H)
        # Solve exit to the boundary using ONLY (Cw,Rw)
        ex = self._solve_exit_local(entry, cur_bounds, cand_nb, Cw, Rw)

        # If the goal is in the current box -> exit is goal; no need for next box
        # if ex is not None and np.linalg.norm(goal - ex) < 1e-2:
        #     return ex, cand_nb
        # Detours if forward is poor
        # if self._too_poor(ex, entry):
        #     ortho = np.array([-path[1], path[0]], dtype=float)
        #     for s in self.detour_fracs:
        #         for sign in (+1.0, -1.0):
        #             cc = nxt_center + sign * (s * H) * ortho
        #             nb = _bounds_from_center(cc[0], cc[1], W, H)
        #             ex2 = self._solve_exit_local(entry, cur_bounds, nb, Cw, Rw)
        #             if not self._too_poor(ex2, entry):
        #                 return ex2, nb
        #     return None, None

        return ex, cand_nb

    def _solve_exit_local(self, entry, cur_bounds, next_bounds, Cw, Rw):
        """
        Exit-to-boundary using only local obstacles (Cw,Rw). If you want to keep Dijkstra,
        call the single-box version here (no global 'world'!).
        """

        # Single-box Dijkstra restricted to cur_bounds and (Cw,Rw)
        # ex = exit_to_boundary_dijkstra(
        #     entry_xy=np.asarray(entry, float),
        #     cur_bounds=cur_bounds,
        #     goal_xy=np.asarray(self._goal_xy, float),   # OK to use global goal
        #     C=Cw, R=Rw,
        #     inflate=self.inflate,
        #     grid_res=(196, 196),
        #     prefer_goal_if_inside=True,
        #     goal_target_frac=0.5,
        #     goal_pull=-1.0
        # )
        # BFS strategy
        ex = exit_to_boundary_reachable(
            entry_xy=entry,
            cur_bounds=cur_bounds,
            goal_xy=np.asarray(self._goal_xy, float),
            C=Cw, R=Rw,                 # sliced obstacles ONLY
            inflate=self.inflate,
            grid_res=(104, 92),
            connectivity=8,
            prefer_goal_if_inside=True
        )

        if ex is None:
            # final fallback: exhaustive reachable boundary
            ex = exit_to_boundary_exhaustive(
                    entry_xy=entry, cur_bounds=cur_bounds, goal_xy=np.asarray(self._goal_xy, float),
                    Cw=Cw, Rw=Rw, inflate=self.inflate, grid_res=(104, 92),
                    connectivity=8, prefer_goal_if_inside=True, min_step_frac=0.12)
        return ex

    def advance_if_needed(self, pos_xy) -> bool:
        Cw, Rw, _ = self.stage_slice(self._world.C_np, self._world.R_np, self._world.W_np)
        self._upd_progress(pos_xy, self._goal_xy)
        return self.advance_if_needed_local(pos_xy, self._goal_xy, Cw, Rw)

    def _too_poor(self, exitp, entry):
        if exitp is None: return True
        v = np.asarray(exitp) - np.asarray(entry)
        return np.linalg.norm(v) < 0.15 * min(self.stage_size)

    def _fallback_exit(self, pos, goal):
        """Always returns a 2D point to keep API satisfied."""
        # project toward goal inside current bounds if possible; else use goal
        if self._live_bounds is None:
            return np.asarray(goal, float)
        x0,x1,y0,y1 = self._live_bounds
        gdir = np.asarray(goal, float) - np.asarray(pos, float)
        n = np.linalg.norm(gdir) + 1e-12
        gdir = -gdir / n
        # shoot a ray to the boundary in gdir; clamp to rectangle edges
        # param t to hit each side
        tx = [((x0 - pos[0]) / gdir[0]) if abs(gdir[0])>1e-12 else np.inf,
              ((x1 - pos[0]) / gdir[0]) if abs(gdir[0])>1e-12 else np.inf]
        ty = [((y0 - pos[1]) / gdir[1]) if abs(gdir[1])>1e-12 else np.inf,
              ((y1 - pos[1]) / gdir[1]) if abs(gdir[1])>1e-12 else np.inf]
        t = min([tt for tt in tx+ty if tt>0], default=1.0)
        p = np.asarray(pos, float) + t * gdir
        # clamp within bounds a tiny bit
        p[0] = np.clip(p[0], x0+1e-4, x1-1e-4)
        p[1] = np.clip(p[1], y0+1e-4, y1-1e-4)
        return p
