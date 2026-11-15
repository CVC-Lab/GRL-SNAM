"""A differentiable twin of a libcvc world gym (``pycvc.WorldGym``), in torch.

libcvc's gym (``cvc::ariadne::WorldGym``) drives a world document's vehicles headless, one
navigator tick at a time, on one of four plants: Jolt, nav's bicycle, or the dynamic single-track
and four-wheel models, whose chassis collide in a physics world of their own. Neither Jolt nor
that world is differentiated. This module is the twin libcvc's P8 built for the Jolt vehicle
(NAV-VEHICLE-PHYSICS-ROADMAP), built for the gym's world. The gym's ``dynamic4w`` plant steps the
four-wheel model (libcvc's ``diff::dyn4_plant``; :func:`grl_snam.dyn4.dyn4_plant` here). The twin
drives that model with the deployed drive's control law (:func:`grl_snam.sdf_nav.bike_control`),
on the gym's own signed distance and ground. Everything it holds is read from the gym:

* ``WorldGym.fields()``: the navigator's grid, its signed distance and gradient, the ground's
  height and grip, and the navigator's settings (nav's normalized frame, its vehicle);
* ``WorldGym.plants()``: each agent's model, its parameters and state in SI, and on the dynamic
  plants what the model took of the last command in each substep;
* ``WorldGym.observe()``: the goals.

The gym stays the judge: a policy trained through the twin is scored by driving the gym itself
(:func:`drive_gym`). Where the twin is the gym, and where it is not:

* **The plant.** On ``dynamic4w``, away from a collision, :meth:`GymTwin.replay` of one tick from
  the gym's state with the command its plant took is the gym's next state to 1e-9
  (``tests/test_gym_twin.py``). On ``jolt`` the model has the vehicle's own numbers and a state
  read back from the chassis (``exact`` false in ``plants()``), so the fitted model's residual
  applies. The bicycle and the single-track model are refused for now.
* **The driver.** The gym's dynamic plants take nav's acceleration through the Jolt plant's
  driver (libcvc's ``dyn_driver``), which has no twin. :meth:`GymTwin.step` applies the
  controller's acceleration raw.
* **Contact.** The gym's chassis collide in Jolt. The four-wheel twin has no contact port
  (:mod:`grl_snam.dyn_contact` couples to the single-track model only), so a rollout into a wall
  is not the gym's. The drive's barrier and :func:`clearance_loss` keep rollouts off walls.
* **Grade.** The gym's four-wheel model runs ``grade_force`` on, which has no torch twin
  (:func:`grl_snam.dyn4._no_grade_force`). The twin turns it off, so it is exact only where the
  ground is flat.
* **The controller.** The gym's navigator (nav@1) runs its carrot FSM, planner and governors
  around this control law. The twin takes goals and coefficients as given: the goal is the
  agent's, and the coefficients are the policy's.

:func:`tick_residuals` measures the last four on any world: the twin's tick against the gym's for
the same commands.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from grl_snam import sdf_nav
from grl_snam.dyn4 import Dyn4Params, PlantLimits, dyn4_plant
from grl_snam.ext_rollout import field_as

__all__ = [
    "GymTwin",
    "arrival_loss",
    "clearance_loss",
    "constant_policy",
    "drive_gym",
    "mlp_policy",
    "progress_loss",
    "tick_residuals",
]

_DYN4_FIELDS = {f.name for f in dataclasses.fields(Dyn4Params)}


def _grid_tensor(a, dtype, device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


class GymTwin:
    """The twin of a gym's world as it is now: its fields, its agents' model and their goals.

    Build it after ``gym.reset(...)`` (or at any tick) with ``GymTwin(gym)``. It reads the gym once;
    :meth:`state` reads the agents' state again, so one twin serves the whole episode while the
    world and the agents' vehicles stay the same.
    """

    def __init__(self, gym, *, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = torch.device(device)
        f = gym.fields()
        plants = gym.plants()
        if not plants:
            raise ValueError("GymTwin: the gym has no agents")
        models = sorted({p["model"] for p in plants})
        if models != ["dyn4"]:
            raise ValueError(
                f"GymTwin: the twin steps the four-wheel model (dynamic4w or jolt agents), "
                f"not {models}"
            )
        first = plants[0]["params"]
        for p in plants[1:]:
            if p["params"] != first:
                raise ValueError("GymTwin: every agent must drive the same vehicle")
        kw = {k: v for k, v in first.items() if k in _DYN4_FIELDS}
        for k in ("iters", "grade_force"):
            kw[k] = int(kw[k])
        # grade_force has no torch twin (dyn4._no_grade_force): exact where the ground is flat.
        kw["grade_force"] = 0
        self.params = Dyn4Params(**kw)
        lim = plants[0]["limits"]
        self.limits = PlantLimits(
            L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
        )
        self.substeps = int(plants[0]["substeps"])
        self.ground_reference = float(plants[0]["ground_reference"])
        self.exact = [bool(p["exact"]) for p in plants]
        self.ids = [int(p["id"]) for p in plants]
        self.plant_names = [p["plant"] for p in plants]

        nav = f["navigator"]
        if not nav["ready"]:
            raise ValueError("GymTwin: the gym's navigator has no grid yet")
        self.S = float(nav["scale"])
        self.tick = float(nav["tick"])
        self.reach = float(nav["reach"])
        self.bounds = tuple(float(b) for b in f["bounds"])
        mnx, mny, mxx, mxy = self.bounds
        # nav's normalized frame: centred on the grid (WorldVehicles' nx/ny), S units per metre.
        self.cx, self.cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        field = sdf_nav.SDFField(
            f["sdf"] * self.S,
            f["sdf_dx"],
            f["sdf_dy"],
            self.bounds,
            (self.cx, self.cy),
            self.S,
            device=self.device,
        )
        self.field = field_as(field, dtype=dtype, device=self.device)
        self.sdf_m = _grid_tensor(f["sdf"], dtype, self.device)
        self.occupancy = np.asarray(f["occupancy"])
        # The ground as the four-wheel model reads it: grip, and height above its reference.
        hm = _grid_tensor(f["height"], dtype, self.device) - self.ground_reference
        mu = _grid_tensor(f["mu"], dtype, self.device)
        self._ground_grid = torch.stack([mu, hm], 0)[None]  # [1, 2, rows, cols]
        self.veh = dict(
            rr=float(nav["rr"]),
            d_hat=float(nav["d_hat"]),
            dt=float(nav["dt"]),  # the controller's step: one navigator tick
            vmax=float(nav["vmax"]),
            L=float(nav["L"]),
            delta_max=float(nav["delta_max"]),
            a_max=float(nav["a_max"]),
            a_lat_max=float(nav["a_lat_max"]),
            k_steer=float(nav["k_steer"]),
            allow_reverse=bool(nav["allow_reverse"]),
        )
        obs = gym.observe()
        self.goal = torch.stack(
            [
                torch.as_tensor(np.asarray(obs["goal_x"]), dtype=dtype),
                torch.as_tensor(np.asarray(obs["goal_y"]), dtype=dtype),
            ],
            -1,
        ).to(self.device)

    # ---- frames ----
    @property
    def n(self) -> int:
        return len(self.ids)

    def to_norm(self, xy: torch.Tensor) -> torch.Tensor:
        """World metres ``[..., 2]`` to nav's normalized frame."""
        c = torch.tensor([self.cx, self.cy], dtype=xy.dtype, device=xy.device)
        return (xy - c) * self.S

    def goal_norm(self) -> torch.Tensor:
        return self.to_norm(self.goal)

    # ---- the gym's world, sampled ----
    def ground(self, wx: torch.Tensor, wy: torch.Tensor):
        """``(mu, height)`` at world points, bilinear on the gym's grid (dyn4's ground callable)."""
        mnx, mny, mxx, mxy = self.bounds
        gx = 2.0 * (wx - mnx) / (mxx - mnx) - 1.0
        gy = 2.0 * (wy - mny) / (mxy - mny) - 1.0
        shape = gx.shape
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)[None, None].to(self.dtype)
        out = F.grid_sample(
            self._ground_grid, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )[0, :, 0, :]
        return out[0].reshape(shape), out[1].reshape(shape)

    def clearance(self, xy: torch.Tensor) -> torch.Tensor:
        """The signed distance (m) at world points ``[B, 2]``."""
        phi, _ = self.field.sample(self.to_norm(xy))
        return phi / self.S

    # ---- the agents ----
    def state(self, gym) -> torch.Tensor:
        """The agents' four-wheel states now, ``[B, 14]`` in :data:`grl_snam.dyn4.ORDER` (SI)."""
        plants = gym.plants()
        if [int(p["id"]) for p in plants] != self.ids:
            raise ValueError("GymTwin.state: the gym's agents changed; build a new twin")
        return torch.as_tensor(
            np.stack([np.asarray(p["state"]) for p in plants], 0), dtype=self.dtype
        ).to(self.device)

    # ---- the plant ----
    def plant_tick(self, state, delta_cmd, accels) -> torch.Tensor:
        """One navigator tick of the four-wheel model: ``substeps`` steps, the steering held, each
        step's acceleration ``accels[:, k]`` (m/s^2), or one acceleration held (``accels [B]``)."""
        for k in range(self.substeps):
            a = accels if accels.dim() == 1 else accels[:, k]
            state = dyn4_plant(self.params, self.limits, state, delta_cmd, a, self.ground)
        return state

    def replay(self, state, gym) -> torch.Tensor:
        """The last tick again: from ``state`` (the agents' state before it), the command each
        agent's plant took, as ``gym.plants()`` reports it after the tick."""
        plants = gym.plants()
        steer = torch.tensor([p["steer_command"] for p in plants], dtype=self.dtype)
        acc = torch.as_tensor(
            np.stack([np.asarray(p["applied_accel"]) for p in plants], 0), dtype=self.dtype
        )
        return self.plant_tick(state, steer.to(self.device), acc.to(self.device))

    # ---- the controller ----
    def control(self, state, coefs, goal_n=None, ext_force=None):
        """The drive's commands from ``state`` and coefficients ``coefs [B, 3]`` (alpha, beta,
        gamma): the steering's angle (rad) and the acceleration along the heading (m/s^2)."""
        o = self.to_norm(state[:, :2])
        goal_n = self.goal_norm() if goal_n is None else goal_n
        delta, a_n, _, _ = sdf_nav.bike_control(
            self.field,
            o,
            state[:, 2],
            state[:, 3] * self.S,
            goal_n,
            coefs[:, 0],
            coefs[:, 1],
            coefs[:, 2],
            ext_force_fn=None if ext_force is None else (lambda _o, _f=ext_force: _f),
            **self.veh,
        )
        return delta, a_n / self.S

    def features(self, state, goal_n=None) -> torch.Tensor:
        """CoefMLP's features (:func:`grl_snam.sdf_nav.coef_feats`) at each agent's rear axle."""
        goal_n = self.goal_norm() if goal_n is None else goal_n
        return sdf_nav.coef_feats(self.field, self.to_norm(state[:, :2]), goal_n)

    def step(self, state, coefs, goal_n=None, ext_force=None):
        """One tick: the drive's commands, then the plant. Returns ``(state, delta, accel)``."""
        delta, acc = self.control(state, coefs, goal_n, ext_force)
        return self.plant_tick(state, delta, acc), delta, acc

    def rollout(self, policy: Callable[[torch.Tensor], torch.Tensor], state, T: int, goal=None):
        """``T`` ticks driven by ``policy(features) -> coefs [B, 3]``. Returns the states
        ``[T + 1, B, 14]`` and the commands ``[T, B, 2]`` (steering, acceleration). ``goal``
        (``[B, 2]``, world metres) is each row's goal, the agents' own by default; rows may be
        states from different ticks or episodes, each with its goal."""
        states, cmds = [state], []
        goal_n = self.goal_norm() if goal is None else self.to_norm(goal)
        for _ in range(T):
            coefs = policy(self.features(state, goal_n))
            state, delta, acc = self.step(state, coefs, goal_n)
            states.append(state)
            cmds.append(torch.stack([delta, acc], -1))
        return torch.stack(states, 0), torch.stack(cmds, 0)


# ---- losses ----
def progress_loss(twin: GymTwin, states: torch.Tensor, goal=None) -> torch.Tensor:
    """The mean distance (m) left to each agent's goal over the rollout's ticks after the first,
    less the reach: what the gym's progress reward pays for, made smooth. It rewards closing
    fast and says nothing about stopping; :func:`arrival_loss` pays for arriving."""
    g = twin.goal if goal is None else goal
    d = (states[1:, :, :2] - g[None]).norm(dim=-1)
    return F.softplus(d - twin.reach).mean()


def arrival_loss(
    twin: GymTwin, states: torch.Tensor, goal=None, *, w_stop: float = 0.5, w_path: float = 0.05
) -> torch.Tensor:
    """What arriving is: ending the rollout inside the reach, stopped. Per row, the distance (m)
    outside the reach at the last tick, plus ``w_stop`` x the speed (m/s) left there in
    proportion to how far inside the reach it is (a smooth step of 1 m), plus ``w_path`` x the
    mean distance outside the reach over the ticks, so a rollout that has not arrived still
    learns to close."""
    g = twin.goal if goal is None else goal
    d = (states[:, :, :2] - g[None]).norm(dim=-1)  # [T + 1, B]
    out_T = F.softplus(d[-1] - twin.reach)
    inside_T = torch.sigmoid(twin.reach - d[-1])
    speed_T = states[-1, :, 3].abs()
    path = F.softplus(d[1:] - twin.reach).mean(0)
    return (out_T + w_stop * inside_T * speed_T + w_path * path).mean()


def clearance_loss(twin: GymTwin, states: torch.Tensor, margin_m: float = 2.0) -> torch.Tensor:
    """A barrier on the signed distance under each rear axle: squared metres inside ``margin_m``.
    The twin has no contact, so this is what keeps a trained rollout off the walls."""
    T1, B, _ = states.shape
    phi = twin.clearance(states[:, :, :2].reshape(T1 * B, 2)).reshape(T1, B)
    return F.relu(margin_m - phi).pow(2).mean()


# ---- the gym as judge ----
def drive_gym(gym, twin: GymTwin, policy, ticks: int, *, stop_when_done: bool = True, states=None):
    """Drive the gym itself for ``ticks`` ticks with the twin's controller and ``policy``: each tick,
    the agents' state from the gym, the drive's commands from the twin, sent as the gym's
    ``step(steer, accel)``. Returns the last ``step`` result and the per-tick observations; with a
    list as ``states``, each tick's agents' state before its step is appended to it."""
    obs_log = []
    out = None
    with torch.no_grad():
        for _ in range(ticks):
            s = twin.state(gym)
            if states is not None:
                states.append(s)
            coefs = policy(twin.features(s))
            delta, acc = twin.control(s, coefs)
            out = gym.step(delta.double().cpu().numpy(), acc.double().cpu().numpy())
            obs_log.append(out[0])
            if stop_when_done and (out[2] or out[3]):
                break
    return out, obs_log


def tick_residuals(gym, twin: GymTwin, policy, ticks: int):
    """The twin's tick against the gym's, for the same commands, over ``ticks`` ticks the twin's
    controller drives in the gym: per tick, the position (m), heading (rad) and speed (m/s) the
    twin's plant reaches from the gym's state less where the gym's plant went, and whether the
    gym's plant collided. Arrays ``[ticks, B]``."""
    pos, head, spd, hit = [], [], [], []
    with torch.no_grad():
        for _ in range(ticks):
            s0 = twin.state(gym)
            coefs = policy(twin.features(s0))
            s1, delta, acc = twin.step(s0, coefs)
            out = gym.step(delta.double().cpu().numpy(), acc.double().cpu().numpy())
            g1 = twin.state(gym)
            pos.append((s1[:, :2] - g1[:, :2]).norm(dim=-1).cpu().numpy())
            dpsi = torch.atan2(torch.sin(s1[:, 2] - g1[:, 2]), torch.cos(s1[:, 2] - g1[:, 2]))
            head.append(dpsi.abs().cpu().numpy())
            spd.append((s1[:, 3] - g1[:, 3]).abs().cpu().numpy())
            hit.append(np.asarray([int(p["collisions_tick"]) > 0 for p in gym.plants()]))
            if out[2] or out[3]:
                break
    return {
        "position_m": np.stack(pos, 0),
        "heading_rad": np.stack(head, 0),
        "speed_mps": np.stack(spd, 0),
        "collided": np.stack(hit, 0),
    }


def constant_policy(coefs) -> Callable[[torch.Tensor], torch.Tensor]:
    """A policy that gives every agent the same ``(alpha, beta, gamma)``."""
    c = torch.as_tensor(coefs, dtype=torch.float64)

    def policy(feats: torch.Tensor) -> torch.Tensor:
        return c.to(feats.dtype).expand(feats.shape[0], 3)

    return policy


def mlp_policy(model: sdf_nav.CoefMLP) -> Callable[[torch.Tensor], torch.Tensor]:
    """A CoefMLP as a policy, in the features' precision."""

    def policy(feats: torch.Tensor) -> torch.Tensor:
        al, be, ga = model(feats.to(next(model.parameters()).dtype))
        return torch.stack([al, be, ga], -1).to(feats.dtype)

    return policy
