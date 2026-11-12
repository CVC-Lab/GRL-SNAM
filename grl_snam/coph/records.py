from dataclasses import dataclass, field, asdict
from typing import Tuple
import numpy as np
from .model import tensor

@dataclass(frozen=True)
class Config:
    stage_blend_mode: str = 'legacy'
    route_field_mode: str = 'anchors'
    controller_mode: str = 'legacy'
    shaping_mode: str = 'legacy'
    shaping_k: float = 1.
    shaping_ell: float = .5
    shaping_eta: float = .5
    safety_margin: float = .05
    obstacle_cap: int = 256
    footprint_cap: int = 120
    route_expansion_cap: int = 4096
    navigation_target: str = 'mission'
    replay_objective: str = 'legacy'
    replay_domain: str = 'single'
    replay_horizon_mode: str = 'fixed'
    replay_update_cap: int | None = None
    boundary_max_steps: int = 32
    boundary_follow_steps: int = 4
    dt: float = .12
    steps: int = 180
    stage_width: float = 3.
    stage_height: float = 4.
    overlap: float = .30
    previous: int = 2
    sample_cap: int = 240
    transition_cap: int = 120
    anchor_cap: int = 12
    rbf_side: int = 4
    rbf_width: float = .8
    update_every: int = 8
    replay_steps: int = 8
    solver_steps: int = 8
    damping_initial: float = .6
    policy_initial: Tuple[float, float] = (1., 2.)
    damping_bounds: Tuple[float, float] = (.15, 1.5)
    policy_bounds: Tuple[float, float] = (0., 10.)
    damping_trust: float = .15
    policy_trust: float = .25
    fit_ridge: float = .02
    validation_slack: float = .003
    sensor_radius: float = 1.25
    force_limit: float = 3.

    def __post_init__(self):
        if self.stage_blend_mode not in ('legacy','smooth'):raise ValueError('invalid stage blend mode')
        if self.stage_blend_mode=='smooth' and (self.replay_domain!='retained' or self.controller_mode not in ('protected','routed')):
            raise ValueError('smooth stage blending requires retained protected/routed control')
        if self.stage_blend_mode=='smooth' and self.overlap<=0:raise ValueError('smooth stage blending requires positive overlap')
        if self.route_field_mode not in ('anchors','supported_field'):raise ValueError('invalid route field mode')
        if self.route_field_mode=='supported_field' and self.controller_mode!='routed':raise ValueError('field routing requires routed control')
        if self.shaping_mode not in ('legacy','progress'):raise ValueError('invalid shaping mode')
        if self.shaping_mode=='progress':
            if self.controller_mode not in ('protected','routed'):raise ValueError('progress requires protected or routed control')
            if self.shaping_k!=1. or self.shaping_ell!=.5 or self.shaping_eta!=.5:raise ValueError('fixed progress settings required')
            if self.policy_bounds[0]<0 or not np.isfinite(self.policy_bounds).all() or any(not self.policy_bounds[0]<=p<=self.policy_bounds[1] for p in self.policy_initial):raise ValueError('progress requires bounded nonnegative shaping')
        if self.controller_mode not in ('legacy','geometry','protected','routed'):raise ValueError('invalid controller mode')
        if self.controller_mode!='legacy':
            if self.replay_domain!='retained' or self.replay_objective!='mission':raise ValueError('new controller requires retained mission replay')
            if self.replay_update_cap is None:object.__setattr__(self,'replay_update_cap',1024)
        if min(self.obstacle_cap,self.footprint_cap,self.route_expansion_cap)<1 or self.safety_margin<0:raise ValueError('invalid navigation limits')
        if self.replay_horizon_mode=='supported_boundary' and self.replay_update_cap is None:
            object.__setattr__(self,'replay_update_cap',1024)
        if self.replay_horizon_mode not in ('fixed','supported_boundary'):
            raise ValueError('invalid replay_horizon_mode')
        if self.replay_horizon_mode=='supported_boundary' and self.replay_domain!='retained':
            raise ValueError('boundary replay requires retained models')
        if self.replay_update_cap is not None and self.replay_update_cap<1:
            raise ValueError('positive replay_update_cap required')
        if self.replay_horizon_mode=='supported_boundary' and (self.boundary_max_steps<self.replay_steps or self.boundary_follow_steps<1):
            raise ValueError('invalid boundary horizon')
        if self.replay_domain not in ('single','retained'):
            raise ValueError('replay_domain must be single or retained')
        if self.replay_domain=='retained' and self.replay_objective!='mission':
            raise ValueError('retained replay requires mission objective')
        if self.replay_objective not in ('legacy', 'mission'):
            raise ValueError('replay_objective must be legacy or mission')
        if self.navigation_target not in ('mission', 'stage'):
            raise ValueError('navigation_target must be mission or stage')
        if not (0 <= self.overlap < 1) or self.previous not in (0, 2, 3):
            raise ValueError('overlap in [0,1); previous must be 0, 2, or 3')
        if min(self.dt, self.stage_width, self.stage_height, self.rbf_width, self.sensor_radius) <= 0:
            raise ValueError('positive physical scales required')
        if min(self.steps, self.sample_cap, self.transition_cap, self.anchor_cap,
               self.update_every, self.replay_steps, self.solver_steps) < 1 or self.rbf_side < 2:
            raise ValueError('positive budgets and rbf_side >= 2 required')

@dataclass(frozen=True)
class Reading:
    uid: str
    time: float
    q: Tuple[float, float]
    value: float

@dataclass(frozen=True)
class Observation:
    time: float
    readings: Tuple[Reading, ...]
    obstacles: Tuple[Tuple[float, float, float], ...]
    sensor_center: tuple | None = None
    sensor_radius: float = 0.

@dataclass(frozen=True)
class Transition:
    uid: int
    time: float
    x: tuple
    u: tuple
    after: tuple
    version: int

@dataclass(frozen=True)
class Counterfactual:
    stage_id: int
    version: int
    interpretation: str
    states: tuple
    cost: float

@dataclass
class Stage:
    id: int
    center: tuple
    entry: tuple
    target: tuple
    centers: object
    weights: object
    readings: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    obstacles: tuple = ()
    anchors: tuple = ()
    version: int = 0
    covariance: object = None
    footprints: list = field(default_factory=list)
    entry_navigation: dict = field(default_factory=dict)
    counterfactuals: list = field(default_factory=list)

    def contains(self, q, cfg):
        return abs(q[0]-self.center[0]) <= cfg.stage_width/2 + 1e-10 and abs(q[1]-self.center[1]) <= cfg.stage_height/2 + 1e-10

    def supported(self, q, cfg):
        # Only fitting observations certify a replay domain; validation data stay reserved.
        return self.contains(q, cfg) and any(np.linalg.norm(np.asarray(q)-r.q) <= .65
                                            for r in self.readings if int(r.uid.split(':')[-1]) % 5)


def new_stage(index, center, entry, goal, cfg):
    xs = np.linspace(center[0]-cfg.stage_width/2, center[0]+cfg.stage_width/2, cfg.rbf_side)
    ys = np.linspace(center[1]-cfg.stage_height/2, center[1]+cfg.stage_height/2, cfg.rbf_side)
    centers = tensor(np.array([(x,y) for x in xs for y in ys]))
    target = (min(goal[0], center[0]+cfg.stage_width/2-.1), goal[1])
    return Stage(index, tuple(center), tuple(entry), target, centers,
                 tensor(np.full(len(centers), -1. / len(centers))))
