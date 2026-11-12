"""Outcome-independent static evaluation scenes; simulator-only truth."""
import math
import numpy as np
from .world import World

SCENARIOS = ('clear','risk_passage','blocked_avoidance','overlap_revelation',
             'long_corridor','damping_in','damping_out','risk','obstacle')


def specification(name, seed, profile='development'):
    if name not in SCENARIOS: raise ValueError(name)
    rng=np.random.default_rng(seed)
    jitter=float(rng.uniform(-.15,.15))
    goal=16. if name=='long_corridor' else 5.
    patches=[(2.2,.25+jitter,.85,.55,.9)]
    obstacles=[]
    if name=='clear':patches=[]
    if name=='blocked_avoidance':obstacles=[(2.0,-.65,.30),(2.65,-.65,.30)]
    if name=='overlap_revelation':patches=[(3.1,.20+jitter,.42,.65,1.1)]
    if name=='long_corridor':
        patches=[(x,(-1)**k*.35+jitter,.7,.55,.9) for k,x in enumerate(np.arange(2.2,15,2.1))]
        obstacles=[(6.6,.9,.25),(10.8,-.9,.25)]
    if name=='obstacle':obstacles=[(3.4,.65,.32)]
    damping=float(rng.uniform(.7,1.05))
    if name=='damping_in':damping=1.35
    if name=='damping_out':damping=1.85
    # Existing fixtures preserve their exact RNG sequence and truth parameters.
    if name in ('risk','obstacle'):
        old=np.random.default_rng(seed);damping=float(old.uniform(.7,1.05))
        patches=[(2.2,.25+float(old.uniform(-.15,.15)),.85,.55,.9)]
    base_steps=96 if profile=='smoke' else 180
    return dict(name=name,seed=seed,initial=[0.,0.,0.,0.],goal=[goal,0.],
                patches=patches,obstacles=obstacles,damping=damping,
                steps=math.ceil(base_steps*goal/5),
                nominal_stage_count=1+math.ceil(max(0,goal-1.4)/2.1),
                damping_regime='outside_fit_range' if name=='damping_out' else 'inside_fit_range',
                bounds=[-1.5,goal+1.5,-2.3,2.3])


class ScenarioWorld(World):
    def __init__(self,spec,cfg):
        self.spec=spec;self.cfg=cfg;self.seed=spec['seed'];self.scene=spec['name']
        self.initial=np.array(spec['initial']);self.goal=tuple(spec['goal'])
        self._damping=spec['damping'];self._obstacles=tuple(tuple(o) for o in spec['obstacles'])

    def risk_truth(self,q):
        q=np.asarray(q);value=np.full(q.shape[:-1],.02)
        for x,y,sx,sy,amplitude in self.spec['patches']:
            z=(q-np.array([x,y]))/np.array([sx,sy])
            value=value+amplitude*np.exp(-.5*np.sum(z*z,axis=-1))
        return value

    def risk_gradient(self,q):
        q=np.asarray(q);gradient=np.zeros_like(q,dtype=float)
        for x,y,sx,sy,amplitude in self.spec['patches']:
            delta=q-np.array([x,y]);scale=np.array([sx,sy])
            bump=amplitude*np.exp(-.5*np.sum((delta/scale)**2,axis=-1))
            gradient-=bump[...,None]*delta/scale**2
        return gradient
