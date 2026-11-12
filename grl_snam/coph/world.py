"""Private simulator truth. Learner receives Observation/Transition, never World."""
import numpy as np
from .records import Observation, Reading
from .model import flow, tensor, balance
from .shapes import is_slab, slab_distance, slab_sweep

class World:
    def __init__(self, seed=0, scene='risk', cfg=None):
        self.cfg=cfg;self.seed=seed;self.scene=scene
        rng=np.random.default_rng(seed)
        self._damping=float(rng.uniform(.7,1.05))
        self._risk_center=np.array([2.2, .25+rng.uniform(-.15,.15)])
        self._obstacles=((3.4,.65,.32),) if scene=='obstacle' else ()
        self.initial=np.array([0.,0.,0.,0.]);self.goal=(5.,0.)

    def risk_truth(self, q):
        q=np.asarray(q)
        z=(q-self._risk_center)/np.array([.85,.55])
        return .02+.9*np.exp(-.5*np.sum(z*z,axis=-1))

    def sense(self,x,tick):
        # Index-addressed noise preserves pairing despite differing policy trajectories.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,tick,7919]))
        offsets=np.array([(a,b) for a in np.linspace(-1.,1.,5) for b in np.linspace(-1.,1.,5)
                          if a*a+b*b<=1.]) * self.cfg.sensor_radius
        points=x[:2]+offsets
        values=np.maximum(.001,self.risk_truth(points)+rng.normal(0,.003,len(points)))
        readings=tuple(Reading(f'{self.seed}:{tick}:{j}',tick*self.cfg.dt,tuple(q),float(y))
                       for j,(q,y) in enumerate(zip(points,values)))
        visible=tuple(o for o in self._obstacles if (float(slab_distance(x[:2],o)) if is_slab(o) else np.linalg.norm(x[:2]-o[:2])-o[2])<=self.cfg.sensor_radius)
        return Observation(tick*self.cfg.dt,readings,visible,tuple(x[:2]) if self.cfg.controller_mode!='legacy' else None,self.cfg.sensor_radius if self.cfg.controller_mode!='legacy' else 0.)

    def step(self,x,u):
        after=flow(tensor(x),tensor(u),self._damping,self.cfg.dt).numpy()
        accounting=balance(tensor(x),tensor(u),self._damping,self.cfg.dt)
        a,b=x[:2],after[:2];segment=b-a
        collision=False
        for o in self._obstacles:
            if is_slab(o):
                collision |= slab_sweep(a,b,o)<=.12
                continue
            ox,oy,r=o
            c=np.array([ox,oy]);frac=np.clip(np.dot(c-a,segment)/(np.dot(segment,segment)+1e-20),0,1)
            collision |= np.linalg.norm(a+frac*segment-c)<=r+.12
        # A collision terminates the mission; no silent position projection/energy reset.
        weight=getattr(self,'spec',{}).get('risk_weight',1.)
        cost=self.cfg.dt*(1+4*weight*float(self.risk_truth(after[:2]))+.03*np.dot(u,u))
        return after,bool(collision),float(cost),accounting
