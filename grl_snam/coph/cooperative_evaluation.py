"""Two-agent evaluation and truth-only causal audits, outside learner interfaces."""
# grl_snam.coph: trimmed to the two-agent scenes (SCENES, specification) and CooperativeWorld, the
# world the demos' runner uses. The evaluation suite, its audits and scheduling are not vendored
# (VENDORED.md).
import numpy as np
from .scenarios import ScenarioWorld

SCENES=('complementary','early_warning','reverse_warning','redundant','irrelevant','changing_hazard','channel_stress')


def specification(name,seed,profile):
    if name not in SCENES:raise ValueError(name)
    rng=np.random.default_rng(seed);jitter=float(rng.uniform(-.12,.12))
    initial=[[0.,-.4,0.,0.],[0.,.4,0.,0.]];goals=[[6.,-.4],[6.,.4]]
    patches=[[2.4,.3+jitter,.7,.5,.9],[4.2,-.4+jitter,.6,.5,.7]]
    if name=='early_warning':initial[0][0]=1.
    if name=='reverse_warning':initial[1][0]=1.
    if name=='redundant':patches=[[2.4,jitter,.7,.7,.9]]
    if name=='irrelevant':initial[1][1]=3.;goals[1][1]=3.
    return dict(name=name,seed=seed,initials=initial,goals=goals,patches=patches,
        initial=initial[0],goal=goals[0],damping=float(rng.uniform(.7,1.05)),obstacles=[],
        bounds=[-1.5,7.5,-2.,4.5 if name=='irrelevant' else 2.],
        steps=96 if profile=='smoke' else 180,dynamic=name=='changing_hazard',
        channel='stress' if name=='channel_stress' else 'normal',
        dynamic_model='smooth sinusoidal amplitudes, period 8 seconds',
        interaction='independent plants sharing information; no inter-agent contact dynamics')


class CooperativeWorld(ScenarioWorld):
    def __init__(self,spec,cfg):super().__init__(spec,cfg);self.time=0.
    def _factor(self,k):return (1+.7*np.sin(2*np.pi*self.time/8+k*np.pi)) if self.spec.get('dynamic') else 1.
    def risk_truth(self,q):
        q=np.asarray(q);value=np.full(q.shape[:-1],.02)
        for k,(x,y,sx,sy,amp) in enumerate(self.spec['patches']):
            z=(q-[x,y])/[sx,sy];value+=self._factor(k)*amp*np.exp(-.5*np.sum(z*z,axis=-1))
        return value
    def risk_gradient(self,q):
        q=np.asarray(q);g=np.zeros_like(q,dtype=float)
        for k,(x,y,sx,sy,amp) in enumerate(self.spec['patches']):
            delta=q-[x,y];scale=np.array([sx,sy]);b=self._factor(k)*amp*np.exp(-.5*np.sum((delta/scale)**2,axis=-1))
            g-=b[...,None]*delta/scale**2
        return g
