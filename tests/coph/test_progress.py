import numpy as np
import torch
from dataclasses import replace
from .test_navigation import NavigationTests
from grl_snam.coph.records import Config,Transition
from grl_snam.coph.model import tensor,potential,command,flow,shaping_parameters
from grl_snam.coph.learner import Learner
from grl_snam.coph.cooperative import CooperativeLearner
from grl_snam.coph.cooperative_evaluation import CooperativeWorld

class ProgressTests(NavigationTests):
    def config(self,**kw):
        return Config(replay_domain='retained',replay_objective='mission',controller_mode='routed',shaping_mode='progress',**kw)
    def test_gradient_radial_bound(self):
        cfg=self.config();rng=np.random.default_rng(3);centers=tensor(rng.normal(size=(16,2)));weights=tensor(rng.normal(size=16)*8);obstacles=((1.,.4,.4),(-1.,.8,.2));target=tensor([.2,-.1])
        bound=shaping_parameters(weights,.8,obstacles,cfg)
        for _ in range(40):
            q=tensor(rng.normal(size=2)).requires_grad_(True);policy=tensor(rng.uniform(0,10,2));grad=torch.autograd.grad(potential(q,target,weights,centers,.8,policy,obstacles,cfg),q)[0];e=q-target
            self.assertGreaterEqual(float(e@grad)+1e-12,.5*float(e@e))
            for j in range(2):
                delta=torch.zeros(2,dtype=torch.float64);delta[j]=1e-5
                fd=(potential(q.detach()+delta,target,weights,centers,.8,policy,obstacles,cfg)-potential(q.detach()-delta,target,weights,centers,.8,policy,obstacles,cfg))/2e-5
                self.assertAlmostEqual(float(grad[j]),float(fd),places=6)
            self.assertEqual(bound,shaping_parameters(weights,.8,obstacles,cfg))
    def test_multistep_potential_gradient(self):
        cfg=self.config();a=Learner(cfg,(0,0,0,0),(2,0));s=a.stages[0]
        def objective(p):
            x=tensor([.1,.2,.1,0.])
            for _ in range(8):
                u,_=command(x,tensor([2.,0]),s.weights,s.centers,.8,p,(),3.,cfg);x=flow(x,u,.8,.12)
            return x.square().sum()
        p=tensor([3.,2.]).requires_grad_(True);g=torch.autograd.grad(objective(p),p)[0];fd=(objective(tensor([3.00001,2.]))-objective(tensor([2.99999,2.])))/.00002
        self.assertAlmostEqual(float(g[0]),float(fd),places=7);self.assertGreater(abs(float(g[0])),1e-7)
    def test_stagnation_shortcut_no_route(self):
        a=Learner(self.config(),(0,0,0,0),(4,0));a.observe(self.observation((0,0),()));events=[]
        for k in range(25):
            a.action((0,0,0,0))
            if a.last_control['stagnation']:events.append(k);self.assertTrue(a.last_control['route_changed'])
        self.assertEqual(events,[16,24]);self.assertEqual(len(a.navigation['progress_history']),17)
        b=Learner(self.config(),(0,0,0,0),(.8,0));b.observe(self.observation((0,0),()));b.action((0,0,0,0));self.assertEqual(b.last_control['waypoint'],[.8,0])
        c=Learner(self.config(),(0,0,0,0),(4,0));u,_=c.action((0,0,0,0));np.testing.assert_array_equal(u,[0,0]);self.assertTrue(c.last_control['no_route'])
    def test_strong_risk_goal_capture_with_learning(self):
        for mode in ('frozen','full'):
            cfg=replace(self.config(steps=160),controller_mode='protected');spec=dict(name='goal_capture',seed=92,initial=[.34,-.14,0.,0.],goal=[0.,0.],damping=.8,patches=[[-.2,0.,.4,.5,8.]],obstacles=[],dynamic=False)
            world=CooperativeWorld(spec,cfg);a=Learner(cfg,spec['initial'],spec['goal'],mode=mode);x=np.array(spec['initial']);a.observe(world.sense(x,0))
            for tick in range(cfg.steps):
                if tick%cfg.update_every==0 and mode=='full':a.update(x)
                u,_=a.action(x);y,hit,_,_=world.step(x,u);self.assertFalse(hit);a.observe(world.sense(y,tick+1),Transition(tick,(tick+1)*cfg.dt,tuple(x),tuple(u),tuple(y),a.version));x=y
                if np.linalg.norm(x[:2])<.25:break
            self.assertLess(np.linalg.norm(x[:2]),.25)
    def test_route_jump_full_potential(self):
        a=Learner(self.config(),(0,0,0,0),(4,0));a.observe(self.observation((0,0),()));s=a.stages[0];x=tensor([0.,0.]);before=potential(x,tensor(a.goal),s.weights,s.centers,.8,a.policy,(),a.cfg)
        a.action((0,0,0,0));after=potential(x,tensor(a.navigation['target']),s.weights,s.centers,.8,a.policy,(),a.cfg);self.assertAlmostEqual(a.last_control['route_shaping_jump'],float(after-before),places=12)

# Avoid collecting the imported base class twice.
del NavigationTests
