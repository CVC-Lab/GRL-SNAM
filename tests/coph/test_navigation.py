import copy,json,unittest
from dataclasses import asdict,replace
import numpy as np
import torch
from grl_snam.coph.records import Config,Observation,Reading,new_stage,Transition
from grl_snam.coph.learner import Learner
from grl_snam.coph.cooperative import CooperativeLearner
from grl_snam.coph.model import tensor,flow,balance
from grl_snam.coph.navigation import intersects,project,edge_clear
from grl_snam.coph.stage_prediction import FrozenStages
from grl_snam.coph.cooperative_evaluation import CooperativeWorld
from grl_snam.coph.diagnostics import Instrumented


# grl_snam.coph: CooperativeLearner no longer subclasses InstrumentedLearner; the hooks
# (replay_calls) come from composing the Instrumented mixin.
class InstrumentedCooperativeLearner(Instrumented,CooperativeLearner):
    pass


class NavigationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def config(self,**kw):return Config(replay_domain='retained',replay_objective='mission',controller_mode='routed',**kw)
    def observation(self,x=(1.,0.),obstacles=((1.8,0.,.4),)):
        return Observation(0.,tuple(Reading(f'0:0:{i*5+1}',0.,(x[0]+dx,x[1]+dy),.02) for i,(dx,dy) in enumerate([(a,b) for a in np.linspace(-1,1,7) for b in np.linspace(-1,1,7)])),obstacles,tuple(x),1.25)
    def test_geometry_intersection_eviction_and_overflow(self):
        cfg=self.config();a=Learner(cfg,(0,0,0,0),(8,0));a.observe(self.observation())
        self.assertFalse(a.stages[0].contains((1.8,0),cfg));self.assertIn((1.8,0.,.4),a.stages[0].obstacles)
        a.advance((1.1,0,0,0));self.assertIn((1.8,0.,.4),a.stages[-1].obstacles)
        for x in (3.3,5.4,7.5):a.advance((x,0,0,0))
        self.assertLessEqual(len(a.stages),3)
        b=Learner(self.config(obstacle_cap=1),(0,0,0,0),(8,0));b.observe(self.observation(obstacles=((1.,1.,.2),(1.,-1.,.2))))
        self.assertTrue(b.geometry_overflow);self.assertEqual(len(b.geometry_store),1)
        b.observe(self.observation(obstacles=()));self.assertTrue(b.geometry_overflow)
        b.action((0,0,0,0));self.assertEqual(b.last_control['reason'],'geometry_overflow')
    def test_projection_gradient_and_bounds(self):
        cfg=self.config();x=tensor([0.,0.,.5,.1]).requires_grad_(True);raw=tensor([4.,.3]).requires_grad_(True)
        u,_,res=project(raw,x,((1.2,0.,.25),),.8,cfg)
        self.assertIsNotNone(u);self.assertLessEqual(res,1e-8);self.assertLessEqual(float(torch.linalg.vector_norm(u)),3.00000001)
        g=torch.autograd.grad(u[1],raw)[0];eps=tensor([0.,1e-5])
        up=project(raw.detach()+eps,x.detach(),((1.2,0.,.25),),.8,cfg)[0]
        dn=project(raw.detach()-eps,x.detach(),((1.2,0.,.25),),.8,cfg)[0]
        self.assertAlmostEqual(float(g[1]),float((up[1]-dn[1])/2e-5),places=5)
        for d in cfg.damping_bounds:self.assertLessEqual(float(torch.linalg.vector_norm(flow(x,u,d,cfg.dt)[2:])),1.0000001)
    def test_shared_controller_energy_and_isolation(self):
        cfg=self.config();a=Learner(cfg,(0.,0.,0.,0.),(4.,0.));a.observe(self.observation((0.,0.),()))
        b=copy.deepcopy(a);atlas=FrozenStages.freeze(a);context={};x=tensor([0.,0.,0.,0.])
        for k in range(5):
            u,p=atlas.control(x,0,a.policy,a.damping,context,k);live,_=b.action(x.detach().numpy())
            np.testing.assert_allclose(u.detach().numpy(),live,atol=1e-12)
            self.assertLess(abs(balance(x.detach(),u.detach(),a.damping,cfg.dt)['residual']),1e-12)
            x=flow(x,u,a.damping,cfg.dt)
        self.assertFalse(a.navigation);self.assertTrue(context)
        self.assertFalse(edge_clear((0.,0.),(3.,0.),(),((0.,0.,1.25,0.),)))
    def test_controlled_disk_and_passage_complete(self):
        for obstacles in ([(2.,0.,.35)],[(2.,-.85,.35),(2.,.85,.35)]):
            cfg=self.config(steps=240);spec=dict(name='controlled',seed=91,initial=[0.,0.,0.,0.],goal=[4.,0.],damping=.8,patches=[],obstacles=obstacles,dynamic=False)
            world=CooperativeWorld(spec,cfg);a=Learner(cfg,spec['initial'],spec['goal'],mode='frozen');x=np.array(spec['initial']);hit=False
            a.observe(world.sense(x,0))
            for tick in range(cfg.steps):
                u,_=a.action(x);y,hit,_,_=world.step(x,u)
                self.assertFalse(hit,(obstacles,tick,x,a.last_control))
                json.dumps(a.last_control,allow_nan=False)
                a.observe(world.sense(y,tick+1),Transition(tick,(tick+1)*cfg.dt,tuple(x),tuple(u),tuple(y),a.version));x=y
                if np.linalg.norm(x[:2]-a.goal)<.25:break
                a.advance(x)
            self.assertLess(np.linalg.norm(x[:2]-a.goal),.25,(obstacles,x,a.last_control))
    def test_multistep_active_projection_gradient(self):
        cfg=self.config();obstacles=((1.2,0.,.25),)
        def objective(z):
            x=tensor([0.,0.,.5,.1])
            for _ in range(5):
                u,_,_=project(torch.stack((z*2, z*.3)),x,obstacles,.8,cfg)
                self.assertIsNotNone(u);x=flow(x,u,.8,cfg.dt)
            return x.square().sum()
        z=tensor(2.).requires_grad_(True);g=float(torch.autograd.grad(objective(z),z)[0]);eps=1e-5
        finite=float((objective(tensor(2.+eps))-objective(tensor(2.-eps)))/(2*eps))
        self.assertAlmostEqual(g,finite,places=5);self.assertGreater(abs(g),1e-5)
    def test_update_budget_rollback_and_route_isolation(self):
        a=InstrumentedCooperativeLearner(self.config(replay_horizon_mode='supported_boundary',replay_update_cap=1024),(0.,0.,0.,0.),(4.,0.),agent_id='0')
        a.observe(self.observation((0,0),()));a.action((0,0,0,0))
        navigation=copy.deepcopy(a.navigation);measurements=copy.deepcopy(a.stages[0].readings);policy=a.policy.clone();version=a.version
        report=a.update((0,0,0,0),force_invalid=True)
        self.assertFalse(report['accepted']);self.assertEqual(version,a.version)
        self.assertEqual(navigation,a.navigation);self.assertEqual(measurements,a.stages[0].readings)
        torch.testing.assert_close(policy,a.policy)
        self.assertLessEqual(report['counterfactual_steps'],1024)
        self.assertEqual(report['counterfactual_steps'],report.get('replay_pilot_steps',0)+sum(c['steps'] for c in a.replay_calls)+report['diagnostic_guard_steps'])
    def test_multiple_obstacles_and_damping_endpoints(self):
        cfg=self.config();x=tensor([0.,0.,.7,0.]);obstacles=((1.1,.55,.2),(1.1,-.55,.2))
        for damping in (*cfg.damping_bounds,.8):
            u,_,res=project(tensor([3.,.2]),x,obstacles,damping,cfg)
            self.assertIsNotNone(u);self.assertLessEqual(res,1e-8)
            for d in (*cfg.damping_bounds,damping):self.assertLessEqual(float(torch.linalg.vector_norm(flow(x,u,d,cfg.dt)[2:])),1.0000001)
        a=Learner(cfg,(0.,0.,.7,0.),(4.,0.));a.observe(self.observation((0,0),obstacles))
        u,_=a.action((0,0,.7,0));self.assertFalse(a.last_control['infeasible'])
        self.assertLessEqual(np.linalg.norm(u),cfg.force_limit+1e-8)

    def test_policy_independent_braking_replay(self):
        a=CooperativeLearner(self.config(replay_update_cap=1024),(0.,0.,0.,0.),(4.,0.),agent_id='0')
        a.observe(self.observation((0,0),()));a.geometry_overflow=True
        policy=a.policy.clone();report=a.update((0,0,0,0))
        self.assertEqual(report['policy_gradient_norm'],0.)
        self.assertTrue(report['policy_independent_replay']);torch.testing.assert_close(policy,a.policy)
        self.assertLessEqual(report['counterfactual_steps'],1024)
    def test_new_geometry_invalidates_future_route_edge(self):
        a=Learner(self.config(sensor_radius=4.),(0.,0.,0.,0.),(4.,0.))
        obs=replace(self.observation((0,0),()),sensor_radius=4.)
        a.observe(obs);a.action((0,0,0,0));self.assertGreater(a.navigation['path'][-1][0],1.3)
        a.observe(replace(obs,time=.12,obstacles=((1.45,0.,.05),)))
        a.action((0,0,0,0));self.assertTrue(a.last_control['route_changed'])
