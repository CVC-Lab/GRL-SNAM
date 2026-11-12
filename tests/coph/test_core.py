import dataclasses
import inspect
import unittest
import numpy as np
import torch
from grl_snam.coph.model import tensor, flow, balance, potential, command
from grl_snam.coph.records import Config, Reading, Observation, Transition
from grl_snam.coph.learner import Learner
from grl_snam.coph.world import World

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=Config(replay_steps=5)

    def fixture(self):
        world=World(0,'risk',self.cfg)
        agent=Learner(self.cfg,world.initial,world.goal)
        for tick in range(12):agent.observe(world.sense(np.array([.09*tick,0.,0.,0.]),tick))
        return agent,world

    def test_exact_energy_and_flow_semigroup(self):
        x=tensor([.3,-.2,.8,-.4]);u=tensor([.2,.9])
        for d in [.15,.6,1.5]:
            self.assertLess(abs(balance(x,u,d,.13)['residual']),1e-12)
            self.assertTrue(torch.allclose(flow(flow(x,u,d,.05),u,d,.08),flow(x,u,d,.13),atol=1e-12))

    def test_navigation_goal_and_stage_advancement_are_independent(self):
        a,w=self.fixture();s=a.stages[-1];x=np.array([.9,-.1,.03,0.])
        expected,_=command(tensor(x),tensor(w.goal),s.weights,s.centers,a.cfg.rbf_width,a.policy,s.obstacles,a.cfg.force_limit)
        np.testing.assert_array_equal(a.action(x)[0],expected.detach().numpy())
        self.assertEqual(a.control_target(),tuple(w.goal))
        self.assertFalse(a.advance(np.array([1.04,0.,0.,0.])))
        self.assertTrue(a.advance(np.array([1.06,0.,0.,0.])))
        self.assertEqual(a.control_target(),tuple(w.goal))
        self.assertAlmostEqual(a.stages[-1].target[0],3.5)

    def test_goal_change_leaves_replay_identical_on_same_evidence(self):
        import copy
        a,_=self.fixture();b=copy.deepcopy(a)
        b.cfg=dataclasses.replace(a.cfg,navigation_target='stage')
        w,_=a._field_fit()
        left,paths=a.replay(a.policy,w,.8);right,other=b.replay(b.policy,w,.8)
        self.assertTrue(torch.equal(left,right))
        self.assertEqual(paths[0][1],other[0][1])

    def test_stage_change_does_not_jump_mission_attraction_energy(self):
        for mode in ('mission','stage'):
            a=Learner(dataclasses.replace(self.cfg,navigation_target=mode),(0.,0.,0.,0.),(5.,0.))
            a.policy=tensor([0.,0.]);a.advance(np.array([1.1,0.,0.,0.]))
            jump=a.context_jumps[-1]['shaping_jump']
            self.assertAlmostEqual(jump,0. if mode=='mission' else (.5*(3.5-1.1)**2-.5*(1.4-1.1)**2))

    def test_potential_force_finite_difference(self):
        a,_=self.fixture();s=a.stages[0];q=tensor([.1,.2]);policy=tensor([1.2,2.])
        u,_=command(torch.cat([q,tensor([0.,0.])]),tensor(s.target),s.weights,s.centers,self.cfg.rbf_width,policy,(),100.)
        g=[]
        for k in range(2):
            e=torch.zeros(2,dtype=torch.float64);e[k]=1e-6
            f=lambda z:potential(z,tensor(s.target),s.weights,s.centers,self.cfg.rbf_width,policy,())
            g.append(float((f(q+e)-f(q-e))/2e-6))
        np.testing.assert_allclose(u.detach().numpy(),-np.array(g),atol=1e-7)

    def test_multistep_replay_gradient(self):
        a,_=self.fixture();w,_=a._field_fit();p=tensor([1.,2.]).requires_grad_(True)
        loss,_=a.replay(p,w,.8);self.assertIsNotNone(loss)
        gradient=torch.autograd.grad(loss,p)[0]
        eps=1e-5;e=tensor([eps,0.])
        plus,_=a.replay(p.detach()+e,w,.8);minus,_=a.replay(p.detach()-e,w,.8)
        self.assertAlmostEqual(float(gradient[0]),float((plus-minus)/(2*eps)),places=5)
        self.assertGreater(abs(float(gradient[0])),1e-7)

    def test_immutable_and_bounded_records(self):
        a,w=self.fixture();r=a.stages[0].readings[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):r.value=9
        for tick in range(30):a.observe(w.sense(w.initial,tick+30))
        self.assertLessEqual(len(a.stages[0].readings),self.cfg.sample_cap)
        for x in [1.1,3.2,5.3]:a.advance(np.array([x,0.,0.,0.]))
        self.assertLessEqual(len(a.stages),self.cfg.previous+1)
        for left,right in zip(a.stages,a.stages[1:]):
            self.assertAlmostEqual(right.center[0]-left.center[0],2.1)

    def test_rollback_nonfinite_and_no_measured_replacement(self):
        a,w=self.fixture();old=a.policy.clone();weights=a.stages[0].weights.clone()
        samples=tuple(a.stages[0].readings)
        result=a.update(w.initial,force_invalid=True)
        self.assertFalse(result['accepted']);self.assertEqual(a.version,0)
        self.assertTrue(torch.equal(a.policy,old));self.assertTrue(torch.equal(a.stages[0].weights,weights))
        self.assertEqual(samples,tuple(a.stages[0].readings))
        self.assertEqual(result['physical_jump'],0.)

    def test_no_hidden_truth_interface(self):
        import grl_snam.coph.learner as module
        src=inspect.getsource(module)
        self.assertNotIn('from .world',src);self.assertNotIn('risk_truth',src)
        self.assertEqual(set(inspect.signature(Learner.observe).parameters),{'self','observation','transition'})
        self.assertEqual(set(Observation.__dataclass_fields__),{'time','readings','obstacles','sensor_center','sensor_radius'})

    def test_overlap_changes_earlier_model_and_policy_gradient(self):
        a,w=self.fixture();a.update(w.initial)
        a.advance(np.array([1.1,0.,0.,0.]))
        before=a.stages[0].weights.clone()
        for tick in range(20,28):a.observe(w.sense(np.array([1.15,0.,0.,0.]),tick))
        report=a.update(np.array([1.15,0.,0.,0.]))
        self.assertTrue(report['accepted'],report)
        self.assertGreater(float(torch.linalg.vector_norm(a.stages[0].weights-before)),1e-6)
        self.assertGreater(report['policy_gradient_norm'],1e-7)
        self.assertTrue(a.stages[-1].anchors)
        self.assertEqual(report['physical_jump'],0.)

    def test_reserved_samples_not_fit(self):
        a,w=self.fixture();weights,_=a._field_fit()
        for s in a.stages:
            s.readings=[dataclasses.replace(r,value=100.) if int(r.uid.split(':')[-1])%5==0 else r for r in s.readings]
        changed,_=a._field_fit()
        for x,y in zip(weights,changed):self.assertTrue(torch.equal(x,y))

    def test_replay_domain_exit_rejected(self):
        a,_=self.fixture();a.stages[0].entry=(1.49,0.,3.,0.)
        cost,_=a.replay(a.policy,[a.stages[0].weights],.8)
        self.assertIsNone(cost)

    def test_identification_uses_executed_inputs(self):
        a,_=self.fixture();x=tensor([0.,0.,0.,0.])
        for n in range(30):
            u=tensor([np.sin(n),np.cos(n)])
            y=flow(x,u,.9,self.cfg.dt)
            a.stages[0].transitions.append(Transition(n,n*self.cfg.dt,tuple(x.tolist()),tuple(u.tolist()),tuple(y.tolist()),0))
            x=y
        estimate=a._fit_damping()
        self.assertLess(abs(estimate-.9),.02)

    def test_stage_eviction_and_observation_deduplication(self):
        a=Learner(self.cfg,(0.,0.,0.,0.),(15.,0.))
        w=World(0,'risk',self.cfg)
        obs=w.sense(w.initial,0)
        a.observe(obs);count=len(a.stages[0].readings);a.observe(obs)
        self.assertEqual(len(a.stages[0].readings),count)
        for x in [1.1,3.2,5.3,7.4,9.5]:a.advance(np.array([x,0.,0.,0.]))
        self.assertEqual([s.id for s in a.stages],[3,4,5])

    def test_feasible_projection_and_reserved_validation_rollback(self):
        a,w=self.fixture()
        a._fit_damping=lambda: -50.
        report=a.update(w.initial)
        self.assertTrue(report['accepted'])
        self.assertGreaterEqual(a.damping,self.cfg.damping_bounds[0])
        self.assertLessEqual(abs(a.damping-self.cfg.damping_initial),self.cfg.damping_trust+1e-12)
        version=a.version
        previous=a._field_fit
        a._field_fit=lambda: ([torch.ones_like(s.weights)*100 for s in a.stages],[None]*len(a.stages))
        report=a.update(w.initial)
        self.assertFalse(report['accepted'])
        self.assertEqual(a.version,version)

    def test_paired_reproducible_rollout_and_energy_separation(self):
        from grl_snam.coph.cli import rollout
        cfg=Config(steps=20)
        x=rollout(cfg,4);y=rollout(cfg,4)
        self.assertEqual(x['trajectory'],y['trajectory'])
        self.assertEqual(x['summary']['mission_cost'],y['summary']['mission_cost'])
        self.assertTrue(all(u['physical_jump']==0 for u in x['updates']))
        self.assertTrue(any(abs(u['shaping_jump'])>0 for u in x['updates']))
