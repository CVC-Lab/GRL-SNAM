import copy
from dataclasses import asdict,replace
import inspect,json,unittest
import numpy as np
import torch
from grl_snam.coph.records import Config,Reading,new_stage
from grl_snam.coph.model import tensor,flow,balance
from grl_snam.coph.cooperative import CooperativeLearner,CommunicationConfig,advertise
from grl_snam.coph.stage_prediction import FrozenStages,DomainExit

class StagePredictionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def fixture(self):
        cfg=Config(replay_objective='mission',replay_domain='retained',replay_steps=6)
        x=np.array([.85,0.,.8,0.]);a=CooperativeLearner(cfg,x,(3.,0.),agent_id='1')
        s0=new_stage(0,(0.,0.),x,a.goal,cfg);s1=new_stage(1,(2.1,0.),(1.1,0.,.8,0.),a.goal,cfg)
        for s in (s0,s1):
            points=[(x,y) for x in np.linspace(s.center[0]-1.5,s.center[0]+1.5,9) for y in np.linspace(-2,2,9)]
            s.readings=[Reading(f'1:fixture:{k*5+1}',0.,q,.1) for k,q in enumerate(points)]
        a.stages=[s0,s1];return a,x

    def test_crossing_execution_gradient_and_frontier(self):
        a,x=self.fixture();p=a.policy.clone().requires_grad_(True);weights=[s.weights for s in a.stages]
        c,paths=a.replay(p,weights,a.damping)
        self.assertIsNotNone(c);self.assertTrue(a.replay_stage_events)
        self.assertTrue(all(e['physical_jump']==0 for e in a.replay_stage_events))
        grad=torch.autograd.grad(c,p)[0];eps=tensor([1e-5,0.])
        plus,_=a.replay(p.detach()+eps,weights,a.damping);minus,_=a.replay(p.detach()-eps,weights,a.damping)
        self.assertAlmostEqual(float(grad[0]),float((plus-minus)/2e-5),places=6)
        live=copy.deepcopy(a);live.stages=[copy.deepcopy(a.stages[0])];state=x.copy()
        for k,expected in enumerate(paths[0][1][1:]):
            live.advance(state)
            u,_=live.action(state);account=balance(tensor(state),tensor(u),live.damping,live.cfg.dt)
            self.assertLess(abs(account['residual']),1e-12)
            state=flow(tensor(state),tensor(u),live.damping,live.cfg.dt).numpy()
            np.testing.assert_allclose(state,expected,atol=1e-12)
        a.stages=a.stages[:1]
        failed,_=a.replay(a.policy,[a.stages[0].weights],a.damping)
        self.assertIsNone(failed);self.assertEqual(a.replay_eligibility[-1]['reason'],'unavailable_stage')

    def test_frozen_and_reserved_support(self):
        a,x=self.fixture();atlas=FrozenStages.freeze(a);before=atlas.stages[0].weights
        a.stages[0].weights+=1;self.assertEqual(atlas.stages[0].weights,before)
        a.stages[0].readings=[Reading('1:reserved:0',0.,(0.,0.),1.)]
        frozen=FrozenStages.freeze(a)
        self.assertFalse(frozen.stages[0].supported((0.,0.),a.cfg))
        with self.assertRaises(DomainExit):frozen.index_at((100.,0.))

    def test_no_evaluator_imports(self):
        # grl_snam.coph: receiver_summary, model_prediction and continuations are not vendored.
        from grl_snam.coph import stage_prediction,replay_budget
        for module in (stage_prediction,replay_budget):
            self.assertNotIn('from .cooperative_evaluation',inspect.getsource(module))
            self.assertNotIn('from .recipient_diagnostics',inspect.getsource(module))


