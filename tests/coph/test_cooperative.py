import copy
from dataclasses import replace
import inspect
import unittest
import numpy as np
import torch
from grl_snam.coph.records import Config,Observation,Reading
from grl_snam.coph.model import tensor,flow,risk,command
from grl_snam.coph.cooperative import CooperativeLearner,CommunicationConfig,CONVENTION,information_gain,advertise,rate_probability
from grl_snam.coph.cooperative_evaluation import specification,CooperativeWorld

class CooperativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=Config(replay_objective='mission',replay_steps=3)
        self.comm=CommunicationConfig()
    def pair(self):
        spec=specification('complementary',34,'smoke');w=CooperativeWorld(spec,self.cfg)
        agents=[]
        for i in range(2):
            x=np.array([.1,.2*i,0.,0.]);a=CooperativeLearner(self.cfg,x,(6.,0.),agent_id=str(i),communication=self.comm)
            for t in range(10):a.observe(w.sense(x,t))
            agents.append(a)
        return agents,w
    def test_replay_matches_execution_and_gradient(self):
        (a,b),w=self.pair();s=a.stages[-1];p=a.policy.clone().requires_grad_(True)
        cost,paths=a.replay(p,[s.weights],a.damping);self.assertIsNotNone(cost)
        x=tensor(s.entry);expected=0.
        for k in range(a.cfg.replay_steps):
            u,_=a.action(x.detach().numpy());x=flow(x,tensor(u),a.damping,a.cfg.dt)
            expected+=a.cfg.dt*(1+4*float(risk(x[:2],s.weights,s.centers,a.cfg.rbf_width))+.03*float(u@u))
        np.testing.assert_allclose(paths[0][1][-1],x.detach().numpy(),atol=1e-12)
        self.assertAlmostEqual(float(cost),expected,places=12)
        grad=torch.autograd.grad(cost,p)[0];eps=tensor([1e-5,0.])
        plus,_=a.replay(p.detach()+eps,[s.weights],a.damping);minus,_=a.replay(p.detach()-eps,[s.weights],a.damping)
        self.assertAlmostEqual(float(grad[0]),float((plus-minus)/2e-5),places=6)
    def test_namespace_export_and_reciprocal_update(self):
        (a,b),w=self.pair();x=np.array([.1,0.,0.,0.]);packets=[a.export(x,'1'),b.export(x,'0')]
        self.assertTrue(set(packets[0].sources).isdisjoint(packets[1].sources))
        for receiver,packet in [(a,packets[1]),(b,packets[0])]:
            measured=copy.deepcopy(receiver.stages[0].readings);state=x.copy();old=receiver.stages[0].weights.clone()
            self.assertEqual(receiver.assimilate(packet)['reason'],'eligible');r=receiver.update(state)
            np.testing.assert_array_equal(state,x);self.assertEqual(measured,receiver.stages[0].readings)
            self.assertTrue(r['accepted']);self.assertGreater(float(torch.linalg.vector_norm(receiver.stages[0].weights-old)),0)
            self.assertEqual(r['physical_jump'],0)
        # Receiving operational knowledge cannot alter locally reconstructed exports.
        again=a.export(x,'1');self.assertEqual(again.anchors,packets[0].anchors);self.assertEqual(again.sources,packets[0].sources)
    def test_message_has_effect_beyond_local_update_and_replacement(self):
        (a,b),_=self.pair();x=np.array([.1,.2,0.,0.]);packet=a.export((.1,0.,0.,0.),'1')
        baseline=copy.deepcopy(b);baseline.update(x)
        b.assimilate(packet);b.update(x)
        self.assertGreater(float(torch.linalg.vector_norm(b.stages[0].weights-baseline.stages[0].weights)),1e-10)
        newer=replace(packet,timestamp=b.now+.01,identity=packet.identity+':new',expires=b.now+1)
        b.now+=.01;b.assimilate(newer)
        self.assertEqual(len(b.received),1);self.assertEqual(b.received['0'].identity,newer.identity)
    def test_order_independent_agents_and_acknowledged_information(self):
        (a,b),_=self.pair();x=(.1,0.,0.,0.)
        packets=[a.export(x,'1'),b.export(x,'0')]
        left=[copy.deepcopy(a),copy.deepcopy(b)];right=[copy.deepcopy(a),copy.deepcopy(b)]
        for i in (0,1):left[1-i].assimilate(packets[i])
        for i in (1,0):right[1-i].assimilate(packets[i])
        for i in (0,1):
            left[i].update(np.array(x));right[i].update(np.array(x))
            self.assertTrue(torch.equal(left[i].stages[0].weights,right[i].stages[0].weights))
        status=advertise(left[1],x,'0',[z[:2] for z in packets[0].anchors])
        self.assertEqual(information_gain(packets[0],status)[0],0.)
    def test_duplicate_expiry_return_reordering_and_bounds(self):
        (a,b),_=self.pair();m=a.export((.1,0.,0.,0.),'1')
        self.assertEqual(b.assimilate(m)['reason'],'eligible')
        self.assertEqual(b.assimilate(m)['reason'],'duplicate_or_reordered')
        self.assertEqual(a.assimilate(replace(m,recipient='0'))['reason'],'returned_evidence')
        self.assertEqual(b.assimilate(replace(m,timestamp=m.timestamp-1))['reason'],'duplicate_or_reordered')
        self.assertEqual(b.assimilate(replace(m,expires=-1))['reason'],'invalid_age')
        self.assertEqual(b.assimilate(replace(m,convention='physical potential'))['reason'],'incompatible_convention')
        self.assertEqual(b.assimilate(replace(m,sender='other',sources=('bad:0',)))['reason'],'invalid_provenance')
        b.now+=3;b._expire();self.assertEqual(b.received,{})
    def test_covariance_common_variables_and_rate_budget(self):
        (a,b),_=self.pair();m=a.export((.1,0.,0.,0.),'1');status=advertise(b,(.1,.2,0.,0.),'0',[z[:2] for z in m.anchors])
        ig,count=information_gain(m,status);self.assertGreater(ig,0);self.assertEqual(count,len(m.anchors))
        self.assertEqual(rate_probability(1,100,1,self.comm,99),(0.,0.))
        self.assertEqual(rate_probability(-1,100,1,self.comm,1000),(0.,0.))
    def test_rollback(self):
        (a,b),w=self.pair();x=np.array([.1,.2,0.,0.]);m=a.export((.1,0.,0.,0.),'1')
        b.assimilate(m);old=b.stages[0].weights.clone();v=b.version
        self.assertFalse(b.update(x,force_invalid=True)['accepted']);self.assertEqual(v,b.version)
        self.assertTrue(torch.equal(old,b.stages[0].weights))
    def test_nonfinite_packet_timestamps_are_rejected(self):
        (a,b),_=self.pair();packet=a.export((.1,0.,0.,0.),'1')
        for changes in ({'timestamp':float('nan')},{'expires':float('nan')}):
            self.assertEqual(b.assimilate(replace(packet,**changes))['reason'],'invalid_age')
        self.assertEqual(b.received,{})
    def test_temporal_refit_expires_measurements(self):
        a=CooperativeLearner(self.cfg,(0,0,0,0),(6,0),agent_id='0',dynamic=True)
        a.observe(Observation(0.,(Reading('0:0:1',0.,(0.,0.),1.),),()))
        a.observe(Observation(4.,(Reading('0:1:1',4.,(0.,0.),.1),),()))
        self.assertEqual(len(a.stages[0].readings),1);self.assertEqual(a.stages[0].readings[0].value,.1)
    def test_truth_exclusion_and_reproducible_scenarios(self):
        import grl_snam.coph.cooperative as m
        self.assertNotIn('risk_truth',inspect.getsource(m));self.assertNotIn('CooperativeWorld',inspect.getsource(m))
        self.assertEqual(specification('complementary',3,'smoke'),specification('complementary',3,'smoke'))
