import pickle,unittest
import numpy as np
import torch
from grl_snam.coph.hazard_demo import specification,learner,communication,ARMS,run
from grl_snam.coph.hazard_oracle import simulate
from grl_snam.coph.maze_family import NetworkWorld
from grl_snam.coph.crossing_demo import configuration
from grl_snam.coph.sharing import BLOCKED
from grl_snam.coph.staggered_demo import sense


class SharingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.spec=specification('moderate');self.cfg=configuration(40)
    def agent(self,arm,i=0,q=(6.,4.)):
        x=np.array([*q,0.,0.]);w=NetworkWorld(dict(self.spec,initial=list(x),goal=self.spec['goals'][i],risk_weight=self.spec['risk_weights'][i]),self.cfg)
        a=learner(self.spec,arm)[i](self.cfg,x,self.spec['goals'][i],agent_id=str(i),communication=communication(arm));a.now=1.
        for k in range(3):a.observe(sense(w,x,k,1.+k*.12))
        a.update(x);return a,x

    def test_modes_change_only_environmental_content(self):
        field=self.agent('field')[0];m=field.export(np.array(field.current_state),'1')
        self.assertTrue(any(abs(z[3])+abs(z[4])>0 for z in m.anchors),'field packets carry gradients')
        for arm in ('occupancy_0p25','decision'):
            a,_=self.agent(arm);p=a.export(np.array(a.current_state),'1')
            self.assertTrue(all(z[2] in (0.,BLOCKED) and z[3]==z[4]==0. for z in p.anchors));self.assertEqual([z[:2] for z in p.anchors],[z[:2] for z in m.anchors])
        raw,_=self.agent('raw');p=raw.export(np.array(raw.current_state),'1');readings={tuple(r.q):r.value for s in raw.stages for r in s.readings}
        self.assertTrue(all(readings[tuple(z[:2])]==z[2] and z[3]==z[4]==0. for z in p.anchors));self.assertGreater(p.bytes(),m.bytes())

    def test_decision_threshold_is_the_senders(self):
        a,_=self.agent('decision');p=a.export(np.array(a.current_state),'1')
        # Scout w=0.25 avoids only above 1/(4*0.25)=1: a 0.5 hazard is reported free.
        self.assertTrue(all(z[2]==0. for z in p.anchors))

    def test_no_learning_receiver_ignores_packets_in_its_fit(self):
        sender,_=self.agent('field',0,(6.,4.));m=sender.export(np.array(sender.current_state),'1')
        for arm,changes in (('field',True),('no_learning',False)):
            r,_=self.agent(arm,1,(6.4,4.));before=[w.clone() for w in r._field_fit()[0]];r.assimilate(m)
            after=r._field_fit()[0];self.assertEqual(any(not torch.equal(a,b) for a,b in zip(before,after)),changes,arm)
            self.assertTrue(r.received,'packet still stored for routing')

    def test_learners_pickle(self):
        for arm in ARMS:
            a,_=self.agent(arm);self.assertEqual(pickle.loads(pickle.dumps(a)).navigation.get('risk_weight'),.25)

    def test_oracle_without_hazards_takes_the_middle_row(self):
        spec=dict(self.spec,patches=[]);o=simulate(spec,'field')['agents']
        for a in o:self.assertAlmostEqual(a['path'],12.,places=6)

    def test_episode_runs_with_weights(self):
        spec=dict(self.spec,steps=30,start_seconds=[0.,1.])
        from grl_snam.coph.staggered_demo import execute
        factories=learner(spec,'field')
        r=execute(spec,'reciprocal_fixed',communication('field'),audit=False,learner_factory=lambda cfg,x,goal,agent_id,communication:factories[int(agent_id)](cfg,x,goal,agent_id=agent_id,communication=communication),
                  world_factory=NetworkWorld,config=configuration(30))
        risk=[t['components']['risk'] for t in r['trajectories'][1]];self.assertTrue(all(np.isfinite(risk)))
        # Carrier's logged risk cost uses its own weight 6.
        t=r['trajectories'][1][-1];self.assertGreater(t['components']['risk'],0.)


if __name__=='__main__':unittest.main()


class LedgerExportTests(unittest.TestCase):
    def test_rotation_covers_retained_stages_and_pickles(self):
        torch.set_num_threads(1);spec=specification('double_mild');cfg=configuration(40)
        a=learner(spec,'field')[0](cfg,np.array([0.,4.,0.,0.]),spec['goals'][0],agent_id='0',communication=communication('field'))
        w=NetworkWorld(dict(spec,initial=[0.,4.,0.,0.],goal=spec['goals'][0],risk_weight=.25),cfg);a.now=0.
        for k,x in enumerate(np.linspace(0,6,40)):
            q=np.array([x,4.,0.,0.]);a.now=k*.12;a.observe(sense(w,q,k,a.now));a.current_state=tuple(q);a.advance(q)
        self.assertGreaterEqual(len(a.stages),2);ids=[a.export(np.array(a.current_state),'1').stage for _ in range(len(a.stages))]
        self.assertEqual(sorted(ids),sorted(s.id for s in a.stages));self.assertEqual(type(pickle.loads(pickle.dumps(a))).__name__,'LedgerFieldLearner')
