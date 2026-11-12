import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch
from grl_snam.coph.staggered_demo import execute,sense,packet_effect
from grl_snam.coph.records import Config
from grl_snam.coph.cooperative import CooperativeLearner
from grl_snam.coph.cooperative_evaluation import CooperativeWorld
import grl_snam.coph.cooperative_evaluation as cooperative_evaluation

# grl_snam.coph: test fixtures. scene() is upstream's complex_environments.scene and specification()
# upstream's staggered_demo.specification, copied here because neither module is vendored.
SCENES=('hazard_slalom','blocked_risk_passage','asymmetric_coverage','long_obstacle_corridor')


def scene(name,seed):
    if name not in SCENES:raise ValueError(name)
    s=cooperative_evaluation.specification('complementary',seed,'development');rng=np.random.default_rng(seed)
    jitter=float(rng.uniform(-.12,.12));length=20. if name=='long_obstacle_corridor' else 10.
    s.update(name=name,initials=[[0.,-.65,0.,0.],[0.,.65,0.,0.]],goals=[[length,-.65],[length,.65]],
        bounds=[-1.5,length+1.5,-3.,3.],steps=int(np.ceil(180*length/6)),dynamic=False,channel='normal')
    if name=='hazard_slalom':
        s['obstacles']=[[2.7,-.65,.5],[5.2,.65,.55],[7.6,-.55,.5]]
        s['patches']=[[2.5,1.+jitter,.8,.6,1.1],[5.,-1.,.9,.65,1.3],[7.8,1.,.8,.6,1.2]]
    elif name=='blocked_risk_passage':
        s['obstacles']=[[3.8,-1.1,.6],[3.8,1.1,.6],[6.6,0.,.65],[7.,1.8,.45]]
        s['patches']=[[3.8,jitter,.7,.75,1.6],[6.8,-1.25,.9,.5,1.3],[8.3,.6,.7,.8,.8]]
    elif name=='asymmetric_coverage':
        s['initials']=[[1.4,-1.1,0.,0.],[0.,1.1,0.,0.]];s['goals']=[[length,.8],[length,-.8]]
        s['obstacles']=[[3.,0.,.65],[5.5,1.15,.55],[7.5,-.9,.55]]
        s['patches']=[[2.5,-.9+jitter,.7,.5,1.5],[4.7,.6,.9,.75,1.2],[7.,-.9,.8,.55,1.4],[8.8,1.,.7,.5,.9]]
    else:
        s['obstacles']=[[x,(-1)**k*.8,.5] for k,x in enumerate(np.arange(2.8,19.,2.6))]
        s['patches']=[[x,(-1)**(k+1)*.9+jitter,.8,.6,1.+.15*(k%3)] for k,x in enumerate(np.arange(2.4,19.,2.6))]
    s.update(initial=s['initials'][0],goal=s['goals'][0]);return s


def specification(name,seed):
    s=scene('long_obstacle_corridor',seed)
    if name=='slalom_passage':
        s.update(steps=420,bounds=[-1.5,13.5,-3.,3.],obstacles=[[2.7,-.65,.5],[5.,-1.1,.6],[5.,1.1,.6],[7.6,0.,.6],[10.,.8,.5]],
            patches=[[2.5,.8,.8,.55,1.4],[5.,0.,.7,.65,1.8],[8.,-1.2,.8,.55,1.5],[10.5,-.7,.7,.6,1.3]])
        goal=[12.,0.]
    elif name=='long_corridor':goal=[20.,0.]
    else:raise ValueError(name)
    s.update(name='staggered_'+name,initials=[[0.,0.,0.,0.]]*2,goals=[goal,goal],initial=[0.,0.,0.,0.],goal=goal,start_seconds=[0.,5.],
        experiment='development demonstration; not held-out validation',interaction='independent agents; no inter-agent collision dynamics',waiting='inactive: no sensing, learning or communication before activation')
    return s


class StaggeredTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_exact_start_full_local_horizon_and_no_prestart_observations(self):
        s=specification('slalom_passage',73000);s['steps']=2
        r=execute(s,'reciprocal_fixed',audit=False)
        self.assertEqual([a['start_time'] for a in r['agents']],[0.,5.])
        self.assertEqual([a['finish_time'] for a in r['agents']],[.24,5.24])
        self.assertEqual([len(t) for t in r['trajectories']],[2,2]);self.assertFalse(r['packets'])
        np.testing.assert_array_equal(r['trajectories'][0][0]['x'],r['trajectories'][1][0]['x'])
        self.assertEqual(r['samples'][1][0]['time'],5.)
        self.assertTrue(all(o['time']>=5 and all(z['time']==o['time'] for z in o['readings']) for o in r['observations'][1]))
        self.assertEqual(r['agents'][1]['components']['time'],.24)
    def test_asynchronous_delivery_clock_and_matched_updates(self):
        s=specification('slalom_passage',73000);s.update(steps=12,start_seconds=[0.,.2])
        a=execute(s,'disabled',audit=False);b=execute(s,'reciprocal_fixed',audit=False)
        self.assertEqual([[u['time'] for u in seq] for seq in a['updates']],[[u['time'] for u in seq] for seq in b['updates']])
        self.assertTrue(b['packets'])
        for e in b['events']:
            if e['kind']=='delivered':
                self.assertGreaterEqual(e['age'],-1e-12);self.assertLessEqual(e['age'],.12+1e-9)
                self.assertIn(e['time'],[s['time'] for s in b['samples'][e['receiver']]])
                self.assertNotEqual(e['reason'],'invalid_age')
        self.assertTrue(all(a['max_update_work']<=1024 for a in b['agents']))
    def test_packet_audit_isolation(self):
        cfg=Config(controller_mode='routed',shaping_mode='progress',replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary')
        s=specification('slalom_passage',73000);w=CooperativeWorld(s,cfg);x=np.array([0.,0.,0.,0.]);agents=[]
        for i in range(2):
            a=CooperativeLearner(cfg,x,s['goal'],agent_id=str(i));a.observe(sense(w,x,0,5.));agents.append(a)
        m=agents[0].export(x,'1');before=pickle.dumps(agents[1]);r=packet_effect(agents[1],x,m)
        self.assertEqual(before,pickle.dumps(agents[1]));self.assertTrue(r['operational_state_unchanged']);self.assertIn('withhold',r['branches'])
    def test_scenes_deterministic_and_shared_origin(self):
        for name in ('slalom_passage','long_corridor'):
            a=specification(name,73000);self.assertEqual(a,specification(name,73000));self.assertEqual(a['initials'][0],a['initials'][1]);self.assertEqual(a['goals'][0],a['goals'][1]);self.assertEqual(a['start_seconds'],[0.,5.])
