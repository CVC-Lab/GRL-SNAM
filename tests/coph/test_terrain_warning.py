import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch
from grl_snam.coph.records import Config
from grl_snam.coph.terrain_warning import TerrainLearner,terrain_values,surveyed_route,ROUTES
from grl_snam.coph.cooperative_evaluation import CooperativeWorld
from grl_snam.coph.staggered_demo import sense,packet_effect
from grl_snam.coph.stage_prediction import FrozenStages
from grl_snam.coph.model import tensor


# grl_snam.coph: test fixture, upstream's terrain_warning_demo.specification (not vendored).
def specification(seed):
    return dict(name='unsafe_shortcut',seed=seed,initials=[[0.,0.,0.,0.]]*2,goals=[[8.,0.]]*2,
        initial=[0.,0.,0.,0.],goal=[8.,0.],start_seconds=[0.,5.],steps=500,damping=.85,
        obstacles=[],patches=[[3.6,0.,.5,.38,4.]],bounds=[-1.,9.,-1.5,3.],dynamic=False,
        surveyed_routes=ROUTES,prior='surveyed route topology only; terrain unknown, fixed risk prior 1',
        experiment='development mechanism demonstration; not held-out validation')


class TerrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=Config(controller_mode='routed',shaping_mode='progress',replay_domain='retained',replay_objective='mission',replay_horizon_mode='supported_boundary')
    def agents(self):
        s=specification(74000);w=CooperativeWorld(s,self.cfg);out=[]
        for i,x in enumerate((np.array([2.4,0.,0.,0.]),np.zeros(4))):
            a=TerrainLearner(self.cfg,x,s['goal'],agent_id=str(i));a.observe(sense(w,x,0,5.));out.append(a)
        return out
    def test_remote_support_does_not_become_measurements_or_replay_support(self):
        a,b=self.agents();m=a.export(np.array(a.current_state),'1');before=pickle.dumps([s.readings for s in b.stages]);atlas=FrozenStages.freeze(b)
        event=b.assimilate(m);self.assertEqual(event['reason'],'eligible');self.assertTrue(b.remote_terrain)
        self.assertEqual(before,pickle.dumps([s.readings for s in b.stages]));self.assertEqual(atlas,FrozenStages.freeze(b))
        self.assertEqual(b.assimilate(m)['reason'],'duplicate_or_reordered')
    def test_invalid_support_expiry_and_provenance(self):
        a,b=self.agents();m=a.export(np.array(a.current_state),'1')
        for wrong in (replace(m,support_points=((100.,100.),)),replace(m,expires=4.),replace(m,sources=('1:fake',))):
            self.assertNotEqual(b.assimilate(wrong)['reason'],'eligible');self.assertFalse(b.remote_terrain)
    def test_supported_predictions_only(self):
        a=[(0.,0.,4.,1.,0.)];np.testing.assert_allclose(terrain_values(np.array([[.1,0.],[2.,0.]]),a),[4.1,1.])
    def test_warning_changes_route_without_truth_or_state_change(self):
        _,a=self.agents();atlas=FrozenStages.freeze(a);x=tensor([1.,0.,0.,0.]);c={'surveyed_routes':ROUTES};surveyed_route(atlas,x,c,0);self.assertEqual(c['branch'],0)
        c['terrain_anchors']=[(float(z),0.,12.,0.,0.) for z in np.linspace(2.5,4.5,8)]
        surveyed_route(atlas,x,c,0);self.assertEqual(c['branch'],1);np.testing.assert_array_equal(x.numpy(),[1.,0.,0.,0.])
    def test_branch_audit_isolated_and_replay_context_present(self):
        a,b=self.agents();before=pickle.dumps(b);m=a.export(np.array(a.current_state),'1');r=packet_effect(b,np.zeros(4),m)
        self.assertTrue(r['operational_state_unchanged']);self.assertEqual(before,pickle.dumps(b));self.assertIn('surveyed_routes',b.stages[0].entry_navigation)

if __name__=='__main__':unittest.main()
