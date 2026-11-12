import unittest
from dataclasses import replace
import numpy as np
import torch
from grl_snam.coph.crossing_demo import specification,configuration,CrossingWorld,CrossingLearner,predicted,door,NODES,EDGES
from grl_snam.coph.maze_geometry import obstacle_clear
from grl_snam.coph.maze_learning import maze_route
from grl_snam.coph.model import tensor,potential,flow
from grl_snam.coph.navigation import project
from grl_snam.coph.shapes import slab_distance,slab_sweep,surface_distance
from grl_snam.coph.stage_prediction import FrozenStages
from grl_snam.coph.staggered_demo import sense


class CrossingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.cfg=configuration(50);self.spec=specification(76000)
    def agent(self,i,x=None,now=1.):
        s=self.spec;x=np.array(s['initials'][i] if x is None else x,dtype=float)
        w=CrossingWorld(dict(s,initial=list(x),goal=s['goals'][i]),self.cfg)
        a=CrossingLearner(self.cfg,x,s['goals'][i],agent_id=str(i));a.now=now;a.observe(sense(w,x,0,now));return a,w

    def test_slab_distances(self):
        o=door('middle')
        self.assertAlmostEqual(float(slab_distance([9.5,4.],o)),-.1)
        self.assertAlmostEqual(float(slab_distance([9.5,5.],o)),1.-.75-.1)
        self.assertAlmostEqual(float(slab_distance([8.,4.3],o)),1.4)
        self.assertLess(slab_sweep([9.,4.5],[10.,4.5],o),0)
        self.assertAlmostEqual(slab_sweep([8.,4.],[9.,4.],o),.4)
        z=tensor([8.,4.3]).requires_grad_(True);d=surface_distance(z,[o])[0]
        np.testing.assert_allclose(torch.autograd.grad(d,z)[0].numpy(),[-1.,0.],atol=1e-12)

    def test_doors_are_invisible_from_starts_and_other_rows(self):
        for i in (0,1):
            a,_=self.agent(i);self.assertFalse(a.local_geometry_reports)
        _,w=self.agent(0)
        for q in ([2.5,4.85],[0.,6.],[9.5,6.6],[11.5,7.5]):
            self.assertNotIn(door('north') if q[1]<6 else door('middle'),sense(w,np.array([*q,0.,0.]),1,1.).obstacles)
        self.assertIn(door('middle'),sense(w,np.array([10.84,4.,0.,0.]),1,1.).obstacles)

    def test_doors_block_motion_and_edges(self):
        _,w=self.agent(0)
        for y in (3.3,4.,4.7):self.assertTrue(w.step(np.array([9.3,y,1.,0.]),np.zeros(2))[1])
        self.assertFalse(w.step(np.array([6.,4.,1.,0.]),np.zeros(2))[1])
        self.assertFalse(obstacle_clear(NODES[1],NODES[4],[door('middle')]))
        self.assertTrue(obstacle_clear(NODES[0],NODES[3],[door('middle'),door('north')]))

    def test_projection_respects_slab_barrier(self):
        o=door('middle');x=tensor([9.,4.2,.6,0.])
        u,_,residual=project(tensor([3.,0.]),x,(o,),.85,self.cfg)
        self.assertIsNotNone(u);self.assertLessEqual(residual,1e-8)
        for _ in range(40):
            x=flow(x,u,.85,self.cfg.dt);u,_,_=project(tensor([3.,0.]),x,(o,),.85,self.cfg)
            self.assertGreater(float(slab_distance(x[:2].numpy(),o)),.12)

    def test_warning_reroutes_before_local_sensing(self):
        sender,_=self.agent(1,[10.8,4.,0.,0.],now=3.);self.assertIn(door('middle'),sender.local_geometry_reports)
        receiver,_=self.agent(0,[1.,4.,.3,0.],now=3.)
        _,before=maze_route(FrozenStages.freeze(receiver),tensor(receiver.current_state),receiver.navigation,0);self.assertIn(4,before['route_nodes'])
        m=sender.export(np.array(sender.current_state),'0');self.assertTrue(any(len(o)==10 for o in m.geometry))
        self.assertEqual(receiver.assimilate(m)['reason'],'eligible');self.assertFalse(receiver.local_geometry_reports)
        _,after=maze_route(FrozenStages.freeze(receiver),tensor(receiver.current_state),receiver.navigation,1)
        self.assertEqual(after['route_nodes'][:2],[1,2]);self.assertIn(EDGES.index((1,4)),after['blocked_edges'])
        self.assertFalse(receiver.export(np.array(receiver.current_state),'1').geometry,'remote door echoed as local evidence')
        # A door claimed from a sensor position 9.5 m away is rejected.
        bad=replace(m,geometry=tuple((*o[:5],0.,4.,1.25,*o[-2:]) if len(o)==10 else o for o in m.geometry))
        fresh,_=self.agent(0,now=3.);self.assertEqual(fresh.assimilate(bad)['reason'],'invalid_geometry')

    def test_landscape_potential_includes_slab_barrier(self):
        a,_=self.agent(0);o=door('middle');target=tensor([12.,4.]);args=(a.stages[0].weights,a.stages[0].centers,self.cfg.rbf_width,a.policy)
        # Unscaled form: progress shaping also rescales by an obstacle-dependent uniform bound.
        barrier=lambda q:float(potential(tensor(q),target,*args,(o,))-potential(tensor(q),target,*args,()))
        self.assertGreater(barrier([9.15,4.]),barrier([8.5,4.]));self.assertGreater(barrier([8.5,4.]),barrier([7.,4.]))
        self.assertAlmostEqual(barrier([9.15,3.4]),barrier([9.15,4.6]))

    def test_predictions_are_superadditive(self):
        waste={arm:sum(v['wasted'] for v in predicted(arm).values()) for arm in ('disabled','one_way_01','one_way_10','reciprocal_fixed')}
        one_way=(waste['disabled']-waste['one_way_01'])+(waste['disabled']-waste['one_way_10'])
        self.assertAlmostEqual(waste['disabled']-waste['one_way_01'],0.)
        self.assertGreater(waste['disabled']-waste['reciprocal_fixed'],one_way+1)

    def test_report_names_doors_from_spec(self):
        # grl_snam.coph: door_name is copied into results from crossing_visuals (not vendored).
        from grl_snam.coph.results import door_name
        self.assertEqual(door_name(self.spec,door('middle')),'door 1');self.assertIsNone(door_name(self.spec,(1.,1.,.3)))


if __name__=='__main__':unittest.main()
