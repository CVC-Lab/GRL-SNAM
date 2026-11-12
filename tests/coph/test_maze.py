import copy,pickle,unittest
from dataclasses import replace
import numpy as np
import torch
from grl_snam.coph.maze_geometry import NODES,EDGES,CORRIDOR_RADIUS,corridor_clear,distance_to_network
from grl_snam.coph.maze_learning import MazeLearner,MazePacket,maze_route
from grl_snam.coph.model import tensor,potential,flow
from grl_snam.coph.stage_prediction import FrozenStages
from grl_snam.coph.staggered_demo import sense
from grl_snam.coph.cooperative_evaluation import CooperativeWorld
from grl_snam.coph.records import Config

# grl_snam.coph: test fixtures, upstream's maze_demo specification, MazeWorld and configuration
# (maze_demo is not vendored).


def specification(seed):
    # Unknown gates span the full traversable corridor. Three overlapping disks
    # are the simulator's conservative rounded-barrier representation, not props.
    gates=[(5.3,0.,'vertical'),(8.3,3.,'vertical'),(11.3,0.,'vertical')]
    obstacles=[]
    for x,y,d in gates:
        obstacles.extend([[x,y+dy,.34] for dy in (-.55,0.,.55)])
    return dict(name='surveyed_maze_hidden_gates',seed=seed,initials=[[0.,0.,0.,0.]]*2,goals=[[12.,0.]]*2,
        initial=[0.,0.,0.,0.],goal=[12.,0.],start_seconds=[0.,5.],steps=1500,damping=.85,
        obstacles=obstacles,patches=[],bounds=[-1.,13.,-1.,7.],dynamic=False,
        gate_groups=gates,network_nodes=NODES,network_edges=EDGES,corridor_radius=CORRIDOR_RADIUS,
        prior='Surveyed corridor layout only. Closures unknown. Independent plants; no inter-agent contact dynamics.',
        experiment='development maze mechanism demo, not held-out evaluation')


class MazeWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        # Same public corridor footprint as the controller; actual agent radius .12.
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12))
        return y,hit,cost,energy


def configuration(steps):
    return Config(controller_mode='routed',shaping_mode='progress',steps=steps,replay_objective='mission',replay_domain='retained',replay_horizon_mode='supported_boundary',replay_update_cap=1024)


class MazeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.cfg=configuration(50)
    def agent(self,x=(0.,0.,0.,0.),i='0',now=10):
        s=specification(75000);w=MazeWorld(s,self.cfg);a=MazeLearner(self.cfg,np.array(x),s['goal'],agent_id=i);a.observe(sense(w,np.array(x),0,now));return a,w
    def test_packet_geometry_changes_route_before_local_sensing(self):
        scout,w=self.agent((4.,0.,0.,0.));carrier,_=self.agent((2.,0.,0.,0.),'1');self.assertFalse(carrier.local_geometry_reports)
        m=scout.export(np.array(scout.current_state),'1');self.assertTrue(m.geometry)
        before=pickle.dumps([s.readings for s in carrier.stages]);state=carrier.current_state
        event=carrier.assimilate(m);self.assertEqual(event['reason'],'eligible');self.assertEqual(state,carrier.current_state)
        self.assertEqual(before,pickle.dumps([s.readings for s in carrier.stages]));atlas=FrozenStages.freeze(carrier)
        _,meta=maze_route(atlas,tensor(state),carrier.navigation,0);self.assertIn(1,meta['blocked_edges']);self.assertNotIn(2,meta['route_nodes'][:2])
        outbound=carrier.export(np.array(state),'0');self.assertFalse(outbound.geometry,'remote geometry was echoed as local evidence')
        self.assertEqual(carrier.assimilate(m)['reason'],'duplicate_or_reordered')
    def test_fast_projection_matches_complete_enumeration(self):
        from grl_snam.coph.navigation import project
        for q,v,raw in [([0.,0.],[0.,0.],[.5,.1]),([0.,0.],[.7,0.],[3.,0.]),([.5,0.],[.5,0.],[2.,0.])]:
            x=tensor(q+v);obstacles=((1.5,0.,.34),)
            a,_,_=project(tensor(raw),x,obstacles,.85,self.cfg)
            b,_,_=project(tensor(raw),x,obstacles,.85,self.cfg,fast=True)
            if a is None:self.assertIsNone(b)
            else:np.testing.assert_allclose(a.detach().numpy(),b.detach().numpy(),atol=1e-12)

    def test_bad_geometry_is_rejected(self):
        scout,_=self.agent((4.,0.,0.,0.));carrier,_=self.agent(i='1');m=scout.export(np.array(scout.current_state),'1')
        for bad in (replace(m,geometry_schema=9),replace(m,geometry=((5.,0.,.3,0.,0.,.1,10.,'0:geometry:0'),))):
            self.assertNotEqual(carrier.assimilate(bad)['reason'],'eligible');self.assertFalse(carrier.remote_geometry_reports)
    def test_real_corridor_and_gate_collisions(self):
        a,w=self.agent();self.assertTrue(corridor_clear([0,0],[3,0]));self.assertFalse(corridor_clear([1,0],[1,3]))
        _,hit,_,_=w.step(np.array([1.,.74,0.,1.]),np.zeros(2));self.assertTrue(hit)
        _,hit,_,_=w.step(np.array([4.8,0.,1.,0.]),np.zeros(2));self.assertTrue(hit)
    def test_spatial_stages_allow_vertical_and_return_moves(self):
        a,w=self.agent();self.assertTrue(a.advance(np.array([0.,1.41,0.,0.])));self.assertAlmostEqual(a.stages[-1].center[1],2.8)
        self.assertTrue(a.advance(np.array([0.,1.39,0.,0.])));self.assertAlmostEqual(a.stages[-1].center[1],0.)
        self.assertEqual(FrozenStages.freeze(a).stage_mode,'spatial');self.assertLessEqual(len(a.stages),3)
    def test_exact_controller_gradients(self):
        # grl_snam.coph: upstream's test_exact_landscape_and_controller_gradients without its
        # maze_visuals.landscape checks (maze_visuals is not vendored).
        a,_=self.agent();a.update(np.array(a.current_state));atlas=FrozenStages.freeze(a);q=np.array([.23,.02,.1,0.]);before=copy.deepcopy(a.navigation)
        u,_=a.action(q);v,_=atlas.control(tensor(q),len(atlas.stages)-1,a.policy,damping=a.damping,context=copy.deepcopy(before),step=0)
        np.testing.assert_allclose(u,v.detach().numpy(),atol=1e-12)
        def objective(z):
            context=copy.deepcopy(before)
            for k in range(3):
                control,_=atlas.control(z,0,a.policy,damping=a.damping,context=context,step=k);z=flow(z,control,a.damping,a.cfg.dt)
            return z.square().sum()
        z=tensor(q).requires_grad_(True);gradient=torch.autograd.grad(objective(z),z)[0].numpy()
        fd=[float((objective(tensor(q+np.eye(4)[i]*1e-6))-objective(tensor(q-np.eye(4)[i]*1e-6)))/(2e-6)) for i in range(4)]
        np.testing.assert_allclose(gradient,fd,atol=1e-5,rtol=1e-4)

if __name__=='__main__':unittest.main()
