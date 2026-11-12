"""Crossing traverses: opposite starts, swapped goals, two hidden full-width doors.

Each agent's starting end is the other's destination, so what one agent has already
seen is exactly what the other is about to need. Door truth lives here and in the
world only; learners know the surveyed corridor layout and nothing else.
"""
import argparse,copy
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .cooperative_evaluation import CooperativeWorld
from .staggered_demo import execute
from .maze_learning import MazeLearner
from .maze_geometry import corridor_clear,CORRIDOR_RADIUS
from .cooperative import CommunicationConfig
from .records import Config
from .evaluation import dump
from .cli import source_hashes,archive_sources

LENGTH=12.
ROWS=dict(south=0.,middle=4.,north=7.5)
# West column x=0, east column x=12; rows join only at the two ends.
NODES=((0.,0.),(0.,4.),(0.,7.5),(12.,0.),(12.,4.),(12.,7.5))
EDGES=((0,1),(1,2),(1,4),(2,5),(0,3),(3,4),(4,5))
DOOR_RADIUS=.1
# Full-width slabs: the core segment plus radius spans the whole corridor.
DOORS=dict(middle=9.5,north=2.5)
ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
SEEDS=(76000,76001)


def door(row):
    return (DOORS[row],ROWS[row],DOOR_RADIUS,0.,CORRIDOR_RADIUS-DOOR_RADIUS)


def specification(seed,steps=2500):
    return dict(name='crossing_traverses',seed=seed,initials=[[0.,4.,0.,0.],[LENGTH,4.,0.,0.]],goals=[[LENGTH,4.],[0.,4.]],
        initial=[0.,4.,0.,0.],goal=[LENGTH,4.],start_seconds=[0.,0.],steps=steps,damping=.85,
        obstacles=[door('middle'),door('north')],patches=[],bounds=[-1.,13.,-1.,8.5],dynamic=False,
        doors={k:door(k) for k in DOORS},network_nodes=NODES,network_edges=EDGES,corridor_radius=CORRIDOR_RADIUS,
        prior='Surveyed corridor layout only. Doors unknown. Independent plants; no inter-agent contact dynamics.',
        experiment='development mechanism demo, not held-out evaluation')


class CrossingWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12,nodes=NODES,edges=EDGES))
        return y,hit,cost,energy


class CrossingLearner(MazeLearner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.navigation['network']=(NODES,EDGES)
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)


def configuration(steps):
    return Config(controller_mode='routed',shaping_mode='progress',steps=steps,replay_objective='mission',replay_domain='retained',replay_horizon_mode='supported_boundary',replay_update_cap=1024)


def predicted(arm,sensor_radius=Config().sensor_radius):
    """Wasted distance from route arithmetic alone: equal speeds, instant turns, zero delay.

    Agent 0 runs west to east, agent 1 east to west. An agent sees a door from
    sensor_radius+DOOR_RADIUS away, then returns to its own end. A warned agent turns
    back at the moment of the warning, so its depth equals the sender's discovery time.
    """
    reach=sensor_radius+DOOR_RADIUS;north=ROWS['north']-ROWS['middle']
    hears={0:arm in ('reciprocal_fixed','one_way_10'),1:arm in ('reciprocal_fixed','one_way_01')}
    own_middle={0:DOORS['middle']-reach,1:LENGTH-DOORS['middle']-reach}
    own_north={0:north+DOORS['north']-reach,1:north+LENGTH-DOORS['north']-reach}
    first=min(own_middle,key=own_middle.get)
    middle={i:min(own_middle[i],own_middle[first]) if hears[i] else own_middle[i] for i in (0,1)}
    back={i:2*middle[i] for i in (0,1)}
    found={i:back[i]+own_north[i] for i in (0,1)};first=min(found,key=found.get)
    depth={i:own_north[i] if i==first or not hears[i] else min(own_north[i],max(0.,found[first]-back[i])) for i in (0,1)}
    shortest=2*ROWS['middle']+LENGTH
    return {i:dict(middle_penetration=middle[i],north_penetration=max(0.,depth[i]-north),wasted=2*(middle[i]+depth[i]),
                   path=shortest+2*(middle[i]+depth[i])) for i in (0,1)}


def run(arg):
    seed,arm,root,steps=arg;torch.set_num_threads(1);s=specification(seed,steps)
    def progress(event):
        event.update(seed=seed,arm=arm);dump(Path(root)/f'progress_{seed}_{arm}.json',event)
    r=execute(s,arm,CommunicationConfig(byte_budget=4000000,ttl=300.),learner_factory=CrossingLearner,world_factory=CrossingWorld,config=configuration(steps),progress=progress)
    path=f'{seed}_{arm}.json.gz';dump(Path(root)/'episodes'/path,r)
    return dict(seed=seed,arm=arm,path=path,goals=sum(a['success'] for a in r['agents']))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=2500)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    if not args.render_only:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'episodes').mkdir()
        # Predictions are written before any episode runs.
        dump(args.out/'manifest.json',dict(specs=[specification(s,args.steps) for s in args.seeds],arms=list(ARMS),animation_seed=args.seeds[0],
            predictions={arm:predicted(arm) for arm in ARMS},config=asdict(configuration(args.steps)),communication=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.)),
            source_hashes=source_hashes()))
        archive_sources(args.out);rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(run,(s,a,str(args.out),args.steps)) for s in args.seeds for a in ARMS]
            for job in as_completed(jobs):rows.append(job.result());dump(args.out/'completion.json',rows);print(rows[-1],flush=True)
    # grl_snam.coph: results.json and REPORT.md from grl_snam.coph.results, in place of
    # crossing_visuals.report (figures and GIF), which is not vendored.
    from .results import crossing
    return crossing(args.out)

if __name__=='__main__':main()
