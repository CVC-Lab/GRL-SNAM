"""Generated grid mazes with hidden doors, screened by the route oracle.

Layouts: 'crossing' (opposite corners, swapped goals) and 'relay' (shared start and
goal, the second agent starts later). Every candidate is predicted by the oracle;
the declared screen keeps those where communication should matter, and the whole
candidate distribution is recorded so selection is visible.
"""
import argparse,copy,json,shutil
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
from .evaluation import dump,load
from .cli import source_hashes,archive_sources
from .shapes import slab_distance
from .route_oracle import predict,shortest,ARM_SENDERS
from . import crossing_demo

ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
DOOR_RADIUS=.1
# Declared screen, fixed before any run.
SCREEN=dict(junction_clearance=1.6,min_reciprocal_saving=8.,selection='top-k accepted candidates by oracle reciprocal saving')


def grid_maze(rng,nx=5,ny=3,sx=5.,sy=4.,loops=3,doors=3,layout='crossing',stagger=20.):
    nodes=[(i*sx,j*sy) for j in range(ny) for i in range(nx)];index=lambda i,j:j*nx+i
    candidates=[(index(i,j),index(i+1,j)) for j in range(ny) for i in range(nx-1)]+[(index(i,j),index(i,j+1)) for j in range(ny-1) for i in range(nx)]
    order=rng.permutation(len(candidates));parent=list(range(len(nodes)))
    def root(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    tree=[];rest=[]
    for k in order:
        a,b=candidates[k]
        if root(a)!=root(b):parent[root(a)]=root(b);tree.append(candidates[k])
        else:rest.append(candidates[k])
    edges=tree+[rest[k] for k in rng.choice(len(rest),min(loops,len(rest)),replace=False)]
    edges=tuple(sorted(tuple(sorted(e)) for e in edges))
    placed=[]
    for k in rng.permutation(len(edges)):
        if len(placed)==doors:break
        a,b=np.asarray(nodes[edges[k][0]]),np.asarray(nodes[edges[k][1]]);L=np.linalg.norm(b-a);u=(b-a)/L
        lo=SCREEN['junction_clearance']+DOOR_RADIUS;hi=L-lo
        if hi<lo:continue
        c=a+u*rng.uniform(lo,hi);h=np.array([-u[1],u[0]])*(CORRIDOR_RADIUS-DOOR_RADIUS)
        placed.append((float(c[0]),float(c[1]),DOOR_RADIUS,float(h[0]),float(h[1])))
    corner=[nodes[0],nodes[-1]]
    if layout=='crossing':initials=[[*corner[0],0.,0.],[*corner[1],0.,0.]];goals=[list(corner[1]),list(corner[0])];starts=[0.,0.]
    else:initials=[[*corner[0],0.,0.]]*2;goals=[list(corner[1])]*2;starts=[0.,stagger]
    return dict(nodes=tuple(nodes),edges=edges,doors=placed,initials=initials,goals=goals,start_seconds=starts)


def screen(m,sensor_radius=1.25):
    """The declared invariants; returns the failed reasons (empty when accepted)."""
    reasons=[];nodes=np.asarray(m['nodes'])
    for d in m['doors']:
        if min(float(slab_distance(n,d)) for n in nodes)<SCREEN['junction_clearance']:reasons.append('door_visible_from_junction')
    for i in range(2):
        if shortest(m['nodes'],m['edges'],m['doors'],m['initials'][i],m['goals'][i]) is None:reasons.append('goal_unreachable')
    return reasons


def specification(m,name,seed,steps):
    return dict(name=name,seed=seed,initials=m['initials'],goals=m['goals'],initial=m['initials'][0],goal=m['goals'][0],start_seconds=m['start_seconds'],
        steps=steps,damping=.85,obstacles=[list(d) for d in m['doors']],patches=[],dynamic=False,network_nodes=[list(n) for n in m['nodes']],
        network_edges=[list(e) for e in m['edges']],corridor_radius=CORRIDOR_RADIUS,
        bounds=[min(n[0] for n in m['nodes'])-1,max(n[0] for n in m['nodes'])+1,min(n[1] for n in m['nodes'])-1,max(n[1] for n in m['nodes'])+1],
        prior='Surveyed corridor layout only. Doors unknown. Independent plants; no inter-agent contact dynamics.',experiment='development maze family, oracle-screened')


class NetworkWorld(CooperativeWorld):
    def step(self,x,u):
        y,hit,cost,energy=super().step(x,u)
        hit=bool(hit or not corridor_clear(x[:2],y[:2],margin=.12,nodes=self.spec['network_nodes'],edges=self.spec['network_edges']))
        return y,hit,cost,energy


class NetworkLearner(MazeLearner):
    def __init__(self,*args,network,risk_weight=1.,unknown_risk=1.,**kwargs):
        super().__init__(*args,**kwargs);self.navigation['network']=network
        # Non-default task costs only; default runs keep their exact navigation records.
        if risk_weight!=1.:self.navigation['risk_weight']=float(risk_weight)
        if unknown_risk!=1.:self.navigation['unknown_risk']=float(unknown_risk)
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)


def learner_for(spec):
    # A partial of a module-level class stays picklable for the runner's packet audits.
    from functools import partial
    return partial(NetworkLearner,network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))))


def run(job):
    spec,arm,root,comm=job;torch.set_num_threads(1)
    r=execute(spec,arm,CommunicationConfig(**comm),learner_factory=learner_for(spec),world_factory=NetworkWorld,config=crossing_demo.configuration(spec['steps']))
    path=f"{spec['seed']}_{arm}.json.gz";dump(Path(root)/'episodes'/path,r)
    return dict(scene=spec['name'],seed=spec['seed'],arm=arm,path=path,goals=sum(a['success'] for a in r['agents']))


def horizon(spec,speed=.25):
    """Steps covering 1.5x the oracle's slowest no-communication mission plus 60 s."""
    worst=max(v['path'] for v in predict(spec,'disabled')['agents'].values())
    return int(np.ceil((1.5*worst/speed+max(spec['start_seconds'])+60)/Config().dt))


def prepare(root,spec,arms,comm,extra=None):
    root.mkdir(parents=True,exist_ok=False);(root/'episodes').mkdir()
    dump(root/'manifest.json',dict(specs=[spec],arms=list(arms),animation_seed=spec['seed'],predictions={a:predict(spec,a,delay=comm['delay_ticks']*Config().dt,speed=.3)['agents'] for a in arms},
        config=asdict(crossing_demo.configuration(spec['steps'])),communication=comm,source_hashes=source_hashes(),**(extra or {})))
    archive_sources(root)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=40)
    ap.add_argument('--layouts',nargs='+',default=['crossing','crossing','relay']);ap.add_argument('--candidates',type=int,default=200)
    ap.add_argument('--seed',type=int,default=81000);ap.add_argument('--delays',type=float,nargs='*',default=[],help='delay sweep on the crossing scene, seconds')
    ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    comm=asdict(CommunicationConfig(byte_budget=4000000,ttl=300.));jobs=[]
    if not args.render_only:
        args.out.mkdir(parents=True,exist_ok=False);catalogue=[]
        for layout in sorted(set(args.layouts)):
            want=args.layouts.count(layout);rng=np.random.default_rng([args.seed,len(layout)]);pool=[]
            for k in range(args.candidates):
                m=grid_maze(rng,layout=layout);reasons=screen(m);name=f'{layout}_{k:03d}';spec=specification(m,name,args.seed,100);saving=oneway=None
                if not reasons:
                    none=predict(spec,'disabled');rec=predict(spec,'reciprocal_fixed');one=[predict(spec,a) for a in ('one_way_01','one_way_10')]
                    saving=none['team_wasted']-rec['team_wasted'];oneway=sum(none['team_wasted']-o['team_wasted'] for o in one)
                    if saving<SCREEN['min_reciprocal_saving']:reasons.append('oracle_saving_below_threshold')
                catalogue.append(dict(name=name,layout=layout,reasons=reasons,selected=False,oracle_reciprocal_saving=saving,oracle_one_way_sum=oneway))
                if not reasons:pool.append((saving,saving-oneway,catalogue[-1],spec))
            # Declared rule: the `want` accepted candidates with the largest oracle reciprocal saving.
            for saving,margin,entry,spec in sorted(pool,key=lambda z:(-z[0],-z[1],z[2]['name']))[:want]:
                entry['selected']=True;spec['steps']=horizon(spec);prepare(args.out/entry['name'],spec,ARMS,comm)
                jobs+=[(spec,a,str(args.out/entry['name']),comm) for a in ARMS]
        dump(args.out/'catalogue.json',dict(screen=SCREEN,candidates=catalogue))
        if args.delays:
            base=crossing_demo.specification(76000,2500);source=Path('results/crossing_traverses_v1/episodes/76000_disabled.json.gz')
            for d in args.delays:
                ticks=int(round(d/Config().dt));c=dict(comm,delay_ticks=ticks);root=args.out/f'delay_{ticks:03d}'
                # The no-communication arm cannot depend on delay; reuse the logged one.
                prepare(root,base,('disabled','one_way_10','reciprocal_fixed'),c,dict(disabled_copied_from=str(source)))
                shutil.copy(source,root/'episodes'/'76000_disabled.json.gz')
                jobs+=[(base,a,str(root),c) for a in ('one_way_10','reciprocal_fixed')]
        completed=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run,j) for j in jobs]
            for f in as_completed(futures):completed.append(f.result());dump(args.out/'completion.json',completed);print(completed[-1],flush=True)
    from .crossing_visuals import report
    for root in sorted(p for p in args.out.iterdir() if (p/'manifest.json').exists()):report(root)


if __name__=='__main__':main()
