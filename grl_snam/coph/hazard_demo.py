"""Share the environmental field, not the decision: graded hazards and heterogeneous agents.

A risk-tolerant scout (w=0.25) and a risk-averse carrier (w=6) leave the west end of
the middle row for its east end; the carrier starts 25 s later. Hazards are traversable
Gaussian risk bumps spanning the middle row. Each agent prices risk with its own w, so
the same hazard means "cross" to one and "detour" to the other. Arms differ only in what
the sender puts in its anchors (see sharing.py).
"""
import argparse,functools,json
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .crossing_demo import NODES,EDGES,configuration
from .maze_geometry import CORRIDOR_RADIUS
from .maze_family import NetworkWorld
from .cooperative import CommunicationConfig
from .staggered_demo import execute
from .sharing import LEDGER_MODES as MODES
from .hazard_oracle import simulate
from .evaluation import dump
from .cli import source_hashes,archive_sources

WEIGHTS=(.25,6.)
UNKNOWN_RISK=.02  # unvisited terrain is assumed to be at background risk
STAGGER=25.
SIGMA=(.6,1.)
SCENES=dict(mild=[(7.,4.,*SIGMA,.15)],moderate=[(7.,4.,*SIGMA,.5)],severe=[(7.,4.,*SIGMA,1.5)],
            double_mild=[(5.5,4.,*SIGMA,.2),(8.5,4.,*SIGMA,.2)])
# label: (learner mode, extra learner kwargs, runner channel)
ARMS=dict(none=('field',{},'disabled'),field=('field',{},'reciprocal_fixed'),occupancy_0p1=('occupancy',dict(tau=.1),'reciprocal_fixed'),
          occupancy_0p25=('occupancy',dict(tau=.25),'reciprocal_fixed'),occupancy_1=('occupancy',dict(tau=1.),'reciprocal_fixed'),decision=('decision',{},'reciprocal_fixed'),
          raw=('raw',{},'reciprocal_fixed'),no_learning=('no_learning',{},'reciprocal_fixed'))
RAW_ANCHORS=12


def specification(scene,seed=79000,steps=1800):
    return dict(name=f'hazard_{scene}',scene=scene,seed=seed,initials=[[0.,4.,0.,0.]]*2,goals=[[12.,4.]]*2,initial=[0.,4.,0.,0.],goal=[12.,4.],
        start_seconds=[0.,STAGGER],risk_weights=list(WEIGHTS),unknown_risk=UNKNOWN_RISK,steps=steps,damping=.85,obstacles=[],patches=[list(p) for p in SCENES[scene]],
        dynamic=False,network_nodes=[list(n) for n in NODES],network_edges=[list(e) for e in EDGES],corridor_radius=CORRIDOR_RADIUS,
        roles=['scout (risk-tolerant)','carrier (risk-averse)'],experiment='development mechanism demo, not held-out evaluation')


def oracle(spec,arm):
    mode,kw,channel=ARMS[arm]
    return simulate(spec,'none' if channel=='disabled' else mode,kw.get('tau'))


def communication(arm):
    return CommunicationConfig(byte_budget=4000000,ttl=300.,anchors=RAW_ANCHORS if ARMS[arm][0]=='raw' else 4)


def learner(spec,arm):
    mode,kw,_=ARMS[arm];cls=MODES[mode]
    return [functools.partial(cls,network=(tuple(map(tuple,spec['network_nodes'])),tuple(map(tuple,spec['network_edges']))),
                              risk_weight=spec['risk_weights'][i],unknown_risk=spec['unknown_risk'],**kw) for i in range(2)]


def run(job):
    spec,arm,root=job;torch.set_num_threads(1);factories=learner(spec,arm)
    def factory(cfg,x,goal,agent_id,communication):return factories[int(agent_id)](cfg,x,goal,agent_id=agent_id,communication=communication)
    r=execute(spec,ARMS[arm][2],communication(arm),learner_factory=factory,world_factory=NetworkWorld,config=configuration(spec['steps']))
    r['arm']=arm;r['channel']=ARMS[arm][2];dump(Path(root)/'episodes'/f"{spec['seed']}_{arm}.json.gz",r)
    return dict(scene=spec['scene'],arm=arm,goals=sum(a['success'] for a in r['agents']),cost=[a['components']['time']+a['components']['risk'] for a in r['agents']])


def table():
    rows=[]
    for scene in SCENES:
        spec=specification(scene)
        for arm in ARMS:
            o=oracle(spec,arm)['agents'];rows.append(dict(scene=scene,arm=arm,scout=o[0]['cost'],carrier=o[1]['cost'],carrier_path=o[1]['path']))
    return rows


def render(root):
    """Per-scene decision-vs-field GIFs and one REPORT.md with measured and oracle carrier excess cost."""
    from .crossing_visuals import report
    from .evaluation import load
    root=Path(root);order=list(ARMS);lines=['# Graded hazards: share the environmental field, not the decision','',
        f'Settings: one noise seed on four designed scenes. Scout risk weight {WEIGHTS[0]}, carrier {WEIGHTS[1]}; carrier starts {STAGGER:.0f} s later. Unknown-terrain prior {UNKNOWN_RISK}. Field packets carry 4 anchors, raw packets {RAW_ANCHORS}. Oracle predictions were frozen in each scene manifest before the runs. Cost = time + weighted risk exposure (effort excluded).','',
        '| Scene | '+' | '.join(order)+' |','|---|'+'---|'*len(order)]
    totals={a:[] for a in order}
    for scene in SCENES:
        d=root/scene;m=load(d/'manifest.json');cost={};goals={}
        for a in order:
            r=load(d/'episodes'/f"{m['specs'][0]['seed']}_{a}.json.gz");z=r['agents'][1];cost[a]=z['components']['time']+z['components']['risk'];goals[a]=sum(x['success'] for x in r['agents'])
        best=min(cost.values());pred={a:m['predictions'][a][1]['cost'] for a in order};pbest=min(pred.values())
        for a in order:totals[a].append(cost[a]-best)
        lines.append(f'| {scene} | '+' | '.join(f'{cost[a]-best:.1f} ({pred[a]-pbest:.1f}){"" if goals[a]==2 else " FAIL"}' for a in order)+' |')
        report(d,pair=('decision','field'))
    lines.append('| **worst-case excess cost** | '+' | '.join(f'**{max(totals[a]):.1f}**' for a in order)+' |')
    lines+=['','Cells: carrier excess cost relative to the best tested arm in that scene (not regret against a globally optimal policy), measured (oracle). Every episode reached both goals unless marked FAIL.',
            'Oracle agreement is qualitative: it idealizes constant speed and immediate full transfer, so magnitudes can differ (for example a timing cost the oracle omits).','',
            'Per-scene animations compare the sender\'s decision with the learned environmental field: `<scene>/figures/decision_vs_field.gif`.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path);ap.add_argument('--workers',type=int,default=40)
    ap.add_argument('--table',action='store_true',help='print oracle predictions and exit');ap.add_argument('--render',action='store_true',help='write GIFs and REPORT.md for an existing --out')
    args=ap.parse_args(argv)
    if args.render:return render(args.out)
    if args.table:
        rows=table()
        for scene in SCENES:
            best=min(r['carrier'] for r in rows if r['scene']==scene)
            print(f'\n{scene}: patches {SCENES[scene]}')
            for r in rows:
                if r['scene']==scene:print(f"  {r['arm']:15s} scout {r['scout']:6.1f}  carrier {r['carrier']:6.1f} ({r['carrier_path']:4.1f} m){'  *' if r['carrier']<=best+.5 else ''}")
        return
    args.out.mkdir(parents=True,exist_ok=False);jobs=[]
    for scene in SCENES:
        spec=specification(scene);root=args.out/scene;(root/'episodes').mkdir(parents=True)
        dump(root/'manifest.json',dict(specs=[spec],arms=list(ARMS),export='retained-stage rotation (LedgerExport), newest first',arm_definitions={k:dict(mode=v[0],kwargs=v[1],channel=v[2]) for k,v in ARMS.items()},
            predictions={arm:oracle(spec,arm)['agents'] for arm in ARMS},config=asdict(configuration(spec['steps'])),
            communication={arm:asdict(communication(arm)) for arm in ARMS},source_hashes=source_hashes(),animation_seed=spec['seed']))
        archive_sources(root);jobs+=[(spec,arm,str(root)) for arm in ARMS]
    done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):done.append(f.result());dump(args.out/'completion.json',done);print(done[-1],flush=True)


if __name__=='__main__':main()
