"""Frozen-manifest development evaluations with evaluator-only truth access."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,copy,gzip,hashlib,json,time,traceback,platform,sys
import numpy as np
import torch
from .records import Config,Transition
from .model import tensor,flow,potential
from .scenarios import SCENARIOS,specification,ScenarioWorld
from .diagnostics import InstrumentedLearner,snapshot,probes,probe_metrics,replay_audit

ARMS=('frozen','evidence','full','no_history','three_history','no_reconciliation','current_replay')


def load(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)


def dump(path,data):
    path=Path(path)
    with (gzip.open(path,'wt') if path.suffix=='.gz' else path.open('w')) as f:json.dump(data,f,allow_nan=False)


def execute(spec,profile,arm,full_budget=None,navigation_target='mission',replay_objective='legacy'):
    cfg=Config(steps=spec['steps'],previous=0 if arm=='no_history' else 3 if arm=='three_history' else 2,
               navigation_target=navigation_target,replay_objective=replay_objective)
    world=ScenarioWorld(spec,cfg);x=world.initial.copy();points=probes(spec)
    mode=arm if arm in ('frozen','evidence') else 'full'
    agent=InstrumentedLearner(cfg,x,world.goal,mode,arm!='no_reconciliation',
                             current_budget=full_budget if arm=='current_replay' else None)
    first=world.sense(x,0);agent.observe(first)
    records=[];updates=[];events=[dict(tick=0,event='entry',stage=0)];audits=[];audited=set()
    cost_parts=dict(time=0.,risk=0.,effort=0.,failure=0.)
    audit_seconds=0.
    observations=[asdict(first)];near_stationary=0.;path_length=0.;prediction_error=[]
    peak_samples=0;peak_stages=0;success=False;collision=False;started=time.perf_counter()
    for n in range(cfg.steps):
        if n%cfg.update_every==0:
            pre=snapshot(agent);before_calls=agent.domain_failures
            report=agent.update(x)
            post=snapshot(agent)
            report['raw_damping_proposal']=getattr(agent,'raw_damping_proposal',None)
            update=dict(tick=n,pre=pre,post=post,diagnostic=report,
                        replay_calls=copy.deepcopy(agent.replay_calls),proposed_fields=None,
                        probes=probe_metrics(agent,world,points),domain_failures=agent.domain_failures-before_calls)
            if agent.reference is not None:
                update['proposed_fields']=[w.tolist() for w in agent.reference]
                eligible=[c for c in agent.replay_calls if c['cost'] is not None]
                if eligible:
                    for s,w in zip(agent.stages,agent.reference):
                        if s.id not in audited and s.supported(s.entry[:2],cfg):
                            audit_start=time.perf_counter()
                            audit=replay_audit(agent,world,s,w,report['damping_proposal'],pre['policy'],eligible[-1]['policy'])
                            audit['wall_seconds']=time.perf_counter()-audit_start
                            audit_seconds+=audit['wall_seconds']
                            audit.update(tick=n,accepted=report['accepted'])
                            audits.append(audit)
                            # First eligible audit is fixed; invalid results are retained, not replaced.
                            audited.add(s.id)
            updates.append(update)
        u,proposed=agent.action(x);pred=flow(tensor(x),tensor(u),agent.damping,cfg.dt).numpy()
        after,collision,cost,energy=world.step(x,u)
        components=dict(time=cfg.dt,risk=cfg.dt*4*float(world.risk_truth(after[:2])),effort=cfg.dt*.03*float(u@u))
        for k,v in components.items():cost_parts[k]+=v
        prediction_error.append(float(np.mean((pred-after)**2)))
        path_length+=float(np.linalg.norm(after[:2]-x[:2]))
        if np.linalg.norm(x[2:])<.05 and np.linalg.norm(x[:2]-world.goal)>.5:near_stationary+=cfg.dt
        records.append(dict(tick=n,x=x.tolist(),after=after.tolist(),u=u.tolist(),proposed=proposed.tolist(),
            version=agent.version,stage=agent.stages[-1].id,cost=cost,components=components,
            transition_mse=prediction_error[-1],energy=energy,
            clipping_work=float((u-proposed)@(after[:2]-x[:2]))))
        t=Transition(n,n*cfg.dt,tuple(x),tuple(u),tuple(after),agent.version)
        observation=world.sense(after,n+1);observations.append(asdict(observation))
        s=agent.stages[-1]
        old=float(potential(tensor(after[:2]),tensor(agent.control_target(s)),s.weights,s.centers,cfg.rbf_width,agent.policy,s.obstacles))
        agent.observe(observation,t)
        new=float(potential(tensor(after[:2]),tensor(agent.control_target(s)),s.weights,s.centers,cfg.rbf_width,agent.policy,s.obstacles))
        if new!=old:events.append(dict(tick=n+1,event='observation_context',shaping_jump=new-old))
        x=after;success=bool(np.linalg.norm(x[:2]-world.goal)<.25)
        peak_stages=max(peak_stages,len(agent.stages));peak_samples=max(peak_samples,sum(len(s.readings) for s in agent.stages))
        if success or collision:break
        previous=[s.id for s in agent.stages]
        if agent.advance(x):events.append(dict(tick=n+1,event='stage_exit_entry',stage=agent.stages[-1].id,
            retained=[s.id for s in agent.stages],evicted=list(set(previous)-{s.id for s in agent.stages}),
            shaping_jump=agent.context_jumps[-1]['shaping_jump']))
    cost_parts['failure']=0. if success else 40.
    useful=agent.counterfactual_steps;padding=agent.spend_unused_budget()
    end_metrics=probe_metrics(agent,world,points)
    summary=dict(arm=arm,scene=spec['name'],seed=spec['seed'],success=success,collision=collision,
        timeout=not success and not collision,mission_cost=sum(cost_parts.values()),components=cost_parts,
        physical_steps=len(records),path_length=path_length,goal_gap=float(np.linalg.norm(x[:2]-world.goal)),
        near_stationary_seconds=near_stationary,transition_mse=float(np.mean(prediction_error)),
        damping_error=abs(agent.damping-world._damping),field_mse=end_metrics['mse'],gradient_mse=end_metrics['gradient_mse'],
        coverage=end_metrics['coverage'],counterfactual_steps=agent.counterfactual_steps,
        useful_replay_steps=useful,padding_steps=padding,identification_work=agent.identification_steps,
        audit_steps=sum(a['audit_steps'] for a in audits),audit_valid=sum(a['valid'] for a in audits),
        audit_count=len(audits),audit_seconds=audit_seconds,
        clipping_work=sum(r['clipping_work'] for r in records),
        actuator_work=sum(r['energy']['work'] for r in records),
        dissipation=sum(r['energy']['heat'] for r in records),
        model_shaping_jumps=sum(u['diagnostic'].get('shaping_jump',0.) for u in updates),
        context_shaping_jumps=sum(e.get('shaping_jump',0.) for e in events),
        mean_update_seconds=float(np.mean([u['diagnostic'].get('wall_seconds',0.) for u in updates])),
        domain_failures=agent.domain_failures,peak_samples=peak_samples,peak_stages=peak_stages,
        visited_stages=agent.stages[-1].id+1,accepted_updates=sum(u['diagnostic']['accepted'] for u in updates),
        max_energy_residual=max(abs(r['energy']['residual']) for r in records),wall_seconds=time.perf_counter()-started,
        update_latency=[u['diagnostic'].get('wall_seconds',0.) for u in updates])
    return dict(schema_version=2,spec=spec,config=asdict(cfg),summary=summary,trajectory=records,
                observations=observations,updates=updates,events=events,audits=audits,
                probe_points=points.tolist(),final_probes=end_metrics)


def task(job):
    root,spec,profile,navigation_target=job;root=Path(root);torch.set_num_threads(1)
    rows=[];budget=None
    for arm in ARMS:
        name=f"{spec['name']}_seed{spec['seed']}_{arm}"
        try:
            if arm=='current_replay' and budget is None:raise RuntimeError('paired full budget unavailable')
            result=execute(spec,profile,arm,budget,navigation_target)
            if arm=='full':budget=result['summary']['counterfactual_steps']
            if arm=='current_replay' and budget is None:raise RuntimeError('paired full budget unavailable')
            dump(root/'episodes'/f'{name}.json.gz',result)
            rows.append(result['summary'])
        except Exception:
            dump(root/'episodes'/f'{name}.error.json',dict(arm=arm,scene=spec['name'],seed=spec['seed'],traceback=traceback.format_exc()))
    return rows


def suite_main(argv):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=['smoke','development'],default='smoke')
    p.add_argument('--navigation-target',choices=['mission','stage'],default='mission')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args(argv)
    if args.out.exists():p.error('use a new output directory')
    if args.workers<1:p.error('workers must be positive')
    count=2 if args.profile=='smoke' else 8;base=10000 if args.profile=='smoke' else 20000
    specs=[specification(scene,base+i,args.profile) for scene in SCENARIOS for i in range(count)]
    package=Path(__file__).parent
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(package.glob('*.py'))}
    manifest=dict(profile=args.profile,seeds=list(range(base,base+count)),scenarios=specs,arms=ARMS,
        config=asdict(Config(navigation_target=args.navigation_target)),animation_seed=base,animation_arm='full',
        source_hashes=hashes,reserved_validation_seeds=[100000,199999],reserved_test_seeds=[200000,299999],
        expected_episodes=len(specs)*len(ARMS),status='development; isolated execution-attraction correction; no cost tuning',
        intervention='execution attraction uses '+args.navigation_target+' target; replay objective and stage advancement unchanged',
        current_replay_budget='paired full episode cap; unused work executed and reported separately as padding',
        audit_schedule='first eligible update per visited stage, including invalid audits',
        torch=torch.__version__,numpy=np.__version__,python=sys.version,platform=platform.platform(),
        workers=args.workers,mission_cost=dict(time=1.,risk=4.,effort=.03,failure=40.),
        replay_cost=dict(risk=4.,effort=.03,target_running=.1,target_terminal=1.,contact=5.))
    manifest['configuration_sha256']=hashlib.sha256(json.dumps(dict(scenarios=specs,config=manifest['config'],arms=ARMS),sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True);(args.out/'episodes').mkdir()
    dump(args.out/'manifest.json',manifest)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(task,(str(args.out),spec,args.profile,args.navigation_target)) for spec in specs]
        for f in as_completed(futures):
            batch=f.result();rows.extend(batch)
            dump(args.out/'summary.json',rows)
            print(f'{len(rows)}/{manifest["expected_episodes"]} episodes complete',flush=True)
    dump(args.out/'completion.json',dict(completed=len(rows),expected=manifest['expected_episodes'],
        errors=len(list((args.out/'episodes').glob('*.error.json')))))
