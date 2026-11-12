import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .records import Config, Transition
from .learner import Learner
from .world import World
from .model import tensor, risk, potential


def rollout(cfg, seed=0, scene='risk', mode='full', reconcile=True):
    world=World(seed,scene,cfg); x=world.initial.copy()
    learner=Learner(cfg,x,world.goal,mode,reconcile)
    learner.observe(world.sense(x,0))
    trajectory=[]; versions=[]; total=0.; collision=False; success=False
    start=time.perf_counter(); obs_jumps=[]; steps=0
    for n in range(cfg.steps):
        if n % cfg.update_every==0:
            learner.update(x)
            versions.append(dict(tick=n,version=learner.version,damping=learner.damping,
                                 policy=learner.policy.tolist(),stages=[dict(id=s.id,center=s.center,
                                 target=s.target,weights=s.weights.tolist()) for s in learner.stages]))
        u,proposed=learner.action(x)
        after,collision,cost,energy=world.step(x,u)
        total+=cost; steps=n+1
        displacement=after[:2]-x[:2]
        trajectory.append(dict(tick=n,x=x.tolist(),u=u.tolist(),after=after.tolist(),
            stage=learner.stages[-1].id,version=learner.version,cost=cost,
            clip_residual_power=float(np.dot(u-proposed,x[2:])),
            clip_residual_work=float(np.dot(u-proposed,displacement)),energy=energy))
        transition=Transition(n,n*cfg.dt,tuple(x),tuple(u),tuple(after),learner.version)
        s=learner.stages[-1]
        before=float(potential(tensor(after[:2]),tensor(learner.control_target(s)),s.weights,s.centers,cfg.rbf_width,learner.policy,s.obstacles))
        learner.observe(world.sense(after,n+1),transition)
        current=float(potential(tensor(after[:2]),tensor(learner.control_target(s)),s.weights,s.centers,cfg.rbf_width,learner.policy,s.obstacles))
        obs_jumps.append(current-before)
        x=after
        success=np.linalg.norm(x[:2]-world.goal)<.25
        if collision or success:break
        learner.advance(x)
    failure=not success
    total+=40*failure
    # Truth-only probes for evaluation are never fed to learner.
    errors=[]
    for s in learner.stages:
        for q in s.centers.numpy():
            if s.supported(q,cfg):
                errors.append((float(risk(tensor(q),s.weights,s.centers,cfg.rbf_width))-float(world.risk_truth(q)))**2)
    summary=dict(seed=seed,scene=scene,mode=mode,reconcile=reconcile,success=bool(success),
        collision=collision,timeout=bool(failure and not collision),mission_cost=float(total),
        physical_steps=steps,counterfactual_steps=learner.counterfactual_steps,
        identification_transition_evaluations=learner.identification_steps,
        accepted_updates=sum(d['accepted'] for d in learner.diagnostics),
        model_field_mse=float(np.mean(errors)) if errors else None,
        damping_error=abs(learner.damping-world._damping),
        max_energy_residual=max(abs(t['energy']['residual']) for t in trajectory),
        goal_gap=float(np.linalg.norm(x[:2]-world.goal)),wall_seconds=time.perf_counter()-start)
    return dict(summary=summary,trajectory=trajectory,updates=learner.diagnostics,
                versions=versions,stage_jumps=learner.context_jumps,observation_shaping_jumps=obs_jumps,
                final_ledger=[dict(id=s.id,center=s.center,entry=s.entry,target=s.target,
                    weights=s.weights.tolist(),centers=s.centers.tolist(),version=s.version,
                    readings=[asdict(r) for r in s.readings],transitions=[asdict(t) for t in s.transitions],
                    obstacles=s.obstacles,anchors=s.anchors,
                    counterfactuals=[asdict(c) for c in s.counterfactuals],
                    curvature_inverse=s.covariance.tolist() if s.covariance is not None else None,
                    uncertainty_status='ridge curvature; not calibrated posterior') for s in learner.stages])


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in str(p)}


def archive_sources(destination):
    import zipfile
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(Path(destination)/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for relative in source_hashes():archive.write(root/relative,relative)


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='campaign':
        from .campaign import main as campaign_main
        return campaign_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='field-route-suite':
        from .field_route_suite import main as field_main
        return field_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='decision-diagnostics':
        from .decision_suite import main as decision_main
        return decision_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='attribution-suite':
        from .attribution_suite import main as attribution_main
        return attribution_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=="receiver-feedback-suite":
        from .receiver_feedback_evaluation import main as feedback_main
        torch.set_num_threads(1)
        return feedback_main(sys.argv[2:])
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='stagewise-diagnostics':
        from .stagewise_diagnostics import main
        torch.set_num_threads(1)
        return main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='receiver-corrections':
        from .receiver_corrections import main as corrections_main
        return corrections_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='receiver-reconstruction':
        from .receiver_reconstruction import main
        return main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='recipient-diagnostics':
        from .recipient_diagnostics import main
        torch.set_num_threads(1)
        return main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='replay-gate':
        from .replay_gate import main
        return main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='cooperative-suite':
        from .cooperative_evaluation import suite_main
        torch.set_num_threads(1)
        return suite_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1] in ('cooperative-render','cooperative-report'):
        from .cooperative_reporting import main
        return main(sys.argv[2:],render_only=sys.argv[1]=='cooperative-render')
    if len(sys.argv)>1 and sys.argv[1] in ('suite','render','report'):
        if sys.argv[1]=='suite':
            from .evaluation import suite_main
            return suite_main(sys.argv[2:])
        from .visualize import render_main,report_main
        return (render_main if sys.argv[1]=='render' else report_main)(sys.argv[2:])
    ap=argparse.ArgumentParser(description='Single-agent controlled development study; not confirmatory evaluation.')
    ap.add_argument('command',choices=['check','rollout','compare'])
    ap.add_argument('--out',type=Path)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--steps',type=int,default=180)
    ap.add_argument('--previous',type=int,choices=[0,2,3],default=2)
    ap.add_argument('--navigation-target',choices=['mission','stage'],default='mission')
    ap.add_argument('--mode',choices=['frozen','evidence','full'],default='full')
    ap.add_argument('--scene',choices=['risk','obstacle'],default='risk')
    args=ap.parse_args();torch.set_num_threads(1)
    if args.command=='check':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.out is None:ap.error('--out is required')
    if args.seeds<1:ap.error('--seeds must be positive')
    if args.out.exists():ap.error('output exists; use a fresh directory')
    cfg=Config(steps=args.steps,previous=args.previous,navigation_target=args.navigation_target)
    args.out.mkdir(parents=True)
    manifest=dict(config=asdict(cfg),source_hashes=source_hashes(),torch=torch.__version__,
                  numpy=np.__version__,command=args.command,seed=args.seed,seeds=args.seeds,
                  status='development only',numerical_precision='float64')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries=[]
    arms=[(args.mode,cfg,True)] if args.command=='rollout' else [
        ('frozen',cfg,True),('evidence',cfg,True),('full',cfg,True),
        ('full',replace(cfg,previous=0),True),('full',replace(cfg,previous=3),True),('full',cfg,False)]
    seeds=range(args.seed,args.seed+(args.seeds if args.command=='compare' else 1))
    for seed in seeds:
        for mode,settings,reconcile in arms:
            name=f'{mode}_memory{settings.previous}_reconcile{int(reconcile)}'
            result=rollout(settings,seed,args.scene,mode,reconcile)
            result['summary']['arm']=name
            summaries.append(result['summary'])
            (args.out/f'{name}_seed{seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            print(json.dumps(result['summary']),flush=True)
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    if args.command=='compare':
        base={s['seed']:s for s in summaries if s['mode']=='frozen'}
        paired=[dict(seed=s['seed'],arm=s['arm'],cost_difference_vs_frozen=s['mission_cost']-base[s['seed']]['mission_cost']) for s in summaries]
        (args.out/'paired.json').write_text(json.dumps(paired,indent=2))

if __name__=='__main__':main()
