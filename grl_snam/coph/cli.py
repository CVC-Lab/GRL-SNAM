# grl_snam.coph: trimmed to rollout (the single-agent reference loop) and to source_hashes and
# archive_sources, which the demos use for provenance. Both of those cover this package's own
# sources (upstream: the whole CoPH-Stagewise tree). The upstream command-line entry point is
# replaced by `grl-snam coph` (VENDORED.md).
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import numpy as np
from .records import Transition
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
    root=Path(__file__).resolve().parent
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in str(p)}


def archive_sources(destination):
    import zipfile
    root=Path(__file__).resolve().parent
    with zipfile.ZipFile(Path(destination)/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for relative in source_hashes():archive.write(root/relative,relative)
