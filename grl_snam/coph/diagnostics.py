"""Read-only learner instrumentation and explicitly labeled current-state comparator."""
from dataclasses import asdict
import copy
import numpy as np
import torch
from .learner import Learner
from .model import tensor,risk,flow,command


def snapshot(agent):
    return dict(**({'receiver_options':dict(fit=getattr(agent,'receiver_fit_mode','legacy'),validation=getattr(agent,'receiver_validation_mode','legacy'),**(dict(localization='compact') if getattr(agent,'receiver_localization_mode','global')=='compact' else {}),**(dict(solver=getattr(agent,'receiver_solver_mode','projected'),search=getattr(agent,'receiver_search_mode','legacy')) if (getattr(agent,'receiver_solver_mode','projected'),getattr(agent,'receiver_search_mode','legacy'))!=('projected','legacy') else {}))}
        if getattr(agent,'receiver_fit_mode','legacy')!='legacy' or getattr(agent,'receiver_validation_mode','legacy')!='legacy' else {}),version=agent.version,damping=agent.damping,policy=agent.policy.tolist(),
        navigation=copy.deepcopy(agent.navigation),geometry=agent.geometry_store,geometry_overflow=agent.geometry_overflow,
        stages=[dict(footprints=s.footprints,entry_navigation=s.entry_navigation,id=s.id,center=s.center,entry=s.entry,target=s.target,control_target=agent.control_target(s),
            bounds=[s.center[0]-agent.cfg.stage_width/2,s.center[0]+agent.cfg.stage_width/2,
                    s.center[1]-agent.cfg.stage_height/2,s.center[1]+agent.cfg.stage_height/2],
            centers=s.centers.tolist(),weights=s.weights.tolist(),obstacles=s.obstacles,
            **(dict(local_residual=copy.deepcopy(s.local_residual)) if getattr(s,'local_residual',None) is not None else {}),
            anchors=s.anchors,readings=[asdict(r) for r in s.readings],version=s.version,
            counterfactuals=[asdict(c) for c in s.counterfactuals]) for s in agent.stages])


class InstrumentedLearner(Learner):
    def __init__(self,*args,current_budget=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.current_budget=current_budget
        self.domain_failures=0;self.replay_calls=[];self.reference=None;self.latest_proposed=None
        self.position=None;self.audit_count=0

    def _field_fit(self):
        weights,cov=super()._field_fit()
        self.reference=[w.detach().clone() for w in weights]
        return weights,cov

    def _fit_damping(self):
        self.raw_damping_proposal=super()._fit_damping()
        return self.raw_damping_proposal

    def replay(self,policy,weights,damping):
        before=self.counterfactual_steps
        original=self.stages
        if self.current_budget is not None:
            # Equal episode-level transition cap from the already completed paired full arm.
            # Reserve baseline + its gradient recomputation before beginning a proposal.
            required=self.cfg.replay_steps*len(original)
            if self.current_budget-before < required*(2 if not self.replay_calls else 1):
                self.replay_calls.append(dict(reason='budget_exhausted',steps=0,cost=None,policy=policy.detach().tolist(),trajectories=[]))
                return None,[]
            clones=[]
            for _ in original:
                s=copy.copy(original[-1]);s.entry=tuple(self.position);clones.append(s)
            self.stages=clones;weights=[weights[-1] for _ in clones]
        try:
            cost,paths=super().replay(policy,weights,damping)
        finally:
            self.stages=original
        if cost is None:self.domain_failures+=1
        self.replay_calls.append(dict(reason='valid' if cost is not None else 'unsupported_domain',
            steps=self.counterfactual_steps-before,cost=float(cost.detach()) if cost is not None else None,
            stage_events=list(getattr(self,'replay_stage_events',[])),
            controls=copy.deepcopy(getattr(self,'replay_controls',[])),
            policy=policy.detach().tolist(),trajectories=[dict(stage=s,states=p,cost=float(c.detach())) for s,p,c in paths]))
        return cost,paths

    def update(self,x,force_invalid=False):
        self.position=tuple(x);self.replay_calls=[];self.reference=None
        return super().update(x,force_invalid)

    def spend_unused_budget(self):
        if self.current_budget is None:return 0
        padding=0;s=self.stages[-1];x=tensor(self.position or s.entry)
        while self.counterfactual_steps<self.current_budget:
            # Explicit non-learning work, separately reported; never disguised as useful replay.
            u,_=command(x,tensor(s.target),s.weights,s.centers,self.cfg.rbf_width,self.policy,s.obstacles,self.cfg.force_limit)
            flow(x,u,self.damping,self.cfg.dt)
            self.counterfactual_steps+=1;padding+=1
        return padding


def probes(spec):
    lo,hi,bot,top=spec['bounds']
    return np.array([(x,y) for x in np.arange(lo,hi+.01,.5) for y in np.linspace(bot,top,9)])


def probe_metrics(agent,world,points):
    values=np.full(len(points),np.nan);grads=np.full((len(points),2),np.nan)
    # Prefer newest supported stage. Undefined probes remain missing, never extrapolated as facts.
    for s in agent.stages:
        samples=np.asarray([r.q for r in s.readings if int(r.uid.split(':')[-1]) % 5])
        if not len(samples):continue
        inside=(np.abs(points[:,0]-s.center[0])<=agent.cfg.stage_width/2+1e-10)&(np.abs(points[:,1]-s.center[1])<=agent.cfg.stage_height/2+1e-10)
        supported=inside & (np.sum((points[:,None,:]-samples[None,:,:])**2,axis=2).min(axis=1)<=.65**2)
        if not supported.any():continue
        z=tensor(points[supported]).requires_grad_(True)
        value=risk(z,s.weights,s.centers,agent.cfg.rbf_width,getattr(s,'local_residual',None))
        values[supported]=value.detach().numpy()
        grads[supported]=torch.autograd.grad(value.sum(),z)[0].detach().numpy()
    valid=np.isfinite(values);truth=world.risk_truth(points);true_gradient=world.risk_gradient(points)
    return dict(support=valid.tolist(),values=[float(v) if np.isfinite(v) else None for v in values],
        gradients=[v.tolist() if np.isfinite(v).all() else None for v in grads],
        squared_error=[float((v-t)**2) if ok else None for v,t,ok in zip(values,truth,valid)],
        gradient_squared_error=[float(np.sum((v-t)**2)) if ok else None for v,t,ok in zip(grads,true_gradient,valid)],
        coverage=float(valid.mean()),mse=float(np.mean((values[valid]-truth[valid])**2)) if valid.any() else None,
        gradient_mse=float(np.mean(np.sum((grads[valid]-true_gradient[valid])**2,axis=1))) if valid.any() else None)


def replay_audit(agent,world,stage,weights,damping,old_policy,new_policy):
    """Evaluator-only fixed-horizon pairs; never calls update or inserts evidence."""
    results={};work=0
    for environment in ('model','truth'):
        for name,policy in [('incumbent',old_policy),('candidate',new_policy)]:
            x=tensor(stage.entry);cost=0.;valid=True;collision=False
            path=[x.tolist()]
            for _ in range(agent.cfg.replay_steps):
                u,_=command(x,tensor(agent.control_target(stage) if agent.cfg.replay_objective=='mission' else stage.target),weights,stage.centers,agent.cfg.rbf_width,
                            tensor(policy),stage.obstacles,agent.cfg.force_limit)
                u=u.detach();work+=1
                if environment=='model':
                    after=flow(x,u,damping,agent.cfg.dt).detach()
                    exposure=float(risk(after[:2],weights,stage.centers,agent.cfg.rbf_width))
                    obstacles=stage.obstacles
                else:
                    actual,hit,_,_=world.step(x.numpy(),u.numpy())
                    after=tensor(actual);collision|=hit
                    exposure=float(world.risk_truth(actual[:2]));obstacles=world._obstacles
                valid &= stage.supported(after[:2].numpy(),agent.cfg)
                if agent.cfg.replay_objective=='mission':
                    cost+=agent.cfg.dt*(1+4*exposure+.03*float(u.square().sum()))
                else:
                    cost+=agent.cfg.dt*(4*exposure+.03*float(u.square().sum())+.1*float((after[:2]-tensor(stage.target)).square().sum()))
                    for o in obstacles:
                        clearance=torch.linalg.vector_norm(after[:2]-tensor(o[:2]))-o[2]-.12
                        cost+=float(5*torch.nn.functional.softplus(-clearance*10)/10)
                if agent.cfg.replay_objective=='mission' and environment=='model':
                    a=x[:2].numpy();b=after[:2].numpy();delta=b-a
                    for ox,oy,radius in obstacles:
                        center=np.array([ox,oy]);fraction=np.clip(np.dot(center-a,delta)/(np.dot(delta,delta)+1e-20),0,1)
                        collision |= np.linalg.norm(a+fraction*delta-center)<=radius+.12
                x=after;path.append(x.tolist())
                if agent.cfg.replay_objective=='mission' and (collision or np.linalg.norm(x[:2].numpy()-agent.goal)<.25):
                    if collision:cost+=40.
                    break
            if agent.cfg.replay_objective=='legacy':cost+=float((x[:2]-tensor(stage.target)).square().sum())
            results[f'{environment}_{name}']=dict(cost=cost,valid=bool(valid),collision=bool(collision),states=path)
    predicted=results['model_incumbent']['cost']-results['model_candidate']['cost']
    actual=results['truth_incumbent']['cost']-results['truth_candidate']['cost']
    return dict(stage=stage.id,start=stage.entry,horizon=agent.cfg.replay_steps,results=results,
                predicted_improvement=predicted,realized_improvement=actual,
                valid=all(r['valid'] for r in results.values()),audit_steps=work,
                objective=('mission running time/risk/effort cost; truncated horizon' if agent.cfg.replay_objective=='mission' else 'same risk/effort/target/soft-contact replay objective; not complete mission cost'))
