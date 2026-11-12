"""Read-only learner instrumentation and explicitly labeled current-state comparator."""
# grl_snam.coph: trimmed. Keeps snapshot (the demos' update records) and the InstrumentedLearner
# hooks, which become the optional Instrumented mixin. probes, probe_metrics and replay_audit are
# evaluator-only and are not vendored (VENDORED.md).
from dataclasses import asdict
import copy
from .learner import Learner
from .model import tensor,flow,command


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


class Instrumented:
    """The InstrumentedLearner hooks as a mixin; list it before the learner class.

    grl_snam.coph: upstream's CooperativeLearner subclasses InstrumentedLearner. Here it subclasses
    Learner, and an evaluation that wants the hooks composes them, e.g.
    ``class X(Instrumented, CooperativeLearner)``. The hooks record replay calls, proposals and
    the raw damping proposal; with ``current_budget=None`` they do not change any result.
    """
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
        # The maze and terrain learners' update() takes x only.
        return super().update(x,force_invalid) if force_invalid else super().update(x)

    def spend_unused_budget(self):
        if self.current_budget is None:return 0
        padding=0;s=self.stages[-1];x=tensor(self.position or s.entry)
        while self.counterfactual_steps<self.current_budget:
            # Explicit non-learning work, separately reported; never disguised as useful replay.
            u,_=command(x,tensor(s.target),s.weights,s.centers,self.cfg.rbf_width,self.policy,s.obstacles,self.cfg.force_limit)
            flow(x,u,self.damping,self.cfg.dt)
            self.counterfactual_steps+=1;padding+=1
        return padding


class InstrumentedLearner(Instrumented,Learner):
    pass
