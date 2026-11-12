"""One-agent local learning. Receives immutable sensor/transition records only."""
import time
import numpy as np
import torch
from .model import tensor, risk, features, flow, potential, command
from .records import new_stage, Counterfactual

class Learner:
    def __init__(self, cfg, initial, goal, mode='full', reconcile=True):
        if mode not in ('frozen', 'evidence', 'full'):
            raise ValueError(mode)
        self.cfg, self.goal, self.mode, self.reconcile = cfg, tuple(goal), mode, reconcile
        self.stages = [new_stage(0, initial[:2], initial, goal, cfg)]
        self.damping = cfg.damping_initial
        self.policy = tensor(cfg.policy_initial)
        self.version = 0
        self.counterfactual_steps = 0
        self.identification_steps = 0
        self.diagnostics = []
        self.context_jumps = []
        self.geometry_store=();self.geometry_overflow=False;self.navigation={};self.navigation_step=0
        self.last_control={};self.current_state=tuple(initial)
        if cfg.route_field_mode=='supported_field':
            from .route_field import sync
            sync(self)

    def observe(self, observation, transition=None):
        if transition is not None:self.current_state=tuple(transition.after)
        if self.cfg.controller_mode!='legacy':
            from .navigation import observe_geometry
            observe_geometry(self,observation)
        for stage in self.stages:
            known = {r.uid for r in stage.readings}
            stage.readings.extend(r for r in observation.readings if r.uid not in known and stage.contains(r.q, self.cfg))
            stage.readings = stage.readings[-self.cfg.sample_cap:]
            if self.cfg.controller_mode=='legacy':
                stage.obstacles = tuple(sorted(set(stage.obstacles + tuple(o for o in observation.obstacles
                    if stage.contains(o[:2], self.cfg)))))
        if transition is not None:
            self.stages[-1].transitions.append(transition)
            self.stages[-1].transitions = self.stages[-1].transitions[-self.cfg.transition_cap:]

    def control_obstacles(self,stage):
        return self.geometry_store if self.cfg.controller_mode!='legacy' else stage.obstacles

    def control_target(self, stage=None):
        """Execution attraction is independent of memory/replay stage targets."""
        stage = self.stages[-1] if stage is None else stage
        if self.cfg.controller_mode=='routed':return tuple(self.navigation.get('target',self.goal))
        return self.goal if self.cfg.navigation_target == 'mission' else stage.target

    def shaping_energy(self,x,stage=None):
        stage=self.stages[-1] if stage is None else stage
        if getattr(self.cfg,'stage_blend_mode','hard')=='smooth':
            from .stage_prediction import FrozenStages
            atlas=FrozenStages.freeze(self)
            index=next(k for k,s in enumerate(self.stages) if s.id==stage.id)
            return atlas.energy(tensor(x[:2]),index,self.policy,self.navigation)
        return potential(tensor(x[:2]),tensor(self.control_target(stage)),stage.weights,stage.centers,
            self.cfg.rbf_width,self.policy,self.control_obstacles(stage),self.cfg,getattr(stage,'local_residual',None))

    def action(self, x):
        if self.cfg.route_field_mode=='supported_field':
            from .route_field import sync
            sync(self)
        if self.cfg.replay_domain == 'retained':
            from .stage_prediction import FrozenStages
            atlas=FrozenStages.freeze(self)
            u,proposed=atlas.control(tensor(x),len(atlas.stages)-1,self.policy,damping=self.damping,context=self.navigation,step=self.navigation_step)
            self.last_control=self.navigation.get('last_control',{}).copy();self.navigation_step+=1
            return u.detach().numpy(),proposed.detach().numpy()
        s = self.stages[-1]
        u, proposed = command(tensor(x), tensor(self.control_target(s)), s.weights, s.centers,
                              self.cfg.rbf_width, self.policy, s.obstacles, self.cfg.force_limit,residual=getattr(s,'local_residual',None))
        return u.detach().numpy(), proposed.detach().numpy()

    def advance(self, x):
        s, cfg = self.stages[-1], self.cfg
        step = cfg.stage_width*(1-cfg.overlap)
        # Shift only along the controlled corridor axis: exact nominal box overlap.
        from .stage_prediction import stage_due
        if not stage_due(x,s,self.goal,cfg):
            return False
        old = float(self.shaping_energy(x,s))
        n = new_stage(s.id+1, (s.center[0]+step, s.center[1]), x, self.goal, cfg)
        n.readings = [r for r in s.readings if n.contains(r.q,cfg)][-cfg.sample_cap:]
        if cfg.controller_mode=='legacy':n.obstacles = tuple(o for o in s.obstacles if n.contains(o[:2],cfg))
        else:
            import copy
            from .navigation import intersects
            n.obstacles=tuple(o for o in self.geometry_store if intersects(o,n.center,cfg))
            n.footprints=list(dict.fromkeys(tuple(f) for oldstage in self.stages for f in oldstage.footprints if intersects((f[0],f[1],f[2]),n.center,cfg)))[-cfg.footprint_cap:]
            n.entry_navigation=copy.deepcopy(self.navigation);n.entry_navigation['planned_at']=-10000
        n.anchors = tuple(r.q for r in n.readings if int(r.uid.split(':')[-1]) % 5)[:cfg.anchor_cap]
        n.version = self.version
        self.stages.append(n)
        self.stages = self.stages[-(cfg.previous+1):]
        from .route_field import sync
        sync(self)
        after = float(self.shaping_energy(x,n))
        self.context_jumps.append(dict(stage=n.id, shaping_jump=after-old, physical_jump=0.))
        return True

    def _field_fit(self):
        cfg = self.cfg
        m = len(self.stages[0].weights)
        rows, targets = [], []
        total = len(self.stages)*m
        for k,s in enumerate(self.stages):
            for r in s.readings:
                if int(r.uid.split(':')[-1]) % 5 == 0:
                    continue
                row = torch.zeros(total, dtype=torch.float64)
                row[k*m:(k+1)*m] = features(tensor(r.q), s.centers, cfg.rbf_width)
                rows.append(row)
                targets.append(float(np.log(np.expm1(4*max(r.value, .001)))))
        for k in range(1,len(self.stages)):
            left,right = self.stages[k-1:k+1]
            for q in right.anchors:
                if not left.supported(q,cfg) or not right.supported(q,cfg):
                    continue
                row=torch.zeros(total,dtype=torch.float64)
                row[(k-1)*m:k*m]=features(tensor(q),left.centers,cfg.rbf_width)
                row[k*m:(k+1)*m]=-features(tensor(q),right.centers,cfg.rbf_width)
                rows.append(2*row); targets.append(0.)
        if not rows:
            return [s.weights.clone() for s in self.stages], [None]*len(self.stages)
        A=torch.stack(rows); y=tensor(targets)
        normal=A.T@A+cfg.fit_ridge*torch.eye(total,dtype=torch.float64)
        prior=torch.cat([s.weights for s in self.stages])
        w=torch.linalg.solve(normal,A.T@y+cfg.fit_ridge*prior)
        covariance=torch.linalg.inv(normal)  # Design curvature only, not calibrated posterior.
        return list(w.split(m)),[covariance[k*m:(k+1)*m,k*m:(k+1)*m] for k in range(len(self.stages))]

    def _transitions(self, validation=False):
        return [t for s in self.stages for t in s.transitions if (t.uid % 5 == 0)==validation]

    def _plant_loss(self, damping, validation=False, energy=True):
        records=self._transitions(validation)
        if not records:
            return tensor(0.)
        x=tensor([t.x for t in records]); u=tensor([t.u for t in records]); y=tensor([t.after for t in records])
        predicted=flow(x,u,damping,self.cfg.dt)
        loss=(predicted-y).square().mean()
        if energy:
            # Trapezoid power on measured states, fixed model; finite-dt tolerance applies.
            v0,v1=x[:,2:],y[:,2:]
            delta=.5*(v1.square().sum(-1)-v0.square().sum(-1))
            power=(u*(v0+v1)/2).sum(-1)-damping*(v0.square().sum(-1)+v1.square().sum(-1))/2
            loss=loss+.05*(delta-self.cfg.dt*power).square().mean()
        return loss

    def _fit_damping(self):
        if len(self._transitions())<3:
            return self.damping
        d=tensor(self.damping).requires_grad_(True)
        for _ in range(self.cfg.solver_steps):
            loss=self._plant_loss(d)
            g=torch.autograd.grad(loss,d,create_graph=True)[0]
            h=torch.autograd.grad(g,d)[0]
            d=(d-g/(h.abs()+1e-5)).detach().requires_grad_(True)
            # Numerical guard is retained even in the no-reconciliation ablation.
            d=d.clamp(.05,3.).detach().requires_grad_(True)
            self.identification_steps += len(self._transitions())
        return float(d.detach())

    def _field_validation(self, weights, residuals=None):
        errors=[]
        for k,(s,w) in enumerate(zip(self.stages,weights)):
            for r in s.readings:
                if int(r.uid.split(':')[-1]) % 5 == 0:
                    residual=getattr(s,'local_residual',None) if residuals is None else residuals[k]
                    errors.append((risk(tensor(r.q),w,s.centers,self.cfg.rbf_width,residual)-r.value).square())
        return float(torch.stack(errors).mean()) if errors else 0.

    def replay(self, policy, weights, damping):
        from .replay_budget import remaining,batch_steps
        self.replay_controls=[]
        if remaining(self)<batch_steps(self):
            self.replay_budget_exhausted=True
            self.replay_stage_events=[]
            return None,[]
        if self.cfg.replay_domain == 'retained':
            from .stage_prediction import retained_replay
            return retained_replay(self,policy,weights,damping)
        costs, trajectories=[],[]
        self.replay_eligibility=[]
        for stage_index,(s,w) in enumerate(zip(self.stages,weights)):
            pending=getattr(self,'receiver_localized_proposal',None)
            residual=pending[stage_index] if getattr(self,'receiver_localization_mode','global')=='compact' and pending is not None else getattr(s,'local_residual',None)
            x=tensor(s.entry)
            if not s.supported(x[:2].detach().numpy(),self.cfg):
                self.replay_eligibility.append(dict(stage=s.id,eligible=False,reason="unsupported_entry"))
                continue
            self.replay_eligibility.append(dict(stage=s.id,eligible=True,reason="supported"))
            target=self.control_target(s) if self.cfg.replay_objective == "mission" else s.target
            path=[tuple(x.detach().tolist())]; cost=tensor(0.)
            valid=True
            for _ in range(self.cfg.replay_steps):
                u,_=command(x,tensor(target),w.detach(),s.centers,self.cfg.rbf_width,
                            policy,s.obstacles,self.cfg.force_limit,residual=residual)
                after=flow(x,u,tensor(damping).detach(),self.cfg.dt)
                self.counterfactual_steps+=1
                if not s.supported(after[:2].detach().numpy(),self.cfg):
                    valid=False;break
                if self.cfg.replay_objective == 'mission':
                    cost=cost+self.cfg.dt*(1+4*risk(after[:2],w.detach(),s.centers,self.cfg.rbf_width,residual)
                                           +.03*u.square().sum())
                else:
                    cost=cost+self.cfg.dt*(4*risk(after[:2],w.detach(),s.centers,self.cfg.rbf_width,residual)
                                           +.03*u.square().sum()+.1*(after[:2]-tensor(s.target)).square().sum())
                    for o in s.obstacles:
                        clearance=torch.linalg.vector_norm(after[:2]-tensor(o[:2]))-o[2]-.12
                        cost=cost+5*torch.nn.functional.softplus(-clearance*10)/10
                # Same swept collision test and goal termination as execution.
                collision=False
                if self.cfg.replay_objective == 'mission':
                    a=x[:2].detach().numpy(); b=after[:2].detach().numpy(); delta=b-a
                    for ox,oy,radius in s.obstacles:
                        center=np.array([ox,oy]); fraction=np.clip(np.dot(center-a,delta)/(np.dot(delta,delta)+1e-20),0,1)
                        collision |= np.linalg.norm(a+fraction*delta-center)<=radius+.12
                x=after;path.append(tuple(x.detach().tolist()))
                if self.cfg.replay_objective == 'mission' and (collision or np.linalg.norm(x[:2].detach().numpy()-self.goal)<.25):
                    if collision: cost=cost+40.
                    break
            if not valid:
                # Candidate domain exit rejects the entire batch, not just its expensive branch.
                self.replay_eligibility[-1]['reason']='domain_exit'
                return None, []
            if self.cfg.replay_objective == 'legacy':
                cost=cost+(x[:2]-tensor(s.target)).square().sum()
            # A truncated replay is not an episode timeout; no invented terminal penalty.
            costs.append(cost);trajectories.append((s.id,path,cost))
        return (torch.stack(costs).mean(),trajectories) if costs else (None,[])

    def update(self, x, force_invalid=False):
        begin=time.perf_counter(); cfg=self.cfg
        compact=getattr(self,'receiver_localization_mode','global')=='compact'
        if compact:self.receiver_localized_proposal=None
        report=dict(version_before=self.version, mode=self.mode, accepted=False,
                    stages=[s.id for s in self.stages], reason='frozen',
                    simulator_steps_before=self.counterfactual_steps)
        if self.mode=='frozen':
            self.diagnostics.append(report);return report
        weights,cov=self._field_fit(); d=self._fit_damping()
        report['field_validation_before']=self._field_validation([s.weights for s in self.stages])
        # Reconcile plant proposal separately from fit.
        if self.reconcile:
            d=float(np.clip(d,max(cfg.damping_bounds[0],self.damping-cfg.damping_trust),
                             min(cfg.damping_bounds[1],self.damping+cfg.damping_trust)))
        report['damping_proposal']=d
        policy=self.policy.clone()
        trace = None
        if getattr(self, 'decision_tracing', False):
            trace = dict(initial=policy.tolist(), gradient=None, unconstrained_proposal=None,
                         bounded_proposal=None, attempts=[], backtracking_accepted=False)
        planned=self.mode=='full' and (cfg.replay_update_cap is not None or cfg.replay_horizon_mode!='fixed')
        if planned:
            from .replay_budget import prepare
            prepare(self,weights,d)
        baseline,paths=self.replay(policy,weights,d) if self.mode=='full' and (not planned or self.replay_baseline_reserved) else (None,[])
        report['replay_eligibility']=list(getattr(self,'replay_eligibility',[]))
        report['replay_before']=float(baseline.detach()) if baseline is not None else None
        report['policy_gradient_norm']=0.
        if baseline is not None:
            p=policy.clone().requires_grad_(True)
            cost,_=self.replay(p,weights,d)
            # A fixed braking branch may be entirely independent of policy.
            gradient=torch.autograd.grad(cost,p,allow_unused=True)[0] if cost.requires_grad else None
            report['policy_independent_replay']=gradient is None
            if gradient is None:gradient=torch.zeros_like(p)
            report['policy_gradient_norm']=float(torch.linalg.vector_norm(gradient))
            # Explicit error-driven proposal, followed by box/trust-region projection.
            proposed=p.detach()-.2*gradient.detach()
            if trace is not None:
                trace.update(gradient=gradient.detach().tolist(), unconstrained_proposal=proposed.tolist())
            if self.reconcile:
                low=torch.maximum(tensor(cfg.policy_bounds[0]),policy-cfg.policy_trust)
                high=torch.minimum(tensor(cfg.policy_bounds[1]),policy+cfg.policy_trust)
                proposed=torch.maximum(low,torch.minimum(high,proposed))
            if trace is not None:trace['bounded_proposal']=proposed.tolist()
            for power in range(cfg.solver_steps):
                if planned:
                    from .replay_budget import remaining,batch_steps
                    if remaining(self)<batch_steps(self):
                        self.replay_budget_exhausted=True;break
                candidate=policy+(proposed-policy)*(0.5**power)
                candidate_cost,candidate_paths=self.replay(candidate,weights,d)
                candidate_ok = candidate_cost is not None and torch.isfinite(candidate_cost) and float(candidate_cost)<=float(baseline)+1e-10
                if trace is not None:
                    trace['attempts'].append(dict(power=power, policy=candidate.tolist(),
                        cost=float(candidate_cost.detach()) if candidate_cost is not None and torch.isfinite(candidate_cost) else None,
                        reason='accepted' if candidate_ok else 'unsupported_or_nonfinite' if candidate_cost is None or not torch.isfinite(candidate_cost) else 'replay_cost',
                        accepted=bool(candidate_ok)))
                if candidate_ok:
                    if trace is not None:trace['backtracking_accepted']=True
                    policy=candidate;paths=candidate_paths;break
        if force_invalid:
            policy=policy*float('nan')
        report['field_validation_after']=self._field_validation(weights,self.receiver_localized_proposal) if compact else self._field_validation(weights)
        report['plant_validation_before']=float(self._plant_loss(tensor(self.damping),True))
        report['plant_validation_after']=float(self._plant_loss(tensor(d),True))
        finite=bool(torch.isfinite(policy).all() and all(torch.isfinite(w).all() for w in weights) and np.isfinite(d))
        feasible=(cfg.damping_bounds[0]<=d<=cfg.damping_bounds[1] and bool((policy>=0).all())
                  and bool((policy<=cfg.policy_bounds[1]).all()))
        valid=(report['field_validation_after']<=report['field_validation_before']+cfg.validation_slack
               and report['plant_validation_after']<=report['plant_validation_before']+cfg.validation_slack)
        accepted=finite and ((feasible and valid) or not self.reconcile)
        receiver_reason=None
        if hasattr(self,'_receiver_acceptance'):
            receiver_ok,receiver_reason,receiver_report=self._receiver_acceptance(weights)
            if receiver_report is not None:
                report['receiver_fit']=receiver_report
                report['whole_model_checks_passed']=bool(accepted)
                accepted=accepted and receiver_ok
        report.update(accepted=accepted,reason='accepted' if accepted else ('nonfinite' if not finite else receiver_reason or 'validation_or_feasibility'))
        s=self.stages[-1]
        old=float(self.shaping_energy(x,s))
        report['older_stage_change']=0.
        if accepted:
            self.version+=1
            for index,(stage,w,c) in enumerate(zip(self.stages,weights,cov)):
                if index<len(self.stages)-1:
                    report['older_stage_change']+=float(torch.linalg.vector_norm(w-stage.weights))
                stage.weights=w.detach().clone(); stage.covariance=c; stage.version=self.version
                if compact:
                    import copy
                    stage.local_residual=copy.deepcopy(self.receiver_localized_proposal[index])
                stage.counterfactuals=[Counterfactual(sid,self.version,'static_current_environment_replay',tuple(path),float(cost.detach()))
                    for sid,path,cost in paths if sid==stage.id]
            self.policy=policy.detach().clone();self.damping=d
        after=float(self.shaping_energy(x,s))
        report.update(version_after=self.version,shaping_jump=after-old,physical_jump=0.,
                      damping=self.damping,policy=self.policy.tolist(),
                      replay_after=float(torch.stack([v[2] for v in paths]).mean().detach()) if paths else None,
                      counterfactual_steps=self.counterfactual_steps-report['simulator_steps_before'],
                      wall_seconds=time.perf_counter()-begin)
        if planned:
            report.update(replay_plan=list(self.replay_plan.values()),replay_pilot_steps=self.replay_pilot_steps,
                          replay_budget_exhausted=self.replay_budget_exhausted,replay_update_cap=cfg.replay_update_cap)
            self.replay_plan_active=False
        if trace is not None:
            trace.update(committed=self.policy.tolist(), model_accepted=bool(accepted),
                         committed_delta=(self.policy-tensor(trace['initial'])).tolist(),
                         policy_changed=bool(torch.linalg.vector_norm(self.policy-tensor(trace['initial']))>1e-10),
                         replay_available=baseline is not None)
            report['policy_trace']=trace
        self.diagnostics.append(report)
        return report
