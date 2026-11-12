"""Frozen, locally supported stage models. No simulator or evaluator access."""
from dataclasses import dataclass
import numpy as np
import torch
from .model import tensor, risk, potential, command, flow
from .shapes import is_slab, slab_sweep

_COMMITTED_RESIDUAL=object()

class DomainExit(Exception):
    pass


def stage_due(q, stage, goal, cfg):
    return bool(float(q[0]) >= stage.center[0]+cfg.stage_width*(1-cfg.overlap)/2
                and stage.target[0] < goal[0])


@dataclass(frozen=True)
class StageModel:
    id: int
    center: tuple
    entry: tuple
    target: tuple
    centers: tuple
    weights: tuple
    obstacles: tuple
    support: tuple
    anchors: tuple
    version: int
    footprints: tuple = ()
    entry_navigation: object = None
    local_residual: object = None

    @classmethod
    def freeze(cls,s,weights=None,residual=_COMMITTED_RESIDUAL):
        return cls(s.id,tuple(s.center),tuple(s.entry),tuple(s.target),tuple(map(tuple,s.centers.tolist())),
            tuple((s.weights if weights is None else weights).detach().tolist()),tuple(map(tuple,s.obstacles)),
            tuple(dict.fromkeys(tuple(r.q) for r in s.readings if int(r.uid.split(':')[-1])%5)),
            tuple(map(tuple,s.anchors)),s.version,tuple(map(tuple,s.footprints)),__import__('copy').deepcopy(s.entry_navigation),
            __import__('copy').deepcopy(getattr(s,'local_residual',None) if residual is _COMMITTED_RESIDUAL else residual))

    def supported(self,q,cfg):
        q=np.asarray(q)
        return bool(abs(q[0]-self.center[0])<=cfg.stage_width/2+1e-10 and
                    abs(q[1]-self.center[1])<=cfg.stage_height/2+1e-10 and self.support and
                    np.min(np.linalg.norm(np.asarray(self.support)-q,axis=1))<=.65)


@dataclass(frozen=True)
class FrozenStages:
    stages: tuple
    cfg: object
    goal: tuple
    geometry: tuple = ()
    footprints: tuple = ()
    overflow: bool = False
    stage_mode: str = "forward"
    # 'strict' (default) exits when a due handoff has no created successor. 'hold'
    # keeps predicting on the current stage while its own measured support holds:
    # prediction cannot create stages, but a missing successor is not missing support.
    transition_mode: str = "strict"

    @classmethod
    def freeze(cls,agent,weights=None):
        ws=[s.weights for s in agent.stages] if weights is None else weights
        pending=getattr(agent,'receiver_localized_proposal',None) if weights is not None and getattr(agent,'receiver_localization_mode','global')=='compact' else None
        residuals=[(pending.get(s.id,getattr(s,'local_residual',None)) if isinstance(pending,dict) else pending[k]) if pending is not None else _COMMITTED_RESIDUAL for k,s in enumerate(agent.stages)]
        return cls(tuple(StageModel.freeze(s,w,r) for s,w,r in zip(agent.stages,ws,residuals)),agent.cfg,tuple(agent.goal),
            tuple(agent.geometry_store),tuple(dict.fromkeys(tuple(f) for s in agent.stages for f in s.footprints)),agent.geometry_overflow,getattr(agent,"stage_mode","forward"),
            getattr(agent,"stage_transition_mode","strict"))

    def advance(self,q,index):
        old=self.stages[index]
        if not old.supported(q,self.cfg):raise DomainExit('unsupported_state')
        due=stage_due(q,old,self.goal,self.cfg)
        if self.stage_mode=="spatial":
            due=bool(np.any(np.abs(np.asarray(q)-old.center)>np.array([self.cfg.stage_width,self.cfg.stage_height])*(1-self.cfg.overlap)/2))
        if due:
            if index+1>=len(self.stages) or self.stages[index+1].id!=old.id+1:
                if self.transition_mode=='hold':return index
                raise DomainExit('unavailable_stage')
            if not self.stages[index+1].supported(q,self.cfg):raise DomainExit('unsupported_overlap')
            return index+1
        return index

    def index_at(self,q):
        for i in range(len(self.stages)-1,-1,-1):
            if self.stages[i].supported(q,self.cfg):return i
        raise DomainExit('unsupported_sample')

    def value(self,q,index):
        if getattr(self.cfg,'stage_blend_mode','legacy')=='smooth':
            members,weights=self.blend_weights(q)
            return sum(w*risk(q,tensor(s.weights),tensor(s.centers),self.cfg.rbf_width,residual=s.local_residual) for s,w in zip(members,weights))
        s=self.stages[index]
        return risk(q,tensor(s.weights),tensor(s.centers),self.cfg.rbf_width,residual=s.local_residual)

    def blend_weights(self,q,member_ids=None):
        """Compact rectangular partition on the already-certified stage set.

        Membership uses the unchanged local-measurement support test. Derivatives
        are valid within a fixed membership region; its hard .65 boundary, model
        changes and stage creation/eviction are not made globally smooth here.
        Optional member_ids is only for fixed-state diagnostic context accounting.
        """
        point=q.detach().numpy()
        members=[s for s in self.stages if (s.supported(point,self.cfg) if member_ids is None else s.id in member_ids)]
        if not members:raise DomainExit('unsupported_blend_state')
        half=tensor([self.cfg.stage_width/2,self.cfg.stage_height/2])
        width=max(min(self.cfg.stage_width,self.cfg.stage_height)*min(self.cfg.overlap,.5),1e-12)
        raw=[]
        for s in members:
            z=((half-(q-tensor(s.center)).abs())/width).clamp(0.,1.)
            raw.append((z.square()*(3-2*z)).prod())
        raw=torch.stack(raw);total=raw.sum()
        if len(members)==1:
            # The normalized singleton is identically one, including a certified
            # rectangular edge; avoid cancellation through a tiny raw/raw ratio.
            return members,raw*0.+1.
        if float(total.detach())<=1e-20:
            raise DomainExit('degenerate_blend_partition')
        return members,raw/total

    def potential_at(self,q,index,policy,target=None,member_ids=None):
        """Shared scalar potential for execution, replay and energy accounting."""
        active=self.stages[index]
        if target is None:target=self.goal if self.cfg.navigation_target=='mission' else active.target
        target=tensor(target)
        if getattr(self.cfg,'stage_blend_mode','legacy')!='smooth':
            return potential(q,target,tensor(active.weights),tensor(active.centers),self.cfg.rbf_width,policy,
                self.geometry if self.cfg.controller_mode!='legacy' else active.obstacles,self.cfg,residual=active.local_residual)
        members,weights=self.blend_weights(q,member_ids)
        # Every stage sees the same physical target and sensed geometry. The full
        # weighted scalar is differentiated, including the partition derivatives.
        return sum(w*potential(q,target,tensor(s.weights),tensor(s.centers),self.cfg.rbf_width,policy,self.geometry,self.cfg,residual=s.local_residual)
            for s,w in zip(members,weights))

    def nominal_control(self,x,index,policy,target):
        q=x[:2]
        if not q.requires_grad:q=q.clone().requires_grad_(True)
        energy=self.potential_at(q,index,policy,target)
        grad=torch.autograd.grad(energy,q,create_graph=True)[0]
        raw=-grad-1.4*x[2:]
        return raw*torch.clamp(self.cfg.force_limit/torch.linalg.vector_norm(raw).clamp_min(1e-12),max=1.),raw

    def energy(self,q,index,policy,context=None):
        s=self.stages[index]
        target=self.goal if self.cfg.navigation_target=='mission' else s.target
        if self.cfg.shaping_mode=='progress' and self.cfg.controller_mode=='routed' and context:target=context.get('target',target)
        return self.potential_at(q,index,policy,target)

    def control(self,x,index,policy,damping=None,context=None,step=0):
        if self.cfg.controller_mode!='legacy':
            from .navigation import guarded_control
            u,p,_=guarded_control(self,x,index,policy,self.cfg.damping_initial if damping is None else damping,{} if context is None else context,step)
            return u,p
        s=self.stages[index]
        target=self.goal if self.cfg.navigation_target=='mission' else s.target
        return command(x,tensor(target),tensor(s.weights),tensor(s.centers),self.cfg.rbf_width,policy,s.obstacles,self.cfg.force_limit,residual=s.local_residual)


def replay_entries(agent, atlas):
    """Select starts without changing retained models or measured stage entries."""
    if getattr(agent, 'replay_start_mode', 'historical') == 'current':
        from dataclasses import replace
        import copy
        i = len(atlas.stages)-1
        return [(i, replace(atlas.stages[i], entry=tuple(agent.position),
                            entry_navigation=copy.deepcopy(agent.navigation)))]
    return list(enumerate(atlas.stages))


def retained_replay(agent,policy,weights,damping):
    atlas=FrozenStages.freeze(agent,weights);cfg=agent.cfg;costs=[];paths=[]
    agent.replay_eligibility=[];agent.replay_stage_events=[];agent.replay_controls=[]
    for start,s in replay_entries(agent, atlas):
        x=tensor(s.entry);index=start;cost=tensor(0.);path=[tuple(x.tolist())]
        context=__import__('copy').deepcopy(s.entry_navigation or {})
        if not s.supported(x[:2].numpy(),cfg):
            agent.replay_eligibility.append(dict(stage=s.id,eligible=False,reason='unsupported_entry'));continue
        agent.replay_eligibility.append(dict(stage=s.id,eligible=True,reason='supported'))
        try:
            plan=getattr(agent,'replay_plan',{}) if getattr(agent,'replay_plan_active',False) else {}
            horizon=plan.get(s.id,{}).get('horizon',cfg.replay_steps)
            for step in range(horizon):
                new=atlas.advance(x[:2].detach().numpy(),index)
                if new!=index:
                    jump=float((atlas.energy(x[:2],new,policy,context)-atlas.energy(x[:2],index,policy,context)).detach())
                    agent.replay_stage_events.append(dict(entry=s.id,step=step,from_stage=atlas.stages[index].id,
                        to_stage=atlas.stages[new].id,shaping_jump=jump,physical_jump=0.))
                index=new;active=atlas.stages[index]
                u,_=atlas.control(x,index,policy,damping=damping,context=context,step=step)
                meta=context.get('last_control',{});agent.counterfactual_steps+=meta.get('guard_steps',0)
                if meta:agent.replay_controls.append(dict(entry=s.id,step=step,**meta))
                if meta.get('infeasible'):raise DomainExit('safety_infeasible')
                after=flow(x,u,tensor(damping).detach(),cfg.dt)
                agent.counterfactual_steps+=1
                if not active.supported(after[:2].detach().numpy(),cfg):raise DomainExit('unsupported_motion')
                cost=cost+cfg.dt*(1+4*context.get('risk_weight',1.)*atlas.value(after[:2],index)+.03*u.square().sum())
                a=x[:2].detach().numpy();b=after[:2].detach().numpy();delta=b-a;collision=False
                for o in active.obstacles:
                    if is_slab(o):
                        collision |= slab_sweep(a,b,o)<=.12
                        continue
                    ox,oy,radius=o
                    c=np.array([ox,oy]);f=np.clip(np.dot(c-a,delta)/(np.dot(delta,delta)+1e-20),0,1)
                    collision |= np.linalg.norm(a+f*delta-c)<=radius+.12
                x=after;path.append(tuple(x.detach().tolist()))
                if collision or np.linalg.norm(b-agent.goal)<.25:
                    if collision:cost=cost+40.
                    break
        except DomainExit as exc:
            agent.replay_eligibility[-1]['reason']=str(exc)
            return None,[]
        costs.append(cost);paths.append((s.id,path,cost))
    return (torch.stack(costs).mean(),paths) if costs else (None,[])
