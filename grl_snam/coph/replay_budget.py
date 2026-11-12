"""Frozen per-update replay plans and explicitly counted boundary pilots."""
import numpy as np
from .model import tensor,flow
from .stage_prediction import FrozenStages,DomainExit,replay_entries


def remaining(agent):
    cap=agent.cfg.replay_update_cap
    if cap is None or not getattr(agent,'replay_plan_active',False):return float('inf')
    return max(0,cap-(agent.counterfactual_steps-agent.replay_budget_start)-getattr(agent,'replay_external_reserve',0))


def multiplier(agent):return 7 if agent.cfg.controller_mode in ('protected','routed') or agent.geometry_overflow else 1


def batch_steps(agent):
    plan=getattr(agent,'replay_plan',{}) if getattr(agent,'replay_plan_active',False) else {}
    starts=([s for _,s in replay_entries(agent,FrozenStages.freeze(agent))]
            if getattr(agent,'replay_start_mode','historical')=='current' else agent.stages)
    return multiplier(agent)*sum(plan.get(s.id,{}).get('horizon',agent.cfg.replay_steps) for s in starts
               if s.supported(s.entry[:2],agent.cfg))


def prepare(agent,weights,damping):
    cfg=agent.cfg;agent.replay_plan_active=True;agent.replay_budget_start=agent.counterfactual_steps
    agent.replay_plan={};agent.replay_budget_exhausted=False;agent.replay_pilot_steps=0
    agent.replay_eligibility=[];agent.replay_stage_events=[]
    atlas=FrozenStages.freeze(agent,weights)
    entries=replay_entries(agent,atlas)
    for i,s in entries:
        record=dict(stage=s.id,entry=s.entry,horizon=cfg.replay_steps,pilot_steps=0,
                    reason='fixed',pilot_controls=[],pilot_handoffs=[],eligible=s.supported(s.entry[:2],cfg))
        agent.replay_plan[s.id]=record
        if not record['eligible']:record['reason']='unsupported_entry';continue
        if cfg.replay_horizon_mode=='fixed':continue
        if i+1>=len(atlas.stages) or atlas.stages[i+1].id!=s.id+1:
            record['reason']='no_known_successor';continue
        # Reserve baseline + gradient for the largest possible chosen batch before
        # spending on pilots. Unused reservation is not counted as executed work.
        reserved=sum(agent.replay_plan.get(t.id,{}).get('horizon',cfg.replay_steps) for _,t in entries if t.supported(t.entry[:2],cfg))
        reserve=(cfg.boundary_max_steps+2*(reserved-cfg.replay_steps+cfg.boundary_max_steps))*multiplier(agent)
        if remaining(agent)<reserve:
            record['reason']='pilot_budget';continue
        x=tensor(s.entry);index=i;target=None;context=__import__('copy').deepcopy(s.entry_navigation or {})
        try:
            for step in range(cfg.boundary_max_steps):
                nxt=atlas.advance(x[:2].numpy(),index)
                if nxt!=index:
                    record['pilot_handoffs'].append(dict(step=step,from_stage=atlas.stages[index].id,to_stage=atlas.stages[nxt].id,
                        shaping_jump=float((atlas.energy(x[:2],nxt,agent.policy,context)-atlas.energy(x[:2],index,agent.policy,context)).detach()),physical_jump=0.))
                    if target is None:
                        target=max(cfg.replay_steps,step+cfg.boundary_follow_steps)
                        if target>cfg.boundary_max_steps:raise DomainExit('post_boundary_cap')
                index=nxt;u,_=atlas.control(x,index,agent.policy,damping=damping,context=context,step=step)
                meta=context.get('last_control',{});guards=meta.get('guard_steps',0)
                record['pilot_controls'].append(dict(step=step,**meta));record['pilot_steps']+=guards;agent.counterfactual_steps+=guards;agent.replay_pilot_steps+=guards
                if meta.get('infeasible'):raise DomainExit('pilot_safety_infeasible')
                after=flow(x,u,damping,cfg.dt).detach()
                record['pilot_steps']+=1;agent.counterfactual_steps+=1;agent.replay_pilot_steps+=1
                if not atlas.stages[index].supported(after[:2].numpy(),cfg):raise DomainExit('pilot_domain_exit')
                # Collision is genuine termination, never an artificial short horizon.
                a=x[:2].numpy();b=after[:2].numpy();delta=b-a;collision=False
                for o in atlas.stages[index].obstacles:
                    if len(o)==5:
                        from .shapes import slab_sweep
                        collision |= slab_sweep(a,b,o)<=.12
                        continue
                    ox,oy,r=o
                    c=np.array([ox,oy]);f=np.clip(np.dot(c-a,delta)/(np.dot(delta,delta)+1e-20),0,1)
                    collision |= np.linalg.norm(a+f*delta-c)<=r+.12
                x=after
                terminal=collision or np.linalg.norm(b-agent.goal)<.25
                if target is not None and (step+1>=target or terminal):
                    record.update(horizon=target,reason='supported_boundary');break
                if terminal:record['reason']='termination_before_handoff';break
            else:record['reason']='handoff_not_reached'
        except DomainExit as exc:record['reason']=str(exc)
    agent.replay_baseline_reserved=remaining(agent)>=2*batch_steps(agent)
    if not agent.replay_baseline_reserved:agent.replay_budget_exhausted=True
