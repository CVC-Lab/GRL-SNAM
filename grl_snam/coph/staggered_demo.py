"""Shared-origin, exactly staggered starts with an explicit asynchronous message timeline.
Existing controller/learner implementations and prior evaluation runners are unchanged.
"""
# grl_snam.coph: trimmed to the runner (sense, packet_effect, execute). specification, job, frame,
# report and main build the staggered demo's own scenes and figures and are not vendored (VENDORED.md).
import copy,heapq,json,pickle,time
from dataclasses import asdict,replace
import numpy as np
from .records import Config,Transition
from .cooperative import CooperativeLearner,CommunicationConfig,advertise
from .cooperative_evaluation import CooperativeWorld
from .diagnostics import snapshot
from .model import tensor,potential


def sense(world,x,tick,now):
    obs=world.sense(x,tick)
    return replace(obs,time=now,readings=tuple(replace(r,time=now) for r in obs.readings))


def packet_effect(agent,x,packet):
    """Evaluator-only intervention: same measured history, local update in both branches."""
    before=pickle.dumps(agent);began=time.perf_counter();branches={}
    for deliver in (False,True):
        a=copy.deepcopy(agent);event=a.assimilate(packet) if deliver else None
        d=a.update(x.copy());u,_=a.action(x.copy());s=a.stages[-1]
        if getattr(a.cfg,'stage_blend_mode','legacy')=='smooth' or getattr(a,'receiver_localization_mode','global')=='compact':
            H=.5*float(x[2:]@x[2:])+float(a.shaping_energy(x,s))
        else:
            H=.5*float(x[2:]@x[2:])+float(potential(tensor(x[:2]),tensor(a.control_target(s)),s.weights,s.centers,a.cfg.rbf_width,a.policy,a.control_obstacles(s),a.cfg))
        branches['deliver' if deliver else 'withhold']=dict(accepted=d['accepted'],reason=d['reason'],version=a.version,
            fields=[s.weights.tolist() for s in a.stages],policy=a.policy.tolist(),control=u.tolist(),H=H,event=event,
            replay_work=d['counterfactual_steps'],identification_work=a.identification_steps-agent.identification_steps)
    assert before==pickle.dumps(agent),'audit mutated operational learner'
    d,b=branches['deliver'],branches['withhold']
    return dict(packet=packet.identity,receiver=int(agent.agent_id),time=agent.now,stage=agent.stages[-1].id,branches=branches,
        field_delta=float(np.linalg.norm(np.array(d['fields'])-b['fields'])),control_delta=float(np.linalg.norm(np.array(d['control'])-b['control'])),
        H_delta=d['H']-b['H'],seconds=time.perf_counter()-began,operational_state_unchanged=True,interpretation='isolated deliver-minus-withhold initial update; not a mission-benefit estimate')


def execute(spec,arm,comm=None,audit=True,learner_factory=CooperativeLearner,world_factory=CooperativeWorld,config=None,progress=None,audit_callback=None,update_callback=None,audit_selector=None,boundary_callback=None,delivery_callback=None):
    if arm not in ('disabled','reciprocal_fixed','one_way_01','one_way_10'):raise ValueError(arm)
    senders={'disabled':(),'reciprocal_fixed':(0,1),'one_way_01':(0,),'one_way_10':(1,)}[arm]
    if spec.get('dynamic'):raise ValueError('static demonstration only')
    cfg=config or Config(controller_mode='routed',shaping_mode='progress',steps=spec['steps'],replay_objective='mission',replay_domain='retained',replay_horizon_mode='supported_boundary',replay_update_cap=1024)
    starts=spec.get('start_seconds',[0.,5.]);assert len(starts)==2 and all(np.isfinite(t) and t>=0 for t in starts)
    comm=comm or CommunicationConfig(byte_budget=2000000)
    x=[np.array(z,dtype=float) for z in spec['initials']];worlds=[];agents=[]
    for i in range(2):
        w=world_factory(dict(spec,initial=spec['initials'][i],goal=spec['goals'][i],risk_weight=spec.get('risk_weights',(1.,1.))[i]),cfg);w.seed=spec['seed']*2+i;worlds.append(w)
        agents.append(learner_factory(cfg,x[i],spec['goals'][i],agent_id=str(i),communication=comm))
    active=[False,False];done=[False,False];pending=[None,None];hits=[False,False];success=[False,False]
    records=[[],[]];samples=[[],[]];updates=[[],[]];observations=[[],[]];parts=[dict(time=0.,risk=0.,effort=0.,failure=0.) for _ in range(2)]
    sent=[dict(payload=0,status=0,ack=0) for _ in range(2)];packets=[];events=[];audits=[];audited=set();queues=[[],[]]
    receive_slots=[{int(np.ceil((starts[1-i]+k*cfg.dt-starts[i])/cfg.dt-1e-9)) for k in range(0,cfg.steps,cfg.update_every) if starts[1-i]+k*cfg.dt>=starts[i]-1e-9} for i in range(2)]
    schedule=[(round(starts[i]*1e6),i,0) for i in range(2)];heapq.heapify(schedule);dt_us=round(cfg.dt*1e6);began=time.perf_counter()
    while schedule:
        stamp=schedule[0][0];now=stamp/1e6;batch=[]
        while schedule and schedule[0][0]==stamp:batch.append(heapq.heappop(schedule))
        # Complete held-input steps and sense every agent at this boundary before exchange.
        for _,i,tick in batch:
            a=agents[i];w=worlds[i];w.time=now
            if pending[i] is not None:
                p=pending[i];after,hit,cost,energy=w.step(x[i],np.array(p['u']))
                components=dict(time=cfg.dt,risk=cfg.dt*4*spec.get('risk_weights',(1.,1.))[i]*float(w.risk_truth(after[:2])),effort=cfg.dt*.03*float(np.dot(p['u'],p['u'])))
                for k,v in components.items():parts[i][k]+=v
                records[i].append(dict(p,after=after.tolist(),end_time=now,cost=cost,components=components,energy=energy,
                    clipping_work=float((np.array(p['navigation']['nominal_clipped'])-p['nominal'])@(after[:2]-x[i][:2])),
                    safety_work=float((np.array(p['u'])-p['navigation']['nominal_clipped'])@(after[:2]-x[i][:2]))))
                tr=Transition(tick-1,now,tuple(x[i]),tuple(p['u']),tuple(after),p['version']);x[i]=after
                obs=sense(w,x[i],tick,now);st=a.stages[-1]
                extended_energy=getattr(cfg,'stage_blend_mode','legacy')=='smooth' or getattr(a,'receiver_localization_mode','global')=='compact'
                beforeV=float(a.shaping_energy(x[i],st)) if extended_energy else float(potential(tensor(x[i][:2]),tensor(a.control_target(st)),st.weights,st.centers,cfg.rbf_width,a.policy,a.control_obstacles(st),cfg))
                a.observe(obs,tr);observations[i].append(asdict(obs))
                afterV=float(a.shaping_energy(x[i],st)) if extended_energy else float(potential(tensor(x[i][:2]),tensor(a.control_target(st)),st.weights,st.centers,cfg.rbf_width,a.policy,a.control_obstacles(st),cfg))
                records[i][-1]['observation_shaping_jump']=afterV-beforeV
                hits[i]=hit;success[i]=bool(np.linalg.norm(x[i][:2]-a.goal)<.25);done[i]=hit or success[i] or tick>=cfg.steps
                if not done[i]:a.advance(x[i])
            else:
                active[i]=True;obs=sense(w,x[i],tick,now);a.observe(obs);observations[i].append(asdict(obs))
                events.append(dict(kind='activation',time=now,agent=i))
            samples[i].append(dict(time=now,x=x[i].tolist(),stage=a.stages[-1].id,version=a.version,done=bool(done[i])))
        if boundary_callback is not None:
            boundary_callback(dict(time=now,batch=batch,agents=agents,states=x,worlds=worlds,
                trajectories=records,samples=samples,observations=observations,updates=updates))
        # Exports occur on each sender's unchanged local 8-step cadence. Receivers
        # accept queued packets only at their next own control boundary (<= .12 s).
        exports=[]
        if arm!='disabled' and now+1e-9>=spec.get('communication_start_seconds',0.):
            for _,i,tick in batch:
                j=1-i
                if i not in senders or done[i] or done[j] or not active[j] or tick%cfg.update_every:continue
                packet=agents[i].export(x[i],str(j))
                if packet is not None:exports.append((i,j,packet))
        for i,j,m in exports:
            query=len(json.dumps([z[:2] for z in m.anchors]).encode());status=advertise(agents[j],x[j],str(i),[z[:2] for z in m.anchors]);reply=status.bytes() if status else 32
            if sum(sent[i].values())+query>comm.byte_budget or sum(sent[j].values())+reply>comm.byte_budget:
                events.append(dict(kind='budget_skip',time=now,sender=i,receiver=j));continue
            sent[i]['status']+=query;sent[j]['status']+=reply
            if sum(sent[i].values())+m.bytes()>comm.byte_budget:events.append(dict(kind='budget_skip',time=now,sender=i,receiver=j));continue
            sent[i]['payload']+=m.bytes();packets.append(asdict(m));queues[j].append((stamp+comm.delay_ticks*dt_us,m))
            events.append(dict(kind='sent',time=now,sender=i,receiver=j,packet=m.identity,bytes=m.bytes(),status_time=agents[j].now))
        for _,i,tick in batch:
            a=agents[i]
            if done[i]:continue
            delivered=[];audit_ids=[]
            # A packet becomes deliverable at its release boundary (send time + declared delay).
            due=[m for release,m in queues[i] if release<=stamp];queues[i]=[(release,m) for release,m in queues[i] if release>stamp]
            for m in sorted(due,key=lambda m:(m.timestamp,m.identity)):
                if delivery_callback is not None:delivery_callback(a,x[i],m,worlds[i],tick)
                trial=copy.deepcopy(a);eligible=trial.assimilate(m)['reason']=='eligible'
                key=(i,a.stages[-1].id)
                selected = (eligible and key not in audited) if audit_selector is None else audit_selector(a,x[i],m,tick,eligible)
                if audit and selected:
                    audits.append(packet_effect(a,x[i],m) if audit_callback is None else audit_callback(a,x[i],m,worlds[i],tick));audited.add(key);audit_ids.append(len(audits)-1)
                before=x[i].copy();event=a.assimilate(m);assert np.array_equal(before,x[i])
                delivered.append(m.identity);events.append(dict(kind='delivered',time=now,sender=int(m.sender),receiver=i,packet=m.identity,
                    age=now-m.timestamp,reason=event['reason'],receiver_q=x[i][:2].tolist(),sender_q=list(m.q),anchors=m.anchors,version_before=a.version,
                    **({'components':event['components']} if 'components' in event else {})))
                ack=len(json.dumps(event,sort_keys=True).encode())
                if sum(sent[i].values())+ack<=comm.byte_budget:sent[i]['ack']+=ack
            if tick%cfg.update_every==0 or tick in receive_slots[i] or delivered:
                pre=snapshot(a);d=a.update(x[i]);post=snapshot(a)
                assert d['counterfactual_steps']<=1024
                updates[i].append(dict(time=now,tick=tick,packets=delivered,pre=pre,post=post,diagnostic=d,audit_ids=audit_ids))
                if update_callback is not None:updates[i][-1]['evaluation']=update_callback(a,worlds[i])
                for e in events:
                    if e['kind']=='delivered' and e['time']==now and e['receiver']==i:e.update(version_after=a.version,accepted=d['accepted'],update_reason=d['reason'],force_change=d['force_change'],shaping_jump=d.get('shaping_jump',0.))
            u,raw=a.action(x[i]);pending[i]=dict(time=now,tick=tick,x=x[i].tolist(),u=u.tolist(),nominal=raw.tolist(),version=a.version,stage=a.stages[-1].id,navigation=copy.deepcopy(a.last_control),control_model=a.control_model() if hasattr(a,"control_model") else None)
            if progress is not None and tick%80==0:progress(dict(agent=i,tick=tick,time=now,q=x[i][:2].tolist(),stage=a.stages[-1].id))
            heapq.heappush(schedule,(stamp+dt_us,i,tick+1))
    outcomes=[]
    for i in range(2):
        parts[i]['failure']=0. if success[i] else 40.
        assert abs(sum(t['cost'] for t in records[i])+parts[i]['failure']-sum(parts[i].values()))<1e-8
        assert sum(sent[i].values())<=comm.byte_budget
        outcomes.append(dict(agent=i,success=bool(success[i]),collision=bool(hits[i]),timeout=not success[i] and not hits[i],start_time=starts[i],finish_time=samples[i][-1]['time'],active_seconds=len(records[i])*cfg.dt,
            goal_gap=float(np.linalg.norm(x[i][:2]-agents[i].goal)),cost=sum(parts[i].values()),components=parts[i],bytes=sent[i],
            stage_traversal=len({s['stage'] for s in samples[i]})-1,interventions=sum(t['navigation']['reason']!='nominal' for t in records[i]),stagnations=sum(t['navigation'].get('stagnation',False) for t in records[i]),
            max_update_work=max(u['diagnostic']['counterfactual_steps'] for u in updates[i]),max_energy_residual=max(abs(t['energy']['residual']) for t in records[i]),
            context_jumps=agents[i].context_jumps,replay_work=agents[i].counterfactual_steps,identification_work=agents[i].identification_steps))
    return dict(**({'receiver_diagnostics':[a.diagnostic_summary() for a in agents]} if all(hasattr(a,'diagnostic_summary') for a in agents) else {}),
        spec=spec,config=asdict(cfg),communication=asdict(comm),arm=arm,agents=outcomes,trajectories=records,samples=samples,observations=observations,updates=updates,packets=packets,events=events,audits=audits,
        geometry_events=[getattr(a,"geometry_events",[]) for a in agents],learner_events=[getattr(a,"terrain_events",[]) for a in agents],receive_update_slots=[sorted(s) for s in receive_slots],pending_packets=sum(map(len,queues)),wall_seconds=time.perf_counter()-began,audit_seconds=sum(a['seconds'] for a in audits),audit_replay_work=sum(b['replay_work'] for a in audits for b in a['branches'].values()),
        semantics='inactive before exact start; each agent has full local horizon; exports each .96s local; delivery next receiver control boundary; matched exogenous receive-update opportunities in both arms; no pre-start buffering or inter-agent contact dynamics')
