"""Shared-origin, exactly staggered starts with an explicit asynchronous message timeline.
Existing controller/learner implementations and prior evaluation runners are unchanged.
"""
import argparse,copy,hashlib,heapq,json,pickle,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from .records import Config,Transition
from .cooperative import CooperativeLearner,CommunicationConfig,advertise
from .cooperative_evaluation import CooperativeWorld
from .complex_environments import scene,world_axes,COLORS
from .diagnostics import snapshot
from .model import tensor,potential
from .evaluation import dump,load
from .cli import source_hashes,archive_sources


def specification(name,seed):
    s=scene('long_obstacle_corridor',seed)
    if name=='slalom_passage':
        s.update(steps=420,bounds=[-1.5,13.5,-3.,3.],obstacles=[[2.7,-.65,.5],[5.,-1.1,.6],[5.,1.1,.6],[7.6,0.,.6],[10.,.8,.5]],
            patches=[[2.5,.8,.8,.55,1.4],[5.,0.,.7,.65,1.8],[8.,-1.2,.8,.55,1.5],[10.5,-.7,.7,.6,1.3]])
        goal=[12.,0.]
    elif name=='long_corridor':goal=[20.,0.]
    else:raise ValueError(name)
    s.update(name='staggered_'+name,initials=[[0.,0.,0.,0.]]*2,goals=[goal,goal],initial=[0.,0.,0.,0.],goal=goal,start_seconds=[0.,5.],
        experiment='development demonstration; not held-out validation',interaction='independent agents; no inter-agent collision dynamics',waiting='inactive: no sensing, learning or communication before activation')
    return s


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


def job(args):
    spec,arm,root=args;torch.set_num_threads(1);path=Path(root)/f"{spec['name']}_{spec['seed']}_{arm}.json.gz"
    try:
        r=execute(spec,arm);dump(path,r);return dict(scene=spec['name'],seed=spec['seed'],arm=arm,path=path.name,complete=True,goals=sum(a['success'] for a in r['agents']))
    except Exception as exc:
        e=dict(scene=spec['name'],seed=spec['seed'],arm=arm,path=path.name,complete=False,error=str(exc),traceback=traceback.format_exc());dump(path,e);return e


def frame(results,now):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,FancyArrowPatch
    from PIL import Image
    fig,axes=plt.subplots(2,2,figsize=(14,8),gridspec_kw={'height_ratios':[1.4,1]});r=results['reciprocal_fixed'];spec=r['spec']
    for ax,arm in zip(axes[0],('disabled','reciprocal_fixed')):
        rr=results[arm];world_axes(ax,spec)
        for i,color in enumerate(COLORS):
            ss=[s for s in rr['samples'][i] if s['time']<=now+1e-9]
            if not ss:
                q=spec['initials'][i][:2];ax.scatter(*q,color='gray',s=40);ax.annotate(f'A{i} waiting until {spec["start_seconds"][i]:.2f}s',q,xytext=(3,15),textcoords='offset points',fontsize=8);continue
            p=np.array([s['x'] for s in ss]);ax.plot(p[:,0],p[:,1],color=color,lw=2);q=p[-1,:2];ax.scatter(*q,color=color,s=35)
            status='GOAL' if ss[-1]['done'] and rr['agents'][i]['success'] else 'FAILED' if ss[-1]['done'] else 'moving'
            ax.annotate(f'A{i} {status}',q,xytext=(3,6+i*9),textcoords='offset points',fontsize=8,color=color)
            controls=[t for t in rr['trajectories'][i] if t['time']<=now]
            if controls:
                n=controls[-1]['navigation']
                if n.get('route'):
                    path=np.array(n['route']);ax.plot(path[:,0],path[:,1],':',color=color,alpha=.7);ax.scatter(*n['waypoint'],marker='x',color=color,s=15)
            us=[u for u in rr['updates'][i] if u['time']<=now]
            if us:
                for ox,oy,rad in us[-1]['post']['geometry']:ax.add_patch(Circle((ox,oy),rad+.17,fill=False,color=color,alpha=.3))
        if arm=='reciprocal_fixed':
            recent=[e for e in r['events'] if e['kind']=='delivered' and 0<=now-e['time']<1.5][-4:]
            for k,e in enumerate(recent):
                accepted=e['reason']=='eligible' and e.get('accepted');color='purple' if accepted else 'red'
                ax.add_patch(FancyArrowPatch(e['sender_q'],e['receiver_q'],arrowstyle='-|>',mutation_scale=15,color=color,connectionstyle='arc3,rad=.2',lw=1.8,alpha=.8))
                z=np.array(e['anchors']);ax.scatter(z[:,0],z[:,1],marker='+',color=color,s=35)
                label='accepted' if accepted else e['reason'] if e['reason']!='eligible' else 'update rejected'
                ax.text(.02,.98-k*.08,f'{e["time"]:.2f}s A{e["sender"]} → A{e["receiver"]}: {label}',transform=ax.transAxes,va='top',fontsize=8,color=color)
        ax.set_title('Communication OFF' if arm=='disabled' else 'Reciprocal fixed-rate: delivery arrows + field anchors')
    ax=axes[1,0]
    for e in r['events']:
        if e['time']>now:continue
        if e['kind']=='sent':ax.scatter(e['time'],int(e['sender'])+.14,s=10,color='purple',marker='.')
        if e['kind']=='delivered':
            color='green' if e['reason']=='eligible' and e.get('accepted') else 'orange' if e['reason']=='eligible' else 'red'
            ax.scatter(e['time'],int(e['sender']),s=18,color=color,marker='|')
    end=max(a['finish_time'] for rr in results.values() for a in rr['agents'])
    ax.axvline(5,color='gray',ls='--');ax.axvline(now,color='black');ax.set(xlim=(0,end),ylim=(-.35,1.5),yticks=[0,1],yticklabels=['A0 → A1','A1 → A0'],xlabel='Global time (s)',title='Purple: sent | Green: accepted | Orange/red: rejected')
    ax=axes[1,1];ax.axis('off');lines=['Packet-only effect: identical receiver history,','deliver + update versus withhold + update.']
    for receiver in (0,1):
        audits=[a for a in r['audits'] if a['receiver']==receiver and a['time']<=now]
        if audits:
            a=audits[-1];lines += ['',f'A{receiver}, audit at {a["time"]:.2f}s (stage {a["stage"]})',f'  field coefficients Δ = {a["field_delta"]:.3e}',f'  applied control Δ = {a["control_delta"]:.3e} N',f'  shaping H Δ = {a["H_delta"]:+.3e} J',f'  deliver accepted: {a["branches"]["deliver"]["accepted"]}']
        else:lines+=['',f'A{receiver}: no eligible packet audit yet.']
    lines+=['','Small effects are shown numerically, not magnified.','No mission benefit is inferred from these local audits.']
    ax.text(0,1,'\n'.join(lines),va='top',fontsize=9,family='monospace')
    fig.suptitle(f'{spec["name"]} | seed {spec["seed"]} | global time {now:.2f}s\nSame origin and goal. A0 starts 0.00s; A1 starts 5.00s. Logged boundary states; independent plants.',fontsize=11)
    fig.tight_layout();fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);return im


def report(root):
    complete=load(root/'completion.json');manifest=load(root/'manifest.json');rows=[];pairs={};failures=[]
    (root/'figures').mkdir(exist_ok=True);gallery=root/'figures/failures';gallery.mkdir(exist_ok=True)
    for c in complete:
        if not c['complete']:failures.append(c);continue
        r=load(root/'episodes'/c['path']);deliveries=[e for e in r['events'] if e['kind']=='delivered']
        row=dict(scene=c['scene'],seed=c['seed'],arm=c['arm'],agents=r['agents'],goals=sum(a['success'] for a in r['agents']),collisions=sum(a['collision'] for a in r['agents']),timeouts=sum(a['timeout'] for a in r['agents']),
            deliveries=len(deliveries),eligible=sum(e['reason']=='eligible' for e in deliveries),accepted=sum(e['reason']=='eligible' and e.get('accepted',False) for e in deliveries),
            reasons={reason:sum(e['reason']==reason for e in deliveries) for reason in sorted({e['reason'] for e in deliveries})},audits=len(r['audits']),nonzero_control_audits=sum(a['control_delta']>1e-10 for a in r['audits']),
            bytes=sum(sum(a['bytes'].values()) for a in r['agents']),wall_seconds=r['wall_seconds'],audit_seconds=r['audit_seconds'],audit_replay_work=r['audit_replay_work'])
        rows.append(row)
        if c['goals']<2:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(11,3));world_axes(ax,r['spec'])
            for i,ss in enumerate(r['samples']):
                p=np.array([s['x'] for s in ss]);ax.plot(p[:,0],p[:,1],color=COLORS[i]);ax.annotate('goal' if r['agents'][i]['success'] else 'FAILED',p[-1,:2],fontsize=8)
            filename=c['path'].replace('.json.gz','.png');ax.set_title(c['path']);fig.tight_layout();fig.savefig(gallery/filename,dpi=110);plt.close(fig)
            failures.append(dict(c,outcomes=r['agents'],figure=filename))
        if c['seed']==manifest['animation_seed']:pairs.setdefault(c['scene'],{})[c['arm']]=r
    dump(root/'rows.json',rows);dump(root/'failures.json',failures)
    (gallery/'index.html').write_text('<h1>All failures and incomplete episodes</h1>'+('None.' if not failures else ''.join('<p>'+c['path']+(f'<br><img width=1000 src="{c["figure"]}">' if 'figure' in c else c.get('error',''))+'</p>' for c in failures)))
    lines=['# Staggered-start cooperative navigation','', 'Development demonstration, separate from the frozen validation experiment. Agents share their start and goal. A0 activates at 0.00 s; A1 activates at exactly 5.00 s. Before activation A1 neither senses, learns nor communicates. Each agent retains the 0.12 s step and its full active horizon. Both arms use matched, predeclared local/receive update opportunities.', '', '| Scene | Seed | Communication | Goals | Collisions | Timeouts | Delivered | Eligible | Accepted update | Nonzero control audits / audits |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:lines.append(f'| {r["scene"]} | {r["seed"]} | {r["arm"]} | {r["goals"]}/2 | {r["collisions"]} | {r["timeouts"]} | {r["deliveries"]} | {r["eligible"]} | {r["accepted"]} | {r["nonzero_control_audits"]}/{r["audits"]} |')
    comparisons=[]
    for name,rs in pairs.items():
        if len(rs)!=2:continue
        end=max(a['finish_time'] for r in rs.values() for a in r['agents']);times=sorted(set([0.,4.92,5.,5.12]+np.linspace(0,end,64).tolist()))
        frames=[frame(rs,t) for t in times];frames[0].save(root/'figures'/f'{name}.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
        frames[-1].save(root/'figures'/f'{name}_final.png');dump(root/'figures'/f'{name}_frames.json',dict(times=times,playback='illustrative time compression; use global-time label and event timeline',animation_seed=manifest['animation_seed']))
        for i in range(2):
            a,b=rs['disabled']['samples'][i],rs['reciprocal_fixed']['samples'][i];n=min(len(a),len(b));dist=np.linalg.norm(np.array([s['x'][:2] for s in a[:n]])-np.array([s['x'][:2] for s in b[:n]]),axis=1)
            comparisons.append(dict(scene=name,seed=manifest['animation_seed'],agent=i,common_boundary_samples=n,max_paired_position_difference=float(dist.max()),finish_time_difference=rs['reciprocal_fixed']['agents'][i]['finish_time']-rs['disabled']['agents'][i]['finish_time']))
        lines+=['',f'![{name}](figures/{name}.gif)']
    dump(root/'paired_visualization_cases.json',comparisons)
    lines+=['','Arrows show delivery (not instantaneous influence). Rejected or unsupported packets are shown explicitly. Each accepted delivery also records its model version and actual update force/shaping revision. The packet-only panel uses isolated deliver/withhold clones at the first eligible packet per receiver stage, with an update in both branches; it separates packet effects from ordinary local fitting. Audits never mutate operational agents and their computation is separate.', '', 'The five-second head start can make the leader’s current anchors fall outside the follower’s retained support; no support is expanded to force acceptance. Fixed-rate sender opportunities are 0.96 s apart on each local clock. Delivery waits until the next receiver control boundary (at most 0.12 s). No pre-start packet buffering, adaptive rates, or inter-agent collision avoidance is introduced. All failures and incomplete runs remain in failures.json and full episode records. Small trajectory differences are not hidden or exaggerated.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--render-only',action='store_true');args=ap.parse_args(argv)
    if not args.out.exists():
        args.out.mkdir(parents=True);specs=[specification(n,s) for n in ('slalom_passage','long_corridor') for s in (73000,73001)]
        old=load(Path('results/routed_validation_v1/manifest.json'))['source_verification']['frozen_source_hashes'];current=source_hashes()
        assert all(current.get(n)==h for n,h in old.items()),'existing implementation changed'
        dump(args.out/'manifest.json',dict(preserved_original_sources=old,specs=specs,arms=['disabled','reciprocal_fixed'],animation_seed=73000,source_hashes=source_hashes(),development_only=True));archive_sources(args.out)
    elif not args.render_only:ap.error('fresh output directory required')
    if not args.render_only:
        (args.out/'episodes').mkdir();completed=[];m=load(args.out/'manifest.json')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(job,(s,a,str(args.out/'episodes'))) for s in m['specs'] for a in m['arms']]
            for f in as_completed(jobs):
                c=f.result();completed.append(c);dump(args.out/'completion.json',completed);print(c,flush=True)
    report(args.out)

if __name__=='__main__':main()
