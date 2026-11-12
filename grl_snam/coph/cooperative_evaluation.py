"""Two-agent evaluation and truth-only causal audits, outside learner interfaces."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .records import Config,Transition
from .cooperative import CooperativeLearner,CommunicationConfig,advertise,information_gain,rate_probability,field_vector,bootstrap_field
from .scheduling import value_features,paired_model_label,fit_predictor,predict
from .scenarios import ScenarioWorld
from .diagnostics import snapshot,probe_metrics
from .evaluation import dump,load
from .cli import source_hashes,archive_sources
from .model import flow,tensor,potential

SCENES=('complementary','early_warning','reverse_warning','redundant','irrelevant','changing_hazard','channel_stress')
ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed','information_gain','recipient_value',
      'evidence_only','no_history','no_reconciliation')


def specification(name,seed,profile):
    if name not in SCENES:raise ValueError(name)
    rng=np.random.default_rng(seed);jitter=float(rng.uniform(-.12,.12))
    initial=[[0.,-.4,0.,0.],[0.,.4,0.,0.]];goals=[[6.,-.4],[6.,.4]]
    patches=[[2.4,.3+jitter,.7,.5,.9],[4.2,-.4+jitter,.6,.5,.7]]
    if name=='early_warning':initial[0][0]=1.
    if name=='reverse_warning':initial[1][0]=1.
    if name=='redundant':patches=[[2.4,jitter,.7,.7,.9]]
    if name=='irrelevant':initial[1][1]=3.;goals[1][1]=3.
    return dict(name=name,seed=seed,initials=initial,goals=goals,patches=patches,
        initial=initial[0],goal=goals[0],damping=float(rng.uniform(.7,1.05)),obstacles=[],
        bounds=[-1.5,7.5,-2.,4.5 if name=='irrelevant' else 2.],
        steps=96 if profile=='smoke' else 180,dynamic=name=='changing_hazard',
        channel='stress' if name=='channel_stress' else 'normal',
        dynamic_model='smooth sinusoidal amplitudes, period 8 seconds',
        interaction='independent plants sharing information; no inter-agent contact dynamics')


class CooperativeWorld(ScenarioWorld):
    def __init__(self,spec,cfg):super().__init__(spec,cfg);self.time=0.
    def _factor(self,k):return (1+.7*np.sin(2*np.pi*self.time/8+k*np.pi)) if self.spec.get('dynamic') else 1.
    def risk_truth(self,q):
        q=np.asarray(q);value=np.full(q.shape[:-1],.02)
        for k,(x,y,sx,sy,amp) in enumerate(self.spec['patches']):
            z=(q-[x,y])/[sx,sy];value+=self._factor(k)*amp*np.exp(-.5*np.sum(z*z,axis=-1))
        return value
    def risk_gradient(self,q):
        q=np.asarray(q);g=np.zeros_like(q,dtype=float)
        for k,(x,y,sx,sy,amp) in enumerate(self.spec['patches']):
            delta=q-[x,y];scale=np.array([sx,sy]);b=self._factor(k)*amp*np.exp(-.5*np.sum((delta/scale)**2,axis=-1))
            g-=b[...,None]*delta/scale**2
        return g


def audit_receiver(agent,state,world,packet,tick,horizon=16):
    """Full receiver snapshots; future communications withheld equally in both branches.

Independent plants mean receiver cost difference equals team difference for this
    intervention. Audit branches are never returned to operational learners.
    """
    start=time.perf_counter();results={};work=0;replay_work=0;identification_work=0
    for deliver in (False,True):
        a=copy.deepcopy(agent);w=copy.deepcopy(world);x=state.copy();cost=0.;path=[x.tolist()]
        event=a.assimilate(packet) if deliver else None
        report=a.update(x);before=snapshot(a);collision=False;success=False
        for n in range(min(horizon,agent.cfg.steps-tick)):
            w.time=(tick+n+1)*a.cfg.dt
            u,_=a.action(x);y,collision,c,_=w.step(x,u);work+=1;cost+=c
            a.observe(w.sense(y,tick+n+1),Transition(tick+n,w.time,tuple(x),tuple(u),tuple(y),a.version))
            x=y;path.append(x.tolist());success=np.linalg.norm(x[:2]-a.goal)<.25
            if collision or success:break
            a.advance(x)
            if (n+1)%a.cfg.update_every==0:a.update(x)
        if collision or (tick+len(path)-1>=agent.cfg.steps and not success):cost+=40.
        replay_work+=a.counterfactual_steps-agent.counterfactual_steps
        identification_work+=a.identification_steps-agent.identification_steps
        results['deliver' if deliver else 'withhold']=dict(cost=cost,path=path,update=report,
            packet_event=event,model=before,success=bool(success),collision=collision)
    return dict(start=state.tolist(),horizon=min(horizon,agent.cfg.steps-tick),tick=tick,
        packet=packet.identity,results=results,realized_improvement=results['withhold']['cost']-results['deliver']['cost'],
        valid=results['deliver']['packet_event']['reason']=='eligible',steps=work,replay_steps=replay_work,identification_work=identification_work,seconds=time.perf_counter()-start,
        future_communication='withheld in both; receiver-only causal intervention, independent plants')


def execute(spec,arm,comm=None,predictor=None,collect_training=False,audits=True,diagnostic_callback=None,replay_domain="single",feedback_mode="none",horizon_mode="fixed",replay_cap=None,feedback_audits=False,controller_mode='legacy',shaping_mode='legacy'):
    if controller_mode!='legacy' and spec['dynamic']:raise ValueError('new controller supports static scenes only')
    if feedback_mode not in ('none','receiver','compact'):raise ValueError('invalid feedback_mode')
    if feedback_mode!='none' and (spec['dynamic'] or spec['channel']!='normal' or arm!='reciprocal_fixed'):
        raise ValueError('feedback requires static reciprocal fixed-rate scenes')
    cfg=Config(controller_mode=controller_mode,shaping_mode=shaping_mode,steps=spec['steps'],replay_objective='mission',replay_domain=replay_domain,
               replay_horizon_mode=horizon_mode,replay_update_cap=replay_cap,previous=0 if arm=='no_history' else 2)
    comm=comm or CommunicationConfig();predictor=predictor or {'ready':False}
    worlds=[];agents=[];states=[np.array(x) for x in spec['initials']]
    for i in range(2):
        ws=dict(spec,initial=spec['initials'][i],goal=spec['goals'][i])
        world=CooperativeWorld(ws,cfg);world.seed=spec['seed']*2+i;worlds.append(world)
        agent=CooperativeLearner(cfg,states[i],spec['goals'][i],mode='evidence' if arm=='evidence_only' else 'full',
            reconcile=arm!='no_reconciliation',agent_id=str(i),communication=comm,dynamic=spec['dynamic'])
        agent.observe(world.sense(states[i],0));agents.append(agent)
    from .receiver_summary import SummaryCache,summarize_receiver
    summary_cache=SummaryCache()
    summary_bytes=[0,0];summary_seconds=0.;prediction_seconds=0.
    feedback_records=[];feedback_seen=set();feedback_bytes=[0,0];feedback_summary_bytes=[0,0];feedback_cache=SummaryCache()
    done=[False,False];success=[False,False];collision=[False,False]
    costs=[dict(time=0.,risk=0.,effort=0.,failure=0.) for _ in agents]
    observations=[[],[]];records=[[],[]];updates=[[],[]];queue=[];packets=[];decisions=[];audit_rows=[];training=[];audited=set()
    sent=[0,0];status_bytes=[0,0];ack_bytes=[0,0];schedule_seconds=0.;peak=[0,0]
    points=np.array([(x,y) for x in np.arange(-.5,6.6,.5) for y in np.linspace(-1.5,3.5 if spec['name']=='irrelevant' else 1.5,7)])
    started=time.perf_counter()
    for tick in range(cfg.steps):
        # Both agents already sensed this time before any packet can be delivered.
        if tick%cfg.update_every==0:
            if arm!='disabled':
                scheduling_start=time.perf_counter()
                exports=[a.export(states[i],str(1-i)) if not done[i] else None for i,a in enumerate(agents)]
                for i,m in enumerate(exports):
                    j=1-i
                    if m is None or done[j] or (arm=='one_way_01' and i!=0) or (arm=='one_way_10' and i!=1):continue
                    # Restrict the common probe vector using the peer's advertised
                    # bounds, never its private model or hidden scene geometry.
                    peer=exports[j]
                    if peer is not None:
                        ids=[k for k,a in enumerate(m.anchors) if peer.bounds[0]<=a[0]<=peer.bounds[1] and peer.bounds[2]<=a[1]<=peer.bounds[3]]
                        if ids:
                            idx=[3*k+d for k in ids for d in range(3)]
                            m=replace(m,anchors=tuple(m.anchors[k] for k in ids),covariance=tuple(map(tuple,np.array(m.covariance)[np.ix_(idx,idx)])))
                    # A query and reply expose only declared status, with both byte charges.
                    query_size=len(json.dumps([a[:2] for a in m.anchors]).encode())
                    status=advertise(agents[j],states[j],str(i),[a[:2] for a in m.anchors])
                    reply_size=status.bytes() if status else 32
                    if sent[i]+query_size>comm.byte_budget or sent[j]+reply_size>comm.byte_budget:continue
                    sent[i]+=query_size;status_bytes[i]+=query_size;sent[j]+=reply_size;status_bytes[j]+=reply_size
                    ig,count=information_gain(m,status);f=value_features(m,status,cfg.dt*cfg.update_every) if status else None
                    value=predict(predictor,f) if f is not None else None
                    receiver_summary=None;summary_reason='disabled';charged_summary=0
                    if comm.prediction_variant not in ('legacy','baseline'):
                        summary_start=time.perf_counter()
                        offered=summarize_receiver(agents[j],states[j],str(i))
                        receiver_summary,charged_summary,summary_reason=summary_cache.transfer(offered,str(i),tick*cfg.dt,comm.byte_budget-sent[j])
                        sent[j]+=charged_summary;status_bytes[j]+=charged_summary;summary_bytes[j]+=charged_summary
                        summary_seconds+=time.perf_counter()-summary_start
                    new_prediction=None
                    if comm.prediction_variant!='legacy':
                        from .model_prediction import variant_prediction
                        prediction_start=time.perf_counter()
                        new_prediction=variant_prediction(m,status,receiver_summary,cfg,comm,comm.prediction_variant,
                            seed=spec['seed']+tick,horizons=(16,))
                        prediction_seconds+=time.perf_counter()-prediction_start
                        value=new_prediction['rows'][0]['benefit']

                    probability=1.;rate=None;reason='fixed'
                    if arm in ('information_gain','recipient_value'):
                        utility=ig if arm=='information_gain' else value
                        if utility is not None and arm=='recipient_value' and comm.prediction_variant=='legacy':utility-=predictor.get('residual_rmse',0.)
                        if utility is None:probability=0.;reason='missing_value_or_status'
                        else:probability,rate=rate_probability(utility,m.bytes(),cfg.dt*cfg.update_every,comm,comm.byte_budget-sent[i]);reason='rate_selection'
                    chosen=np.random.default_rng(np.random.SeedSequence([spec['seed'],tick,i,835])).random()<probability
                    if sent[i]+m.bytes()>comm.byte_budget:chosen=False;reason='byte_budget'
                    decision=dict(tick=tick,sender=i,packet=m.identity,information_gain=ig,common_anchors=count,
                        predicted_benefit=value,probability=probability,rate=rate,send=bool(chosen),reason=reason,
                        summary_reason=summary_reason,summary_bytes=charged_summary,
                        receiver_summary=asdict(receiver_summary) if receiver_summary else None,
                        model_prediction=new_prediction,
                        status=asdict(status) if status else None,features=f.tolist() if f is not None else None)
                    decisions.append(decision)
                    if collect_training and status and len(training)<12:
                        label=(dict(new_prediction['rows'][0],steps=sum(b['steps'] for b in new_prediction['branches'].values()))
                               if new_prediction is not None else paired_model_label(m,status,cfg,comm,seed=spec['seed']+tick+i))
                        training.append(dict(features=f.tolist(),label=label,packet=m.identity))
                    if not chosen:continue
                    sent[i]+=m.bytes();packets.append(asdict(m))
                    delay=comm.delay_ticks
                    if spec['channel']=='stress':delay=[0,8,24,4][(tick//cfg.update_every)%4]
                    queue.append((tick+delay,i,m))
                    if spec['channel']=='stress':
                        # Duplicates and returned packets consume real transmission budget.
                        if sent[i]+m.bytes()<=comm.byte_budget:
                            sent[i]+=m.bytes();queue.append((tick+delay+1,i,m))
                        returned=replace(m,recipient=str(i),identity=m.identity+':return')
                        if sent[j]+returned.bytes()<=comm.byte_budget:
                            sent[j]+=returned.bytes();queue.append((tick+delay+2,j,returned))
                schedule_seconds+=time.perf_counter()-scheduling_start
        due=[item for item in queue if item[0]<=tick];queue=[item for item in queue if item[0]>tick]
        touched=set();pending_feedback={}
        # Sort by delivery time and identity; never by mutable learner iteration order.
        for delivery,i,m in sorted(due,key=lambda item:(item[0],item[2].identity,item[1])):
            j=int(m.recipient)
            if done[j]:continue
            a=agents[j];key=(j,a.stages[-1].id)
            if audits and key not in audited:
                trial=copy.deepcopy(a);event=trial.assimilate(m)
                if event['reason']=='eligible':
                    if diagnostic_callback is not None:
                        diagnostic_callback(copy.deepcopy(a),states[j].copy(),copy.deepcopy(worlds[j]),m,tick)
                    ar=audit_receiver(a,states[j],worlds[j],m,tick);ar['recipient']=j
                    status=advertise(copy.deepcopy(a),states[j],str(i),[z[:2] for z in m.anchors])
                    model_audit_start=time.perf_counter()
                    predicted=paired_model_label(m,status,cfg,comm,horizon=ar['horizon'],seed=spec['seed']+tick) if status else dict(valid=False,reason='missing_status',steps=0)
                    ar['model_pair']=predicted;ar['model_seconds']=time.perf_counter()-model_audit_start;ar['predicted_improvement']=predicted.get('benefit');audit_rows.append(ar);audited.add(key)
            feedback_record=None
            if feedback_mode!='none' and key not in feedback_seen:
                trial=copy.deepcopy(a);eligibility=trial.assimilate(m)
                if eligibility['reason']=='eligible':
                    from .packet_feedback import evaluate_compact
                    feedback_seen.add(key);feedback_start=time.perf_counter();seed=spec['seed']+tick*17+j
                    summary_charge=0;assessment=None;summary_reason='not_needed'
                    if feedback_mode=='receiver':assessment=a.evaluate_packet(m,seed=seed)
                    else:
                        offered=summarize_receiver(a,states[j],str(i))
                        received,summary_charge,summary_reason=feedback_cache.transfer(offered,str(i),a.now,comm.byte_budget-sent[j])
                        sent[j]+=summary_charge;status_bytes[j]+=summary_charge;feedback_summary_bytes[j]+=summary_charge
                        if received is not None:
                            status=advertise(copy.deepcopy(a),states[j],str(i),[z[:2] for z in m.anchors])
                            # This is a real reply, not free sender access to current receiver status.
                            extra_status=status.bytes() if status else 32
                            if sent[j]+extra_status<=comm.byte_budget:
                                sent[j]+=extra_status;status_bytes[j]+=extra_status
                                summary_charge+=extra_status
                                assessment=evaluate_compact(received,m,status,seed)
                            else:summary_reason='status_budget'
                    feedback_record=dict(packet=m.identity,receiver=j,tick=tick,stage=a.stages[-1].id,mode=feedback_mode,
                        version_before=a.version,state=states[j].tolist(),assessment=assessment,
                        summary_bytes=summary_charge,summary_reason=summary_reason,seconds=time.perf_counter()-feedback_start,
                        feedback_sent=False,feedback_bytes=0)
                    if feedback_audits:
                        from .continuations import continuation
                        feedback_record['truth_audit']=continuation(a,states[j],worlds[j],m,tick,(8,16,32))
                    feedback_records.append(feedback_record);pending_feedback[j]=feedback_record
            a.assimilate(m);touched.add(j)
            ack_size=len(json.dumps(dict(packet=m.identity,version=a.version,reason=a.packet_events[-1]['reason'])).encode())
            if feedback_record is not None and feedback_mode=='receiver':
                size=feedback_record['assessment']['wire_bytes']
                if sent[j]+size<=comm.byte_budget:
                    sent[j]+=size;ack_bytes[j]+=size;feedback_bytes[j]+=size
                    feedback_record.update(feedback_sent=True,feedback_bytes=size,feedback_reason='sent_in_ack');ack_size=0
                else:feedback_record['feedback_reason']='byte_budget'
            elif feedback_record is not None:feedback_record['feedback_reason']='sender_local_estimate'
            if sent[j]+ack_size<=comm.byte_budget:sent[j]+=ack_size;ack_bytes[j]+=ack_size
        scheduled=tick%cfg.update_every==0
        if spec['channel']=='stress':
            # Exogenous update opportunities are matched even when an arm sends nothing.
            scheduled |= any(tick == origin+[0,8,24,4][(origin//cfg.update_every)%4]+offset
                             for origin in range(0,tick+1,cfg.update_every) for offset in (0,1,2))
        for j in range(2):
            if done[j] or not (scheduled or j in touched):continue
            a=agents[j];pre=snapshot(a);r=a.update(states[j]);updates[j].append(dict(tick=tick,phase='message' if j in touched else 'local',pre=pre,
                post=snapshot(a),diagnostic=r,proposals=[w.tolist() for w in a.reference] if a.reference is not None else None,replay_calls=copy.deepcopy(a.replay_calls),probes=probe_metrics(a,worlds[j],points)))
            if j in pending_feedback:
                pending_feedback[j].update(committed_version=a.version,actual_accepted=r['accepted'],actual_reason=r['reason'])
        for i,a in enumerate(agents):
            if done[i]:continue
            x=states[i];u,proposed=a.action(x);worlds[i].time=(tick+1)*cfg.dt
            after,hit,cost,energy=worlds[i].step(x,u)
            parts=dict(time=cfg.dt,risk=cfg.dt*4*float(worlds[i].risk_truth(after[:2])),effort=cfg.dt*.03*float(u@u))
            for k,v in parts.items():costs[i][k]+=v
            records[i].append(dict(tick=tick,x=x.tolist(),after=after.tolist(),u=u.tolist(),proposed=proposed.tolist(),
                stage=a.stages[-1].id,version=a.version,components=parts,cost=cost,energy=energy,
                clipping_work=float((np.asarray(a.last_control.get('nominal_clipped',u))-proposed)@(after[:2]-x[:2])),
                navigation=copy.deepcopy(a.last_control),
                safety_work=float((u-np.asarray(a.last_control.get('nominal_clipped',u)))@(after[:2]-x[:2])),
                route_shaping_jump=a.last_control.get('route_shaping_jump',0.)))
            obs=worlds[i].sense(after,tick+1);observations[i].append(asdict(obs))
            s=a.stages[-1]
            old_shaping=float(potential(tensor(after[:2]),tensor(a.control_target(s)),s.weights,s.centers,cfg.rbf_width,a.policy,a.control_obstacles(s),cfg))
            a.observe(obs,Transition(tick,(tick+1)*cfg.dt,tuple(x),tuple(u),tuple(after),a.version))
            new_shaping=float(potential(tensor(after[:2]),tensor(a.control_target(s)),s.weights,s.centers,cfg.rbf_width,a.policy,a.control_obstacles(s),cfg))
            records[i][-1]['observation_shaping_jump']=new_shaping-old_shaping
            records[i][-1]['transition_mse']=float(np.mean((flow(tensor(x),tensor(u),a.damping,cfg.dt).numpy()-after)**2))
            states[i]=after;success[i]=bool(np.linalg.norm(after[:2]-a.goal)<.25);collision[i]=hit;done[i]=hit or success[i]
            peak[i]=max(peak[i],sum(len(s.readings) for s in a.stages))
            if not done[i]:a.advance(after)
        if all(done):break
    metrics=[]
    for i,a in enumerate(agents):
        costs[i]['failure']=0. if success[i] else 40.
        metric=probe_metrics(a,worlds[i],points)
        trajectory=records[i]
        metrics.append(dict(agent=i,success=success[i],collision=collision[i],timeout=not success[i] and not collision[i],
            cost=sum(costs[i].values()),components=costs[i],goal_gap=float(np.linalg.norm(states[i][:2]-a.goal)),
            transition_mse=float(np.mean([r['transition_mse'] for r in records[i]])),
            near_stationary_seconds=cfg.dt*sum(np.linalg.norm(r['x'][2:])<.05 and np.linalg.norm(np.array(r['x'][:2])-a.goal)>.5 for r in records[i]),
            path_length=sum(float(np.linalg.norm(np.array(r['after'][:2])-r['x'][:2])) for r in trajectory),
            field=metric,damping_error=abs(a.damping-worlds[i]._damping),bytes=sent[i],status_bytes=status_bytes[i],ack_bytes=ack_bytes[i],
            payload_bytes=sent[i]-status_bytes[i]-ack_bytes[i],replay_steps=a.counterfactual_steps,
            identification_work=a.identification_steps,export_work=a.export_work,message_fit_work=a.message_fit_work,
            peak_samples=peak[i],retained_stages=len(a.stages),
            max_energy_residual=max(abs(r['energy']['residual']) for r in trajectory),
            clipping_work=sum(r['clipping_work'] for r in trajectory),
            shaping_jumps=sum(u['diagnostic'].get('shaping_jump',0) for u in updates[i]),
            context_jumps=a.context_jumps,observation_shaping_jumps=sum(r['observation_shaping_jump'] for r in records[i]),update_seconds=sum(u['diagnostic'].get('wall_seconds',0) for u in updates[i])))
    return dict(schema=1,spec=spec,config=asdict(cfg),communication=asdict(comm),arm=arm,
        feedback_mode=feedback_mode,feedback_records=feedback_records,
        summary=dict(scene=spec['name'],seed=spec['seed'],arm=arm,team_cost=sum(m['cost'] for m in metrics),
            summary_bytes=sum(summary_bytes),summary_seconds=summary_seconds,prediction_seconds=prediction_seconds,
            feedback_bytes=sum(feedback_bytes),feedback_summary_bytes=sum(feedback_summary_bytes),
            feedback_seconds=sum(f['seconds'] for f in feedback_records),
            feedback_audit_seconds=sum(f.get('truth_audit',{}).get('seconds',0) for f in feedback_records),
            team_success=all(success),failures=sum(not s for s in success),bytes=sum(sent),
            replay_steps=sum(a.counterfactual_steps for a in agents),scheduler_seconds=schedule_seconds,
            audit_steps=sum(a['steps']+a['model_pair']['steps'] for a in audit_rows),audit_seconds=sum(a['seconds']+a['model_seconds'] for a in audit_rows),
            audit_replay_steps=sum(a['replay_steps']+a['model_pair'].get('replay_steps',0) for a in audit_rows),
            audit_identification_work=sum(a['identification_work']+a['model_pair'].get('identification_work',0) for a in audit_rows),
            wall_seconds=time.perf_counter()-started),agents=metrics,trajectories=records,observations=observations,updates=updates,
        packet_events=[a.packet_events for a in agents],packets=packets,decisions=decisions,audits=audit_rows,
        training=training,probe_points=points.tolist(),pending_packets=len(queue),
        final_models=[snapshot(a) for a in agents])


def calibrate(seed_base=40000):
    """Separate calibration scenes; temporal batch bootstrap coverage, values + gradients."""
    rows=[];cfg=Config(replay_objective='mission');comm=CommunicationConfig()
    for k in range(6):
        spec=specification('changing_hazard' if k%2 else 'complementary',seed_base+k,'smoke')
        world=CooperativeWorld(spec,cfg);a=CooperativeLearner(cfg,world.initial,world.goal,agent_id='cal',communication=comm,dynamic=spec['dynamic'])
        for tick in range(24):
            x=np.array([tick*.035,0.,0.,0.]);world.time=tick*cfg.dt;a.observe(world.sense(x,tick))
        packet=a.export(x,'probe');points=np.array([z[:2] for z in packet.anchors])
        truth=np.column_stack((world.risk_truth(points),world.risk_gradient(points))).reshape(-1)
        mean=np.array(packet.anchors)[:,2:].reshape(-1);var=np.diag(packet.covariance)
        rows.append(dict(seed=seed_base+k,dynamic=spec['dynamic'],squared_standardized=((truth-mean)**2/var).tolist()))
    # Scale fitting and coverage assessment use distinct scenes.
    ratios=np.array([v for row in rows[:3] for v in row['squared_standardized']])
    scale=max(1.,float(np.quantile(ratios,.95)/1.96**2))
    holdout=np.array([v for row in rows[3:] for v in row['squared_standardized']])
    coverage=float(np.mean(holdout<=scale*1.96**2))
    return dict(scale=scale,coverage_95=coverage,passed=coverage>=.90,rows=rows,
        scale_seeds=[r['seed'] for r in rows[:3]],assessment_seeds=[r['seed'] for r in rows[3:]],
        limitations='small development calibration; marginal coverage is not a calibrated joint posterior')


def suite_main(argv):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=['smoke','development'],default='smoke')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--budgets',type=int,nargs='+',default=[120000,240000]);args=p.parse_args(argv)
    if args.out.exists():p.error('use a fresh output directory')
    if args.workers<1 or any(b<=0 for b in args.budgets):p.error('positive workers and budgets required')
    args.out.mkdir(parents=True);(args.out/'episodes').mkdir()
    count=2 if args.profile=='smoke' else 8;base=50000 if args.profile=='smoke' else 60000
    specs=[specification(s,base+k,args.profile) for s in SCENES for k in range(count)]
    manifest=dict(profile=args.profile,scenarios=specs,arms=ARMS,budgets=args.budgets,
        config=asdict(Config(replay_objective='mission')),communication=asdict(CommunicationConfig()),
        source_hashes=source_hashes(),animation_seed=base,animation_arm='reciprocal_fixed',
        calibration_seeds=list(range(40000,40006)),training_seeds=list(range(30000,30005)),
        reserved_validation=[100000,199999],reserved_test=[200000,299999],
        expected=len(specs)*len(ARMS)*len(args.budgets),
        audit='first eligible packet per receiver stage; 16 steps, future communication withheld equally',
        value_labels='8-step legal-advertisement model pairs; approximation to remaining mission benefit',
        status='development only; no superiority acceptance criterion')
    dump(args.out/'manifest.json',manifest);archive_sources(args.out)
    calibration=calibrate();dump(args.out/'calibration.json',calibration)
    comm=replace(CommunicationConfig(),covariance_scale=calibration['scale'])
    training=[]
    for k,scene in enumerate(SCENES[:5]):
        result=execute(specification(scene,30000+k,'smoke'),'reciprocal_fixed',comm,collect_training=True,audits=False)
        training.extend(result['training']);dump(args.out/f'training_{scene}.json.gz',result)
    predictor=fit_predictor(training);dump(args.out/'predictor.json',predictor);dump(args.out/'training_labels.json',training)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    rows=[]
    jobs=[(spec,arm,replace(comm,byte_budget=budget),predictor,str(args.out)) for spec in specs for budget in args.budgets for arm in ARMS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,job) for job in jobs]
        for f in as_completed(futures):
            row=f.result()
            if row is not None:rows.append(row)
            dump(args.out/'summary.json',rows)
            print(f'{len(rows)}/{len(jobs)} cooperative episodes complete',flush=True)
    dump(args.out/'completion.json',dict(completed=len(rows),expected=len(jobs),errors=len(list((args.out/'episodes').glob('*.error.json')))))
    from .cooperative_reporting import report,render
    report(args.out);render(args.out)


def run_job(job):
    spec,arm,comm,predictor,root=job;torch.set_num_threads(1)
    name=f"{spec['name']}_{spec['seed']}_{arm}_b{comm.byte_budget}"
    try:
        result=execute(spec,arm,comm,predictor);result['summary']['budget']=comm.byte_budget
        dump(Path(root)/'episodes'/(name+'.json.gz'),result)
        return result['summary']
    except Exception:
        dump(Path(root)/'episodes'/(name+'.error.json'),dict(traceback=traceback.format_exc(),job=name))
        return None
