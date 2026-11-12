"""Learner-side cooperation. No simulator truth or evaluator imports.

H is explicitly kinetic + navigation shaping, not physical potential energy.
Exports are rebuilt exclusively from local measurements, never operational priors.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import time
import numpy as np
import torch
from .diagnostics import InstrumentedLearner
from .model import tensor, features, risk, potential

CONVENTION='unit_mass; H=kinetic+navigation_shaping; physical_H=kinetic; SI; world_xy'

@dataclass(frozen=True)
class CommunicationConfig:
    ensemble_size: int = 8
    anchors: int = 4
    received_cap: int = 8
    ttl: float = 2.0
    temporal_window: float = 3.0
    covariance_floor: float = .0025
    covariance_scale: float = 1.
    byte_budget: int = 240000
    byte_price: float = .00001
    processing_price: float = .0001
    rate_regularizer: float = .01
    max_rate: float = 4.
    delay_ticks: int = 0
    prediction_variant: str = "legacy"

    def __post_init__(self):
        if self.prediction_variant not in ("legacy","baseline","compact","cross_stage","both"):
            raise ValueError("invalid prediction_variant")

@dataclass(frozen=True)
class Packet:
    schema: int
    sender: str
    recipient: str
    identity: str
    timestamp: float
    expires: float
    version: int
    convention: str
    q: tuple
    p: tuple
    v: tuple
    H: float
    V_task: float
    goal: tuple
    bounds: tuple
    anchors: tuple  # (qx,qy,value,gx,gy)
    covariance: tuple  # joint value/gradient covariance in anchor-major ordering
    sources: tuple
    stage: int
    support_points: tuple
    uncertainty: str
    acknowledged_sources: tuple = ()

    def bytes(self):
        return len(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def field_vector(weights,centers,points,width,residual=None):
    q=tensor(points).requires_grad_(True)
    v=risk(q,weights,centers,width,residual)
    g=torch.autograd.grad(v.sum(),q)[0]
    return torch.cat((v[:,None],g),dim=1).detach().numpy().reshape(-1)


def bootstrap_field(stage,cfg,comm,now,seed,dynamic=False):
    readings=[r for r in stage.readings if int(r.uid.split(':')[-1])%5 and
              (not dynamic or now-r.time<=comm.temporal_window)]
    if not readings:return None
    # Whole time blocks, not individual spatial sensor points, are resampled.
    batches={}
    for r in readings:batches.setdefault(int(r.time/(cfg.dt*cfg.update_every)),[]).append(r)
    blocks=list(batches.values());rng=np.random.default_rng(seed)
    prior=tensor(np.full(len(stage.weights),-1/len(stage.weights)))
    def fit(rs):
        A=features(tensor([r.q for r in rs]),stage.centers,cfg.rbf_width)
        y=tensor([np.log(np.expm1(4*max(.001,min(r.value,20.)))) for r in rs])
        return torch.linalg.solve(A.T@A+cfg.fit_ridge*torch.eye(len(prior),dtype=torch.float64),
                                  A.T@y+cfg.fit_ridge*prior)
    mean=fit(readings)
    ensemble=[fit([r for k in rng.integers(0,len(blocks),len(blocks)) for r in blocks[k]])
              for _ in range(comm.ensemble_size)]
    return mean,ensemble,readings


class CooperativeLearner(InstrumentedLearner):
    def __init__(self,*args,agent_id,communication=None,dynamic=False,receiver_fit_mode="legacy",receiver_validation_mode="legacy",receiver_solver_mode="projected",receiver_search_mode="legacy",receiver_localization_mode="global",**kwargs):
        super().__init__(*args,**kwargs)
        if receiver_fit_mode not in ('legacy','precision_bounded'):raise ValueError('receiver_fit_mode')
        if receiver_validation_mode not in ('legacy','relative_local'):raise ValueError('receiver_validation_mode')
        if receiver_solver_mode not in ('projected','active_set'):raise ValueError('receiver_solver_mode')
        if receiver_search_mode not in ('legacy','diagnostic'):raise ValueError('receiver_search_mode')
        if (receiver_solver_mode,receiver_search_mode)!=('projected','legacy') and receiver_fit_mode!='precision_bounded':raise ValueError('nondefault receiver solver requires precision_bounded fitting')
        if receiver_localization_mode not in ('global','compact'):raise ValueError('receiver_localization_mode')
        if receiver_localization_mode=='compact' and receiver_fit_mode!='precision_bounded':raise ValueError('compact receiver requires precision_bounded fitting')
        self.receiver_localization_mode=receiver_localization_mode
        self.receiver_solver_mode=receiver_solver_mode;self.receiver_search_mode=receiver_search_mode
        self.receiver_fit_mode=receiver_fit_mode;self.receiver_validation_mode=receiver_validation_mode
        self.agent_id=str(agent_id);self.communication=communication or CommunicationConfig()
        self.current_state=tuple(args[1] if len(args)>1 else kwargs['initial'])
        self.dynamic=dynamic;self.now=0.;self.received={};self.packet_events=[]
        self.last_sequences={};self.export_work=0;self.message_fit_work=0

    def observe(self,observation,transition=None):
        self.now=observation.time
        if transition is not None:self.current_state=tuple(transition.after)
        observation=replace(observation,readings=tuple(replace(r,uid=f'{self.agent_id}:{r.uid}') for r in observation.readings))
        super().observe(observation,transition)
        if self.dynamic:
            for s in self.stages:
                s.readings=[r for r in s.readings if self.now-r.time<=self.communication.temporal_window]
        self._expire()

    def evaluate_packet(self,packet,seed=None):
        from .packet_feedback import evaluate_receiver
        return evaluate_receiver(self,packet,seed=seed)

    def _expire(self):
        for key,m in list(self.received.items()):
            if m.expires<self.now or not any(s.contains(a[:2],self.cfg) for s in self.stages for a in m.anchors):
                del self.received[key]

    def anchor_candidates(self,x,stage):
        return [np.asarray(x[:2])+offset for offset in ((0.,0.),(.4,0.),(0.,.4),(0.,-.4))]

    def export(self,x,recipient):
        s=self.stages[-1];comm=self.communication
        seed=int(hashlib.sha256(f'{self.agent_id}:{self.now}:{s.id}'.encode()).hexdigest()[:8],16)
        result=bootstrap_field(s,self.cfg,comm,self.now,seed,self.dynamic)
        if result is None:return None
        mean,ensemble,rs=result
        candidates=self.anchor_candidates(x,s)
        points=np.array([q for q in candidates if s.supported(q,self.cfg)])[:comm.anchors]
        if not len(points):return None
        vector=field_vector(mean,s.centers,points,self.cfg.rbf_width)
        vectors=np.array([field_vector(w,s.centers,points,self.cfg.rbf_width) for w in ensemble])
        covariance=comm.covariance_scale*(np.cov(vectors,rowvar=False)+np.eye(len(vector))*comm.covariance_floor)
        self.export_work+=len(rs)*(comm.ensemble_size+1)
        task=.5*float(np.sum((np.asarray(x[:2])-np.asarray(self.control_target(self.stages[-1])))**2))
        # Fixed convention: no sender policy coefficient is transferred as environment truth.
        H=.5*float(np.sum(np.asarray(x[2:])**2))+float(potential(tensor(x[:2]),tensor(self.control_target(s)),
            mean,s.centers,self.cfg.rbf_width,self.policy,self.control_obstacles(s),self.cfg))
        anchors=tuple(tuple(q)+tuple(z) for q,z in zip(points,vector.reshape(-1,3)))
        return Packet(1,self.agent_id,str(recipient),f'{self.agent_id}:{self.now:.9f}:{s.id}',self.now,
            self.now+comm.ttl,self.version,CONVENTION,tuple(x[:2]),tuple(x[2:]),tuple(x[2:]),H,task,tuple(self.goal),
            (s.center[0]-self.cfg.stage_width/2,s.center[0]+self.cfg.stage_width/2,
             s.center[1]-self.cfg.stage_height/2,s.center[1]+self.cfg.stage_height/2),anchors,
            tuple(map(tuple,covariance)),tuple(sorted({r.uid for r in rs})),s.id,tuple(dict.fromkeys(r.q for r in rs)),
            'block_bootstrap; calibration_scale='+str(comm.covariance_scale))

    def assimilate(self,packet):
        """Validate and stage evidence only; update() commits atomically afterwards."""
        m=packet;reason='eligible'
        if m.schema!=1 or m.convention!=CONVENTION:reason='incompatible_convention'
        elif m.sender==self.agent_id or any(s.startswith(self.agent_id+':') for s in m.sources):reason='returned_evidence'
        elif m.recipient!=self.agent_id:reason='wrong_recipient'
        elif not m.sources or any(not s.startswith(m.sender+':') for s in m.sources):reason='invalid_provenance'
        elif not np.isfinite([m.timestamp,m.expires]).all():reason='invalid_age'
        elif m.timestamp>self.now+1e-9 or m.expires<self.now or m.expires<m.timestamp:reason='invalid_age'
        elif (m.timestamp,m.stage)<=self.last_sequences.get(m.sender,(-1.,-1)):reason='duplicate_or_reordered'
        else:
            try:
                cov=np.asarray(m.covariance);a=np.asarray(m.anchors)
                finite=np.isfinite(a).all() and np.isfinite(cov).all() and np.isfinite([*m.q,*m.p,*m.v,m.H,m.V_task,*m.bounds]).all() and np.isfinite(np.asarray(m.support_points)).all()
                valid=(len(a)>0 and len(a)<=self.communication.anchors and a.shape==(len(a),5) and
                    cov.shape==(3*len(a),3*len(a)) and np.allclose(cov,cov.T) and np.linalg.eigvalsh(cov).min()>0)
                if not finite or not valid or len(m.q)!=2 or len(m.p)!=2 or len(m.v)!=2 or not np.allclose(m.p,m.v) or np.any(a[:,2]<0) or len(m.support_points)>self.cfg.sample_cap:reason='invalid_payload'
                elif any(not (m.bounds[0]<=z[0]<=m.bounds[1] and m.bounds[2]<=z[1]<=m.bounds[3]) for z in a):reason='invalid_support'
                elif not any(s.contains(z[:2],self.cfg) for s in self.stages for z in a):reason='unsupported'
            except (ValueError,IndexError,TypeError):reason='invalid_payload'
        if reason=='eligible':
            # Replace the sender's whole prior contribution, even when stages overlap.
            self.received[m.sender]=m
            self.last_sequences[m.sender]=(m.timestamp,m.stage)
            while len(self.received)>self.communication.received_cap:
                del self.received[min(self.received,key=lambda k:self.received[k].timestamp)]
        event=dict(packet=m.identity,sender=m.sender,recipient=self.agent_id,time=self.now,reason=reason,
                   version_before=self.version)
        self.packet_events.append(event)
        return event

    def _field_fit(self):
        # Local reconstruction may use operational history; exports never do.
        weights,cov=super()._field_fit()
        self.receiver_fit_diagnostic=None
        if (getattr(self,'receiver_fit_mode','legacy')!='legacy' or getattr(self,'receiver_validation_mode','legacy')!='legacy'
            or getattr(self,'receiver_diagnostic_variant',None) is not None):
            from .receiver_fitting import fit
            try:return fit(self,weights,cov)
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                self.receiver_local_proposal=[w.clone() for w in weights]
                self.receiver_fit_diagnostic=dict(success=False,eligible_rows=None,stages=[],error=str(exc),status='numerical_failure')
                self.reference=[w.clone() for w in weights]
                return weights,cov
        if not self.received:return weights,cov
        adjusted=[]
        for s,w in zip(self.stages,weights):
            rows=[];targets=[]
            for m in self.received.values():
                ids=[k for k,a in enumerate(m.anchors) if s.contains(a[:2],self.cfg)]
                if not ids:continue
                points=tensor([m.anchors[k][:2] for k in ids]);z=tensor([m.anchors[k][2:] for k in ids])
                # Gauss-Newton message residual in actual value/gradient units.
                def vector(v):
                    f=features(points,s.centers,self.cfg.rbf_width); raw=f@v; sigmoid=torch.sigmoid(raw)
                    df=-(points[:,None,:]-s.centers)/self.cfg.rbf_width**2*f[:,:,None]
                    val=torch.nn.functional.softplus(raw)/4
                    grad=sigmoid[:,None]/4*torch.einsum('nmd,m->nd',df,v)
                    return torch.cat((val[:,None],grad),1).reshape(-1)
                J=torch.autograd.functional.jacobian(vector,w)
                indices=[3*k+j for k in ids for j in range(3)]
                C=tensor(np.asarray(m.covariance)[np.ix_(indices,indices)])
                # Representation error and age diffusion are explicit, not new observations.
                C=C+torch.eye(len(indices),dtype=torch.float64)*(.01+.005*(self.now-m.timestamp))
                L=torch.linalg.cholesky(C)
                rows.append(torch.linalg.solve_triangular(L,J,upper=False))
                targets.append(torch.linalg.solve_triangular(L,(z.reshape(-1)-vector(w))[:,None],upper=False).flatten())
                self.message_fit_work+=len(indices)
            if rows:
                A=torch.cat(rows);y=torch.cat(targets)
                # Anchor to local evidence curvature, not a purported posterior.
                local=torch.linalg.inv(cov[len(adjusted)]) if cov[len(adjusted)] is not None else torch.eye(len(w),dtype=torch.float64)
                delta=torch.linalg.solve(local+A.T@A,A.T@y)
                w=w+delta.clamp(-.5,.5)  # declared per-fit field trust bound
            adjusted.append(w)
        self.reference=[w.clone() for w in adjusted]
        return adjusted,cov

    def _receiver_acceptance(self,weights):
        from .receiver_fitting import acceptance
        return acceptance(self,weights)

    def update(self,x,force_invalid=False):
        import copy
        self._expire()
        state=copy.deepcopy(self.navigation);step=self.navigation_step;last=copy.deepcopy(self.last_control)
        before=self.action(x)[0];pre_meta=copy.deepcopy(self.last_control);pre_guards=self.last_control.get('guard_steps',0)
        self.navigation=copy.deepcopy(state);self.navigation_step=step;self.last_control=last
        self.replay_external_reserve=12 if self.cfg.controller_mode in ('protected','routed') or self.geometry_overflow else 0
        self.counterfactual_steps+=pre_guards
        result=super().update(x,force_invalid)
        from .route_field import sync
        sync(self,publish=result['accepted'])
        result['messages']=[m.identity for m in self.received.values()]
        after=self.action(x)[0];post_meta=copy.deepcopy(self.last_control);post_guards=self.last_control.get('guard_steps',0)
        self.navigation=state;self.navigation_step=step;self.last_control=last
        sync(self)
        self.counterfactual_steps+=post_guards;self.replay_external_reserve=0
        result['force_change']=float(np.linalg.norm(after-before))
        result['diagnostic_guard_steps']=pre_guards+post_guards
        result['diagnostic_controls']=[pre_meta,post_meta]
        result['simulator_steps_before']-=pre_guards
        result['counterfactual_steps']=result.get('counterfactual_steps',0)+pre_guards+post_guards
        result['temporal_estimator']='sliding_window_refit' if self.dynamic else 'static'
        for event in self.packet_events:
            if 'version_after' not in event:
                event.update(version_after=self.version,update_accepted=result['accepted'],update_reason=result['reason'])
        return result


def information_gain(packet,status):
    """Common advertised anchor variables; no access to recipient learner state."""
    if packet is None or status is None:return 0.,0
    a=np.array(packet.anchors);b=np.array(status.anchors)
    # Only exact common coordinates are comparable; callers generate common status below.
    if a.shape!=b.shape or not np.allclose(a[:,:2],b[:,:2]):return 0.,0
    novel=set(packet.sources)-set(status.acknowledged_sources)
    if not novel:return 0.,len(a)
    fraction=len(novel)/max(1,len(packet.sources))
    prior=np.array(status.covariance);noise=(np.array(packet.covariance)+np.eye(len(prior))*.01)/fraction
    post=np.linalg.inv(np.linalg.inv(prior)+np.linalg.inv(noise))
    return max(0.,float(.5*(np.linalg.slogdet(prior)[1]-np.linalg.slogdet(post)[1]))),len(a)


def advertise(agent,x,recipient,points=None):
    packet=agent.export(x,recipient)
    if packet is None or points is None:return packet
    s=agent.stages[-1]
    if not all(s.contains(q,agent.cfg) for q in points):return None
    seed=int(hashlib.sha256(f'status:{agent.agent_id}:{agent.now}'.encode()).hexdigest()[:8],16)
    mean,ensemble,rs=bootstrap_field(s,agent.cfg,agent.communication,agent.now,seed,agent.dynamic)
    z=field_vector(mean,s.centers,points,agent.cfg.rbf_width).reshape(-1,3)
    vectors=np.array([field_vector(w,s.centers,points,agent.cfg.rbf_width) for w in ensemble])
    cov=agent.communication.covariance_scale*(np.cov(vectors,rowvar=False)+np.eye(z.size)*agent.communication.covariance_floor)
    for k,q in enumerate(points):
        if not s.supported(q,agent.cfg):cov[3*k:3*k+3,3*k:3*k+3]+=np.eye(3)
    return replace(packet,anchors=tuple(tuple(q)+tuple(v) for q,v in zip(points,z)),covariance=tuple(map(tuple,cov)),acknowledged_sources=tuple(sorted({source for m in agent.received.values() for source in m.sources})))


def rate_probability(value,size,interval,comm,remaining):
    """Solve the scalar MD rate objective by derivative bisection."""
    if size>remaining:return 0.,0.
    utility=value-comm.byte_price*size-comm.processing_price
    if utility<=0:return 0.,0.
    lo,hi=0.,comm.max_rate
    for _ in range(48):
        mid=(lo+hi)/2
        derivative=interval*np.exp(-mid*interval)*utility-comm.rate_regularizer*mid
        if derivative>0:lo=mid
        else:hi=mid
    rate=(lo+hi)/2
    return float(1-np.exp(-rate*interval)),rate
