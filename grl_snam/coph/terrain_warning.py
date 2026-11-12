"""Opt-in surveyed-route / unknown-terrain experiment. No world truth in learner.
Remote anchor Taylor fields support route scoring only; they do not certify replay,
geometry or measurements. Surveyed polylines are a shared mission prior.
"""
import copy
from dataclasses import replace
import numpy as np
from .cooperative import CooperativeLearner,bootstrap_field,field_vector
from .model import tensor

ROUTES=(((0.,0.),(1.5,0.),(5.5,0.),(8.,0.)),
        ((0.,0.),(1.5,0.),(1.5,1.7),(5.5,1.7),(5.5,0.),(8.,0.)))


def dense(path,spacing=.15):
    out=[]
    for a,b in zip(path[:-1],path[1:]):
        a,b=np.array(a),np.array(b)
        out.extend((a+(b-a)*t).tolist() for t in np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/spacing)+1),endpoint=False))
    return np.asarray(out+[list(path[-1])])


def terrain_values(points,anchors,unknown=1.):
    """Unknown terrain costs `unknown` (default 1). Local/remote predictions have radius .65 support."""
    values=np.ones(len(points)) if unknown==1. else np.full(len(points),float(unknown))
    if not anchors:return values
    a=np.asarray(anchors);dist=np.linalg.norm(points[:,None,:]-a[None,:,:2],axis=2)
    supported=dist<=.65
    predictions=a[None,:,2]+np.sum(a[None,:,3:5]*(points[:,None,:]-a[None,:,:2]),axis=2)
    # Conservative envelope across supported estimates, not a calibrated confidence bound.
    # A nearby low-valued estimate must not erase an overlapping hazard warning.
    ok=supported.any(1)
    values[ok]=np.clip(np.max(np.where(supported,predictions,-np.inf),axis=1)[ok],0,20)
    return values


def surveyed_route(atlas,x,context,step):
    q=x[:2].detach().numpy();paths=[dense(p) for p in context['surveyed_routes']]
    # Rejoining a different branch goes through the fork, never across unsurveyed terrain.
    candidates=[];old=context.get('branch',0)
    for j,path in enumerate(paths):
        if j!=old and q[0]>1.65 and q[0]<5.4:
            fork=np.array([1.5,0.]);prior=paths[old];idx=int(np.argmin(np.linalg.norm(prior-q,axis=1)))
            f=int(np.argmin(np.linalg.norm(prior-fork,axis=1)));start=int(np.argmin(np.linalg.norm(path-fork,axis=1)))
            remaining=np.vstack([q,prior[f:idx+1][::-1],path[start:]])
        else:
            idx=int(np.argmin(np.linalg.norm(path-q,axis=1)));remaining=np.vstack([q,path[idx+1:]])
        if len(remaining)<2:remaining=np.vstack([q,atlas.goal])
        mid=(remaining[:-1]+remaining[1:])/2;lengths=np.linalg.norm(np.diff(remaining,axis=0),axis=1)
        score=float(np.sum(lengths*(1+4*terrain_values(mid,context.get('terrain_anchors',[])))))
        candidates.append((score,j,remaining))
    # Deterministic, no hand-coded hazard label or route switch condition.
    score,j,path=min(candidates,key=lambda a:(a[0],a[1]))
    target=path[-1]
    for z in path[1:]:
        target=z
        if np.linalg.norm(z-q)>=.65:break
    if np.linalg.norm(q-np.asarray(atlas.goal))<1.:target=np.asarray(atlas.goal)
    changed=j!=old
    context.update(branch=j,path=path.tolist(),target=target.tolist(),route_scores=[v[0] for v in candidates],planned_at=step)
    return tensor(target),dict(route_expansions=sum(len(p) for p in paths),route_reason='surveyed_risk_route',no_route=False,
        route_changed=changed,branch=j,route_scores=context['route_scores'],target=target.tolist(),stagnation=False)


class TerrainLearner(CooperativeLearner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.navigation['surveyed_routes']=copy.deepcopy(ROUTES)
        self.remote_terrain={};self.local_terrain=[];self.terrain_events=[];self.local_model_anchors=[];self.remote_cells={};self.local_cells={}
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)

    def observe(self,observation,transition=None):
        super().observe(observation,transition)
        # Local route field is measured evidence; no remote prediction enters this list.
        self.local_terrain=(self.local_terrain+[(r.time,(*r.q,r.value,0.,0.)) for r in observation.readings])[-800:]
        self.remote_terrain={k:m for k,m in self.remote_terrain.items() if m.expires>=self.now}
        self.refresh_terrain()

    def refresh_terrain(self):
        self.remote_cells={k:v for k,v in self.remote_cells.items() if v[0]>=self.now}
        remote=[v[1] for v in self.remote_cells.values()]
        # Last local sample wins exact-coordinate ties, while remote information stays separate.
        anchors=remote+self.local_model_anchors
        self.navigation['terrain_anchors']=anchors

    def update(self,x):
        # Freeze the current evidence for every historical entry in this update batch.
        for stage in self.stages:
            if stage.entry_navigation is None:stage.entry_navigation={}
            stage.entry_navigation["surveyed_routes"]=copy.deepcopy(ROUTES)
            stage.entry_navigation["terrain_anchors"]=copy.deepcopy(self.navigation.get("terrain_anchors",[]))
        result=super().update(x)
        anchors=[]
        for s in self.stages:
            points=list(dict.fromkeys(r.q for r in s.readings if int(r.uid.split(":")[-1])%5))
            points=points[::max(1,len(points)//80)][:80]
            if points:
                vectors=field_vector(s.weights,s.centers,points,self.cfg.rbf_width,getattr(s,'local_residual',None)).reshape(-1,3)
                anchors.extend(tuple(q)+tuple(v) for q,v in zip(points,vectors))
        for a in anchors:self.local_cells[tuple(np.round(np.asarray(a[:2])/.2).astype(int))]=(self.now,a)
        while len(self.local_cells)>256:del self.local_cells[min(self.local_cells,key=lambda k:self.local_cells[k][0])]
        self.local_model_anchors=[v[1] for v in self.local_cells.values()]
        self.refresh_terrain()
        return result

    def export(self,x,recipient):
        packet=super().export(x,recipient)
        if packet is None:return None
        s=self.stages[-1];result=bootstrap_field(s,self.cfg,self.communication,self.now,12345+int(self.now*1000))
        mean,ensemble,rs=result
        # Send supported locations with the largest locally measured risk first.
        points=[]
        for r in sorted(rs,key=lambda r:(-r.value,r.uid)):
            if s.supported(r.q,self.cfg) and all(np.linalg.norm(np.asarray(r.q)-p)>.3 for p in points):points.append(r.q)
            if len(points)==self.communication.anchors:break
        if not points:return packet
        v=field_vector(mean,s.centers,points,self.cfg.rbf_width).reshape(-1,3)
        vs=np.array([field_vector(w,s.centers,points,self.cfg.rbf_width) for w in ensemble])
        cov=np.cov(vs,rowvar=False)+np.eye(3*len(points))*self.communication.covariance_floor
        self.export_work+=len(rs)*(len(ensemble)+1)
        return replace(packet,anchors=tuple(tuple(q)+tuple(z) for q,z in zip(points,v)),covariance=tuple(map(tuple,cov)))

    def assimilate(self,packet):
        # Existing validation handles identity, units, age, shape, covariance and provenance.
        # "unsupported" is accepted ONLY into a separate bounded route-evidence store.
        before=copy.deepcopy((self.received,self.last_sequences))
        event=super().assimilate(packet)
        if event['reason'] in ('eligible','unsupported'):
            support=np.asarray(packet.support_points)
            valid=bool(len(support) and all(np.min(np.linalg.norm(support-np.asarray(a[:2]),axis=1))<=.65 for a in packet.anchors))
            key=(packet.sender,packet.stage)
            previous=self.remote_terrain.get(key)
            if not valid or (previous and previous.timestamp>=packet.timestamp):
                self.received,self.last_sequences=before;event['reason']='invalid_remote_support' if not valid else 'duplicate_or_reordered'
            else:
                scope='retained_and_remote_route' if event['reason']=='eligible' else 'remote_route_only'
                self.remote_terrain[key]=packet
                for a in packet.anchors:
                    cell=(packet.sender,*np.round(np.asarray(a[:2])/.2).astype(int))
                    self.remote_cells[cell]=(packet.expires,a,packet.identity,packet.sources)
                while len(self.remote_cells)>128:del self.remote_cells[min(self.remote_cells,key=lambda k:self.remote_cells[k][0])]
                while len(self.remote_terrain)>8:del self.remote_terrain[min(self.remote_terrain,key=lambda k:self.remote_terrain[k].timestamp)]
                self.last_sequences[packet.sender]=(packet.timestamp,packet.stage)
                event.update(reason='eligible',scope=scope)
                self.refresh_terrain()
        self.terrain_events.append(copy.deepcopy(event));return event
