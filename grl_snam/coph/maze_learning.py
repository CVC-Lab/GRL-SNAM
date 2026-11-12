"""Opt-in geometric warning exchange and graph routing on a surveyed maze.
Unknown closures are learned only from local sensors or provenance-bearing reports.
"""
import copy,heapq,time
from dataclasses import dataclass,asdict
import numpy as np
from .cooperative import Packet,CooperativeLearner
from .terrain_warning import TerrainLearner,terrain_values,dense
from .maze_geometry import corridor_clear,obstacle_clear,network
from .shapes import is_slab,slab_distance
from .model import tensor,potential
from .records import new_stage
from .navigation import intersects

@dataclass(frozen=True)
class MazePacket(Packet):
    geometry_schema:int=1
    geometry:tuple=()  # (*shape,sensor_x,sensor_y,sensor_radius,time,source_id); shape is a disk or slab


from functools import lru_cache

@lru_cache(maxsize=64)
def graph_costs(anchors,obstacles,layout,field='',weight=1.,unknown=1.):
    nodes=np.asarray(layout[0])
    graph=[[] for _ in nodes];blocked=[];field_evaluations=0
    for ei,(i,j) in enumerate(layout[1]):
        if not obstacle_clear(nodes[i],nodes[j],obstacles):blocked.append(ei);continue
        points=dense([nodes[i],nodes[j]],.3);mid=(points[:-1]+points[1:])/2
        if field:
            import json
            from .route_field import values
            risks,meta=values(mid,dict(terrain_anchors=anchors,route_field_mode='supported_field',route_field_snapshot=json.loads(field),unknown_risk=unknown),details=True)
            field_evaluations+=meta['field_evaluations']
        else:risks=terrain_values(mid,anchors,unknown)
        # The agent's own risk weight prices shared environmental risk (default 1).
        cost=float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)*(1+4*weight*risks)))
        graph[i].append((j,cost));graph[j].append((i,cost))
    return tuple(tuple(g) for g in graph),tuple(blocked),field_evaluations


def maze_route(atlas,x,context,step):
    started=time.perf_counter();misses=graph_costs.cache_info().misses;count=0
    q=x[:2].detach().numpy();obstacles=context.get('reported_geometry',[]);layout=network(context);nodes=np.asarray(layout[0]);edges=layout[1]
    from .route_field import field_key,values
    additive=context.get('route_integral_mode','legacy')=='additive_v1'
    if additive:
        from .additive_routing import graph_and_connections,route_path
        graph,blocked,additive_connections,integral_work=graph_and_connections(q,context)
        graph_evals=integral_work['field_evaluations'];count+=graph_evals
    else:
        graph,blocked,graph_evals=graph_costs(tuple(tuple(a) for a in context.get('terrain_anchors',[])),tuple(tuple(o) for o in obstacles),layout,field_key(context),context.get('risk_weight',1.),context.get('unknown_risk',1.))
    count+=graph_evals if graph_costs.cache_info().misses>misses else 0
    def finish(meta):
        if additive:meta['route_integral_work']=integral_work
        if context.get('route_field_trace_enabled') or context.get('route_field_mode')=='supported_field':
            meta['route_field_work']=dict(field_evaluations=count,graph_cache_miss=graph_costs.cache_info().misses>misses,seconds=time.perf_counter()-started)
        return meta
    # Virtual start connects along the occupied corridor, never through a closure/wall.
    connections={}
    for i,j in (() if additive else edges):
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/np.sum((b-a)**2),0,1)
        if np.linalg.norm(q-(a+t*(b-a)))>.7:continue
        for k in (i,j):
            if corridor_clear(q,nodes[k],nodes=layout[0],edges=edges) and obstacle_clear(q,nodes[k],obstacles):
                segment=dense([q,nodes[k]],.15);mid=(segment[:-1]+segment[1:])/2
                vv,vm=values(mid,context,details=True);count+=vm['field_evaluations']
                connections[k]=float(np.sum(np.linalg.norm(np.diff(segment,axis=0),axis=1)*(1+4*context.get('risk_weight',1.)*vv)))
    if additive:connections=additive_connections
    distances={};parents={};queue=[]
    for k,v in connections.items():distances[k]=v;parents[k]=None;heapq.heappush(queue,(v,k))
    target_node=int(np.argmin(np.linalg.norm(nodes-np.asarray(atlas.goal),axis=1)));expanded=0
    while queue:
        cost,i=heapq.heappop(queue)
        if cost>distances[i]+1e-12:continue
        expanded+=1
        if i==target_node:break
        for j,edge_cost in graph[i]:
            proposed=cost+edge_cost
            if proposed<distances.get(j,float('inf'))-1e-12:
                distances[j]=proposed;parents[j]=i;heapq.heappush(queue,(proposed,j))
    if target_node not in distances:
        context.update(path=[],target=q.tolist(),blocked_edges=list(blocked))
        return tensor(q),finish(dict(no_route=True,route_reason='maze_no_route',route_expansions=expanded,route_changed=True,blocked_edges=list(blocked),route_nodes=[],waypoint=q.tolist()))
    ids=[target_node]
    while parents[ids[-1]] is not None:ids.append(parents[ids[-1]])
    ids.reverse();path=dense([q,*nodes[ids].tolist()]);target=path[-1]
    if additive:path=route_path(ids,q,context,integral_work);target=path[-1]
    if context.get('route_candidate_schema')==1 and context.get('route_candidate_nodes'):
        from .route_decision import conditioned_path
        forced,forced_cost,forced_work=conditioned_path(context['route_candidate_nodes'],q,context)
        count+=forced_work
        if forced is None:
            context.update(path=[],target=q.tolist(),blocked_edges=list(blocked))
            return tensor(q),finish(dict(no_route=True,route_reason='invalid_frozen_candidate',route_expansions=expanded,
                route_changed=True,blocked_edges=list(blocked),route_nodes=[],waypoint=q.tolist()))
        ids=forced;path=dense([q,*nodes[ids].tolist()]);target=path[-1]
        if additive:path=route_path(ids,q,context,integral_work);target=path[-1]
    for p in path[1:]:
        if not corridor_clear(q,p,nodes=layout[0],edges=edges) or not obstacle_clear(q,p,obstacles):break
        target=p
        if np.linalg.norm(p-q)>=.65:break
    if np.linalg.norm(q-nodes[target_node])<1 and corridor_clear(q,nodes[target_node],nodes=layout[0],edges=edges) and obstacle_clear(q,nodes[target_node],obstacles):target=nodes[target_node]
    # Fixed .1 m grid targets preserve state derivatives away from grid switches.
    target=np.round(np.asarray(target)/.1)*.1
    if not corridor_clear(q,target,nodes=layout[0],edges=edges) or not obstacle_clear(q,target,obstacles):
        return tensor(q),finish(dict(no_route=True,route_reason='unsafe_connector',route_expansions=expanded,route_changed=True,blocked_edges=list(blocked),route_nodes=ids,waypoint=q.tolist()))
    changed=ids!=context.get('route_nodes') or list(blocked)!=context.get('blocked_edges')
    context.update(path=path.tolist(),target=target.tolist(),route_nodes=ids,blocked_edges=list(blocked),planned_at=step)
    return tensor(target),finish(dict(no_route=False,route_reason='maze_graph',route_expansions=expanded,route_changed=changed,
        route_nodes=ids,blocked_edges=list(blocked),route=path.tolist(),waypoint=target.tolist(),stagnation=False))


class MazeLearner(TerrainLearner):
    stage_mode='spatial'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.navigation.pop('surveyed_routes',None);self.navigation['maze_graph']=True
        self.local_geometry_reports={};self.remote_geometry_reports={};self.geometry_events=[]
        self.stages[0].entry_navigation=copy.deepcopy(self.navigation)

    def refresh_geometry(self):
        self.remote_geometry_reports={k:v for k,v in self.remote_geometry_reports.items() if v['expires']>=self.now}
        geometry=set(self.local_geometry_reports)|set(self.remote_geometry_reports)
        self.navigation['reported_geometry']=sorted(geometry)
        # Only geometrically relevant disks enter the existing local potential/CBFs.
        relevant=[o for o in geometry if any(intersects(o,s.center,self.cfg) for s in self.stages)]
        if len(relevant)>self.cfg.obstacle_cap:self.geometry_overflow=True
        self.geometry_store=tuple(sorted(set(self.geometry_store)|set(relevant)))[:self.cfg.obstacle_cap]
        for s in self.stages:s.obstacles=tuple(o for o in self.geometry_store if intersects(o,s.center,self.cfg))

    def observe(self,observation,transition=None):
        super().observe(observation,transition)
        for o in observation.obstacles:
            key=tuple(o)
            if key not in self.local_geometry_reports:
                source=f'{self.agent_id}:geometry:{len(self.local_geometry_reports)}'
                self.geometry_events.append(dict(kind='locally_sensed',time=self.now,obstacle=key,source=source,q=observation.sensor_center))
            else:source=self.local_geometry_reports[key][-1]
            self.local_geometry_reports[key]=(*key,*observation.sensor_center,observation.sensor_radius,self.now,source)
        if len(self.local_geometry_reports)>32:self.geometry_overflow=True
        self.local_geometry_reports=dict(list(self.local_geometry_reports.items())[-32:]);self.refresh_geometry()

    def export(self,x,recipient):
        m=super().export(x,recipient)
        if m is None:return None
        return MazePacket(**asdict(m),geometry=tuple(self.local_geometry_reports.values()))

    def assimilate(self,m):
        if not isinstance(m,MazePacket) or m.geometry_schema!=1:return dict(reason='incompatible_geometry',packet=m.identity,sender=m.sender,recipient=self.agent_id,time=self.now,version_before=self.version)
        valid=len(m.geometry)<=32
        try:
            for o in m.geometry:
                # Shape, then sensor centre/radius, observation time and source.
                valid &= len(o) in (8,10) and np.isfinite(o[:-1]).all() and o[2]>0 and o[-3]>0 and o[-2]<=m.timestamp+1e-9 and o[-2]>=0 and str(o[-1]).startswith(m.sender+':geometry:')
                shape,sensor=o[:-5],np.asarray(o[-5:-3])
                gap=float(slab_distance(sensor,shape)) if is_slab(shape) else np.linalg.norm(np.asarray(o[:2])-sensor)-o[2]
                valid &= gap<=o[-3]+1e-9
        except (TypeError,ValueError,IndexError):valid=False
        if not valid:return dict(reason='invalid_geometry',packet=m.identity,sender=m.sender,recipient=self.agent_id,time=self.now,version_before=self.version)
        event=super().assimilate(m)
        if event['reason']=='eligible':
            for o in m.geometry:
                key=tuple(o[:-5]);new=key not in self.remote_geometry_reports and key not in self.local_geometry_reports
                self.remote_geometry_reports[key]=dict(expires=m.expires,packet=m.identity,source=o[-1],observation_time=o[-2],sender=m.sender)
                if new:self.geometry_events.append(dict(kind='received_geometry',time=self.now,obstacle=key,packet=m.identity,sender=m.sender,q=self.current_state[:2]))
            if len(self.remote_geometry_reports)>32:self.geometry_overflow=True
            self.remote_geometry_reports=dict(list(self.remote_geometry_reports.items())[-32:]);self.refresh_geometry()
        return event

    def update(self,x):
        for s in self.stages:
            s.entry_navigation.pop('surveyed_routes',None);s.entry_navigation['maze_graph']=True
            s.entry_navigation['reported_geometry']=copy.deepcopy(self.navigation.get('reported_geometry',[]))
        result=super().update(x)
        # TerrainLearner supplies its legacy surveyed-route key; maze dispatch takes priority.
        return result

    def advance(self,x):
        s=self.stages[-1];delta=np.abs(np.asarray(x[:2])-s.center)
        if not np.any(delta>np.array([self.cfg.stage_width,self.cfg.stage_height])*(1-self.cfg.overlap)/2):return False
        old=float(self.shaping_energy(x,s))
        sizes=np.array([self.cfg.stage_width,self.cfg.stage_height]);axis=int(np.argmax(delta/sizes))
        center=np.asarray(s.center).copy();center[axis]+=np.sign(x[axis]-center[axis])*sizes[axis]*(1-self.cfg.overlap)
        n=new_stage(s.id+1,tuple(center),x,self.goal,self.cfg)
        n.readings=[r for oldstage in self.stages for r in oldstage.readings if n.contains(r.q,self.cfg)]
        n.readings=list({r.uid:r for r in n.readings}.values())[-self.cfg.sample_cap:]
        n.footprints=list(dict.fromkeys(f for oldstage in self.stages for f in oldstage.footprints))[-self.cfg.footprint_cap:]
        n.entry_navigation=copy.deepcopy(self.navigation);n.version=self.version
        self.stages=(self.stages+[n])[-(self.cfg.previous+1):];self.refresh_geometry()
        from .route_field import sync
        sync(self)
        new=float(self.shaping_energy(x,n))
        self.context_jumps.append(dict(stage=n.id,shaping_jump=new-old,physical_jump=0.));return True

    def control_model(self):
        s=self.stages[-1]
        result=dict(stage=s.id,version=self.version,center=s.center,weights=s.weights.tolist(),centers=s.centers.tolist(),policy=self.policy.tolist(),
            geometry=self.geometry_store,target=self.control_target(s),support=list(dict.fromkeys(r.q for r in s.readings if int(r.uid.split(':')[-1])%5)),
            local_geometry=sorted(self.local_geometry_reports),remote_geometry=sorted(self.remote_geometry_reports),damping=self.damping)
        if getattr(self,'receiver_localization_mode','global')=='compact':
            result['local_residual']=copy.deepcopy(getattr(s,'local_residual',None))
        if getattr(self.cfg,'stage_blend_mode','legacy')=='smooth':
            result['stage_blend_mode']='smooth'
            result['blend_models']=[dict(stage=stage.id,version=stage.version,center=stage.center,weights=stage.weights.tolist(),
                centers=stage.centers.tolist(),support=list(dict.fromkeys(r.q for r in stage.readings if int(r.uid.split(':')[-1])%5)),
                local_residual=copy.deepcopy(getattr(stage,'local_residual',None))) for stage in self.stages]
        return result
