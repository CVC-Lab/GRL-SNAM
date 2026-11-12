"""Local sensed geometry, deterministic routing and differentiable control projection.
No world, scene specification, or evaluator input is accepted here.
"""
import copy,heapq,math
import numpy as np
import torch
from .model import tensor,command,flow,risk,potential,shaping_parameters
from .shapes import reach,is_slab,slab_distance,slab_sweep,slab_offset


def intersects(obstacle,center,cfg):
    d=np.maximum(np.abs(np.asarray(obstacle[:2])-center)-[cfg.stage_width/2,cfg.stage_height/2],0)
    return float(d@d)<=(reach(obstacle)+.12+cfg.safety_margin)**2+1e-12


def observe_geometry(agent,observation):
    cfg=agent.cfg
    if observation.sensor_center is not None:
        footprint=(*observation.sensor_center,observation.sensor_radius,observation.time)
        for s in agent.stages:
            if intersects((*observation.sensor_center,observation.sensor_radius),s.center,cfg):
                s.footprints=(s.footprints+[footprint])[-cfg.footprint_cap:]
    incoming=set(agent.geometry_store)|set(map(tuple,observation.obstacles))
    center=np.asarray(observation.sensor_center if observation.sensor_center is not None else agent.current_state[:2])
    velocity=np.asarray(agent.current_state[2:]) if hasattr(agent,'current_state') else np.zeros(2)
    stop=np.linalg.norm(velocity)**2/(2*cfg.force_limit)+cfg.sensor_radius
    kept=sorted(o for o in incoming if any(intersects(o,s.center,cfg) for s in agent.stages) or np.linalg.norm(center-o[:2])<=stop+reach(o)+.17)
    agent.geometry_overflow=agent.geometry_overflow or len(kept)>cfg.obstacle_cap
    # Overflow latches braking for this mission; discarded geometry cannot be silently forgotten.
    agent.geometry_store=tuple(sorted(kept,key=lambda o:(float(slab_distance(center,o)) if is_slab(o) else np.linalg.norm(center-o[:2])-o[2],o))[:cfg.obstacle_cap])
    for s in agent.stages:s.obstacles=tuple(o for o in agent.geometry_store if intersects(o,s.center,cfg))


def edge_clear(a,b,obstacles,footprints,margin=.17):
    a=np.asarray(a);b=np.asarray(b);v=b-a;vv=float(v@v)
    for o in obstacles:
        t=np.clip(np.dot(np.asarray(o[:2])-a,v)/(vv+1e-20),0,1)
        if np.linalg.norm(a+t*v-o[:2])<=o[2]+margin+1e-9:return False
    if not footprints:return False
    fp=np.asarray(footprints);r=fp[:,2];delta=fp[:,:2]-a
    if vv<1e-18:return bool(np.any(np.sum(delta*delta,axis=1)<=r*r))
    t=delta@v/vv;perp=np.sum(delta*delta,axis=1)-t*t*vv;good=perp<=r*r
    intervals=sorted((max(0.,float(c-d)),min(1.,float(c+d))) for c,d in zip(t[good],np.sqrt(np.maximum(0,r[good]**2-perp[good])/vv)))
    end=0.
    for lo,hi in intervals:
        if hi<0 or lo>1:continue
        if lo>end+1e-9:return False
        end=max(end,hi)
        if end>=1-1e-9:return True
    return False


def route(atlas,x,context,step):
    if context.get("maze_graph"):
        from .maze_learning import maze_route
        return maze_route(atlas,x,context,step)
    if "surveyed_routes" in context:
        from .terrain_warning import surveyed_route
        return surveyed_route(atlas,x,context,step)
    cfg=atlas.cfg;q=x[:2].detach().numpy();goal=np.array(atlas.goal)
    obstacles=atlas.geometry;footprints=atlas.footprints
    path=context.get('path',[]);old=context.get('target',goal.tolist())
    stagnation=False;progress=None
    if cfg.shaping_mode=='progress':
        history=context.get('progress_history',[])
        if history and step<=history[-1][0]:history=[]
        history=(history+[[step,*q.tolist()]])[-17:]
        if len(history)==17 and step-history[0][0]==16:
            prior=np.array(history[0][1:]);target=np.array(old)
            progress=float(np.linalg.norm(prior-target)-np.linalg.norm(q-target))
            stagnation=bool(np.linalg.norm(q-prior)<.05 and progress<.05 and np.linalg.norm(q-goal)>.25 and step-context.get('last_stagnation',-10000)>=cfg.update_every)
        context['progress_history']=history
        if stagnation:context['last_stagnation']=step
    cursor=context.get('cursor',0)
    if path:
        while cursor<len(path)-1 and np.linalg.norm(q-path[cursor])<.2:cursor+=1
    stale=(stagnation or not path or step-context.get('planned_at',-10000)>=cfg.update_every or
           min((np.linalg.norm(q-p) for p in path),default=1e9)>.25 or
           not edge_clear(q,path[cursor] if path else q,obstacles,footprints,.12+cfg.safety_margin))
    # A newly sensed disk invalidates any remaining route edge immediately,
    # including edges beyond the current lookahead segment.
    if path and obstacles:
        points=np.asarray([q.tolist()]+path[cursor:]);a=points[:-1];delta=points[1:]-a
        lengths=np.sum(delta*delta,axis=1)
        for ox,oy,rad in obstacles:
            t=np.clip(np.sum((np.array([ox,oy])-a)*delta,axis=1)/(lengths+1e-20),0,1)
            if np.any(np.linalg.norm(a+t[:,None]*delta-[ox,oy],axis=1)<=rad+.12+cfg.safety_margin):stale=True;break
    expansions=0;reason='cached'
    if stale:
        centers=np.array([s.center for s in atlas.stages]);lo=centers.min(0)-[cfg.stage_width/2,cfg.stage_height/2];hi=centers.max(0)+[cfg.stage_width/2,cfg.stage_height/2]
        spacing=.1;lower=np.ceil(lo/spacing).astype(int);upper=np.floor(hi/spacing).astype(int)
        nodes=np.array([(i,j) for i in range(lower[0],upper[0]+1) for j in range(lower[1],upper[1]+1)],dtype=int)
        points=nodes*spacing;free=np.zeros(len(points),bool)
        # Sensor footprints certify geometry only. They never certify field predictions.
        for fx,fy,rad,_ in footprints:free |= np.sum((points-[fx,fy])**2,axis=1)<=max(0,rad-.075)**2
        for ox,oy,rad in obstacles:free &= np.sum((points-[ox,oy])**2,axis=1)>(rad+.12+cfg.safety_margin+.075)**2
        points=points[free];nodes=nodes[free]
        if len(nodes):
            values=np.ones(len(nodes));unassigned=np.ones(len(nodes),bool)
            for s in reversed(atlas.stages):
                mask=(np.abs(points[:,0]-s.center[0])<=cfg.stage_width/2)&(np.abs(points[:,1]-s.center[1])<=cfg.stage_height/2)&unassigned
                if s.support:
                    ids=np.flatnonzero(mask);supported=np.min(np.sum((points[ids,None,:]-np.asarray(s.support)[None,:,:])**2,axis=2),axis=1)<=.65**2
                    ids=ids[supported]
                    if len(ids):values[ids]=risk(tensor(points[ids]),tensor(s.weights),tensor(s.centers),cfg.rbf_width,residual=s.local_residual).detach().numpy();unassigned[ids]=False
            lookup={tuple(n):i for i,n in enumerate(nodes)}
            starts=sorted(range(len(points)),key=lambda i:(np.linalg.norm(points[i]-q),tuple(nodes[i])))
            source=next((i for i in starts[:16] if edge_clear(q,points[i],obstacles,footprints,.12+cfg.safety_margin)),None)
            if source is not None:
                destination=min(range(len(points)),key=lambda i:(np.linalg.norm(points[i]-goal),tuple(nodes[i])))
                distances={source:0.};parents={};queue=[(float(np.linalg.norm(points[source]-points[destination])),source)];visited=set()
                offsets=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
                # Vectorized exact edge checks preserve the graph and fixed ordering.
                adjacency=[[] for _ in nodes]
                for dx,dy in offsets:
                    js=np.array([lookup.get((n[0]+dx,n[1]+dy),-1) for n in nodes])
                    valid=js>=0
                    if dx and dy:
                        valid &= np.array([(n[0]+dx,n[1]) in lookup and (n[0],n[1]+dy) in lookup for n in nodes])
                    ids=np.flatnonzero(valid);delta=np.array([dx,dy])*spacing
                    for o in obstacles:
                        t=np.clip((np.asarray(o[:2])-points[ids])@delta/(delta@delta),0,1)
                        keep=np.linalg.norm(points[ids]+t[:,None]*delta-o[:2],axis=1)>o[2]+.12+cfg.safety_margin
                        ids=ids[keep]
                    length=float(np.linalg.norm(delta))
                    for i in ids:adjacency[i].append((int(js[i]),length*(1+4*(values[i]+values[js[i]])/2)))
                goal_distances=np.linalg.norm(points-goal,axis=1)
                nearest=set(np.flatnonzero(goal_distances<=goal_distances[destination]+1e-12))
                heuristic=np.linalg.norm(points-points[destination],axis=1)
                while queue and expansions<cfg.route_expansion_cap:
                    _,i=heapq.heappop(queue)
                    if i in visited:continue
                    visited.add(i);expansions+=1
                    if nearest.issubset(visited):break
                    for j,edge_cost in adjacency[i]:
                        cost=distances[i]+edge_cost
                        if cost<distances.get(j,float('inf'))-1e-12:
                            distances[j]=cost;parents[j]=i;heapq.heappush(queue,(cost+float(heuristic[j]),j))
                if visited:
                    closest=min(goal_distances[i] for i in visited)
                    dest=min((i for i in visited if goal_distances[i]<=closest+1e-12),key=lambda i:(distances[i],tuple(nodes[i])))
                    ids=[dest]
                    while ids[-1]!=source:ids.append(parents[ids[-1]])
                    path=[p.tolist() for p in points[ids[::-1]]]
                    if edge_clear(path[-1],goal,obstacles,footprints,.12+cfg.safety_margin) and np.linalg.norm(points[dest]-goal)<.2:path.append(goal.tolist())
                    reason='goal_route' if np.linalg.norm(np.array(path[-1])-goal)<.01 else 'frontier_route'
                else:path=[];reason='no_route'
            else:path=[];reason='no_start_connector'
        else:path=[];reason='no_observed_free_cells'
        cursor=0
        # Fixed grid waypoint chosen at planning time; no detached moving lookahead.
        for k,p in enumerate(path):
            if np.linalg.norm(np.asarray(p)-q)<=1. and edge_clear(q,p,obstacles,footprints,.12+cfg.safety_margin):cursor=k
            elif k>cursor:break
        context.update(path=path,cursor=cursor,planned_at=step)
    elif path and cursor!=context.get('cursor',0):
        for k in range(cursor,len(path)):
            if np.linalg.norm(np.asarray(path[k])-q)<=1. and edge_clear(q,path[k],obstacles,footprints,.12+cfg.safety_margin):cursor=k
            else:break
    target=path[cursor] if path else q.tolist()
    if cfg.shaping_mode=='progress' and np.linalg.norm(q-goal)<=1. and edge_clear(q,goal,obstacles,footprints,.12+cfg.safety_margin):
        target=goal.tolist();path=[goal.tolist()];cursor=0;reason='direct_goal'
        context['path']=path
    if path and cursor==len(path)-1 and np.linalg.norm(q-target)<.2 and np.linalg.norm(q-goal)>.25:
        context['planned_at']=-10000
    jump=.5*(np.sum((q-target)**2)-np.sum((q-old)**2))
    context.update(target=list(target),cursor=cursor,reason=reason)
    return tensor(target),dict(route_reason=reason,route_expansions=expansions,route_changed=bool(stale),
                               route=path,waypoint=list(target),route_shaping_jump=float(jump),no_route=not bool(path),
                               stagnation=stagnation,target_progress=progress)


def project(raw,x,obstacles,damping,cfg,fast=False):
    """Closest point in the intersection of disks and robust CBF halfplanes."""
    q,v=x[:2],x[2:];lines=[];disks=[(torch.zeros_like(raw),tensor(cfg.force_limit))]
    for d in sorted(set((*cfg.damping_bounds,float(damping)))):
        a=math.exp(-d*cfg.dt);b=-math.expm1(-d*cfg.dt)/d
        disks.append((-a/b*v,tensor(1./b)))
    for o in obstacles:
        r=o[2]
        if is_slab(o):
            z,interior=slab_offset(q,o)
            # Beside the core segment, d²|z|²/dt² = 2(n·v)² + 2z·a; at an end cap it is the disk term.
            curvature=(z@v).square()/z.square().sum().clamp_min(1e-20) if bool(interior) else v.square().sum()
        else:
            z=q-tensor(o[:2]);curvature=v.square().sum()
        h=z.square().sum()-(r+.12+cfg.safety_margin)**2;rv=z@v
        worst=torch.maximum(2*cfg.damping_bounds[0]*rv,2*cfg.damping_bounds[1]*rv)
        bound=worst-2*curvature-8*rv-4*h
        lines.append((2*z,bound))
    if fast and bool(torch.isfinite(raw).all()) and all(float((torch.linalg.vector_norm(raw-c)-r).detach())<=1e-8 for c,r in disks) and all(float((b-a@raw).detach())<=1e-8 for a,b in lines):
        residual=max([float((torch.linalg.vector_norm(raw-c)-r).detach()) for c,r in disks]+[float((b-a@raw).detach()) for a,b in lines])
        return raw,1,residual
    candidates=[raw]
    for center,radius in disks:
        z=raw-center;candidates.append(center+radius*z/torch.linalg.vector_norm(z).clamp_min(1e-12))
    for a,b in lines:
        aa=a@a
        if float(aa.detach())<1e-14:continue
        candidates.append(raw+(b-a@raw)/aa*a)
        perp=torch.stack((-a[1],a[0]))/torch.sqrt(aa)
        for center,radius in disks:
            t=(b-a@center)/aa;base=center+t*a;disc=radius**2-t*t*aa
            if float(disc.detach())>=0:
                offset=torch.sqrt(disc.clamp_min(1e-20))*perp;candidates.extend((base+offset,base-offset))
    for i,(a,b) in enumerate(lines):
        for c,d in lines[i+1:]:
            mat=torch.stack((a,c))
            if abs(float(torch.linalg.det(mat).detach()))>1e-10:candidates.append(torch.linalg.solve(mat,torch.stack((b,d))))
    for i,(a,ra) in enumerate(disks):
        for b,rb in disks[i+1:]:
            z=b-a;length=torch.linalg.vector_norm(z)
            if float(length.detach())<1e-12:continue
            t=(ra*ra-rb*rb+length*length)/(2*length);h=ra*ra-t*t
            if float(h.detach())>=0:
                unit=z/length;perp=torch.stack((-unit[1],unit[0]));base=a+t*unit
                candidates.extend((base+torch.sqrt(h.clamp_min(1e-20))*perp,base-torch.sqrt(h.clamp_min(1e-20))*perp))
    valid=[]
    for i,p in enumerate(candidates):
        if bool(torch.isfinite(p).all()) and all(float((torch.linalg.vector_norm(p-c)-r).detach())<=1e-8 for c,r in disks) and all(float((b-a@p).detach())<=1e-8 for a,b in lines):valid.append(i)
    if not valid:return None,len(candidates),None
    index=min(valid,key=lambda i:(float((candidates[i]-raw).square().sum().detach()),i));chosen=candidates[index]
    residual=max([float((torch.linalg.vector_norm(chosen-c)-r).detach()) for c,r in disks]+[float((b-a@chosen).detach()) for a,b in lines])
    return chosen,len(candidates),residual


def guarded_control(atlas,x,index,policy,damping,context,step):
    cfg=atlas.cfg;s=atlas.stages[index];meta=dict(guard_steps=0,projection_candidates=0,route_expansions=0,route_shaping_jump=0.)
    target=tensor(atlas.goal if cfg.navigation_target=='mission' else s.target)
    old_target=tensor(context.get('target',target.tolist()))
    if cfg.controller_mode=='routed':target,extra=route(atlas,x,context,step);meta.update(extra)
    if cfg.shaping_mode=='progress':
        meta.update(shaping_mode='progress',**shaping_parameters(tensor(s.weights),cfg.rbf_width,atlas.geometry,cfg,residual=s.local_residual))
        d2=float((x[:2]-target).square().sum().detach());meta['taper']=d2/(d2+cfg.shaping_ell**2)
        if cfg.controller_mode=='routed':
            if getattr(cfg,'stage_blend_mode','legacy')=='smooth':
                meta['route_shaping_jump']=float((atlas.potential_at(x[:2],index,policy,target)-atlas.potential_at(x[:2],index,policy,old_target)).detach())
            else:
                meta['route_shaping_jump']=float((potential(x[:2],target,tensor(s.weights),tensor(s.centers),cfg.rbf_width,policy,atlas.geometry,cfg,residual=s.local_residual)-potential(x[:2],old_target,tensor(s.weights),tensor(s.centers),cfg.rbf_width,policy,atlas.geometry,cfg,residual=s.local_residual)).detach())
    if getattr(cfg,'stage_blend_mode','legacy')=='smooth':
        nominal,raw=atlas.nominal_control(x,index,policy,target)
        members,weights=atlas.blend_weights(x[:2])
        meta['stage_blend']=dict(mode='smooth',members=[s.id for s in members],weights=weights.detach().tolist(),
            versions=[s.version for s in members],certification='unchanged local measurements and stage rectangles',
            field_evaluations=len(members)*(3 if cfg.controller_mode=='routed' and cfg.shaping_mode=='progress' else 1),
            weight_evaluations=len(members)*(4 if cfg.controller_mode=='routed' and cfg.shaping_mode=='progress' else 2),
            certification_checks=len(atlas.stages)*(4 if cfg.controller_mode=='routed' and cfg.shaping_mode=='progress' else 2),
            regularity='fixed support membership only; spatial blend terms invalidate the original per-stage radial bound')
        previous=context.get('stage_blend_context')
        current=dict(atlas=[[m.id,m.version] for m in atlas.stages],geometry=[list(g) for g in atlas.geometry],
            members=[s.id for s in members],target=target.detach().tolist())
        jump=dict(available=True,shaping_jump=0.,physical_jump=0.,reason='initial' if previous is None else 'unchanged_membership')
        if previous is not None and previous['members']!=current['members']:
            if all(previous.get(key)==current[key] for key in ('atlas','geometry','target')):
                try:
                    # Diagnostic evaluation of the previous membership at this
                    # fixed state does not certify it or feed an applied control.
                    old=atlas.potential_at(x[:2],index,policy,target,member_ids=previous['members'])
                    new=atlas.potential_at(x[:2],index,policy,target)
                    jump.update(shaping_jump=float((new-old).detach()),reason='support_membership_changed')
                    meta['stage_blend']['field_evaluations']+=len(previous['members'])+len(members)
                    meta['stage_blend']['weight_evaluations']+=len(previous['members'])+len(members)
                    meta['stage_blend']['certification_checks']+=len(atlas.stages)
                except Exception as error:
                    jump.update(available=False,shaping_jump=None,reason='previous_partition_unavailable:'+type(error).__name__)
            else:jump.update(available=False,shaping_jump=None,reason='model_geometry_or_target_context_changed')
        meta['stage_blend_context_jump']=jump;context['stage_blend_context']=current
    else:
        nominal,raw=command(x,target,tensor(s.weights),tensor(s.centers),cfg.rbf_width,policy,atlas.geometry,cfg.force_limit,cfg,residual=s.local_residual)
    if cfg.shaping_mode=='progress':
        attraction=-cfg.shaping_k*(x[:2]-target)
        meta.update(attraction_force=attraction.detach().tolist(),shaping_force=(raw+1.4*x[2:]-attraction).detach().tolist())
    applied=nominal;meta.update(nominal=raw.detach().tolist(),nominal_clipped=nominal.detach().tolist(),constrained=nominal.detach().tolist(),reason='nominal',infeasible=False)
    if cfg.controller_mode in ('protected','routed') or atlas.overflow:
        if atlas.overflow or meta.get('no_route'):candidate=None;count=0;residual=None
        else:candidate,count,residual=project(raw,x,atlas.geometry,damping,cfg,fast=bool(context.get("maze_graph")))
        meta.update(projection_candidates=count,constraint_residual=residual)
        def guard(u):
            valid=True
            for d in sorted(set((*cfg.damping_bounds,float(damping)))):
                y=flow(x,u,d,cfg.dt);meta['guard_steps']+=1
                a=x[:2].detach().numpy();b=y[:2].detach().numpy();vel=y[2:].detach().numpy();delta=b-a
                stopping=np.dot(vel,vel)/(2*cfg.force_limit)
                if context.get("maze_graph"):
                    from .maze_geometry import corridor_clear,network
                    nodes,edges=network(context)
                    valid &= corridor_clear(a,b,nodes=nodes,edges=edges)
                    endpoint=b+vel/(np.linalg.norm(vel)+1e-20)*stopping
                    valid &= corridor_clear(b,endpoint,nodes=nodes,edges=edges)
                for shape in atlas.geometry:
                    if is_slab(shape):
                        valid &= slab_sweep(a,b,shape)>=.12+cfg.safety_margin-1e-9
                        valid &= float(slab_distance(b,shape))>=.12+cfg.safety_margin+stopping-1e-9
                        continue
                    ox,oy,r=shape
                    o=np.array([ox,oy]);t=np.clip(np.dot(o-a,delta)/(delta@delta+1e-20),0,1)
                    valid &= np.linalg.norm(a+t*delta-o)>=r+.12+cfg.safety_margin-1e-9
                    valid &= np.linalg.norm(b-o)>=r+.12+cfg.safety_margin+stopping-1e-9
                valid &= np.linalg.norm(vel)<=1.+1e-8
            return bool(valid)
        if candidate is not None and guard(candidate):applied=candidate;meta['reason']='projected' if float(torch.linalg.vector_norm(applied-nominal).detach())>1e-9 else 'nominal'
        else:
            speed=torch.linalg.vector_norm(x[2:]);applied=-cfg.force_limit*x[2:]/speed.clamp_min(1e-12)
            # Do not reverse velocity at a near-zero-speed stop.
            applied=applied*torch.clamp(speed/(cfg.force_limit*cfg.dt),max=1.)
            ok=guard(applied);meta.update(reason='geometry_overflow' if atlas.overflow else 'brake',infeasible=not ok)
        meta['constrained']=candidate.detach().tolist() if candidate is not None else None
    meta['applied']=applied.detach().tolist();context['last_control']=meta
    return applied,raw,meta
