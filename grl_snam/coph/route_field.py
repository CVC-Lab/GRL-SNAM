"""Committed-field route evaluation. No truth, raw values, or support expansion."""
import copy,json,time
import numpy as np
from .terrain_warning import terrain_values
from .model import risk,tensor

# Process-local accounting only; never a controller or learner input.
FIELD_EVALUATIONS=0


def field_key(context):
    if context.get('route_field_mode','anchors')=='anchors':return ''
    return json.dumps(context['route_field_snapshot'],sort_keys=True,separators=(',',':'),allow_nan=False)


def sync(agent,publish=False):
    enabled=agent.cfg.route_field_mode=='supported_field'
    if not enabled and not getattr(agent,'track_route_field',False):return
    fields=getattr(agent,'routing_fields',agent.navigation.get('route_field_snapshot',{}).get('stages',[]))
    if publish:
        fields=[dict(id=s.id,version=s.version,center=list(s.center),centers=s.centers.tolist(),weights=s.weights.tolist(),
                     support=[list(q) for q in dict.fromkeys(r.q for r in s.readings if int(r.uid.split(':')[-1])%5)],
                     **(dict(local_residual=copy.deepcopy(s.local_residual)) if getattr(s,'local_residual',None) is not None else {})) for s in agent.stages]
    retained={s.id for s in agent.stages};agent.routing_fields=copy.deepcopy([s for s in fields if s['id'] in retained])
    for f in agent.routing_fields:
        stage=next(s for s in agent.stages if s.id==f['id'])
        coordinates={tuple(r.q) for r in stage.readings if int(r.uid.split(':')[-1])%5}
        f['support']=[q for q in f['support'] if tuple(q) in coordinates]
    if enabled:
        snapshot=dict(schema=1,width=agent.cfg.rbf_width,stage_width=agent.cfg.stage_width,stage_height=agent.cfg.stage_height,stages=agent.routing_fields)
        for context in [agent.navigation]+[s.entry_navigation for s in agent.stages if s.entry_navigation is not None]:
            context['route_field_mode']='supported_field';context['route_field_snapshot']=copy.deepcopy(snapshot)


def hypothetical_context(agent):
    context=copy.deepcopy(agent.navigation)
    context.update(route_field_mode='supported_field',route_field_snapshot=dict(schema=1,width=agent.cfg.rbf_width,
        stage_width=agent.cfg.stage_width,stage_height=agent.cfg.stage_height,stages=copy.deepcopy(getattr(agent,'routing_fields',[]))))
    return context


def values(points,context,details=False):
    global FIELD_EVALUATIONS
    points=np.asarray(points,dtype=float).reshape(-1,2);anchors=context.get('terrain_anchors',[])
    result=terrain_values(points,anchors,context.get('unknown_risk',1.));raw=result.copy();n=len(points)
    support=(np.linalg.norm(points[:,None,:]-np.asarray(anchors)[None,:,:2],axis=2)<=.65).any(1) if anchors else np.zeros(n,bool)
    # Recover the pre-clipping anchor envelope only for diagnostic output.
    if details and anchors:
        a=np.asarray(anchors);delta=points[:,None,:]-a[None,:,:2];ok=np.linalg.norm(delta,axis=2)<=.65
        predicted=a[None,:,2]+np.sum(delta*a[None,:,3:5],axis=2)
        raw[support]=np.max(np.where(ok,predicted,-np.inf),axis=1)[support]
    source=np.where(support,'anchor','unknown').astype(object);stage_ids=[None]*n;versions=[None]*n;selected=np.zeros(n,bool);evaluations=0
    if context.get('route_field_mode','anchors')=='supported_field':
        snap=context['route_field_snapshot']
        for s in sorted(snap['stages'],key=lambda s:s['id'],reverse=True):
            if not s['support']:continue
            mask=(np.abs(points[:,0]-s['center'][0])<=snap['stage_width']/2+1e-10)&(np.abs(points[:,1]-s['center'][1])<=snap['stage_height']/2+1e-10)
            mask &= (np.linalg.norm(points[:,None,:]-np.asarray(s['support'])[None,:,:],axis=2)<=.65).any(1)&~selected
            ids=np.flatnonzero(mask)
            if not len(ids):continue
            v=risk(tensor(points[mask]),tensor(s['weights']),tensor(s['centers']),snap['width'],s.get('local_residual')).detach().numpy()
            FIELD_EVALUATIONS+=len(ids)
            if not np.isfinite(v).all():raise ValueError('nonfinite_committed_route_field')
            raw[mask]=v;result[mask]=np.clip(v,0,20);source[mask]='field';selected[mask]=True;evaluations+=len(ids)
            for i in ids:stage_ids[i]=s['id'];versions[i]=s['version']
    if not details:return result
    return result,dict(source=source.tolist(),support=(support|selected).tolist(),field_support=selected.tolist(),stage=stage_ids,version=versions,raw=raw.tolist(),clipped=result.tolist(),field_evaluations=evaluations)


def validate_context(context,cfg,retained_ids):
    from .additive_routing import validate as validate_integral
    validate_integral(context)
    mode=context.get('route_field_mode','anchors')
    if mode!=cfg.route_field_mode:raise ValueError('incompatible_route_field_mode')
    if mode=='anchors':return
    s=context.get('route_field_snapshot')
    if not isinstance(s,dict) or s.get('schema')!=1:raise ValueError('missing_route_field_snapshot')
    if (s['width'],s['stage_width'],s['stage_height'])!=(cfg.rbf_width,cfg.stage_width,cfg.stage_height):raise ValueError('route_field_scales')
    ids=[x['id'] for x in s['stages']]
    if len(ids)>cfg.previous+1 or ids!=sorted(set(ids)) or not set(ids)<=set(retained_ids):raise ValueError('route_field_memory')
    for f in s['stages']:
        from .localized_receiver import validate
        validate(f.get('local_residual'),cfg.anchor_cap)
        if len(f['weights'])!=cfg.rbf_side**2 or np.asarray(f['centers']).shape!=(cfg.rbf_side**2,2) or len(f['support'])>cfg.sample_cap:raise ValueError('route_field_shape')
        if np.asarray(f['center']).shape!=(2,) or (f['support'] and np.asarray(f['support']).shape!=(len(f['support']),2)):raise ValueError('route_field_support')
        if f['version']<1 or not np.isfinite([*f['center'],*f['weights'],*np.asarray(f['centers']).ravel(),*np.asarray(f['support']).ravel()]).all():raise ValueError('invalid_route_field')
