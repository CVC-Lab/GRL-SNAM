"""Public surveyed corridor geometry; no hidden gate locations or world handles."""
import numpy as np
from .shapes import is_slab,slab_sweep

NODES=((0.,0.),(3.,0.),(6.,0.),(9.,0.),(12.,0.),(0.,3.),(3.,3.),(6.,3.),(9.,3.),(12.,3.),(3.,6.),(6.,6.),(9.,6.),(12.,6.))
EDGES=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(1,6),(2,7),(3,8),(4,9),(6,7),(7,8),(8,9),(6,10),(10,11),(11,12),(12,8),(12,13),(13,9))
CORRIDOR_RADIUS=.85


def network(context):
    """Surveyed layout carried in the navigation context; the original maze by default."""
    nodes,edges=context.get('network',(NODES,EDGES))
    return tuple(map(tuple,nodes)),tuple(map(tuple,edges))


def distance_to_network(points,nodes=NODES,edges=EDGES):
    p=np.asarray(points);a=np.asarray([nodes[i] for i,j in edges]);d=np.asarray([np.asarray(nodes[j])-nodes[i] for i,j in edges])
    delta=p[...,None,:]-a;t=np.clip(np.sum(delta*d,axis=-1)/np.sum(d*d,axis=-1),0,1)
    return np.min(np.linalg.norm(delta-t[...,None]*d,axis=-1),axis=-1)


def corridor_clear(a,b,margin=.17,nodes=NODES,edges=EDGES):
    # 1-Lipschitz distance + half the sample spacing makes this conservative.
    a,b=np.asarray(a),np.asarray(b);n=max(2,int(np.ceil(np.linalg.norm(b-a)/.04))+1)
    points=a+(b-a)*np.linspace(0,1,n)[:,None];padding=np.linalg.norm(b-a)/(2*(n-1))
    return bool(np.max(distance_to_network(points,nodes,edges))+padding<=CORRIDOR_RADIUS-margin+1e-10)


def obstacle_clear(a,b,obstacles,margin=.17):
    a,b=np.asarray(a),np.asarray(b);d=b-a
    for o in obstacles:
        if is_slab(o):
            if slab_sweep(a,b,o)<=margin:return False
            continue
        ox,oy,r=o
        z=np.array([ox,oy]);t=np.clip((z-a)@d/(d@d+1e-20),0,1)
        if np.linalg.norm(a+t*d-z)<=r+margin:return False
    return True
