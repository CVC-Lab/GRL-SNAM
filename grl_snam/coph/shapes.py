"""Obstacle shapes. A disk is (x,y,r); a slab is (x,y,r,hx,hy): every point within
r of the segment (x,y)±(hx,hy). Callers keep their original disk formulas and use
these helpers only for slabs, so earlier disk results are reproduced bit for bit.
"""
import numpy as np
import torch


def is_slab(o):
    return len(o)==5


def reach(o):
    """Radius about (x,y) that contains the shape; exact for disks."""
    return o[2]+float(np.hypot(o[3],o[4])) if is_slab(o) else o[2]


def _point_segment(p,a,b):
    d=b-a;t=np.clip(np.dot(p-a,d)/(d@d+1e-20),0,1)
    return float(np.linalg.norm(a+t*d-p))


def _crosses(a,b,p,s):
    def side(o,x,y):return (x[0]-o[0])*(y[1]-o[1])-(x[1]-o[1])*(y[0]-o[0])
    return side(p,s,a)*side(p,s,b)<0 and side(a,b,p)*side(a,b,s)<0


def ends(o):
    c=np.asarray(o[:2],float);h=np.asarray(o[3:5],float)
    return c-h,c+h


def slab_distance(q,o):
    """Distance from point(s) q to the slab surface; negative inside."""
    p,s=ends(o);q=np.asarray(q,float);d=s-p
    t=np.clip((q-p)@d/(d@d+1e-20),0,1)
    return np.linalg.norm(q-(p+t[...,None]*d),axis=-1)-o[2]


def slab_sweep(a,b,o):
    """Distance from the motion segment a->b to the slab surface."""
    a=np.asarray(a,float);b=np.asarray(b,float);p,s=ends(o)
    if _crosses(a,b,p,s):return -o[2]
    return min(_point_segment(a,p,s),_point_segment(b,p,s),_point_segment(p,a,b),_point_segment(s,a,b))-o[2]


def slab_offset(q,o):
    """Differentiable q minus its nearest core point, and whether that point is interior."""
    p,s=(torch.as_tensor(z,dtype=q.dtype) for z in ends(o));d=s-p
    raw=((q-p)*d).sum(-1)/(d@d)
    t=torch.clamp(raw,0,1)
    return q-(p+t[...,None]*d),(raw>0)&(raw<1)


def surface_distance(q,obstacles):
    """Torch distance from q (...,2) to every obstacle surface, shape (...,n)."""
    out=[]
    for o in obstacles:
        if is_slab(o):z,_=slab_offset(q,o)
        else:z=q-torch.as_tensor(o[:2],dtype=q.dtype)
        out.append(torch.linalg.vector_norm(z,dim=-1)-o[2])
    return torch.stack(out,-1)
