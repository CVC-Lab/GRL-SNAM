"""Abstract route oracle: what communication should save, from geometry alone.

Agents move at a common speed along centerline routes on the surveyed graph,
planning optimistically: an edge is open until its door is known. A door becomes
known when its surface is within sensor range (Euclidean, as in the simulator) or
when a warning from the other agent arrives after a fixed delay. Warnings carry
only locally sensed doors, never relayed ones. The planner ignores terrain cost;
it is a declared idealization whose error the real runs measure.
"""
import heapq
import numpy as np
from .shapes import slab_distance
from .maze_geometry import obstacle_clear


def _routes(nodes,edges,known,q,goal):
    """Shortest centerline path from point q to the goal node, avoiding known doors."""
    graph={i:[] for i in range(len(nodes))}
    for i,j in edges:
        if obstacle_clear(nodes[i],nodes[j],known,margin=0.):
            w=float(np.linalg.norm(nodes[i]-nodes[j]));graph[i].append((j,w));graph[j].append((i,w))
    dist={};parent={};queue=[]
    for i,j in edges:
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/((b-a)@(b-a)),0,1)
        if np.linalg.norm(q-(a+t*(b-a)))>1e-6:continue
        for k in (i,j):
            if obstacle_clear(q,nodes[k],known,margin=0.):
                d=float(np.linalg.norm(q-nodes[k]))
                if d<dist.get(k,np.inf):dist[k]=d;parent[k]=None;heapq.heappush(queue,(d,k))
    while queue:
        d,i=heapq.heappop(queue)
        if d>dist[i]:continue
        if i==goal:break
        for j,w in graph[i]:
            if d+w<dist.get(j,np.inf)-1e-12:dist[j]=d+w;parent[j]=i;heapq.heappush(queue,(d+w,j))
    if goal not in dist:return None
    ids=[goal]
    while parent[ids[-1]] is not None:ids.append(parent[ids[-1]])
    return [q]+[nodes[k] for k in reversed(ids)]


def simulate(nodes,edges,doors,starts,goals,senders=(0,1),delay=0.,start_times=(0.,0.),speed=1.,sensor_radius=1.25,ds=.01,horizon=1e4):
    """Returns per-agent path length, finish time and door events.

    `senders` lists the agents whose warnings are transmitted. `delay` and
    `start_times` are in seconds; motion is at `speed` metres per second.
    """
    nodes=np.asarray(nodes,float);doors=[tuple(d) for d in doors];n=len(starts)
    goal_ids=[int(np.argmin(np.linalg.norm(nodes-np.asarray(g),axis=1))) for g in goals]
    q=[np.asarray(s[:2],float) for s in starts];known=[set() for _ in range(n)];path=[None]*n
    length=[0.]*n;done=[False]*n;finish=[None]*n;events=[];inbox=[]
    dt=ds/speed;t=0.
    def replan(i):
        path[i]=_routes(nodes,edges,sorted(known[i]),q[i],goal_ids[i])
    for i in range(n):replan(i)
    while not all(done) and t<horizon:
        for when,i,d,sender in [m for m in inbox if m[0]<=t+1e-12]:
            if d not in known[i]:known[i].add(d);events.append(dict(kind='received',time=when,agent=i,door=d,sender=sender));replan(i)
        inbox=[m for m in inbox if m[0]>t+1e-12]
        for i in range(n):
            if done[i] or t<start_times[i]-1e-12:continue
            for d in doors:
                if d not in known[i] and float(slab_distance(q[i],d))<=sensor_radius:
                    known[i].add(d);events.append(dict(kind='sensed',time=t,agent=i,door=d,q=q[i].tolist()));replan(i)
                    # An inactive receiver has no pre-start buffer; it hears at activation.
                    if i in senders:inbox+=[(max(t+delay,start_times[j]),j,d,i) for j in range(n) if j!=i]
            if path[i] is None:done[i]=True;continue
            step=ds
            while step>1e-12 and len(path[i])>1:
                seg=path[i][1]-q[i];L=float(np.linalg.norm(seg))
                if L<=step:q[i]=path[i][1].copy();path[i].pop(0);step-=L;length[i]+=L
                else:q[i]=q[i]+seg/L*step;length[i]+=step;step=0.
            if len(path[i])<=1:done[i]=True;finish[i]=t+dt
        t+=dt
    return dict(length=length,finish=finish,success=[f is not None for f in finish],events=events)


def shortest(nodes,edges,doors,start,goal):
    nodes=np.asarray(nodes,float);g=int(np.argmin(np.linalg.norm(nodes-np.asarray(goal),axis=1)))
    p=_routes(nodes,edges,sorted(map(tuple,doors)),np.asarray(start[:2],float),g)
    return None if p is None else float(sum(np.linalg.norm(b-a) for a,b in zip(p[:-1],p[1:])))


ARM_SENDERS={'disabled':(),'one_way_01':(0,),'one_way_10':(1,),'reciprocal_fixed':(0,1)}


def predict(spec,arm,delay=0.,speed=1.,sensor_radius=1.25):
    """Wasted distance per agent for a scene specification."""
    nodes,edges=spec['network_nodes'],spec['network_edges'];doors=[o for o in spec['obstacles'] if len(o)==5]
    r=simulate(nodes,edges,doors,spec['initials'],spec['goals'],ARM_SENDERS[arm],delay,spec.get('start_seconds',(0.,0.)),speed,sensor_radius)
    out={}
    for i in range(len(spec['initials'])):
        best=shortest(nodes,edges,doors,spec['initials'][i],spec['goals'][i])
        out[i]=dict(path=r['length'][i],wasted=r['length'][i]-best,shortest=best,success=r['success'][i],finish=r['finish'][i])
    return dict(agents=out,team_wasted=sum(v['wasted'] for v in out.values()),events=r['events'])


def true_distance(spec,q,goal):
    """Evaluator-only: corridor distance from q to goal with every door known."""
    nodes=np.asarray(spec['network_nodes'],float);edges=spec['network_edges'];doors=[tuple(o) for o in spec['obstacles'] if len(o)==5]
    q=np.asarray(q[:2],float);best=None
    for i,j in edges:
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/((b-a)@(b-a)),0,1);p=a+t*(b-a);d=float(np.linalg.norm(q-p))
        if best is None or d<best[0]:best=(d,p)
    g=int(np.argmin(np.linalg.norm(nodes-np.asarray(goal),axis=1)))
    path=_routes(nodes,edges,doors,best[1],g)
    return np.nan if path is None else float(sum(np.linalg.norm(b-a) for a,b in zip(path[:-1],path[1:])))
