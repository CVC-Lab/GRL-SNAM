"""Abstract oracle for graded hazards: what each sharing mode should cost each agent.

Agents move at a common speed along centerline routes and plan with their own risk
weight w: a segment costs length*(1+4*w*background) plus 4*w times the believed exposure
of every hazard ahead on it. A hazard becomes known on sensing (within sensor range of
its 2.45-sigma core) or through the other agent, encoded by the sharing mode. Mission
cost is time plus weighted true exposure, as in the simulator (effort excluded).
"""
import heapq
import numpy as np

BACKGROUND=.02
BLOCKED=20.
CORE=2.45


def exposure(patch):
    """Centerline integral of a corridor-spanning Gaussian bump (amplitude times length)."""
    return patch[4]*patch[2]*np.sqrt(2*np.pi)


def encode(mode,value,sender_weight,tau=None):
    if mode in ('field','raw','no_learning'):return value
    if mode=='occupancy':return BLOCKED if value+BACKGROUND>=tau else 0.
    if mode=='decision':return BLOCKED if value+BACKGROUND>=1/(4*sender_weight) else 0.
    return None


def true_risk(q,patches):
    r=BACKGROUND
    for x,y,sx,sy,a in patches:r+=a*np.exp(-.5*((q[0]-x)/sx)**2-.5*((q[1]-y)/sy)**2)
    return r


def segment_cost(a,b,w,patches,belief):
    a,b=np.asarray(a,float),np.asarray(b,float);d=b-a;L=float(np.linalg.norm(d));cost=L*(1+4*w*BACKGROUND)
    if L<1e-12:return cost
    for k,p in enumerate(patches):
        if belief.get(k) is None:continue
        c=np.array(p[:2]);t=(c-a)@d/(L*L)
        if 0<t<1 and np.linalg.norm(a+t*d-c)<.85:cost+=4*w*belief[k]*p[2]*np.sqrt(2*np.pi)
    return cost


def route(nodes,edges,q,goal,w,patches,belief):
    graph={i:[] for i in range(len(nodes))}
    for i,j in edges:
        graph[i].append((j,segment_cost(nodes[i],nodes[j],w,patches,belief)));graph[j].append((i,segment_cost(nodes[j],nodes[i],w,patches,belief)))
    dist={};parent={};queue=[]
    for i,j in edges:
        a,b=nodes[i],nodes[j];t=np.clip((q-a)@(b-a)/((b-a)@(b-a)),0,1)
        if np.linalg.norm(q-(a+t*(b-a)))>1e-6:continue
        for k in (i,j):
            c=segment_cost(q,nodes[k],w,patches,belief)
            if c<dist.get(k,np.inf):dist[k]=c;parent[k]=None;heapq.heappush(queue,(c,k))
    while queue:
        c,i=heapq.heappop(queue)
        if c>dist[i]:continue
        if i==goal:break
        for j,e in graph[i]:
            if c+e<dist.get(j,np.inf)-1e-12:dist[j]=c+e;parent[j]=i;heapq.heappush(queue,(c+e,j))
    ids=[goal]
    while parent[ids[-1]] is not None:ids.append(parent[ids[-1]])
    return [q]+[nodes[k] for k in reversed(ids)]


def simulate(spec,mode,tau=None,speed=.3,sensor_radius=1.25,ds=.02):
    nodes=np.asarray(spec['network_nodes'],float);edges=spec['network_edges'];patches=[tuple(p) for p in spec['patches']]
    w=spec['risk_weights'];starts=spec['start_seconds'];n=len(starts)
    goal=[int(np.argmin(np.linalg.norm(nodes-np.asarray(g),axis=1))) for g in spec['goals']]
    q=[np.asarray(s[:2],float) for s in spec['initials']];belief=[{} for _ in range(n)];own=[set() for _ in range(n)]
    plan=[route(nodes,edges,q[i],goal[i],w[i],patches,belief[i]) for i in range(n)]
    cost=[dict(time=0.,risk=0.) for _ in range(n)];length=[0.]*n;done=[False]*n;t=0.;dt=ds/speed;inbox=[];events=[]
    while not all(done) and t<5000:
        for m in [m for m in inbox if m[0]<=t+1e-12]:
            _,j,k,value=m
            if k not in own[j] and belief[j].get(k)!=value:belief[j][k]=value;plan[j]=route(nodes,edges,q[j],goal[j],w[j],patches,belief[j]);events.append(dict(kind='received',time=t,agent=j,patch=k,value=value))
        inbox=[m for m in inbox if m[0]>t+1e-12]
        for i in range(n):
            if done[i] or t<starts[i]-1e-12:continue
            for k,p in enumerate(patches):
                if k not in own[i] and np.linalg.norm(q[i]-p[:2])<=sensor_radius+CORE*p[2]:
                    own[i].add(k);belief[i][k]=p[4];plan[i]=route(nodes,edges,q[i],goal[i],w[i],patches,belief[i]);events.append(dict(kind='sensed',time=t,agent=i,patch=k))
                    value=encode(mode,p[4],w[i],tau)
                    if value is not None:inbox+=[(max(t,starts[j]),j,k,value) for j in range(n) if j!=i]
            step=ds
            while step>1e-12 and len(plan[i])>1:
                seg=plan[i][1]-q[i];L=float(np.linalg.norm(seg));move=min(L,step)
                mid=q[i]+seg/max(L,1e-12)*move/2
                cost[i]['time']+=move/speed;cost[i]['risk']+=move/speed*4*w[i]*true_risk(mid,patches)
                if L<=step:q[i]=plan[i][1].copy();plan[i].pop(0)
                else:q[i]=q[i]+seg/L*step
                step-=move;length[i]+=move
            if len(plan[i])<=1:done[i]=True
        t+=dt
    return dict(agents=[dict(path=length[i],time=cost[i]['time'],risk=cost[i]['risk'],cost=cost[i]['time']+cost[i]['risk']) for i in range(n)],events=events)
