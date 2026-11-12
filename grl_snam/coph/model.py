"""Differentiable models in float64.

Exact held-force flow/work adapted from CoPH-Sense belief_planning.port_step,
complete_potential.energy_balance and CoPH-v2 navigation.plant_step.
Unlike the old NavigationPotential.force, position is NOT detached during replay.
"""
import torch

DTYPE = torch.float64

def tensor(x):
    return torch.as_tensor(x, dtype=DTYPE)


def flow(x, u, damping, dt):
    d = torch.as_tensor(damping, dtype=x.dtype, device=x.device)
    a = torch.exp(-d * dt)
    integral = -torch.expm1(-d * dt) / d
    q, v = x[..., :2], x[..., 2:]
    return torch.cat((q + integral * v + (dt - integral) * u / d,
                      a * v + integral * u), -1)


def balance(x, u, damping, dt):
    d = torch.as_tensor(damping, dtype=x.dtype)
    steady = u / d
    transient = x[2:] - steady
    first = -torch.expm1(-d * dt) / d
    second = -torch.expm1(-2 * d * dt) / (2 * d)
    after = flow(x, u, d, dt)
    heat = d * ((steady * steady).sum() * dt + 2 * (steady * transient).sum() * first
                + (transient * transient).sum() * second)
    work = (u * (after[:2] - x[:2])).sum()
    kinetic = (after[2:].square().sum() - x[2:].square().sum()) / 2
    return dict(work=float(work), heat=float(heat), kinetic_change=float(kinetic),
                residual=float(kinetic - work + heat))


def features(q, centers, width):
    return torch.exp(-.5 * ((q[..., None, :] - centers) / width).square().sum(-1))


def compact_features(q, centers, radius):
    """Wendland C2 kernels: exactly zero outside the declared correction radius."""
    squared = ((q[..., None, :] - centers)/radius).square().sum(-1)
    distance = squared.clamp_min(1e-24).sqrt()
    # Expanded polynomial avoids the norm's undefined second derivative at r=0.
    value = 1-10*squared+20*squared*distance-15*squared.square()+4*squared.square()*distance
    return torch.where(squared<1,value,torch.zeros_like(value))


def risk(q, weights, centers, width, residual=None):
    # Smooth positive prediction. Coefficients are fit to inverse-softplus readings.
    raw = features(q, centers, width) @ weights
    if residual is not None and residual.get('coefficients'):
        raw = raw + compact_features(q,tensor(residual['centers']),residual['radius']) @ tensor(residual['coefficients'])
    return torch.nn.functional.softplus(raw) / 4


def shaping_parameters(weights, width, obstacles, cfg, residual=None):
    """Uniform bound, independent of position and candidate policy.

    Field/geometry are frozen during replay optimization; using policy upper bounds
    keeps alpha identical for incumbent, gradient and validation evaluations.
    """
    risk_bound = float(torch.exp(tensor(-.5)) * weights.detach().abs().sum() / (4*width))
    if residual is not None and residual.get('coefficients'):
        risk_bound += sum(abs(c) for c in residual['coefficients'])*135/(256*residual['radius'])
    barrier_bound = sum(float(torch.nn.functional.softplus(tensor(6*(o[2]+.47))))/3 for o in obstacles)
    bound = cfg.policy_bounds[1]*(risk_bound+barrier_bound)
    alpha = 1. if bound == 0 else min(1.,2*cfg.shaping_eta*cfg.shaping_k*cfg.shaping_ell/bound)
    return dict(bound=bound, alpha=alpha, k=cfg.shaping_k, ell=cfg.shaping_ell, eta=cfg.shaping_eta)


def potential(q, target, weights, centers, width, policy, obstacles, cfg=None, residual=None):
    squared = (q-target).square().sum()
    shaping = policy[0]*risk(q,weights,centers,width,residual)
    if obstacles:
        if any(len(o)==5 for o in obstacles):
            from .shapes import surface_distance
            clearance = surface_distance(q,obstacles)-.12
        else:
            obs = tensor(obstacles)
            clearance = torch.linalg.vector_norm(q-obs[:,:2],dim=-1)-obs[:,2]-.12
        shaping = shaping + policy[1]*torch.nn.functional.softplus((.35-clearance)*6).square().sum()/36
    if cfg is not None and cfg.shaping_mode=='progress':
        params=shaping_parameters(weights,width,obstacles,cfg,residual)
        return .5*params['k']*squared + params['alpha']*squared/(squared+params['ell']**2)*shaping
    return .5*squared+shaping


def command(x, target, weights, centers, width, policy, obstacles, limit=3., cfg=None, residual=None):
    q = x[:2]
    if not q.requires_grad:
        q = q.clone().requires_grad_(True)
    energy = potential(q, target, weights, centers, width, policy, obstacles, cfg,residual)
    grad = torch.autograd.grad(energy, q, create_graph=True)[0]
    proposed = -grad - 1.4 * x[2:]
    applied = proposed * torch.clamp(limit / torch.linalg.vector_norm(proposed).clamp_min(1e-12), max=1.)
    return applied, proposed
