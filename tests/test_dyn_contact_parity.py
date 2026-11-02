"""Parity for the dynamic plant's compliant contact, and its step under the contact's wrench.

libcvc's ``dyn_footprint_points``, ``dyn_contact_field``, ``dyn_contact_pair`` and
``dyn_plant_books`` under an external wrench (``inc/cvc/nav/detail/dyn_contact.h``,
``dyn_bike.h``) against the torch twins in :mod:`grl_snam.dyn_contact` and
:mod:`grl_snam.dyn_bike`. The reference is libcvc's ``src/cvc/tests/data/dyn_contact_golden.json``
(copied to ``tests/fixtures``; written by libcvc's ``DynContact.Golden`` with
``CVC_DYN_CONTACT_GOLDEN_WRITE=1``): the Humvee's footprint points; its contact (the barrier's and
the damper's wrenches, the barrier's energy, the dissipation, the points touching) against an
analytic wall and a pillar at five poses and against another Humvee at three; and one
``dyn_plant_books`` step under a wrench, with the port's energy. That step's plant is
``DynBikePH.Golden``'s Humvee, its parameters recorded with the step.

Tier: DOUBLE contract. libcvc's contact is double, and its plant runs in double from the same float
parameters, so every value agrees to 1e-9 relative. libcvc does not differentiate the contact;
autograd does here, so the structural gates hold the barrier to minus its energy's gradient in the
pose exactly (libcvc's own gate uses central differences) and a pair's forces to balance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from grl_snam.dyn_bike import DynBikeParams, PlantLimits, dyn_plant, dyn_plant_books  # noqa: E402
from grl_snam.dyn_contact import (  # noqa: E402
    DynContactParams,
    DynFootprint,
    PillarField,
    WallField,
    dyn_contact_field,
    dyn_contact_pair,
    dyn_footprint_points,
    dyn_rounded_box_sdf,
)

GOLDEN = Path(__file__).parent / "fixtures" / "dyn_contact_golden.json"
RTOL = 1e-9
BOOKS = ("dH", "dissipated", "supplied", "floor", "grade", "limiter", "residual", "port")
GROUPS = ("points", "field", "pair", "next", "books", "free")


def _golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["format"] == "cvc-dyn-contact-golden-1"
    return data


def _rel(a: float, b: float) -> float:
    """|a - b| relative to the larger magnitude (at least 1)."""
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _setup(data):
    p = DynContactParams(**data["params"])
    f = DynFootprint(**data["footprint"])
    return p, f, dyn_footprint_points(f, p)


def _field(data, name):
    return WallField(data["wall_x0"]) if name == "wall" else PillarField(*data["pillar"])


def _body(v, grad=False):
    return torch.tensor(v, dtype=torch.float64, requires_grad=grad)


def _contact_rows(label, c, want):
    """Torch's contact beside libcvc's, value by value."""
    rows = [(f"{label} Wc[{i}]", c.Wc[i].item(), want["Wc"][i]) for i in range(3)]
    rows += [(f"{label} Wd[{i}]", c.Wd[i].item(), want["Wd"][i]) for i in range(3)]
    rows += [(f"{label} {k}", getattr(c, k).item(), want[k]) for k in ("energy", "damping")]
    rows.append((f"{label} touching", float(c.touching.item()), float(want["touching"])))
    return rows


def _step(data):
    """The golden's plant step under its wrench: (next, books, free, whether the books' step is
    the plant's to the last bit)."""
    st = data["step"]
    p = DynBikeParams(**st["params"])
    lim = st["limits"]
    limits = PlantLimits(
        L=lim["L"], vmax=lim["vmax"], hdt=lim["hdt"], allow_reverse=bool(lim["allow_reverse"])
    )
    s = torch.tensor(st["state"], dtype=torch.float64)
    dcmd = torch.tensor(st["delta_cmd"], dtype=torch.float64)
    acmd = torch.tensor(st["a_cmd"], dtype=torch.float64)
    W = torch.tensor(st["wrench"], dtype=torch.float64)
    nxt, books = dyn_plant_books(p, limits, s, dcmd, acmd, st["grade"], wrench=W)
    plant, free = dyn_plant(p, limits, s, dcmd, acmd, st["grade"], with_free=True, wrench=W)
    return plant, books, free, torch.equal(nxt, plant)


def _compare(data):
    """Torch's value beside libcvc's for everything the golden records, by group."""
    p, f, pts = _setup(data)
    out = {g: [] for g in GROUPS}
    assert len(pts) == len(data["points"])
    for k, (got, want) in enumerate(zip(pts, data["points"])):
        out["points"] += [(f"[{k}].x", got[0], want[0]), (f"[{k}].y", got[1], want[1])]
    for k, fc in enumerate(data["field_cases"]):
        c = dyn_contact_field(p, pts, _field(data, fc["field"]), _body(fc["body"]))
        out["field"] += _contact_rows(f"{fc['field']} {k}", c, fc["contact"])
    for k, pc in enumerate(data["pair_cases"]):
        ca, cb = dyn_contact_pair(p, pts, f, _body(pc["a"]), pts, f, _body(pc["b"]))
        out["pair"] += _contact_rows(f"{k} on a", ca, pc["on_a"])
        out["pair"] += _contact_rows(f"{k} on b", cb, pc["on_b"])
    st = data["step"]
    nxt, books, free, _ = _step(data)
    order = ("x", "y", "psi", "vx", "vy", "r", "delta", "a")
    out["next"] = [(k, nxt[i].item(), st["next"][i]) for i, k in enumerate(order)]
    out["books"] = [(k, getattr(books, k).item(), st["books"][k]) for k in BOOKS]
    out["free"] = [(k, free[i].item(), st["free"][i]) for i, k in enumerate(("vx", "vy", "r"))]
    return out


@pytest.mark.parametrize("group", GROUPS)
def test_matches_libcvc(group):
    for name, got, want in _compare(_golden())[group]:
        assert _rel(got, want) <= RTOL, f"{group} {name}: torch {got!r}, libcvc {want!r}"


def test_the_books_step_is_the_plants():
    assert _step(_golden())[3], "the books' step under a wrench is not dyn_plant's"


def test_worst_difference_per_group():
    worst = {g: (0.0, "") for g in GROUPS}
    out = _compare(_golden())
    for g in GROUPS:
        for name, got, want in out[g]:
            if _rel(got, want) > worst[g][0]:
                worst[g] = (_rel(got, want), name)
    print("torch twin against libcvc (max relative, where):")
    for g in GROUPS:
        print(f"  {g:6s} {worst[g][0]:.2e}  {worst[g][1]}")
    for g in GROUPS:
        assert worst[g][0] <= RTOL, (g, worst[g])


def _minus_gradient(energy, body):
    """-dE/d(pose) as body wrenches ``(..., 3)``: the world force -(dE/dx, dE/dy) in body axes,
    and -dE/dpsi."""
    (g,) = torch.autograd.grad(energy.sum(), body)
    fx, fy = -g[..., 0], -g[..., 1]
    c, s = torch.cos(body[..., 2]), torch.sin(body[..., 2])
    return torch.stack((c * fx + s * fy, -s * fx + c * fy, -g[..., 2]), dim=-1).detach()


def _uniform(gen, n):
    return 2.0 * torch.rand(n, generator=gen, dtype=torch.float64) - 1.0


def _skin_poses(n, at_wall, gen):
    """libcvc's DynContact gate's poses: within the skin of a wall at x = 0 or a pillar of radius
    1.5 at the origin with some point, at any heading and twist."""
    psi, vx, vy, r = (
        math.pi * _uniform(gen, n),
        5.0 * _uniform(gen, n),
        _uniform(gen, n),
        0.5 * _uniform(gen, n),
    )
    if at_wall:
        x, y = -(1.6 + 1.0 * _uniform(gen, n)), 3.0 * _uniform(gen, n)
    else:
        ang, d = math.pi * _uniform(gen, n), 1.5 + 1.6 + 0.8 * _uniform(gen, n)
        x, y = d * torch.cos(ang), d * torch.sin(ang)
    return torch.stack((x, y, psi, vx, vy, r), dim=-1)


def test_the_barrier_is_minus_its_energys_gradient_and_the_damper_dissipates():
    data = _golden()
    p, _, pts = _setup(data)
    gen = torch.Generator().manual_seed(3)
    cases = [(_field(data, fc["field"]), _body([fc["body"]])) for fc in data["field_cases"]]
    cases += [(WallField(0.0), _skin_poses(100, True, gen))]
    cases += [(PillarField(0.0, 0.0, 1.5), _skin_poses(100, False, gen))]
    worst, touching = 0.0, 0
    for field, bodies in cases:
        body = bodies.clone().requires_grad_(True)
        c = dyn_contact_field(p, pts, field, body)
        G = _minus_gradient(c.energy, body)
        Wc, Wd, damping = c.Wc.detach(), c.Wd.detach(), c.damping.detach()
        scale = 1.0 + Wc.abs().sum(-1)
        worst = max(worst, ((Wc - G).abs().max(-1).values / scale).max().item())
        assert torch.all((Wc - G).abs().max(-1).values <= RTOL * scale), (Wc, G)
        # The damper: power -v^T R_c v.
        power = (Wd * body.detach()[..., 3:]).sum(-1)
        assert torch.all(damping >= 0.0)
        assert torch.all((power + damping).abs() <= RTOL * (1.0 + damping)), (power, damping)
        touching += int((c.touching > 0).sum())
    print(f"{touching} poses in the skin: the barrier's wrench within {worst:.2e} of -dPhi/dpose")
    assert touching > 50


def test_a_pair_keeps_its_momentum():
    data = _golden()
    p, f, pts = _setup(data)
    gen = torch.Generator().manual_seed(5)
    n = 400
    zero = torch.zeros(n, dtype=torch.float64)
    a = torch.stack(
        (
            zero,
            zero,
            math.pi * _uniform(gen, n),
            4.0 * _uniform(gen, n),
            zero,
            0.3 * _uniform(gen, n),
        ),
        dim=-1,
    )
    b = torch.stack(
        (
            4.5 * _uniform(gen, n),
            4.5 * _uniform(gen, n),
            math.pi * _uniform(gen, n),
            4.0 * _uniform(gen, n),
            _uniform(gen, n),
            zero,
        ),
        dim=-1,
    )
    a = torch.cat((a, torch.tensor([pc["a"] for pc in data["pair_cases"]], dtype=torch.float64)))
    b = torch.cat((b, torch.tensor([pc["b"] for pc in data["pair_cases"]], dtype=torch.float64)))
    a = a.requires_grad_(True)
    ca, cb = dyn_contact_pair(p, pts, f, a, pts, f, b)
    G = _minus_gradient(ca.energy + cb.energy, a)
    a, keep = a.detach(), ca.touching > 0

    def world(q, W):
        c, s = torch.cos(q[..., 2]), torch.sin(q[..., 2])
        Fx, Fy = c * W[..., 0] - s * W[..., 1], s * W[..., 0] + c * W[..., 1]
        return Fx, Fy, W[..., 2] + q[..., 0] * Fy - q[..., 1] * Fx

    fax, fay, ma = world(a, ca.wrench().detach())
    fbx, fby, mb = world(b, cb.wrench().detach())
    scale = 1.0 + torch.hypot(fax, fay)
    assert torch.all((fax + fbx).abs() <= RTOL * scale)
    assert torch.all((fay + fby).abs() <= RTOL * scale)
    assert torch.all((ma + mb).abs() <= RTOL * scale * (1.0 + torch.hypot(b[..., 0], b[..., 1])))
    # Each vehicle's barrier is minus the pair's energy's gradient in its pose.
    Wc = ca.Wc.detach()
    sc = 1.0 + Wc.abs().sum(-1)
    worst = ((Wc - G).abs().max(-1).values / sc)[keep].max().item()
    assert worst <= RTOL, worst
    # The damper only dissipates, reading the relative velocity.
    power = (ca.Wd.detach() * a[..., 3:]).sum(-1) + (cb.Wd.detach() * b[..., 3:]).sum(-1)
    damping = (ca.damping + cb.damping).detach()
    assert torch.all((power + damping).abs() <= RTOL * (1.0 + damping))
    print(
        f"{int(keep.sum())} touching pairs: forces and moments balance; the barrier within "
        f"{worst:.2e} of -dPhi/dpose"
    )
    assert int(keep.sum()) > 50


def test_the_corner_radius_is_held_to_the_outline():
    # As libcvc's dyn_rounded_box_sdf: rho is held to [0, min(half width, half length)].
    f = DynFootprint(**_golden()["footprint"])
    gen = torch.Generator().manual_seed(7)
    lx, ly = 4.0 * _uniform(gen, 64), 2.5 * _uniform(gen, 64)
    for rho, held in ((5.0, f.half_width), (-0.2, 0.0)):
        got, want = dyn_rounded_box_sdf(f, rho, lx, ly), dyn_rounded_box_sdf(f, held, lx, ly)
        for g, w in zip(got, want):
            assert torch.equal(g, w), rho


def test_no_wrench_and_a_zero_wrench_step_alike():
    # The port is opt-in: without a wrench the step and the books are dyn_plant's as before, and a
    # zero wrench adds exactly nothing.
    data = _golden()
    st = data["step"]
    p = DynBikeParams(**st["params"])
    limits = PlantLimits(L=st["limits"]["L"], vmax=st["limits"]["vmax"], hdt=st["limits"]["hdt"])
    s = torch.tensor(st["state"], dtype=torch.float64)
    dcmd = torch.tensor(st["delta_cmd"], dtype=torch.float64)
    acmd = torch.tensor(st["a_cmd"], dtype=torch.float64)
    nxt, books = dyn_plant_books(p, limits, s, dcmd, acmd, st["grade"])
    nxt0, books0 = dyn_plant_books(
        p, limits, s, dcmd, acmd, st["grade"], wrench=torch.zeros(3, dtype=torch.float64)
    )
    assert books.port is None and books0.port.item() == 0.0
    assert torch.equal(nxt, nxt0)
    for name in BOOKS[:-1]:
        assert torch.equal(getattr(books, name), getattr(books0, name)), name
