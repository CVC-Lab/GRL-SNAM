"""M10 guardrail: the IPC barrier derivative must BE the derivative of the barrier.

`ipc_piecewise` returned a `dbdd` that differed from d(b)/dd by exactly (d - d_hat) + 1,
an attraction band over ~(0.39-0.80)*d_hat where the obstacle barrier pulled the agent
toward the obstacle. This fails if that regresses.
"""

import torch

from train_coef_energy import ipc_piecewise


def test_dbdd_is_the_autograd_derivative_of_b():
    dh = torch.tensor(2.0, dtype=torch.float64)
    d = torch.linspace(0.05, float(dh) - 1e-6, 4000, dtype=torch.float64, requires_grad=True)
    b, dbdd = ipc_piecewise(d, dh)
    (g,) = torch.autograd.grad(b.sum(), d)
    assert torch.allclose(
        dbdd, g, atol=1e-9
    ), f"dbdd is not b'(d); max err {(dbdd - g).abs().max().item()}"


def test_no_attraction_band_barrier_is_purely_repulsive():
    dh = torch.tensor(2.0, dtype=torch.float64)
    d = torch.linspace(0.01, float(dh) - 1e-6, 4000, dtype=torch.float64)
    _, dbdd = ipc_piecewise(d, dh)
    assert (
        dbdd <= 1e-9
    ).all(), f"{int((dbdd > 1e-9).sum())} points form an attraction band (dbdd > 0)"
