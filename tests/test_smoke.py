"""Smoke tests confirming the `grl_snam` package is importable and
exposes its documented public API.  These tests avoid heavy CUDA
initialisation paths by deferring imports until needed.
"""

from __future__ import annotations


def test_package_imports():
    import grl_snam

    assert hasattr(grl_snam, "__version__")
    assert isinstance(grl_snam.__version__, str)


def test_public_api_attribute_access():
    """`grl_snam.CoefEnergyNet` must resolve via lazy `__getattr__`."""
    import grl_snam

    cls = grl_snam.CoefEnergyNet
    assert cls.__name__ == "CoefEnergyNet"


def test_dynamics_reexports():
    from grl_snam.dynamics import integrate_surrogate_v2, multi_start_penalty

    assert callable(integrate_surrogate_v2)
    assert callable(multi_start_penalty)
