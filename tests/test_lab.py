"""Tests for grl_snam.demos.lab — the analytic lab demo.

The pure geometry (terrain height, agent loop, track) always runs; the end-to-end
scene build is skipped where the compiled pycvc / pycvc_gl bindings aren't importable.
"""

import pytest

from grl_snam.demos import lab


def test_terrain_height_peaks_at_centre():
    assert lab.terrain_height(0.0, 0.0) > lab.terrain_height(95.0, 95.0)


def test_track_is_closed_and_draped():
    assert len(lab.TRACK) == lab._N_TRACK + 1
    assert lab.TRACK[0] == pytest.approx(lab.TRACK[-1], abs=1e-9)  # closed loop
    x, y, z = lab.TRACK[0]
    assert abs(z - (lab.terrain_height(x, y) + lab.AGENT_LIFT)) < 1e-6


def test_agent_position_wraps_one_lap():
    a = lab.agent_position(0.0)
    b = lab.agent_position(lab.LOOP_SECONDS)  # exactly one lap -> back to start
    assert all(abs(u - v) < 1e-6 for u, v in zip(a, b))


def test_terrain_heights_grid_shape():
    g = lab._terrain_heights(8)
    assert len(g) == 8 and all(len(row) == 8 for row in g)


# ── end-to-end (requires the compiled bindings) ─────────────────────────
pytest.importorskip("pycvc", reason="pycvc bindings not installed")
pytest.importorskip("pycvc_gl", reason="pycvc_gl bindings not installed")


def test_lab_demo_builds_a_scene(tmp_path):
    out = lab.run_standalone(png=str(tmp_path / "lab.png"))
    assert out.num_nodes() >= 3
