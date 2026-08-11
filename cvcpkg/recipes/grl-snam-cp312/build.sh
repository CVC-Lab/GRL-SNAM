#!/usr/bin/env bash
# recipes/grl-snam-cp31X/build.sh — install the GRL-SNAM pure-Python package
# (source.type python_sdist) into the column interpreter's site-packages.
#
# This script is column-generic: it is byte-identical across the
# grl-snam-cp311/cp312/cp313 recipe dirs (keep the copies in lockstep) and is
# parameterized entirely off CVC_PYTHON_INTERPRETER, which the builder exports
# from the recipe's python.interpreter — the same pattern as libcvc-deps'
# recipes/_common/python-wheel.sh.
#
# cvcpkg fetches the sdist and verifies its sha256 (source.type python_sdist)
# and extracts it to $CVC_SOURCE_DIR before this runs.  GRL-SNAM is pure Python
# (py3-none-any), so the build produces a noarch wheel and the install below is
# fully offline (--no-index).
set -euo pipefail

: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"
: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
: "${CVC_DEPS_PREFIX:?CVC_DEPS_PREFIX must be set}"

# Resolve the target interpreter inside the prefix from the recipe's
# python.interpreter (e.g. python312 -> python3.12), falling back to python311
# (the original single-column recipe's interpreter) if the builder did not
# export it.  We install into that interpreter's own site-packages so a single
# activatable prefix carries both libcvc's pycvc bindings and the importable
# grl_snam package.
interp="${CVC_PYTHON_INTERPRETER:-python311}"
digits="${interp#python}"            # python311 -> 311
ver="${digits:0:1}.${digits:1}"      # 311 -> 3.11
py="${CVC_DEPS_PREFIX}/bin/python${ver}"
if [ ! -x "${py}" ]; then
  echo "build.sh: interpreter not found: ${py}" >&2
  echo "  (does this recipe depend on ${interp}?)" >&2
  exit 1
fi

echo "installing grl_snam into ${CVC_INSTALL_DIR} using ${py}"

# --no-deps:            torch/numpy/matplotlib/imageio and libcvc are resolved
#                       by cvcpkg's depends graph, not by pip going behind
#                       cvcpkg's back and pulling unpinned copies.
# --no-build-isolation: build against the prefix (python.build_isolation=false)
#                       using the pinned poetry-core backend (python.build_requires).
# --no-index:           the sdist is already on disk and pinned; forbid any
#                       network resolution, which is what makes air-gapped
#                       installs work.
# --no-compile:         ship .py only; no host-specific .pyc in the bundle.
# --no-build-isolation means pip does NOT download the PEP-517 backend into a
# throwaway venv: poetry-core must ALREADY be importable by ${py}.  It is a
# depends.build edge (poetry-core-cp31X), extracted into the deps prefix on
# the remote builder; a local build with build/deps prefix SEPARATION stages
# it in CVC_BUILD_PREFIX instead, so bridge that site-packages onto
# PYTHONPATH too (same pattern as libcvc-deps' generated backend recipes).
if [ -n "${CVC_BUILD_PREFIX:-}" ]; then
  _bp_site="${CVC_BUILD_PREFIX}/lib/python${ver}/site-packages"
  [ -d "${_bp_site}" ] && export PYTHONPATH="${_bp_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi

"${py}" -m pip install \
  --no-deps \
  --no-build-isolation \
  --no-index \
  --no-compile \
  --prefix "${CVC_INSTALL_DIR}" \
  "${CVC_SOURCE_DIR}"

# Smoke-test: grl_snam must import under the target interpreter.  __init__ does
# only a lightweight importlib.metadata version() lookup at import time (torch
# and the flat research modules are imported lazily via __getattr__), so this
# check does not require the heavy runtime deps to be present.
libdir="$(find "${CVC_INSTALL_DIR}" -maxdepth 3 -type d -name 'site-packages' -print -quit)"
if [ -z "${libdir}" ]; then
  echo "build.sh: no site-packages found under ${CVC_INSTALL_DIR}" >&2
  exit 1
fi
# Assert only what the PINNED v0.1.0 sdist actually ships: grl_snam +
# grl_snam_lab (Lab/terrain_mesh).  The selftest module, the lab run_* entry
# points and the grl-snam-selftest / grl-snam-lab-demo console scripts all
# POSTDATE the v0.1.0 release (the sdist declares no [project.scripts] at
# all) — asserting them here can never pass until the recipes pin a newer
# release, at which point this smoke should grow back with it.
PYTHONPATH="${libdir}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" -c "
import grl_snam, grl_snam_lab
for fn in ('Lab', 'terrain_mesh'):
    assert hasattr(grl_snam_lab, fn), 'grl_snam_lab missing ' + fn
print('grl_snam', getattr(grl_snam, '__version__', '(no __version__)'),
      '| grl_snam_lab', grl_snam_lab.__version__, 'from', grl_snam_lab.__file__)
"
