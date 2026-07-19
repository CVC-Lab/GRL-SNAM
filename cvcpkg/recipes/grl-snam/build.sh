#!/usr/bin/env bash
# recipes/grl-snam/build.sh — install the GRL-SNAM pure-Python package
# (source.type python_sdist) into the prefix interpreter's site-packages.
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
# python.interpreter (e.g. python311 -> python3.11).  We install into that
# interpreter's own site-packages so a single activatable prefix carries both
# libcvc's pycvc bindings and the importable grl_snam package.
interp="${CVC_PYTHON_INTERPRETER:?CVC_PYTHON_INTERPRETER must be set (recipe python.interpreter)}"
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
PYTHONPATH="${libdir}${PYTHONPATH:+:${PYTHONPATH}}" "${py}" -c "
import grl_snam
print('grl_snam', getattr(grl_snam, '__version__', '(no __version__)'), 'from', grl_snam.__file__)
"
