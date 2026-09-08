#!/usr/bin/env bash
# cvcpkg/recipes/grl-snam-cp31X/test.sh — bundle self-test for a grl-snam
# per-interpreter column package.
#
# Column-generic: byte-identical across grl-snam-cp311/cp312/cp313 (keep the
# copies in lockstep), parameterized off CVC_PYTHON_INTERPRETER (the recipe's
# python.interpreter) when the packager exports it, else off the one python3.X
# the column's deps closure staged into the prefix (see the resolve block below).
#
# Invoked by the packager after build.sh installs grl_snam into
# $CVC_INSTALL_DIR. It runs under the build prefix, where:
#   * depends.build staged the test/lint tools (pytest-cp31X, ruff,
#     black-cp31X), and
#   * depends.runtime staged the runtime closure (python31X, pycvc-gl-cp31X,
#     numpy-cp31X, torch-cp31X) into $CVC_DEPS_PREFIX's own interpreter.
# Non-zero exit => the bundle is broken and must not ship.
#
# This is the cvcpkg-native counterpart to the GRL-SNAM CI test job. The v0.1.0
# sdist ships no tests/ dir, so we run `pytest -q` against a bundle smoke test
# that imports the just-built package (proving the runner works and the runtime
# closure resolves); when a future release carries its suite, we run that
# instead. The FULL lint+test over the working tree runs in CI, which installs
# this exact closure via `cvcpkg install-deps` — once per interpreter column.
set -euo pipefail

: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
: "${CVC_DEPS_PREFIX:?CVC_DEPS_PREFIX must be set}"
# NOTE: CVC_SOURCE_DIR is intentionally NOT required — the bundle smoke test below
# imports the INSTALLED package, not the source tree (pack does not export
# CVC_SOURCE_DIR to the test phase).

# Column interpreter. `cvcpkg pack` exports CVC_PYTHON_INTERPRETER to the BUILD
# phase but NOT to this TEST phase, so the old env fallback to python311 silently
# ran the cp312/cp313 tests against a python3.11 that is NOT in their deps prefix
# (only cp311 happened to match) -> "FAIL: no python3.11 in deps prefix". Resolve
# it robustly: honour CVC_PYTHON_INTERPRETER when present, else discover the one
# python3.X the column's deps closure staged into the prefix.
if [ -n "${CVC_PYTHON_INTERPRETER:-}" ]; then
  digits="${CVC_PYTHON_INTERPRETER#python}" # python312 -> 312
  ver="${digits:0:1}.${digits:1}"           # 312 -> 3.12
  PY="${CVC_DEPS_PREFIX}/bin/python${ver}"
else
  PY="$(ls "${CVC_DEPS_PREFIX}"/bin/python3.1? 2>/dev/null | sort -V | tail -1 || true)"
  ver="$(basename "${PY:-python?}" | sed 's/^python//')"
fi
[ -n "${PY:-}" ] && [ -x "${PY}" ] || { echo "FAIL: no python3.X in deps prefix (${CVC_DEPS_PREFIX})"; exit 1; }

# Make the just-built grl_snam importable alongside the deps
# already on the prefix interpreter's path (pycvc-gl, numpy, torch).
SP="$(find "${CVC_INSTALL_DIR}" -maxdepth 3 -type d -name site-packages -print -quit || true)"
export PYTHONPATH="${SP:-}${PYTHONPATH:+:${PYTHONPATH}}"

echo "-- grl-snam bundle self-test (python${ver}) --"
echo "-- test/lint tools staged via depends.build --"
"${PY}" -m pytest --version
"${PY}" -m black --version
# ruff is a native binary in the deps prefix bin/ (no `python -m` needed).
"${CVC_DEPS_PREFIX}/bin/ruff" --version

echo "-- pytest -q (bundle smoke test) --"
# This is the BUNDLE self-test: prove the just-built package imports against the
# staged runtime closure (runner + deps resolve). The FULL lint+test over the
# working tree is the dev CI's job (ci.yml) — running it here would fail the pack
# on GPU/GL/headless-only tests the shipped bundle does not depend on. Now that
# source is the vendored checkout (which carries tests/), we must NOT branch on
# tests/ presence — always run the smoke test.
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
cat > "${TMP}/test_bundle_smoke.py" <<'EOF'
def test_grl_snam_imports():
    import grl_snam
    assert isinstance(grl_snam.__version__, str)
EOF
"${PY}" -m pytest -q "${TMP}"

echo "-- grl-snam recipe test passed --"
