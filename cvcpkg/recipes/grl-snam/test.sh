#!/usr/bin/env bash
# cvcpkg/recipes/grl-snam/test.sh — bundle self-test for the grl-snam package.
#
# Invoked by the packager after build.sh installs grl_snam / grl_snam_lab into
# $CVC_INSTALL_DIR. It runs under the build prefix, where:
#   * depends.build staged the test/lint tools (pytest, ruff, black), and
#   * depends.runtime staged the runtime closure (python311, pycvc-gl, numpy,
#     torch) into $CVC_DEPS_PREFIX's own interpreter.
# Non-zero exit => the bundle is broken and must not ship.
#
# This is the cvcpkg-native counterpart to the GRL-SNAM CI test job. The v0.1.0
# sdist ships no tests/ dir, so we run `pytest -q` against a bundle smoke test
# that imports the just-built package (proving the runner works and the runtime
# closure resolves); when a future release carries its suite, we run that
# instead. The FULL lint+test over the working tree runs in CI, which installs
# this exact closure via `cvcpkg install-deps`.
set -euo pipefail

: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"
: "${CVC_DEPS_PREFIX:?CVC_DEPS_PREFIX must be set}"

PY="${CVC_DEPS_PREFIX}/bin/python3.11"
[ -x "${PY}" ] || { echo "FAIL: no python3.11 in deps prefix (${CVC_DEPS_PREFIX})"; exit 1; }

# Make the just-built grl_snam / grl_snam_lab importable alongside the deps
# already on the prefix interpreter's path (pycvc-gl, numpy, torch).
SP="$(find "${CVC_INSTALL_DIR}" -maxdepth 3 -type d -name site-packages -print -quit || true)"
export PYTHONPATH="${SP:-}${PYTHONPATH:+:${PYTHONPATH}}"

echo "-- grl-snam bundle self-test --"
echo "-- test/lint tools staged via depends.build --"
"${PY}" -m pytest --version
"${PY}" -m black --version
# ruff is a native binary in the deps prefix bin/ (no `python -m` needed).
"${CVC_DEPS_PREFIX}/bin/ruff" --version

echo "-- pytest -q --"
if [ -d "${CVC_SOURCE_DIR}/tests" ]; then
    # A release that ships its suite: run it against the built package + deps.
    "${PY}" -m pytest -q "${CVC_SOURCE_DIR}/tests"
else
    # v0.1.0 sdist ships no tests/: exercise the runner against a bundle smoke
    # test that imports the built package (proves closure + runner).
    TMP="$(mktemp -d)"
    trap 'rm -rf "${TMP}"' EXIT
    cat > "${TMP}/test_bundle_smoke.py" <<'EOF'
def test_grl_snam_imports():
    import grl_snam
    assert isinstance(grl_snam.__version__, str)
EOF
    "${PY}" -m pytest -q "${TMP}"
fi

echo "-- grl-snam recipe test passed --"
