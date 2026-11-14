#!/usr/bin/env bash
# pages/build-ariglx-web.sh — the browser host the Pages site runs GRL-SNAM's Ariadne ports on:
# libcvc's ariglx-web (arigl-web with CPython 3.12, pycvc, pycvc_gl and numpy linked in), built
# from a libcvc source tree by its own recipe script, cvcpkg/recipes/ariglx-web/build-wasm.sh.
#
#   pages/build-ariglx-web.sh LIBCVC_SRC WORK OUT
#
# LIBCVC_SRC is a libcvc checkout at the pinned tag (pages.yml's LIBCVC_REF). WORK holds the
# tools, the Emscripten SDK, the wasm deps and the build tree. The page files land in
# OUT/share/ariglx-web/. It is the single-threaded build (CVC_PLATFORM=wasm), the lean tier (no
# vtk-python): GitHub Pages sends no COOP/COEP headers, so the threaded build cannot run there.
#
# Everything comes from cvcpkg's catalog, as libcvc's own CI builds it (its ariglx-web job): cmake,
# ninja, protoc and emsdk for the host, and the recipe's wasm deps. Nothing from apt.
set -euo pipefail

SRC="${1:?usage: build-ariglx-web.sh LIBCVC_SRC WORK OUT}"
WORK="${2:?usage: build-ariglx-web.sh LIBCVC_SRC WORK OUT}"
OUT="${3:?usage: build-ariglx-web.sh LIBCVC_SRC WORK OUT}"
SRC="$(cd "$SRC" && pwd)"
mkdir -p "$WORK" "$OUT"
WORK="$(cd "$WORK" && pwd)"
OUT="$(cd "$OUT" && pwd)"
RECIPE="$SRC/cvcpkg/recipes/ariglx-web"
[[ -f "$RECIPE/build-wasm.sh" ]] || { echo "build-ariglx-web: $SRC has no ariglx-web recipe" >&2; exit 1; }

command -v cvcpkg >/dev/null || {
  curl -fsSL https://cvcpkg.org/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

# Host tools: cmake and ninja for the build, protoc for the conference client's messages (the
# recipe's host tool; the wasm protobuf ships none), and the Emscripten SDK.
TOOLS="$WORK/tools"
[[ -x "$TOOLS/bin/cmake" ]] ||
  cvcpkg install cmake ninja protobuf --prefix "$TOOLS" --config release --link shared
export PATH="$TOOLS/bin:$PATH"
EMSDK="$WORK/emsdk"
[[ -f "$EMSDK/emsdk_env.sh" ]] ||
  cvcpkg install emsdk --platform linux --prefix "$EMSDK" \
    --config release --link shared --no-fallback-to-source

# The recipe's wasm deps, the newest of each. The configure checks python312's build itself
# (libcvc's src/ariadne_python/ariadne_python.cmake).
DEPS="$WORK/wasm-deps"
[[ -f "$DEPS/lib/libpython3.12.a" ]] ||
  cvcpkg install-deps "$RECIPE" --prefix "$DEPS" \
    --platform wasm --arch wasm32 --config release --link static --no-fallback-to-source
grep -m1 '^  version: ' "$DEPS/share/libcvc-deps/python312/manifest.yaml" || true

CVC_SOURCE_DIR="$SRC" CVC_BUILD_DIR="$WORK/build" CVC_INSTALL_DIR="$OUT" \
CVC_DEPS_PREFIX="$DEPS" CVC_EMSDK_DIR="$EMSDK" CVC_PLATFORM=wasm CVC_ARIGLX_WEB_TIER=lean \
  bash "$RECIPE/build-wasm.sh"
ls -la "$OUT/share/ariglx-web"
