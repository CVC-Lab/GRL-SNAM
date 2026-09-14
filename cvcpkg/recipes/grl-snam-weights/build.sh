#!/bin/bash
# Stage the committed base GRL-SNAM nav weights into the prefix. Data-only
# (source.type: none), so the payload comes from the recipe directory
# (CVC_RECIPE_DIR/payload), not a fetched source tree.
set -euo pipefail

dst="$CVC_INSTALL_DIR/share/grl-snam-weights"
mkdir -p "$dst"
cp "$CVC_RECIPE_DIR/payload/coef_sdf.cvcnav" "$dst/coef_sdf.cvcnav"
cp "$CVC_RECIPE_DIR/payload/coef_sdf.pt" "$dst/coef_sdf.pt"
cp "$CVC_RECIPE_DIR/payload/PROVENANCE.md" "$dst/PROVENANCE.md"

# Guard: the native blob must be present, non-empty, and carry the CVNV magic the
# C++ cvc::nav forward requires.
test -s "$dst/coef_sdf.cvcnav" || { echo "grl-snam-weights: coef_sdf.cvcnav missing/empty" >&2; exit 1; }
test -s "$dst/coef_sdf.pt" || { echo "grl-snam-weights: coef_sdf.pt missing/empty" >&2; exit 1; }
magic=$(head -c 4 "$dst/coef_sdf.cvcnav")
test "$magic" = "CVNV" || { echo "grl-snam-weights: bad blob magic '$magic'" >&2; exit 1; }
echo "grl-snam-weights: staged coef_sdf.cvcnav ($(wc -c < "$dst/coef_sdf.cvcnav") B) + coef_sdf.pt -> $dst"
