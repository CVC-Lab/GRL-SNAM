#!/bin/bash
# Stage the learning check's checkpoints into the prefix. Data-only (source.type: none), and the
# .pt files are not committed, so they come from GRL_SNAM_LEARNING_RUNS: a directory holding
# retrain/, fixed/retrain/ and world-units/retrain/, each with seed{0..4}/ and in it the
# coef_sdf.step*.pt its run saved
# (`grl-snam train ... --checkpoint-every 100`). Every file SHA256SUMS lists is copied and checked
# against it; nothing else is staged.
set -euo pipefail

src="${GRL_SNAM_LEARNING_RUNS:?set GRL_SNAM_LEARNING_RUNS to the directory with retrain/, fixed/retrain/, world-units/retrain/}"
sums="$CVC_RECIPE_DIR/SHA256SUMS"
dst="$CVC_INSTALL_DIR/share/grl-snam-learning-checkpoints"
mkdir -p "$dst"
n=0
while read -r _ rel; do
  test -f "$src/$rel" || { echo "grl-snam-learning-checkpoints: $src/$rel missing" >&2; exit 1; }
  mkdir -p "$dst/$(dirname "$rel")"
  cp "$src/$rel" "$dst/$rel"
  n=$((n + 1))
done < "$sums"
(cd "$dst" && sha256sum --quiet --strict -c "$sums") || {
  echo "grl-snam-learning-checkpoints: a checkpoint is not the one SHA256SUMS names" >&2
  exit 1
}
cp "$sums" "$dst/SHA256SUMS"
cp "$CVC_RECIPE_DIR/PROVENANCE.md" "$dst/PROVENANCE.md"
want=$(grep -c . "$sums")
test "$n" -eq "$want" || { echo "grl-snam-learning-checkpoints: $n checkpoints, not $want" >&2; exit 1; }
echo "grl-snam-learning-checkpoints: staged $n checkpoints ($(du -sk "$dst" | cut -f1) KiB) -> $dst"
