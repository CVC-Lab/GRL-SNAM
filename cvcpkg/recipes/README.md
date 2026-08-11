# cvcpkg recipes for grl-snam

This packages GRL-SNAM as a [cvcpkg](https://cvcpkg.org) component so it can
be installed into an activatable prefix alongside `libcvc` (whose `pycvc` /
`pycvc_gl` bindings ship as the per-interpreter `pycvc-gl-cp31X` columns) —
one prefix carries both the `grl_snam` API and the bindings it can drive.

**Per-interpreter columns.** Like every python package in the cvcpkg
ecosystem, grl-snam is a matrix of per-interpreter column recipes:

| recipe           | interpreter | runtime closure                                  |
|------------------|-------------|--------------------------------------------------|
| `grl-snam-cp311` | python311   | pycvc-gl-cp311, numpy-cp311, torch-cp311          |
| `grl-snam-cp312` | python312   | pycvc-gl-cp312, numpy-cp312, torch-cp312          |
| `grl-snam-cp313` | python313   | pycvc-gl-cp313, numpy-cp313, torch-cp313          |

These columns supersede the published bare `grl-snam` package (which targeted
python311 only). There is no `cp313t` (free-threaded) column because
`pycvc-gl` has none — SWIG is not free-threaded-safe.

The `build.sh` / `test.sh` scripts are byte-identical across the three column
dirs (parameterized off `CVC_PYTHON_INTERPRETER`, the recipe's
`python.interpreter`); keep them in lockstep when editing.

**Project owns its recipe.** Per cvcpkg convention, a project's own cvcpkg
recipes live in the project repo (here), not in the central `libcvc-deps`
recipe set — which is reserved for the shared dependency ecosystem. See the
cvcpkg roadmap's "Recipe ownership" note.

## Build / validate locally
```bash
cvcpkg validate --recipes-dir cvcpkg/recipes cvcpkg/recipes/grl-snam-cp311
cvcpkg pack grl-snam-cp311 --recipes-dir cvcpkg/recipes --local --output-dir dist
```

## Publish to the `cvc` org
```bash
for col in grl-snam-cp311 grl-snam-cp312 grl-snam-cp313; do
  cvcpkg publish "$col" --org cvc --recipes-dir cvcpkg/recipes \
    --output-dir dist --token "$CVCPKG_TOKEN"
done
```

Follow-ups (see each recipe's `notes`): publish the `v0.1.0` GitHub release
asset (or mirror the sdist) so `source.url` resolves; `matplotlib`/`imageio`
wheel recipes and a `poetry-core` backend recipe to close the full research
core's closure.
