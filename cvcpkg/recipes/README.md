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

`--recipes-dir cvcpkg/recipes` alone does NOT validate: the runtime closure
names `pycvc-gl-cp31X`, which libcvc owns, and `python311` / `torch-cp31X` /
`poetry-core-cp31X`, which libcvc-deps owns. With only this repo's recipes
resolvable the check fails on `unknown dependency 'pycvc-gl-cp311'`. Point it
at all three sets:

```bash
cvcpkg validate \
  --recipes-dir cvcpkg/recipes \
  --recipes-dir /path/to/libcvc/cvcpkg/recipes \
  --recipes-dir /path/to/libcvc-deps/recipes \
  cvcpkg/recipes/grl-snam-cp311

cvcpkg pack grl-snam-cp311 --recipes-dir cvcpkg/recipes --local --output-dir dist
```

## Publish to the `cvc` org
```bash
for col in grl-snam-cp311 grl-snam-cp312 grl-snam-cp313; do
  cvcpkg publish "$col" --org cvc --recipes-dir cvcpkg/recipes \
    --output-dir dist --token "$CVCPKG_TOKEN"
done
```

The matplotlib/imageio/poetry-core columns exist, so the closure is complete.

**Source is the checkout.** As of revision 4 these columns are
`source.type: vendored` with `path: ../../..` — the same arrangement libcvc uses
for `pycvc-cp31X` and `cvc-cli`. `cvcpkg build grl-snam-cp31X` therefore builds
*this tree*.

Revisions 1–3 pinned `python_sdist` to the `v0.1.0` GitHub release asset and its
sha256. That froze the columns at whatever had last been released: a code change
meant rebuilding the sdist, replacing the asset, moving the tag, re-pinning
`sha256` in all three columns and bumping `cvc_revision` — and until you did,
`cvcpkg build` silently built the *released* code rather than your working tree.
Vendoring removes that loop entirely.

**Re-publishing after a code change** is now just a `cvc_revision` bump in all
three columns (the recipes stay in lockstep), then `cvcpkg publish`. The version
itself lives in `pyproject.toml`; when it moves, update `upstream_version` and
the `package.files` dist-info path with it.

Revision 3 was the first whose artifact declared `[project.scripts]`:
0.1.0+cvc.1 shipped the importable package with no `grl-snam` console script at
all, so `cvcpkg install` gave you the library and no command. `build.sh` asserts
the script exists, so that cannot ship again unnoticed.
