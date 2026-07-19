# cvcpkg recipe for grl-snam

This packages GRL-SNAM as a [cvcpkg](https://cvcpkg.org) component so it can
be installed into an activatable prefix alongside `libcvc` (which ships the
`pycvc` Python bindings) — one prefix carries both the `grl_snam` API and the
libcvc/pycvc bindings it can drive.

**Project owns its recipe.** Per cvcpkg convention, a project's own cvcpkg
recipe lives in the project repo (here), not in the central `libcvc-deps`
recipe set — which is reserved for the shared dependency ecosystem. See the
cvcpkg roadmap's "Recipe ownership" note.

## Build / validate locally
```bash
cvcpkg validate --recipes-dir cvcpkg/recipes cvcpkg/recipes/grl-snam
cvcpkg pack grl-snam --recipes-dir cvcpkg/recipes --local --output-dir dist
```

## Publish to the `cvc` org
```bash
cvcpkg publish grl-snam --org cvc --recipes-dir cvcpkg/recipes \
  --output-dir dist --token "$CVCPKG_TOKEN"
```

Follow-ups (see recipe `notes`): publish the `v0.1.0` GitHub release asset
(or mirror the sdist) so `source.url` resolves; add `torch`/`matplotlib`
wheel recipes and a `poetry-core` backend to close the build closure; a
per-interpreter matrix for full 3.10–3.13 coverage.
