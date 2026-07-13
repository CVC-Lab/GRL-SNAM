# PyPI Publishing Roadmap

**Goal:** publish **`grl-snam`** to PyPI under a group-owned account, following
current Python packaging standards, so it installs with a plain
`pip install grl-snam`.

> **Scope.** This covers the public `grl-snam` package only. The downstream
> `grl-snam-dbg` extension is sensitive and **not** publicly distributed; it is
> out of scope here and must never be published. Once `grl-snam` is on PyPI,
> downstreams can depend on it by version instead of a GitHub URL.

---

## Current state

| Aspect | Status |
|---|---|
| Build system | Poetry / PEP 621 (`poetry-core`), version `0.1.0` |
| Layout | Flat: importable `grl_snam/` plus `scripts/`, `experiments/`, and top-level `train_coef_energy.py`, `eval_coef_energy.py`, `surrogate_robust.py` in `[tool.poetry] packages` |
| CLI | `argparse` across ~15 entry points (`train_coef_energy.py`, `eval_coef_energy.py`, `scripts/*`, `experiments/*`). No `click`, no console entry points. |
| CI | `.github/workflows/ci.yml` — GitHub-hosted `ubuntu-latest`, Python 3.10–3.12, lint + tests. No release/publish workflow. |
| PyPI presence | None. |
| License | Confirm SPDX identifier + add a `LICENSE` file. |

---

## Phase 0 — Account & name

- Verify the name `grl-snam` is free on PyPI; if taken, choose an alternative.
- Create/confirm a group-owned PyPI account/organization with 2FA enabled; add maintainers.

## Phase 1 — Packaging hygiene

- Complete `pyproject.toml` metadata: `description`, `readme` (+ content-type),
  `authors`/`maintainers`, **`license`** (SPDX), `keywords`, `classifiers`
  (incl. `License ::`, supported Python versions, Development Status), and
  `[project.urls]` (Homepage, Repository, Documentation, Issues).
- Add a `LICENSE` file matching the SPDX identifier.
- Adopt a `src/` layout and ship only the importable package. Publishing
  `experiments/` and loose top-level modules as importable names pollutes the
  global namespace — move library code under `src/grl_snam/`, keep
  experiments/scripts in the repo but out of the wheel, and expose runnable
  pieces as CLI entry points (Phase 2).
- Declare runtime deps with sane lower bounds (`torch`, `numpy`, `matplotlib`,
  `imageio`, and `click`).
- Build & check:
  ```bash
  python -m build            # sdist + wheel
  twine check dist/*         # metadata / readme render
  ```

## Phase 2 — Standardize the CLI on `click`

- Add `click`; create `src/grl_snam/cli.py` with a `click.Group` and migrate
  argparse entry points to subcommands:
  - `grl-snam train …`       ← `train_coef_energy.py`
  - `grl-snam eval …`        ← `eval_coef_energy.py`
  - `grl-snam gen-dataset …` ← `scripts/stagewise_dataset.py`
- Register the console entry point:
  ```toml
  [project.scripts]
  grl-snam = "grl_snam.cli:main"
  ```
- Add CLI tests with `click.testing.CliRunner`.

## Phase 3 — Versioning & changelog

- Adopt SemVer; single source of version truth (`pyproject` `version` or
  `__version__` via `importlib.metadata`).
- Add `CHANGELOG.md` (Keep a Changelog); tag releases `vX.Y.Z`.

## Phase 4 — Automated publishing (Trusted Publishing / OIDC)

- Use **PyPI Trusted Publishers (OIDC)** — no long-lived API tokens.
- Add `.github/workflows/release.yml`, triggered on `v*` tags:
  1. build (`python -m build`) + `twine check`;
  2. publish to **TestPyPI** first;
  3. publish to **PyPI** via `pypa/gh-action-pypi-publish` with
     `permissions: id-token: write`.
- Protect releases with a GitHub `pypi` environment (required reviewers).
- Keep CI/release on **GitHub-hosted** runners. Do not attach a self-hosted
  runner to this public repo — a fork's pull request can execute arbitrary code
  on a self-hosted runner. GitHub-hosted minutes are free for public repos.

## Phase 5 — Verify & document

- From a clean environment: `pip install grl-snam`, run `grl-snam --help`, run a smoke test.
- Add README badges (PyPI version, CI status).

---

## Checklist

- [ ] PyPI name verified + group account with 2FA (P0)
- [ ] Metadata complete, `LICENSE` added, `src/` layout, `twine check` passes (P1)
- [ ] `click` CLI + `[project.scripts]`, CLI tests (P2)
- [ ] SemVer, `CHANGELOG.md`, tagging (P3)
- [ ] Trusted-publishing release workflow, TestPyPI → PyPI, `pypi` environment (P4)
- [ ] Clean-env install verified, badges (P5)
