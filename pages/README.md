# GRL-SNAM's pages

A static landing page, and GRL-SNAM's Ariadne ports running in the browser on libcvc's
`ariglx-web`: CPython 3.12, pycvc, pycvc_gl and numpy compiled to WebAssembly.
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds and checks them on every
change here. It deploys nothing: the site is published by another repository's workflow, which
checks this repo out and calls the scripts below.

| File | What it does |
|---|---|
| `build-ariglx-web.sh` | builds `ariglx-web` from a libcvc checkout with its recipe's own script: the single-threaded build (`wasm`), lean tier; tools, emsdk and the wasm deps from cvcpkg |
| `assemble.py` | writes the site (`site/` at the root, `learn/`, and with `--scrub` `scrub/`), or with `--learn-only` / `--scrub-only` one page alone: the host, `main.py`, `grl_snam_web.zip`, the page's data, `ariadne.config.json` with every preload |
| `main.py` | the program a page runs in place of a document: unpacks `grl_snam_web.zip` and runs a port (`args`: `["learn", ...]`, `["scrub", "--view", ...]`) |
| `smoke.mjs` | loads the site in headless Chrome as Pages serves it: learn must start, run its frames and make learn-report's claims in the page; scrub (`--scrub`) must start, show a checkpoint and run its frames |
| `site/` | the landing page |

## From another repository's workflow

Check out CVC-Lab/GRL-SNAM (here `grl-snam/`) and transfix/libcvc at the tag `pages.yml` pins
(`LIBCVC_REF`, here `libcvc/`), then:

```bash
# ariglx-web (cache ariglx-web/ by libcvc's commit and a hash of the script): ~20-30 min cold
bash grl-snam/pages/build-ariglx-web.sh libcvc "$RUNNER_TEMP/ariglx-web-work" ariglx-web

# the learn page, self-contained, at any path of your site
python3 grl-snam/pages/assemble.py --host ariglx-web/share/ariglx-web --learn-only \
  --out _site/learn [--histories H.jsonl[.gz]...] [--arms ARMS.json]

# the scrub page, self-contained: its runs deployed here, natively, by grl_snam's closure
# (cvcpkg's grl-snam-cp312: torch and pycvc-gl, as pages.yml installs it), ~10 min for the five
python3 grl-snam/pages/assemble.py --host ariglx-web/share/ariglx-web --scrub-only \
  --out _site/scrub --python "$PREFIX/bin/python3.12" [--scrub-cache DIR] \
  [--scrub-histories H...] [--checkpoints DIR] [--scrub-world generated|flat|data]

# check them where your site serves them (Node 22; Chrome or Chromium)
node grl-snam/pages/smoke.mjs --site _site --base /your-site/ --learn learn/ --scrub scrub/ \
  --no-landing --chrome "$CHROME" --shots shots
```

- `--histories` picks the learning histories the page shows (`.jsonl` or `.jsonl.gz`; default:
  the learning check's retrain, `docs/learning-validation/retrain/seed*.history.jsonl.gz`). Each
  run is labelled by its file's stem, so the names must differ.
- `--arms` takes a `grl_snam.arms/v1` table (what `grl-snam arms-eval` writes) for learn's Arms
  window. `docs/learning-validation/arms.json` is the learning check's own schema
  (`grl_snam.learnproof_arms/v1`), which learn refuses, so there is no default.
- Once learn runs, the page's stderr (the browser console, and `window.ariadneWeb.log`) has
  `grl-snam: learn: started in N s`, then `grl-snam: learn: running` once its frames run.

### The scrub page

`grl-snam ariadne scrub` deploys every checkpoint of a training run and shows checkpoint k beside
step 0. Deploying a `.pt` checkpoint needs torch and libcvc's sim_world, so the page cannot: the
runs are deployed when the site is built, and the page runs `scrub --view`, which only reads
them (the reports, `grl_snam.scrub/v1`, and each checkpoint's run log, `grl_snam.run_log/v1`).

- **The runs.** `--scrub-histories` (default: the retrain's five, as learn's) are deployed with
  `--python` (default: the Python running assemble), each history's saved policies found in
  `--checkpoints`/<run>/ (<run> is the history's name without `.history.jsonl[.gz]`; default
  `docs/learning-validation/retrain/checkpoints/`, the retrain's 41 a seed, 4.9 MB). A cvcpkg
  prefix's `bin/python3.12` gets its `lib/` on the loader path. `--scrub-cache DIR` keeps the run
  logs for the next build: a checkpoint already deployed on the same scenario is read back.
  `--scrub-reports R.json...` takes runs deployed before and deploys nothing (no torch).
- **The world.** `--scrub-world generated` (the default) is libcvc's generator on scrub.ari's
  `world.*` keys. The page builds the same world again from the report, with nothing to fetch,
  provided its libcvc is the deploy's (ariglx-web's tag, `LIBCVC_REF`, and the catalog's
  pycvc-gl are both 3.7.13; the page says when the world differs). The policies were trained on
  Austin, but Austin's bundle is 228 MB, cvcpkg.org sends no CORS for github.io, and nothing cuts
  it down yet, so a `data` world's page shows the windows alone. On `flat` ground every
  checkpoint drives the same line.
- **The page.** `scrub/data/<run>/` holds each report (its paths made the file names beside
  it), the history and the run logs: about 8 MB a run (already compressed), 41 MB for the five.
  It opens on the first run's pick, from above (`--camera orbit`), with the Scrub, Episode and
  Stats windows (`--windows`); the Run combo picks another run, and the Windows menu has
  Training, Ledger and Simulation.
- Once scrub runs, the page's stderr has `grl-snam: scrub: started in N s`, a line for each
  checkpoint it shows (`scrub: seed0: checkpoint 22 of 41: step 2100, the pick; ...`), then
  `grl-snam: scrub: running`.

## Locally

About 20 minutes for the first build; Chrome or Chromium and Node 22 for the smoke:

```bash
git clone --depth 1 --branch v3.7.13 https://github.com/transfix/libcvc /tmp/libcvc
bash pages/build-ariglx-web.sh /tmp/libcvc /tmp/agw-work /tmp/agw
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site
node pages/smoke.mjs --site _site --chrome "$(command -v google-chrome)" --shots shots

# with the scrub page: a Python with grl_snam's closure (cvcpkg install grl-snam-cp312)
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site --scrub \
  --python "$PREFIX/bin/python3.12" --scrub-cache /tmp/scrub-runs
node pages/smoke.mjs --site _site --scrub scrub/ --chrome "$(command -v google-chrome)"
```

To look at it, serve the site's parent at `/GRL-SNAM/` (for example a directory with a
`GRL-SNAM` link to `_site`, then `python3 -m http.server`) and open `/GRL-SNAM/learn/` or
`/GRL-SNAM/scrub/`.

## Constraints

- **Single-threaded.** GitHub Pages sends no COOP/COEP headers, so the threaded `wasm-mt` build
  (SharedArrayBuffer) cannot run there.
- **Same origin.** Everything a page fetches is beside it: the page loads documents and data only
  from its own origin, and cvcpkg.org sends no CORS headers for github.io.
- **Size.** A first visit to the learn page downloads about 34 MB gzip (28 MB of it
  `ariglx.wasm`), which the browser then caches. learn takes about 20 s to start in the page.
  The scrub page is about 72 MB: its own copy of the host, and 41 MB of run logs. Each page
  carries its own host, so a visit to both downloads `ariglx.wasm` twice.
  Gzipped histories ship as they are (the retrain's five: 1.1 MB, 5.4 MB unzipped); learn's
  reader unzips them in the page, whether or not the server marks them `Content-Encoding: gzip`.
- **Charts.** libcvc v3.7.13 has no `chart:` widget (3.7.14), so learn's windows show text.
