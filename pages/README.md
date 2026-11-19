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

# the scrub page, self-contained: its runs deployed here, natively, by a closure whose libcvc is
# the one ariglx-web is built from (V: LIBCVC_REF without the v), on the learning check's saved
# policies (cvcpkg's grl-snam-learning-checkpoints); ~10 min a five-seed set
cvcpkg install python312 numpy-cp312 torch-cp312 "libcvc==$V" "cvcgl==$V" "pycvc-cp312==$V" \
  "pycvc-gl-cp312==$V" --platform linux --prefix "$PREFIX" --config release --link shared
cvcpkg install grl-snam-learning-checkpoints --prefix "$PREFIX" --platform any --config release \
  --link shared --no-fallback-to-source
python3 grl-snam/pages/assemble.py --host ariglx-web/share/ariglx-web --scrub-only \
  --out _site/scrub --python "$PREFIX/bin/python3.12" --expect-libcvc "$V" [--scrub-cache DIR] \
  [--scrub-sets SET...] [--checkpoints DIR] [--scrub-world generated|flat|data]

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

- **The data sets.** `--scrub-sets` (default `retrain fixed/retrain world-units/retrain`) names the learning check's
  data sets: directories under `docs/learning-validation/` whose `*.history.jsonl.gz` are their
  runs. `--scrub-histories H...` deploys those histories instead, as one set
  (`--scrub-set-name`).
- **The plants.** `--scrub-plants` (default `humvee kinematic`) deploys every set on each, and the
  page's Plant combo switches between them at the same run and step. `humvee` (the Jolt Humvee)
  collides with the buildings, and the page opens on it. `kinematic` (nav's bicycle) drives through
  them, and it is there to compare.
- **The runs.** Each is deployed with `--python` (default: the Python running assemble), as
  `scrub HISTORY --headless --lean --data-set SET --plant PLANT --report`, its saved policies found in
  `--checkpoints`/<set>/<run>/ (else `--checkpoints`/<run>/; <run> is the history's name without
  `.history.jsonl[.gz]`). A cvcpkg prefix's `bin/python3.12` gets its `lib/` on the loader path.
  `--scrub-cache DIR` keeps the run logs for the next build: a checkpoint already deployed on the
  same scenario is read back. `--scrub-reports R.json...` takes runs deployed before and deploys
  nothing (no torch).
- **The saved policies** are not in the repository (`.gitignore` keeps `*.pt` out). They are the
  cvcpkg data package `grl-snam-learning-checkpoints`
  ([its recipe](../cvcpkg/recipes/grl-snam-learning-checkpoints/PROVENANCE.md)): every checkpoint
  of the learning check's three retrains, `retrain/`, `fixed/retrain/` and `world-units/retrain/`
  (`seed{0..4}/` each), 41 a seed, 615 `.pt` files, 11 MB packed (1.0.0+cvc.2). Its recipe commits their sha256 (`SHA256SUMS`), and
  its build stages them from the training runs' directories and refuses any that differ, so the
  package holds the files the committed histories name. `--checkpoints` defaults to the package
  in `--python`'s prefix; pages.yml installs it in a prefix of its own and names it.
- **The world.** `--scrub-world generated` (the default) is libcvc's generator on scrub.ari's
  `world.*` keys. The page builds the same world again from the report, with nothing to fetch,
  provided its libcvc is the deploy's. pages.yml pins the deploy's libcvc family (libcvc, cvcgl,
  pycvc, pycvc-gl) to `LIBCVC_REF`'s version, the tag ariglx-web is built from;
  `--expect-libcvc` refuses runs whose report names another libcvc; and the page says when the
  world it built differs from the deploy's, which fails the smoke. The policies were trained on
  Austin, but Austin's bundle is 228 MB, cvcpkg.org sends no CORS for github.io, and nothing cuts
  it down yet, so a `data` world's page shows the windows alone. On `flat` ground every
  checkpoint drives the same line.
- **The page.** `scrub/data/<plant>/<set>/<run>.scrub.zip` (`data/<set>/<run>.scrub.zip` when the page
  has one plant) is a run's bundle: one zip (stored) with its
  report (`scrub.json`, its paths made the names inside), its history and its run logs, which
  `scrub --view` reads from it when a checkpoint is shown. The page's loader fetches at most 512
  files and a run is 43, so three sets as loose files (646) would not load; as bundles they are
  16. The logs are lean (`--lean`: single precision, the
  cars template's derived fields left out; about 40% smaller, read back the same): about
  5.1–5.3 MB a run's bundle on kinematic and 5.8–6.1 MB on the Humvee, 26–30 MB a five-seed set
  on one plant. It
  opens on the first plant's first set's first run, at its pick,
  from above (`--camera orbit`), with the Scrub, Episode and Stats windows (`--windows`). The
  Set combo picks a data set and shows its run of the same name at the same step; the Run combo
  picks a run in the set; the Windows menu has Training, Ledger and Simulation.
- **Every run is preloaded.** The page's loader fetches its files before the program starts,
  and ariglx-web has no supported fetch after that (non-blocking downloads in the browser
  are libcvc's data roadmap D4, not done), so a set cannot be loaded when it is picked. Each
  five-seed set adds 26–30 MB a plant to the first visit. The run logs are already compressed,
  so the server's gzip barely shrinks them: three sets on two plants are 167 MB of bundles
  (154 MB at gzip -9; 89 MB of them the Humvee's, 78 MB kinematic's), so the page is about
  190 MB on a first visit, against about 105 MB on one plant. `--scrub-plants humvee` builds the
  lighter page.
- Once scrub runs, the page's stderr has `grl-snam: scrub: started in N s`, the runs and data
  sets (`scrub: 15 runs in 3 data sets (retrain, fixed/retrain, world-units/retrain)`), a line for each checkpoint it
  shows (`scrub: retrain/seed0: checkpoint 22 of 41: step 2100, the pick; ...`), then
  `grl-snam: scrub: running`.

## Locally

About 20 minutes for the first build; Chrome or Chromium and Node 22 for the smoke:

```bash
git clone --depth 1 --branch v4.0.0 https://github.com/transfix/libcvc /tmp/libcvc
bash pages/build-ariglx-web.sh /tmp/libcvc /tmp/agw-work /tmp/agw
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site
node pages/smoke.mjs --site _site --chrome "$(command -v google-chrome)" --shots shots

# with the scrub page: a Python whose libcvc is ariglx-web's (the closure above, V=4.0.0)
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site --scrub \
  --python "$PREFIX/bin/python3.12" --expect-libcvc 4.0.0 --scrub-cache /tmp/scrub-runs
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
- **Size.** A first visit to the learn page downloads about 35 MB gzip (29.5 MB of it
  `ariglx.wasm`, 105.6 MB raw at 4.0.0), which the browser then caches. GitHub documents no
  per-file limit for a Pages site deployed from an Actions artifact (1 GB a site). learn takes about 20 s to start in the page.
  The scrub page is about 190 MB: its own copy of the host, and 167 MB of run bundles (three
  data sets on two plants, lean). Each page carries its own host, so a visit to both downloads `ariglx.wasm` twice.
  Gzipped histories ship as they are (the retrain's five: 1.1 MB, 5.4 MB unzipped); learn's
  reader unzips them in the page, whether or not the server marks them `Content-Encoding: gzip`.
- **Charts.** libcvc 4.0.0 has Ariadne's `chart:` widget (3.7.14+), so learn and scrub draw their
  curves as charts (on 3.7.13 they showed the same numbers as text).
- **One libcvc.** ariglx-web is built from `LIBCVC_REF`, and the scrub page's runs are deployed
  with the catalog's libcvc of the same version: pages.yml pins libcvc, cvcgl, pycvc and pycvc-gl
  to it, `assemble.py --expect-libcvc` refuses runs from another, and the smoke fails when the page
  says the world it built is not the deploy's. A newer libcvc in the catalog changes nothing until
  `LIBCVC_REF` moves.
