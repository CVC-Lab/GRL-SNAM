# GRL-SNAM's pages

A static landing page, and GRL-SNAM's Ariadne ports running in the browser on libcvc's
`ariglx-web`: CPython 3.12, pycvc, pycvc_gl and numpy compiled to WebAssembly.
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds and checks them on every
change here. It deploys nothing: the site is published by another repository's workflow, which
checks this repo out and calls the scripts below.

| File | What it does |
|---|---|
| `build-ariglx-web.sh` | builds `ariglx-web` from a libcvc checkout with its recipe's own script: the single-threaded build (`wasm`), lean tier; tools, emsdk and the wasm deps from cvcpkg |
| `assemble.py` | writes the site (`site/` at the root, and `learn/`), or with `--learn-only` the learn page alone: the host, `main.py`, `grl_snam_web.zip`, the histories unzipped, `ariadne.config.json` with every preload |
| `main.py` | the program the learn page runs in place of a document: unpacks `grl_snam_web.zip` and runs a port (`args`: `["learn", ...]`) |
| `smoke.mjs` | loads the site in headless Chrome as Pages serves it: learn must start, run its frames and make learn-report's claims in the page |
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

# check it where your site serves it (Node 22; Chrome or Chromium)
node grl-snam/pages/smoke.mjs --site _site --base /your-site/ --learn learn/ --no-landing \
  --chrome "$CHROME" --shots shots
```

- `--histories` picks the learning histories the page shows (`.jsonl` or `.jsonl.gz`; default:
  the learning check's retrain, `docs/learning-validation/retrain/seed*.history.jsonl.gz`). Each
  run is labelled by its file's stem, so the names must differ.
- `--arms` takes a `grl_snam.arms/v1` table (what `grl-snam arms-eval` writes) for learn's Arms
  window. `docs/learning-validation/arms.json` is the learning check's own schema
  (`grl_snam.learnproof_arms/v1`), which learn refuses, so there is no default.
- Once learn runs, the page's stderr (the browser console, and `window.ariadneWeb.log`) has
  `grl-snam: learn: started in N s`, then `grl-snam: learn: running` once its frames run.

## Locally

About 20 minutes for the first build; Chrome or Chromium and Node 22 for the smoke:

```bash
git clone --depth 1 --branch v3.7.13 https://github.com/transfix/libcvc /tmp/libcvc
bash pages/build-ariglx-web.sh /tmp/libcvc /tmp/agw-work /tmp/agw
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site
node pages/smoke.mjs --site _site --chrome "$(command -v google-chrome)" --shots shots
```

To look at it, serve the site's parent at `/GRL-SNAM/` (for example a directory with a
`GRL-SNAM` link to `_site`, then `python3 -m http.server`) and open `/GRL-SNAM/learn/`.

## Constraints

- **Single-threaded.** GitHub Pages sends no COOP/COEP headers, so the threaded `wasm-mt` build
  (SharedArrayBuffer) cannot run there.
- **Same origin.** Everything a page fetches is beside it: the page loads documents and data only
  from its own origin, and cvcpkg.org sends no CORS headers for github.io.
- **Size.** A first visit to the learn page downloads about 34 MB gzip (28 MB of it
  `ariglx.wasm`), which the browser then caches. learn takes about 20 s to start in the page.
- **Charts.** libcvc v3.7.13 has no `chart:` widget (3.7.14), so learn's windows show text.
