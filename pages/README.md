# GRL-SNAM's Pages site

A static landing page, and GRL-SNAM's Ariadne ports running in the browser on libcvc's
`ariglx-web`: CPython 3.12, pycvc, pycvc_gl and numpy compiled to WebAssembly. Built, checked and
(by hand) deployed by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml).

| File | What it does |
|---|---|
| `build-ariglx-web.sh` | builds `ariglx-web` from a libcvc checkout with its recipe's own script: the single-threaded build (`wasm`), lean tier; tools, emsdk and the wasm deps from cvcpkg |
| `assemble.py` | writes the site: `site/` at the root, and `learn/` (the host, `main.py`, `grl_snam_web.zip`, the retrain histories unzipped, `ariadne.config.json`) |
| `main.py` | the program `learn/` runs in place of a document: unpacks `grl_snam_web.zip` and runs a port (`args`: `["learn", ...]`) |
| `smoke.mjs` | loads the site in headless Chrome as Pages serves it: learn must start, run its frames and make learn-report's claims in the page |
| `site/` | the landing page |

Locally (about 20 minutes for the first build; Chrome or Chromium for the smoke, Node 22):

```bash
git clone --depth 1 --branch v3.7.13 https://github.com/transfix/libcvc /tmp/libcvc
bash pages/build-ariglx-web.sh /tmp/libcvc /tmp/agw-work /tmp/agw
python3 pages/assemble.py --host /tmp/agw/share/ariglx-web --out _site
node pages/smoke.mjs --site _site --chrome "$(command -v google-chrome)" --shots shots
```

To look at it, serve the site's parent at `/GRL-SNAM/` (for example a directory with a
`GRL-SNAM` link to `_site`, then `python3 -m http.server`) and open `/GRL-SNAM/learn/`.

Constraints:

- **Single-threaded.** GitHub Pages sends no COOP/COEP headers, so the threaded `wasm-mt` build
  (SharedArrayBuffer) cannot run there.
- **Same origin.** Everything a page fetches is on the site: the page loads documents and data
  only from its own origin, and cvcpkg.org sends no CORS headers for github.io.
- **Size.** A first visit to `learn/` downloads about 34 MB gzip (28 MB of it `ariglx.wasm`),
  which the browser then caches.
- **Arms.** `assemble.py --arms` takes a `grl_snam.arms/v1` table (what `grl-snam arms-eval`
  writes) for learn's Arms window. `docs/learning-validation/arms.json` is the learning check's
  own schema (`grl_snam.learnproof_arms/v1`), which learn does not read, so the page has none yet.
