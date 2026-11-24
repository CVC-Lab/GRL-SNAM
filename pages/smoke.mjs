// pages/smoke.mjs — the assembled Pages site in a real (headless) Chrome, before it deploys. Modelled
// on libcvc's src/arigl/web/test/arigl_web_smoke.mjs, whose server and DevTools driver it follows.
//
//   node pages/smoke.mjs --site _site --chrome <chrome or chromium> [--base /GRL-SNAM/] [--shots DIR]
//        [--learn learn/] [--no-learn] [--scrub scrub/] [--city-scrub scrub-austin/]...
//        [--no-landing]
//
// It serves the site at --base (the project site's path on github.io) on 127.0.0.1, with an
// ordinary site's headers: no COOP/COEP, as GitHub Pages sends none (the single-threaded ariglx-web
// needs none), and a strict Content-Security-Policy (no inline code or style, no eval, nothing from
// another origin but, for a scrub page on a city, the cvcpkg registry the city comes from), so a
// pass also says the pages fetch only from their own origin, and a city's page from the registry
// too. Then:
//
//   - the landing page: every link and image on the site itself must be there (--no-landing skips
//     it, for a site whose landing page is not GRL-SNAM's);
//   - the learn page (--learn, the path under --base; learn/ by default): grl-snam ariadne learn
//     must start in the page (main.py says so on stderr), its frames must run, and the learn-report
//     it makes in the page (?arg=--report, read back from the page's file system) must hold
//     claims; no Python traceback, no CSP violation (--no-learn skips it);
//   - the scrub page (--scrub, the path under --base; none unless given): grl-snam ariadne scrub
//     --view must start in the page, show a checkpoint of the runs it was given (its stderr says
//     which), draw a 3-D world, the one they were deployed on (it says which, or that its own
//     libcvc built another; Austin comes from cvcpkg.org), draw its cars with the Humvee's model
//     when the page ships it, run its frames, and still run 5 s later; no traceback, no CSP
//     violation. It says how many frames a second the page ran (headless Chrome's software GL
//     here, so a floor, not the figure a GPU gets);
//   - each scrub page on a city (--city-scrub, the path under --base, as many as given: the runs
//     deployed in Austin, assemble.py --scrub-austin's scrub-austin/): the same, its city fetched
//     from cvcpkg.org, so its connect-src has the registry too.
//
// --shots writes each page as DIR/<name>.png. Exit status 0 when every case passes.

import {spawn} from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const SITE = opt('--site'), CHROME = opt('--chrome'), SHOTS = opt('--shots');
const BASE = opt('--base', '/GRL-SNAM/'), LEARN = opt('--learn', 'learn/'), SCRUB = opt('--scrub');
const CITY_SCRUBS = argv.flatMap((a, i) => a === '--city-scrub' ? [argv[i + 1]] : []);
const LANDING = !argv.includes('--no-landing'), DO_LEARN = !argv.includes('--no-learn');
const sub = p => p === undefined || (p.endsWith('/') && !p.startsWith('/'));
if (!SITE || !CHROME || !BASE.startsWith('/') || !BASE.endsWith('/') || !sub(LEARN) ||
    !sub(SCRUB) || !CITY_SCRUBS.every(p => p !== undefined && sub(p))) {
  console.error('usage: smoke.mjs --site DIR --chrome PATH [--base /PATH/] [--shots DIR] ' +
                '[--learn PATH/] [--no-learn] [--scrub PATH/] [--city-scrub PATH/]... ' +
                '[--no-landing]');
  process.exit(2);
}
const TIMEOUT_MS = Number(process.env.GRL_SNAM_SMOKE_TIMEOUT_MS || 600000);

// A scrub page on a city may also reach the registry: runs deployed on a data world (Austin) are
// drawn on the city the deploy fetched, a cvcpkg package, which the page's data store fetches from
// cvcpkg.org (it lets any origin read a public package). Every other page reaches nothing else.
const REGISTRY = 'https://cvcpkg.org';
const csp = connect => "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; " +
                       `style-src 'self'; connect-src ${connect}; img-src 'self' data: blob:`;
const CSP = csp("'self'"), CITY_CSP = csp(`'self' ${REGISTRY}`);
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
               '.wasm': 'application/wasm', '.json': 'application/json', '.ari': 'text/plain',
               '.py': 'text/plain', '.zip': 'application/zip', '.png': 'image/png',
               '.jpg': 'image/jpeg', '.jsonl': 'text/plain', '.md': 'text/markdown',
               '.npz': 'application/octet-stream'};

function fileFor(urlPath) {
  const p = decodeURIComponent(urlPath);
  if (!p.startsWith(BASE) || p.includes('..')) return null;
  let f = path.join(SITE, p.slice(BASE.length));
  if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
  return fs.existsSync(f) && fs.statSync(f).isFile() ? f : null;
}

const server = http.createServer((req, res) => {
  const f = fileFor(new URL(req.url, 'http://x').pathname);
  if (!f) {
    res.writeHead(404, {'Content-Type': 'text/plain'});
    res.end('not found');
    return;
  }
  const city = CITY_SCRUBS.some(p => req.url.startsWith(BASE + p));
  res.writeHead(200, {'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream',
                      'Cache-Control': 'no-cache',
                      'Content-Security-Policy': city ? CITY_CSP : CSP});
  fs.createReadStream(f).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const ORIGIN = `http://127.0.0.1:${server.address().port}`;

const results = [];
const record = (name, ok, detail) => {
  results.push({name, ok});
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '\n      ' + detail : ''}`);
};

// ── the landing page: its own links and images resolve ─────────────────────────────────────────
if (LANDING) {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  const refs = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map(m => m[1])
                 .filter(r => !/^[a-z]+:/i.test(r));
  const missing = [];
  for (const r of new Set(refs)) {
    const resp = await fetch(new URL(r, ORIGIN + BASE));
    if (!resp.ok) missing.push(`${r} (HTTP ${resp.status})`);
  }
  record(`the landing page's ${new Set(refs).size} links and images on the site resolve`,
         refs.length > 0 && missing.length === 0, missing.join(', '));
}

// ── Chrome, headless, its DevTools endpoint on a port it picks ─────────────────────────────────
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'grl-snam-smoke-'));
const chrome = spawn(CHROME, [
  '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist',
  '--window-size=1280,860', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const devtools = await new Promise((resolve, reject) => {
  let err = '';
  const t = setTimeout(() => reject(new Error('Chrome did not start:\n' + err)), 30000);
  chrome.stderr.on('data', d => {
    err += d;
    const m = /DevTools listening on (ws:\/\/[^\s]+)/.exec(err);
    if (m) { clearTimeout(t); resolve(new URL(m[1])); }
  });
  chrome.on('exit', c => reject(new Error(`Chrome exited (${c}):\n` + err)));
});

// A tab driven through its own DevTools socket: it opens blank, gets a listener that records the
// page's CSP violations (injected by DevTools, so the policy does not apply to it), then goes to
// the URL.
async function openPage(url) {
  const r = await fetch(`http://${devtools.host}/json/new?about:blank`, {method: 'PUT'});
  const target = await r.json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  };
  const send = (method, params = {}) => new Promise(res => {
    pending.set(++id, res);
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expr => {
    const m = await send('Runtime.evaluate', {expression: expr, returnByValue: true,
                                                awaitPromise: true});
    return m.result && m.result.result ? m.result.result.value : undefined;
  };
  await send('Page.enable');
  await send('Page.addScriptToEvaluateOnNewDocument', {
    source: 'window.__csp = []; document.addEventListener("securitypolicyviolation", ' +
            'e => window.__csp.push(e.effectiveDirective + " " + (e.blockedURI || "inline")));'});
  await send('Page.navigate', {url});
  const close = async () => {
    ws.close();
    await fetch(`http://${devtools.host}/json/close/${target.id}`).catch(() => {});
  };
  const screenshot = async file => {
    const m = await send('Page.captureScreenshot', {format: 'png'});
    if (m.result && m.result.data) fs.writeFileSync(file, Buffer.from(m.result.data, 'base64'));
  };
  return {evaluate, screenshot, close};
}

// The page's state (arigl-web.js's window.ariadneWeb) until `done` holds or the page fails.
async function watch(page, done) {
  const until = Date.now() + TIMEOUT_MS;
  for (;;) {
    const s = await page.evaluate(
        'JSON.stringify(window.ariadneWeb ? {state: ariadneWeb.state, log: ariadneWeb.log,' +
        ' csp: window.__csp || [], warnings: ariadneWeb.warnings || [],' +
        ' status: document.getElementById("status").textContent} : null)');
    const v = s ? JSON.parse(s) : null;
    if (v && (done(v) || v.state === 'failed' || v.state === 'ended')) return v;
    if (Date.now() > until) return v || {state: 'timeout', log: [], csp: [], warnings: []};
    await new Promise(r => setTimeout(r, 500));
  }
}

if (SHOTS) fs.mkdirSync(SHOTS, {recursive: true});
let failed = false;
try {
  if (LANDING) {
    const landing = await openPage(ORIGIN + BASE);
    await new Promise(r => setTimeout(r, 1500));
    const csp = JSON.parse(await landing.evaluate('JSON.stringify(window.__csp || [])'));
    record('the landing page loads under a strict CSP', csp.length === 0, csp.join(', '));
    if (SHOTS) await landing.screenshot(path.join(SHOTS, 'landing.png'));
    await landing.close();
  }

  if (DO_LEARN) {
    // learn/, writing its report into the page's file system (the page's own directory, since
    // main.py runs the port there; the page writes each file at its URL path).
    const t0 = Date.now();
    const page = await openPage(`${ORIGIN}${BASE}${LEARN}?arg=--report&arg=report.json`);
    const v = await watch(page, v => v.log.some(l => /^grl-snam: learn: running/.test(l)));
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const started = v.log.find(l => /^grl-snam: learn: started in/.test(l)) || '';
    const tb = v.log.findIndex(l => /^Traceback/.test(l));
    const running = v.state === 'running' && v.log.some(l => /^grl-snam: learn: running/.test(l));
    record(`learn starts in the page and runs its frames (${secs} s; ${started})`,
           running && tb < 0,
           running && tb < 0 ? '' : `state ${v.state}: ${v.status || ''}\n      ` +
             v.log.slice(tb >= 0 ? tb : -25).join('\n      '));
    // Its frames keep running: an exception in a frame hook would close the document.
    await new Promise(r => setTimeout(r, 5000));
    const later = await watch(page, () => true);
    record('learn is still running 5 s later, with no CSP violation',
           later.state === 'running' && later.csp.length === 0 &&
             !later.log.some(l => /^Traceback/.test(l)),
           `state ${later.state}; csp ${later.csp.join(', ')}; warnings ${later.warnings.length}`);
    const report = await page.evaluate(
        `(() => { try { return FS.readFile(${JSON.stringify(BASE + LEARN + 'report.json')}, ` +
        `{encoding: 'utf8'}); } catch (e) { return 'ERROR ' + e; } })()`);
    let summary = null;
    try { summary = JSON.parse(report).summary; } catch (e) {}
    record('learn-report, made in the page, holds claims',
           summary !== null && summary.pass + summary.fail > 0,
           summary ? `pass ${summary.pass}, fail ${summary.fail}` : String(report).slice(0, 300));
    if (later.warnings.length) console.log('      warnings: ' + later.warnings.slice(0, 5).join(' | '));
    if (SHOTS) await page.screenshot(path.join(SHOTS, 'learn.png'));
    await page.close();
  }

  // scrub/: the runs deployed at build time, viewed. Its stderr says how many runs and data sets
  // it has ("scrub: 15 runs in 3 data sets (retrain, fixed/retrain, world-units/retrain)"), names the checkpoint it
  // shows ("scrub: retrain/seed0: checkpoint 22 of 41: step 2100, the pick; ...") and, once the
  // world is in (a frame after it starts; a city once it has been fetched), whether it is the one
  // the runs drove on ("scrub: the world is the one deployed on (... compared)", "scrub: this
  // world is not quite the one deployed on: ...") or why there is none ("scrub: no 3-D world: ...").
  // The default scrub page, then each on a city: one case each, named in its records.
  for (const where of [...(SCRUB ? [SCRUB] : []), ...CITY_SCRUBS]) {
    const name = where === SCRUB ? 'scrub' : `scrub ${where}`;
    const t0 = Date.now();
    const page = await openPage(`${ORIGIN}${BASE}${where}`);
    const v = await watch(page, v => v.log.some(l => /^grl-snam: scrub: running/.test(l)));
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const started = v.log.find(l => /^grl-snam: scrub: started in/.test(l)) || '';
    const tb = v.log.findIndex(l => /^Traceback/.test(l));
    const running = v.state === 'running' && v.log.some(l => /^grl-snam: scrub: running/.test(l));
    record(`${name} starts in the page and runs its frames (${secs} s; ${started})`,
           running && tb < 0,
           running && tb < 0 ? '' : `state ${v.state}: ${v.status || ''}\n      ` +
             v.log.slice(tb >= 0 ? tb : -25).join('\n      '));
    const shown = v.log.map(l => /^scrub: (.+?): checkpoint (\d+) of (\d+): step ([^,;\s]+)/.exec(l))
                       .filter(Boolean).pop();
    // A city comes a while after the page starts (fetched, then decoded): wait for the line.
    const WORLD = /^scrub: (no 3-D world|the windows alone|this world is not|the world is the one)/;
    const w = await watch(page, v => v.log.some(l => WORLD.test(l)));
    const world = w.log.find(l => WORLD.test(l)) || '';
    const came = ((Date.now() - t0) / 1000).toFixed(1);
    const sets = (v.log.find(l => /^scrub: \d+ runs in \d+ data sets/.test(l)) || '').slice(7);
    record(`${name} shows a checkpoint of the runs deployed at build time`,
           shown !== undefined && Number(shown[3]) > 1 && sets !== '',
           shown ? `${shown[1]}: checkpoint ${shown[2]} of ${shown[3]}, step ${shown[4]}; ${sets}`
                 : 'no "scrub: ...: checkpoint k of n" line');
    // Its frames a second over 3 s, from the page's own animation frames (each runs one frame).
    const fps = await page.evaluate(
        'new Promise(res => { let n = 0; const t0 = performance.now(); const f = t => { ++n; ' +
        'if (t - t0 < 3000) requestAnimationFrame(f); else res(n / ((t - t0) / 1000)); }; ' +
        'requestAnimationFrame(f); })');
    console.log(`      ${name} runs at ${Number(fps).toFixed(1)} frames a second here`);
    await new Promise(r => setTimeout(r, 2000));
    const later = await watch(page, () => true);
    // The page builds the runs' world again with its own libcvc: it must be the one they drove on
    // (scrub --view compares its record field by field, the navigator's grid among them; a grid
    // that differs in a few cells, as the page's wasm libcvc rasterizes it, is still that world).
    const differs = later.log.find(l => /^scrub: this world is not quite the one deployed on/.test(l));
    const same = later.log.find(l => /^scrub: the world is the one deployed on/.test(l));
    record(`${name} draws the world its runs were deployed on (in ${came} s)`,
           differs === undefined && same !== undefined,
           differs ? later.log.slice(later.log.indexOf(differs), later.log.indexOf(differs) + 4)
                       .join(' / ') : same || world || 'no line says whether the 3-D world came');
    // A page that ships the Humvee's model (assemble.py --vehicle-model) draws the cars with it.
    const shipped = fs.existsSync(path.join(SITE, where, 'share/cvc-scenes/shared/Humvee.glb'));
    const cars = later.log.find(l => /^scrub: the cars are /.test(l)) || '';
    record(`${name} draws its cars${shipped ? " with the Humvee's model" : ''}`,
           shipped ? cars === "scrub: the cars are the Humvee's model" : cars !== '',
           cars || world || 'no "scrub: the cars are ..." line');
    record(`${name} is still running 5 s later, with no CSP violation`,
           later.state === 'running' && later.csp.length === 0 &&
             !later.log.some(l => /^Traceback/.test(l)),
           `state ${later.state}; csp ${later.csp.join(', ')}; warnings ${later.warnings.length}`);
    if (later.warnings.length) console.log('      warnings: ' + later.warnings.slice(0, 5).join(' | '));
    // scrub.png, as before; a city's page by its path (scrub-austin/ -> scrub-austin.png).
    const shot = where === SCRUB ? 'scrub.png' : `${where.slice(0, -1).replace(/\//g, '-')}.png`;
    if (SHOTS) await page.screenshot(path.join(SHOTS, shot));
    await page.close();
  }
} catch (e) {
  console.error(e);
  failed = true;
} finally {
  // Chrome writes its profile until it has exited.
  const exited = new Promise(r => chrome.once('exit', r));
  chrome.kill();
  await Promise.race([exited, new Promise(r => setTimeout(r, 5000))]);
  server.close();
  try { fs.rmSync(profile, {recursive: true, force: true}); } catch (e) {}
}
failed = failed || results.some(r => !r.ok);
console.log(failed ? 'smoke: FAILED' : `smoke: all ${results.length} cases passed`);
process.exit(failed ? 1 : 0);
