// pages/smoke.mjs — the assembled Pages site in a real (headless) Chrome, before it deploys. Modelled
// on libcvc's src/arigl/web/test/arigl_web_smoke.mjs, whose server and DevTools driver it follows.
//
//   node pages/smoke.mjs --site _site --chrome <chrome or chromium> [--base /GRL-SNAM/] [--shots DIR]
//        [--learn learn/] [--no-landing]
//
// It serves the site at --base (the project site's path on github.io) on 127.0.0.1, with an
// ordinary site's headers: no COOP/COEP, as GitHub Pages sends none (the single-threaded ariglx-web
// needs none), and a strict Content-Security-Policy (no inline code or style, no eval, nothing from
// another origin), so a pass also says the pages fetch only from their own origin. Then:
//
//   - the landing page: every link and image on the site itself must be there (--no-landing skips
//     it, for a site whose landing page is not GRL-SNAM's);
//   - the learn page (--learn, the path under --base; learn/ by default): grl-snam ariadne learn
//     must start in the page (main.py says so on stderr), its frames must run, and the learn-report
//     it makes in the page (?arg=--report, read back from the page's file system) must hold
//     claims; no Python traceback, no CSP violation.
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
const BASE = opt('--base', '/GRL-SNAM/'), LEARN = opt('--learn', 'learn/');
const LANDING = !argv.includes('--no-landing');
if (!SITE || !CHROME || !BASE.startsWith('/') || !BASE.endsWith('/') || !LEARN.endsWith('/') ||
    LEARN.startsWith('/')) {
  console.error('usage: smoke.mjs --site DIR --chrome PATH [--base /PATH/] [--shots DIR] ' +
                '[--learn PATH/] [--no-landing]');
  process.exit(2);
}
const TIMEOUT_MS = Number(process.env.GRL_SNAM_SMOKE_TIMEOUT_MS || 600000);

const CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; " +
            "connect-src 'self'; img-src 'self' data: blob:";
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
               '.wasm': 'application/wasm', '.json': 'application/json', '.ari': 'text/plain',
               '.py': 'text/plain', '.zip': 'application/zip', '.png': 'image/png',
               '.jsonl': 'text/plain', '.md': 'text/markdown'};

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
  res.writeHead(200, {'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream',
                      'Cache-Control': 'no-cache', 'Content-Security-Policy': CSP});
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
    const m = await send('Runtime.evaluate', {expression: expr, returnByValue: true});
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
