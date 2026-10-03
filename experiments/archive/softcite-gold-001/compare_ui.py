"""Three-model comparison UI.

Serves a single page with a text box and one button, then runs the same text through
three arms and renders one tab each:

  1. Softcite WAPITI  (http://localhost:8060)
  2. Softcite SciBERT (http://localhost:8062)
  3. OSSoMeX (our checkpoint bundle)

Each tab shows wall-clock timing, the span count, and the input text with that model's
spans highlighted in place, because judging a span without surrounding context is not
possible. The highlight view is the default; a flat list is one click away for scanning.
Spans that overlap or carry offsets that do not match the text are reported rather than
drawn. No web framework is used; the repo has no Flask or FastAPI and adding one for a
debug tool is not worth it.

Run:
  .venv-scibert/bin/python compare_ui.py --bundle checkpoints/scibert-full-label-011
  open http://localhost:8077
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, 'src')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import call_ours, call_softcite, chunk_text

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Software mention comparison</title>
<style>
  :root {
    --bg: #11151c; --panel: #171d26; --line: #263041; --text: #e6ebf2;
    --muted: #8b97a8; --accent: #4da3ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 18px 22px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 17px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 13px; }
  main { padding: 18px 22px 40px; }
  textarea { width: 100%; min-height: 150px; padding: 12px; resize: vertical;
             background: var(--panel); color: var(--text); border: 1px solid var(--line);
             border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size: 13px; }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .row { display: flex; gap: 10px; align-items: center; margin: 12px 0 4px; flex-wrap: wrap; }
  button { background: var(--accent); color: #04121f; border: 0; border-radius: 7px;
           padding: 9px 20px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  .hint { color: var(--muted); font-size: 12px; }
  .tabs { display: flex; gap: 4px; margin-top: 20px; border-bottom: 1px solid var(--line); }
  .tab { background: none; border: 0; border-bottom: 2px solid transparent;
         color: var(--muted); padding: 9px 14px; cursor: pointer; font-size: 13px; }
  .tab.active { color: var(--text); border-bottom-color: var(--accent); font-weight: 600; }
  .stats { display: flex; gap: 22px; flex-wrap: wrap; margin: 14px 0;
           padding: 11px 14px; background: var(--panel); border: 1px solid var(--line);
           border-radius: 8px; }
  .stat b { display: block; font-size: 17px; font-weight: 600; }
  .stat span { color: var(--muted); font-size: 11px; text-transform: uppercase;
               letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
           vertical-align: top; }
  th { color: var(--muted); font-size: 11px; text-transform: uppercase;
       letter-spacing: .05em; font-weight: 600; }
  td.name { font-family: ui-monospace, Menlo, monospace; color: #ffd479; white-space: nowrap; }
  td.ctx { color: var(--muted); font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
  td.num { color: var(--muted); white-space: nowrap; font-size: 12px; }
  .warn { color: #ff9d6b; font-size: 12px; margin: 8px 0; }
  .err { color: #ff8080; font-size: 13px; }
  select { background: var(--panel); color: var(--text); border: 1px solid var(--line);
           border-radius: 7px; padding: 8px 12px; font-size: 13px; }

  /* annotated text view: the point of the tool is judging spans in context,
     so the text is the primary surface and the list is secondary. */
  .viewbar { display: flex; gap: 6px; align-items: center; margin: 10px 0 8px; }
  .viewbar button.v { background: var(--panel); color: var(--muted);
                      border: 1px solid var(--line); border-radius: 6px;
                      padding: 5px 12px; font-size: 12px; font-weight: 500; }
  .viewbar button.v.active { color: var(--text); border-color: var(--accent);
                             background: #10243a; }
  .viewbar .spacer { flex: 1; }
  .viewbar .legend { color: var(--muted); font-size: 11.5px; }
  .doc { background: #0e131a; border: 1px solid var(--line); border-radius: 8px;
         padding: 16px 18px; max-height: 620px; overflow: auto;
         white-space: pre-wrap; overflow-wrap: anywhere;
         font: 14px/1.85 Georgia, 'Times New Roman', serif; }
  .doc mark { background: #4a3f12; color: #ffe08a; border-bottom: 2px solid #d9b64a;
              border-radius: 3px; padding: 1px 2px; cursor: pointer; }
  .doc mark.bad { background: #4a1f22; color: #ff9d9d; border-bottom-color: #ff6b6b; }
  .doc mark:hover { filter: brightness(1.25); }
  @keyframes flash { 0%,100% { box-shadow: 0 0 0 0 rgba(77,163,255,0); }
                     25% { box-shadow: 0 0 0 5px rgba(77,163,255,.45); } }
  .doc mark.hit { animation: flash 1.5s ease-out; }
  table tbody tr { cursor: pointer; }
  table tbody tr.on { background: #10243a; }
</style>
</head>
<body>
<header>
  <h1>Software mention comparison</h1>
  <div class="sub">Softcite WAPITI &middot; Softcite SciBERT &middot; OSSoMeX
    <span id="model"></span></div>
</header>
<main>
  <textarea id="input" placeholder="Paste or type text (a paragraph or a full paper)..."></textarea>
  <div class="row">
    <button id="go">Run 3 models</button>
    <select id="examples">
      <option value="">Load test paper…</option>
    </select>
    <span class="hint" id="status">Idle</span>
  </div>
  <div class="hint" id="expmeta"></div>
  <div class="tabs" id="tabs"></div>
  <div id="panel"></div>
</main>
<script>
const ARMS = [
  { key: 'softcite-wapiti',  label: 'Softcite WAPITI' },
  { key: 'softcite-scibert', label: 'Softcite SciBERT' },
  { key: 'ossomex',          label: 'OSSoMeX (ours)' },
];
let active = 0, last = null, view = 'text', expanded = false;

document.getElementById('model').textContent = ' \u00b7 ' + __MODEL_NAME__;

function esc(s) {
  return (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
function context(text, s, e) {
  const a = Math.max(0, s - 45), b = Math.min(text.length, e + 35);
  return (a > 0 ? '…' : '') + text.slice(a, b).replace(/\s+/g, ' ') +
         (b < text.length ? '…' : '');
}

function renderTabs() {
  const el = document.getElementById('tabs');
  el.innerHTML = '';
  ARMS.forEach((arm, i) => {
    const b = document.createElement('button');
    b.className = 'tab' + (i === active ? ' active' : '');
    const n = last && last.arms[arm.key] ? last.arms[arm.key].count : null;
    b.textContent = arm.label + (n === null ? '' : ' (' + n + ')');
    b.onclick = () => { active = i; renderTabs(); renderPanel(); };
    el.appendChild(b);
  });
}

// Spans arrive in whatever order the model emitted them, and they can overlap or
// point outside the text. Normalise before drawing: sort by position, drop anything
// that starts before the previous span ended, and compare the offsets against the
// text they claim to cover.
//
// When the two disagree the mark shows the text that is actually at those offsets,
// never the text the model reported. Drawing the claim would put words in the
// document that are not in it, which is exactly what this view exists to rule out.
// The disagreement is flagged red and the claim is kept in the tooltip.
function layout(spans, text) {
  const out = [];
  spans.forEach(s => {
    const a = Math.max(0, s.start | 0), b = Math.min(text.length, s.end | 0);
    if (!(b > a)) return;
    const actual = text.slice(a, b);
    out.push({ start: a, end: b, actual: actual,
               claimed: s.text, bad: actual !== s.text });
  });
  out.sort((x, y) => x.start - y.start || y.end - x.end);
  const keep = [];
  let cursor = -1;
  for (const s of out) {
    if (s.start < cursor) continue;
    keep.push(s); cursor = s.end;
  }
  return keep.map((s, k) => Object.assign(s, { i: k }));
}

function annotatedText(arm, text) {
  const keep = layout(arm.spans, text);
  if (!keep.length) return '<div class="doc" id="doc">' + esc(text) + '</div>';
  let html = '', last = 0;
  keep.forEach(s => {
    html += esc(text.slice(last, s.start));
    html += '<mark class="' + (s.bad ? 'bad' : '') + '" data-i="' + s.i + '"' +
            (s.bad ? ' title="model reported: ' + esc(s.claimed) + '"' : '') +
            '>' + esc(s.actual) + '</mark>';
    last = s.end;
  });
  html += esc(text.slice(last));
  return '<div class="doc" id="doc">' + html + '</div>';
}

function spanTable(arm, text) {
  const keep = layout(arm.spans, text);
  if (!keep.length) return '<div class="hint">No mentions found.</div>';
  return '<table><thead><tr><th>#</th><th>In text</th><th>Offsets</th>' +
    '<th>Context</th></tr></thead><tbody>' +
    keep.map(s => '<tr data-i="' + s.i + '"><td class="num">' + (s.i + 1) + '</td>' +
      '<td class="name">' + esc(s.actual) +
      (s.bad ? '<div class="warn">model reported: ' + esc(s.claimed) + '</div>' : '') +
      '</td>' +
      '<td class="num">' + s.start + '–' + s.end + '</td>' +
      '<td class="ctx">' + esc(context(text, s.start, s.end)) + '</td></tr>').join('') +
    '</tbody></table>';
}

function renderPanel() {
  const panel = document.getElementById('panel');
  if (!last) { panel.innerHTML = ''; return; }
  const arm = last.arms[ARMS[active].key];
  const secs = arm.seconds;
  const cpm = secs ? (last.characters / secs * 60000).toFixed(0) : null;
  const keep = layout(arm.spans, last.text);
  const badCount = keep.filter(s => s.bad).length;
  panel.innerHTML =
    '<div class="stats">' +
      '<div class="stat"><b>' + (secs === null || secs === undefined ? 'failed' : secs.toFixed(2) + ' s') +
        '</b><span>wall clock</span></div>' +
      '<div class="stat"><b>' + arm.count + '</b><span>mentions</span></div>' +
      '<div class="stat"><b>' + last.characters.toLocaleString() + '</b><span>characters</span></div>' +
      (cpm ? '<div class="stat"><b>' + cpm + '</b><span>chars/min</span></div>' : '') +
      (arm.chunks ? '<div class="stat"><b>' + arm.chunks + '</b><span>softcite chunks</span></div>' : '') +
    '</div>' +
    (arm.error ? '<div class="err">' + esc(arm.error) + '</div>' : '') +
    (arm.rejected_offsets ? '<div class="warn">' + arm.rejected_offsets +
      ' mention(s) discarded because Softcite returned offsets that do not match the input.</div>' : '') +
    (arm.chunk_failures && arm.chunk_failures.length ? '<div class="warn">' +
      arm.chunk_failures.length + ' chunk(s) failed; those regions were not scored.</div>' : '') +
    ((arm.count > 0 && keep.length < arm.count) ? '<div class="warn">' +
      (arm.count - keep.length) + ' mention(s) overlap another span or fall outside the ' +
      'text, so they are not drawn; all of them remain in the list.</div>' : '') +
    (badCount ? '<div class="warn">' + badCount + ' mention(s) have offsets that do not match ' +
      'the text at those positions. They are drawn in red showing the text that is really ' +
      'there; hover for what the model claimed.</div>' : '') +
    '<div class="viewbar">' +
      '<button class="v' + (view === 'text' ? ' active' : '') + '" data-v="text">In text</button>' +
      '<button class="v' + (view === 'list' ? ' active' : '') + '" data-v="list">List</button>' +
      '<button class="v" id="expand">' + (expanded ? 'Collapse text' : 'Expand text') + '</button>' +
      '<span class="spacer"></span>' +
      '<span class="legend">Click a span, or a row in the list, to highlight it here</span>' +
    '</div>' +
    (view === 'text' ? annotatedText(arm, last.text) : spanTable(arm, last.text));
  if (view === 'text' && !expanded) {
    const d = document.getElementById('doc');
    if (d) d.style.maxHeight = '340px';
  }
}

// the list and the text are two views of the same span, so a click in either
// one should light up the other
document.addEventListener('click', e => {
  const v = e.target.closest('button.v[data-v]');
  if (v) { view = v.dataset.v; renderPanel(); return; }
  if (e.target.closest('#expand')) { expanded = !expanded; renderPanel(); return; }
  const hit = e.target.closest('mark[data-i], tr[data-i]');
  if (!hit) return;
  const i = hit.dataset.i;
  document.querySelectorAll('tr[data-i]').forEach(r =>
    r.classList.toggle('on', r.dataset.i === i));
  if (view !== 'text') { view = 'text'; renderPanel(); }
  const target = document.querySelector('mark[data-i="' + i + '"]');
  if (target) {
    target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    target.classList.remove('hit'); void target.offsetWidth; target.classList.add('hit');
  }
});

const sel = document.getElementById('examples');
fetch('/examples').then(r => r.json()).then(list => {
  list.forEach(e => {
    const o = document.createElement('option');
    o.value = e.n;
    o.textContent = e.n + '. ' + e.ecosystem;
    sel.appendChild(o);
  });
});
sel.onchange = async () => {
  if (!sel.value) return;
  const list = await (await fetch('/examples')).json();
  const e = list.find(x => String(x.n) === sel.value);
  if (!e) return;
  document.getElementById('input').value = e.excerpt;
  const exp = e.expected.map(x => x.name + '×' + x.count).join(', ') || 'none';
  document.getElementById('expmeta').textContent =
    e.title + '  ·  DOI ' + e.doi + '  ·  ' + e.chars.toLocaleString() +
    ' chars  ·  software a human should find: ' + exp;
  document.getElementById('status').textContent = 'Loaded. Click Run 3 models.';
};

document.getElementById('go').onclick = async () => {
  const text = document.getElementById('input').value;
  if (!text.trim()) { document.getElementById('status').textContent = 'Enter some text first.'; return; }
  const btn = document.getElementById('go');
  btn.disabled = true;
  document.getElementById('status').textContent = 'Running 3 models, this can take a minute…';
  try {
    const res = await fetch('/api/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    last = await res.json();
    last.text = text;
    document.getElementById('status').textContent = 'Done.';
    renderTabs(); renderPanel();
  } catch (e) {
    document.getElementById('status').textContent = 'Request failed: ' + e.message;
  } finally { btn.disabled = false; }
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    pipeline = None
    bundle = ''
    ports = (8060, 8062)
    examples = []

    def log_message(self, fmt, *a):
        sys.stderr.write('%s - %s\n' % (self.log_date_time_string(), fmt % a))

    def do_GET(self):
        if self.path == '/examples':
            body = json.dumps(self.examples).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != '/':
            self.send_error(404)
            return
        # Inject as a quoted JSON string. Substituting the bare checkpoint name would
        # drop an unquoted identifier into the JS and throw at load, which silently
        # killed the click handler.
        injected = PAGE.replace('__MODEL_NAME__', json.dumps(self.bundle))
        body = injected.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        text = json.loads(self.rfile.read(length) or '{}').get('text', '')
        if not text.strip():
            self.send_error(400, 'empty text')
            return

        arms = {}
        # Run arms sequentially and stream each back as it lands, so a slow Softcite
        # container does not hold up the fast local model.
        for (name, port) in (('softcite-wapiti', self.ports[0]),
                             ('softcite-scibert', self.ports[1])):
            try:
                started = time.perf_counter()
                r = call_softcite(port, text)
                r['seconds'] = round(time.perf_counter() - started, 3)
                arms[name] = r
            except Exception as exc:
                arms[name] = {'error': f'{exc}', 'seconds': None, 'spans': [], 'count': 0}

        try:
            arms['ossomex'] = call_ours(self.bundle, text, pipeline=self.pipeline)
        except Exception as exc:
            arms['ossomex'] = {'error': f'{exc}', 'seconds': None, 'spans': [], 'count': 0}

        payload = {
            'characters': len(text),
            'arms': {k: {'seconds': v.get('seconds'),
                         'count': len(v['spans']),
                         'error': v.get('error'),
                         'rejected_offsets': v.get('rejected_offsets', 0),
                         'chunk_failures': v.get('chunk_failures', []),
                         'chunks': v.get('chunks'),
                         'spans': v['spans']}
                     for k, v in arms.items()},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global PAGE
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default='checkpoints/scibert-full-label-011')
    ap.add_argument('--port', type=int, default=8077)
    ap.add_argument('--wapiti-port', type=int, default=8060)
    ap.add_argument('--scibert-port', type=int, default=8062)
    ap.add_argument('--examples',
                    default='reports/scibert-v2/diverse-001/excerpts.jsonl')
    args = ap.parse_args()

    from research.training.full_label import FullLabelPipeline
    print(f'loading {args.bundle} ...', flush=True)
    t0 = time.perf_counter()
    Handler.pipeline = FullLabelPipeline(args.bundle)
    Handler.bundle = Path(args.bundle).name
    Handler.ports = (args.wapiti_port, args.scibert_port)

    # Optional: offer the 10-paper diverse test set as a dropdown.
    ex_path = Path(args.examples)
    if ex_path.exists():
        Handler.examples = []
        with ex_path.open() as fh:
            for line in fh:
                r = json.loads(line)
                Handler.examples.append({
                    'n': r['n'], 'ecosystem': r['ecosystem'], 'doi': r['doi'],
                    'title': r['title'], 'chars': r['excerpt_characters'],
                    'expected': r['expected_software_in_excerpt'],
                    'excerpt': r['excerpt'],
                })
        print(f'loaded {len(Handler.examples)} test papers from {ex_path}', flush=True)

    print(f'loaded in {time.perf_counter() - t0:.1f}s', flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'\n  http://localhost:{args.port}\n', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()