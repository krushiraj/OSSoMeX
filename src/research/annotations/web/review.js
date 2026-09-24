const $ = id => document.getElementById(id);
let csrf = '', current = null, queueItems = [], target = null, versions = [], evidenceSpans = {intents: [], sentiment: []}, revealed = false, selection = null;
const chars = value => Array.from(value);
const intentBits = ['created', 'used', 'shared'];
const sourceSlice = (start, end) => chars(current.task.text).slice(start - current.task.offset_base, end - current.task.offset_base).join('');
const report = (text, error = false) => { $('message').textContent = text; $('message').className = error ? 'error' : ''; };
const api = async (path, body) => {
  const response = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
};
const safe = action => async event => { try { await action(event); } catch (error) { report(error.message, true); } };
const element = (tag, text) => { const el = document.createElement(tag); el.textContent = text; return el; };
const refreshQueue = async () => {
  const params = new URLSearchParams(['status', 'source', 'reason', 'field'].map(k => [k, $(`filter-${k}`).value]));
  const data = await api(`/api/queue?${params}`); queueItems = data.items;
  $('progress').textContent = `${data.progress.reviewed} of ${data.progress.total} name audits done`;
  $('queue').replaceChildren();
  for (const item of queueItems) {
    const row = element('div', ''); row.className = 'queue-item';
    row.append(element('p', item.document_id), element('p', item.reasons.join(', ') || item.status));
    const open = element('button', 'Open passage'); open.onclick = safe(() => openTask(item.task_id)); row.append(open); $('queue').append(row);
  }
  if (!queueItems.length) $('queue').textContent = 'No passages match these filters.';
};
const captureSelection = () => {
  const selected = window.getSelection();
  if (!selected.rangeCount || selected.isCollapsed) return;
  const range = selected.getRangeAt(0), passage = $('passage');
  if (!passage.contains(range.startContainer) || !passage.contains(range.endContainer)) return;
  const prefix = document.createRange(); prefix.selectNodeContents(passage); prefix.setEnd(range.startContainer, range.startOffset);
  const start = chars(prefix.toString()).length + current.task.offset_base;
  selection = {start, end: start + chars(range.toString()).length};
};
const renderPassage = () => {
  $('passage').replaceChildren();
  const all = revealed ? current.annotation.occurrences.flatMap(o => [{...o.name_span, kind: 'name'}, ...o.version_links.map(v => ({...v.span, kind: 'version'}))]) : [];
  const spans = all.sort((a, b) => a.start - b.start || b.end - a.end);
  let offset = current.task.offset_base;
  for (const span of spans) {
    if (span.start < offset) continue;
    $('passage').append(document.createTextNode(sourceSlice(offset, span.start)));
    const mark = element('mark', sourceSlice(span.start, span.end)); mark.className = span.kind; $('passage').append(mark); offset = span.end;
  }
  $('passage').append(document.createTextNode(sourceSlice(offset, current.task.context_span.end)));
  $('occurrences').replaceChildren();
  if (revealed) for (const o of current.annotation.occurrences) {
    const button = element('button', `${o.name} (${o.name_span.start}:${o.name_span.end})`); button.onclick = () => edit(o); $('occurrences').append(button);
  }
  $('reveal').hidden = revealed;
};
const openTask = async id => {
  current = await api(`/api/tasks/${encodeURIComponent(id)}`); target = null; selection = null;
  revealed = !current.task.whole_passage_audit;
  $('workspace').hidden = false; $('empty').hidden = true; $('editor-form').hidden = true;
  $('source-title').textContent = current.task.document_id;
  const region = current.task.annotation_region;
  $('boundaries').textContent = `Owned region ${region.start}:${region.end}. Revision ${current.annotation_revision}. ${current.task.split} / ${current.task.source}`;
  $('audit-message').textContent = revealed ? 'Check proposed spans and their evidence.' : 'Whole-passage audit: inspect the text for missed names before showing proposals.';
  $('selection-label').textContent = 'Select a proposed name or add one from the passage.';
  $('passage-reason').value = ''; report(''); renderPassage();
};
const showVersions = () => {
  $('version-list').replaceChildren();
  versions.forEach((v, index) => {
    const row = element('div', `${v.text} (${v.span.start}:${v.span.end})`);
    const remove = element('button', 'Unlink'); remove.type = 'button'; remove.onclick = () => { versions.splice(index, 1); showVersions(); }; row.append(remove); $('version-list').append(row);
  });
};
const showEvidence = () => { $('evidence-view').textContent = current ? sourceSlice(Number($('evidence-start').value), Number($('evidence-end').value)) : ''; };
const showEvidenceList = () => {
  $('evidence-list').replaceChildren();
  for (const field of ['intents', 'sentiment']) evidenceSpans[field].forEach((span, index) => {
    const row = element('div', `${field} ${span.start}:${span.end}: ${sourceSlice(span.start, span.end)}`);
    const remove = element('button', `Remove ${field} evidence ${index + 1}`); remove.type = 'button';
    remove.onclick = () => { evidenceSpans[field].splice(index, 1); showEvidenceList(); };
    row.append(remove); $('evidence-list').append(row);
  });
};
const addEvidence = () => {
  const start = $('evidence-start').value, end = $('evidence-end').value;
  if (start === '' && end === '') return;
  if (start === '' || end === '') throw new Error('Evidence needs both start and end.');
  const span = {start: Number(start), end: Number(end)};
  if (!Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < current.task.offset_base || span.end > current.task.context_span.end || span.start >= span.end) throw new Error('Evidence must be inside the source passage.');
  const list = evidenceSpans[$('evidence-field').value];
  if (!list.some(s => s.start === span.start && s.end === span.end)) list.push(span);
  $('evidence-start').value = ''; $('evidence-end').value = ''; showEvidence(); showEvidenceList();
};
const edit = o => {
  target = o ? {...o.name_span} : null; versions = o ? structuredClone(o.version_links) : [];
  evidenceSpans = o ? structuredClone(o.evidence) : {intents: [], sentiment: []};
  const span = o ? o.name_span : selection;
  if (!span) throw new Error('Select a name in the source passage first.');
  $('name-start').value = span.start; $('name-end').value = span.end;
  $('selection-label').textContent = sourceSlice(span.start, span.end); $('editor-form').hidden = false;
  for (const label of ['created', 'used', 'shared', 'mentioned']) $(`intent-${label}`).checked = (o?.intents || []).includes(label);
  for (const label of intentBits) $(`known-${label}`).value = o?.known?.[label] ? 'known' : 'unknown';
  $('sentiment').value = o?.sentiment || ''; $('version-state').value = o?.version_status || 'unannotated';
  $('evidence-field').value = 'intents'; $('evidence-start').value = ''; $('evidence-end').value = '';
  $('reason').value = ''; showEvidence(); showEvidenceList(); showVersions();
};
const save = async (action, extra = {}, reason = $('reason').value) => {
  if (!reason.trim()) throw new Error('Add a reason for this decision.');
  const result = await api('/api/decisions', {decision_id: crypto.randomUUID(), task_id: current.task.task_id,
    document_id: current.task.document_id, text_revision: current.task.text_revision, base_annotation_revision: current.annotation_revision,
    reviewer: 'Krushi', action, reason, value: null, target_name_span: target, ...extra});
  const wasRevealed = revealed; await openTask(current.task.task_id); revealed = wasRevealed; renderPassage(); await refreshQueue();
  report(`Saved revision ${result.annotation_revision}`);
};
$('editor-form').onsubmit = safe(async event => {
  event.preventDefault();
  const span = {start: Number($('name-start').value), end: Number($('name-end').value)};
  const knownIntents = Object.fromEntries(intentBits.map(k => [k, $(`known-${k}`).value === 'known']));
  const intents = intentBits.filter(k => $(`intent-${k}`).checked);
  if (!intents.length && Object.values(knownIntents).every(Boolean)) intents.push('mentioned');
  const sentiment = $('sentiment').value || null, versionStatus = $('version-state').value;
  addEvidence();
  const value = {schema_version: '2.0', document_id: current.task.document_id, text_revision: current.task.text_revision,
    name: sourceSlice(span.start, span.end), name_span: span, context_sentence: current.task.text, context_span: current.task.context_span,
    context_kind: 'paragraph', version_links: versions, version_status: versionStatus, intents: Object.values(knownIntents).some(Boolean) ? intents : null, sentiment,
    known: {software: true, versions: ['explicit', 'absent'].includes(versionStatus), ...knownIntents, sentiment: sentiment !== null},
    evidence: structuredClone(evidenceSpans), review: {status: 'pending', reasons: []}};
  await save('upsert_occurrence', {value});
});
$('passage').onmouseup = captureSelection;
$('add-name').onpointerdown = captureSelection;
$('add-name').onclick = safe(() => edit(null));
$('add-version').onpointerdown = captureSelection;
$('add-version').onclick = safe(() => {
  if (!selection || $('editor-form').hidden) throw new Error('Open an occurrence, then select its version.');
  versions.push({text: sourceSlice(selection.start, selection.end), span: {...selection}, status: 'explicit_local'});
  $('version-state').value = 'explicit'; showVersions();
});
$('use-evidence').onpointerdown = captureSelection;
$('use-evidence').onclick = safe(() => { if (!selection || $('editor-form').hidden) throw new Error('Open an occurrence, then select evidence.'); $('evidence-start').value = selection.start; $('evidence-end').value = selection.end; showEvidence(); });
$('add-evidence').onclick = safe(addEvidence);
for (const label of ['created', 'used', 'shared', 'mentioned']) $(`intent-${label}`).onchange = () => {
  if (label === 'mentioned') {
    for (const other of intentBits) {
      $(`intent-${other}`).checked = false;
      $(`known-${other}`).value = $('intent-mentioned').checked ? 'known' : 'unknown';
    }
  } else {
    $(`known-${label}`).value = 'known';
    $('intent-mentioned').checked = intentBits.every(k => !$(`intent-${k}`).checked && $(`known-${k}`).value === 'known');
  }
};
for (const label of intentBits) $(`known-${label}`).onchange = () => {
  if ($(`known-${label}`).value === 'unknown') $(`intent-${label}`).checked = false;
  $('intent-mentioned').checked = intentBits.every(k => !$(`intent-${k}`).checked && $(`known-${k}`).value === 'known');
};
for (const id of ['evidence-start', 'evidence-end']) $(id).oninput = showEvidence;
$('reveal').onclick = () => { revealed = true; renderPassage(); };
$('accept').onclick = safe(() => save('accept_occurrence'));
$('remove').onclick = safe(() => save('remove_occurrence'));
$('finish-audit').onclick = safe(() => save('accept_passage', {}, $('passage-reason').value));
$('filter').onclick = safe(refreshQueue);
$('export').onclick = safe(async () => { const result = await api('/api/export', {}); $('export-result').textContent = `${result.quality} export: ${result.path}`; });
const move = async delta => { const index = queueItems.findIndex(t => t.task_id === current?.task.task_id); const next = queueItems[index + delta]; if (next) await openTask(next.task_id); };
$('next').onclick = safe(() => move(1)); $('previous').onclick = safe(() => move(-1));
document.onkeydown = safe(event => { if (event.altKey && ['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); return move(event.key === 'ArrowRight' ? 1 : -1); } });
safe(async () => { const session = await api('/api/session'); csrf = session.csrf_token; $('mode').textContent = `${session.role === 'demo' ? 'Synthetic demo. No research labels.' : session.role + ' review.'} Local only. References remain provisional.`; await refreshQueue(); })();
