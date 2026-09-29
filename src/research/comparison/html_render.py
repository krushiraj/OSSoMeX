"""Self-contained, offline presentation for verified comparison evidence."""

import html
import json
import math
from urllib.parse import urlsplit


_CSS = """
:root{color-scheme:light;--paper:#f5f8fc;--white:#fff;--ink:#1c3047;--muted:#52667c;
--rule:#cad6e5;--blue:#2459b3;--green:#176d54;--amber:#805900;--error:#a82d44}
*{box-sizing:border-box}body{margin:0;background:var(--paper);
color:var(--ink);font:16px/1.55 'Avenir Next','Trebuchet MS',Arial,sans-serif}
a{color:var(--blue);text-underline-offset:.16em;overflow-wrap:anywhere}a:focus-visible,summary:focus-visible{
outline:3px solid var(--blue);outline-offset:3px}h1,h2,h3,h4,p{margin-top:0}
h1{font-size:clamp(2rem,4vw,3.35rem);line-height:1.13;letter-spacing:-.035em;max-width:22ch}
h2{font-size:1.65rem;line-height:1.25;letter-spacing:-.02em}h3{font-size:1.18rem;line-height:1.3}
header,main,footer{max-width:1480px;margin:auto;padding-left:clamp(18px,3vw,52px);
padding-right:clamp(18px,3vw,52px)}header{padding-top:52px;padding-bottom:28px;
border-bottom:2px solid var(--ink)}header p{max-width:72ch;color:var(--muted)}
.masthead{display:grid;grid-template-columns:minmax(0,2fr) minmax(235px,1fr);gap:32px;align-items:end}
.evidence-stamp{border-left:4px solid var(--blue);padding:8px 0 8px 17px;font-size:.92rem}
.evidence-stamp strong,.evidence-stamp span{display:block;color:var(--ink)}nav{display:flex;gap:6px 22px;flex-wrap:wrap;
margin-top:27px}nav a{font-size:.94rem;font-weight:650;text-decoration:none;border-bottom:2px solid transparent}
nav a:hover,nav a:focus-visible{border-bottom-color:currentColor}main{padding-top:30px;padding-bottom:80px}
main>section{padding:38px 0;border-bottom:1px solid var(--rule)}.section-head{max-width:78ch}
.section-head p,.muted{color:var(--muted)}.summary-lines{border-left:3px solid var(--blue);
padding-left:15px;max-width:78ch;margin:18px 0 24px}.summary-lines p{margin:0 0 5px}
.meta-list{display:grid;grid-template-columns:minmax(150px,210px) minmax(0,1fr);gap:7px 18px;
max-width:1000px;margin:0}.meta-list dt{font-weight:650;color:var(--muted)}.meta-list dd{margin:0;overflow-wrap:anywhere}
.limitations{margin:22px 0 0;padding:15px 18px;background:#fff7e6;border-left:4px solid var(--amber);
max-width:1000px}.limitations p:last-child{margin-bottom:0}.table-scroll{overflow-x:auto;background:var(--white);
border:1px solid var(--rule)}table{border-collapse:collapse;width:100%;min-width:760px;text-align:left}
.metric-primer{max-width:1000px;background:#eaf1f9;padding:16px 18px;margin:22px 0}
.metric-primer h3{margin-bottom:5px}.metric-primer p{margin:0 0 6px}.metric-primer p:last-child{margin:0}
th,td{padding:11px 13px;border-bottom:1px solid #dfe7f1;vertical-align:top}th{font-size:.85rem;
font-weight:750;background:#eaf1f9;color:var(--ink)}tbody tr:last-child td{border-bottom:0}
.arm-name{font-weight:700;overflow-wrap:anywhere}.status{font-size:.82rem;font-weight:700;
display:inline-block;padding:2px 8px;background:#e8f1ed;color:var(--green)}
.status.unsupported,.status.unavailable{background:#edf1f6;color:var(--muted)}
.status.failure,.status.partial{background:#fbe8ec;color:var(--error)}
.metric{font-variant-numeric:tabular-nums;white-space:nowrap}.na{color:var(--muted)}
.counts{font-variant-numeric:tabular-nums;white-space:nowrap}.bar-track{width:128px;height:9px;
background:#e8eef5;margin:6px 0 4px}.bar-fill{display:block;height:100%;background:var(--blue)}
.metric-version .bar-fill{background:var(--green)}.metric-links .bar-fill{background:var(--amber)}
.secondary-group{margin-top:16px}.secondary-group summary{font-weight:700;color:var(--blue);
cursor:pointer;padding:8px 0}.table-note{font-size:.88rem;color:var(--muted);margin:10px 0 0}
.passage{padding:30px 0 36px;border-top:1px solid var(--rule)}.passage:first-of-type{border-top:0}
.passage-index{margin:18px 0 28px;gap:7px 10px}.passage-index a{background:#eaf1f9;
border:1px solid var(--rule);padding:5px 9px;font-size:.86rem;max-width:100%}
.back-link{display:inline-block;margin-top:18px;font-size:.88rem;font-weight:700}
.passage-heading{display:flex;align-items:baseline;gap:12px 25px;flex-wrap:wrap;margin-bottom:16px}
.passage-heading h3{margin:0}.revision{font-size:.8rem;color:var(--muted);overflow-wrap:anywhere}
.passage-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.18fr);gap:18px;align-items:start}
.reference-panel,.model-panel{background:var(--white);border:1px solid var(--rule);padding:19px;
min-width:0}.reference-panel{border-top:4px solid var(--blue)}.model-panel{border-top:4px solid var(--green)}
.panel-heading{font-size:.85rem;font-weight:750;color:var(--muted);margin-bottom:10px}
.source-text{font:1.05rem/1.75 Georgia,'Times New Roman',serif;white-space:pre-wrap;
overflow-wrap:anywhere;background:#f7faff;padding:14px 16px;border-left:2px solid var(--rule);
margin:10px 0 17px}mark.span{border-radius:2px;padding:0 1px;color:inherit}
mark.software{background:#dce8ff;border-bottom:2px solid var(--blue)}
mark.version{background:#e0f2e9;border-bottom:2px solid var(--green)}
mark.software.version{background:linear-gradient(90deg,#dce8ff 50%,#e0f2e9 50%);
border-bottom:2px solid var(--amber)}
.legend{font-size:.82rem;color:var(--muted)}.legend span{margin-right:15px;white-space:nowrap}
.model-output{border-top:1px solid var(--rule);padding:12px 0}.model-output:first-of-type{border-top:0}
.model-output summary{cursor:pointer;list-style-position:outside;font-weight:700;padding:2px 0 4px}
.model-output summary .status{margin-left:8px}.model-output .source-text{font-size:.98rem}
.span-audit table{min-width:650px}.field-table table{min-width:800px}.span-list{margin:0;padding-left:18px}
.span-list li{margin:2px 0}.coverage-note{font-size:.87rem;color:var(--muted);margin:8px 0}
.minor-heading{font-size:.9rem;font-weight:750;margin:17px 0 4px}.edge-list{margin:5px 0 10px;padding-left:22px}
.edge-list li{margin:3px 0}.mask{background:#fff7e6;border-left:3px solid var(--amber);
padding:9px 12px;margin:12px 0}.mask p{margin:0 0 4px}.mask p:last-child{margin:0}
.error{color:var(--error)}.note-list{margin:8px 0;padding-left:20px}.note-list li{margin:3px 0}
.feedback{background:#fff7e6;border-left:4px solid var(--amber);padding:16px 19px;
max-width:90ch;margin:14px 0}.feedback h3{margin-bottom:6px}.feedback p{margin-bottom:5px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff4fa;padding:13px;font:12px/1.5 ui-monospace,
SFMono-Regular,Menlo,Consolas,monospace;max-height:420px;overflow:auto}
.appendix-example{padding:18px 0;border-top:1px solid var(--rule);min-width:0}
.appendix-example p,.appendix-example li{overflow-wrap:anywhere}.appendix-example:first-of-type{border:0}
.appendix-example h3{margin-bottom:5px}footer{padding-bottom:32px;color:var(--muted);font-size:.86rem}
@media(max-width:780px){.masthead,.passage-grid{grid-template-columns:1fr}.evidence-stamp{max-width:60ch}
header{padding-top:30px}.reference-panel,.model-panel{padding:15px}}
@media(max-width:390px){body{font-size:15px}.source-text{padding:11px}.bar-track{width:95px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:#fff;color:#000}header,main,footer{max-width:none;padding-left:0;padding-right:0}
nav{display:none}main>section,.passage{break-inside:avoid}.reference-panel,.model-panel,.table-scroll{
border-color:#999}details:not([open])>*:not(summary){display:block}.bar-fill{print-color-adjust:exact}}
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_span(source, span):
    start, end = span.get('start'), span.get('end')
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or end > len(source) or span.get('text') != source[start:end]):
        raise ValueError('span offsets/text differ from frozen source')
    return start, end


def _highlight(source, spans):
    validated = []
    breaks = {0, len(source)}
    for span in spans:
        start, end = _validate_span(source, span)
        label = span.get('label')
        if label not in ('SOFTWARE', 'VERSION'):
            raise ValueError('span label must be SOFTWARE or VERSION')
        validated.append((start, end, label.lower()))
        breaks.update((start, end))
    points = sorted(breaks)
    parts = []
    for start, end in zip(points, points[1:]):
        labels = [label for label in ('software', 'version')
                  if any(lo <= start < end <= hi and kind == label for lo, hi, kind in validated)]
        fragment = _escape(source[start:end])
        if labels:
            parts.append(f'<mark class="span {" ".join(labels)}">{fragment}</mark>')
        else:
            parts.append(fragment)
    return ''.join(parts)


def _score(value):
    if value is None:
        return '<span class="na">N/A</span>'
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('metric score must be finite and between zero and one')
    return f'{value * 100:.1f}%'


def _count(value):
    return '<span class="na">N/A</span>' if value is None else _escape(value)


def _status(value):
    status = str(value)
    style = status if status in ('unsupported', 'unavailable', 'failure', 'partial') else 'ready'
    return f'<span class="status {style}">{_escape(status)}</span>'


def _summary(lines):
    if not lines:
        return ''
    return '<div class="summary-lines">' + ''.join(f'<p>{_escape(line)}</p>' for line in lines) + '</div>'


def _metric_row(model, field):
    metric = model.get('metrics', {}).get(field)
    arm = _escape(model['arm_id'])
    source = model.get('span_source_arm') if field != 'links' else None
    source_note = f'<div class="table-note">Span source: {_escape(source)}</div>' if source else ''
    note = '<div class="table-note">' + _escape('; '.join(model.get('notes', []))) + '</div>' if model.get('notes') else ''
    if metric is None:
        counts = '<span class="na">N/A</span>'
        p = r = f1 = '<span class="na">N/A</span>'
        bar = ''
    else:
        counts = ' / '.join(_count(metric.get(key)) for key in ('tp', 'fp', 'fn'))
        p, r, f1 = (_score(metric.get(key)) for key in ('precision', 'recall', 'f1'))
        value = metric.get('f1')
        bar = '' if value is None else f'<div class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:{value * 100:.1f}%"></span></div>'
    excluded = model.get('excluded_predictions')
    extra = f'<div class="table-note">{_escape(excluded)} excluded predictions</div>' if field == 'links' and excluded is not None else ''
    return (f'<tr><td><span class="arm-name">{arm}</span>{source_note}{note}</td><td>{_status(model["status"])}</td>'
            f'<td class="counts">{counts}</td><td class="metric">{p}</td><td class="metric">{r}</td>'
            f'<td class="metric">{bar}{f1}{extra}</td></tr>')


def _metric_table(models, field):
    return (f'<div class="table-scroll"><table class="metric-{field}"><thead><tr>'
            '<th scope="col">Model</th><th scope="col">Status</th><th scope="col">TP / FP / FN</th>'
            '<th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th>'
            '</tr></thead><tbody>' + ''.join(_metric_row(model, field) for model in models) + '</tbody></table></div>')


def _metric_section(data, field, title, description):
    primary = [m for m in data['models'] if m['kind'] in ('base', 'pipeline', 'softcite')]
    secondary = [m for m in data['models'] if m['kind'] not in ('base', 'pipeline', 'softcite')]
    section = [f'<section id="{field}"><div class="section-head"><h2>{title}</h2><p>{description}</p></div>',
               _summary(data.get('section_summaries', {}).get(field, [])), _metric_table(primary, field)]
    if secondary:
        section += ['<details class="secondary-group"><summary>Other comparison arms</summary>',
                    _metric_table(secondary, field), '</details>']
    section.append('</section>')
    return ''.join(section)


def _model_inventory(models):
    parts = ['<h3 class="minor-heading">Model identities</h3>',
             '<p class="muted">Expand an arm for its supplied identity and operational context.</p>']
    for model in models:
        parts.append('<details class="model-output"><summary>' + _escape(model['arm_id']) + ' ' +
                     _status(model['status']) + '</summary>')
        if model.get('span_source_arm'):
            parts.append('<p>Span score source: ' + _escape(model['span_source_arm']) + '</p>')
        for title, key in (('Identity', 'identity'), ('Operations', 'operations'), ('Timing', 'timing')):
            if model.get(key):
                parts.append('<h4 class="minor-heading">' + title + '</h4><pre>' + _json(model[key]) + '</pre>')
        parts.append(_notes(model.get('notes', [])))
        parts.append('</details>')
    return ''.join(parts)


def _endpoint(source, edge, key):
    endpoint = edge[key]
    start, end = _validate_span(source, endpoint)
    return source[start:end]


def _edge_list(source, edges, *, empty='None'):
    if not edges:
        return f'<p class="muted">{empty}</p>'
    return '<ul class="edge-list">' + ''.join(
        f'<li>{_escape(_endpoint(source, edge, "software"))} → '
        f'{_escape(_endpoint(source, edge, "version"))}</li>' for edge in edges) + '</ul>'


def _offset_edges(source, edges):
    if not edges:
        return '<p class="muted">None</p>'
    items = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 4:
            raise ValueError('link edge requires four source offsets')
        a, b, c, d = edge
        for start, end in ((a, b), (c, d)):
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
                raise ValueError('link edge offset outside frozen source')
        items.append(f'<li>{_escape(source[a:b])} → {_escape(source[c:d])}</li>')
    return '<ul class="edge-list">' + ''.join(items) + '</ul>'


def _notes(lines):
    return '<ul class="note-list">' + ''.join(f'<li>{_escape(note)}</li>' for note in lines) + '</ul>' if lines else ''


def _mask(source, document):
    versions, software = document.get('ignored_versions', []), document.get('ignored_software', [])
    if not versions and not software:
        return '<p class="muted">No ignored ownership endpoints in this passage.</p>'
    for span in versions + software:
        _validate_span(source, span)
    names = ', '.join(_escape(span['text']) for span in software) or 'none'
    values = ', '.join(_escape(span['text']) for span in versions) or 'none'
    return (f'<div class="mask"><p><strong>Ignored versions:</strong> {values}</p>'
            f'<p><strong>Ignored software:</strong> {names}</p>'
            '<p class="muted">These endpoints are excluded from ownership scoring, including wrong predicted links that touch them.</p></div>')


def _coverage_regions(source, rows, review_kind, label=None):
    regions = []
    for row in rows:
        if row.get('review_kind') != review_kind or row.get('complete') is not True:
            continue
        if label is not None and row.get('label') != label:
            continue
        start, end = row.get('start'), row.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
            raise ValueError('coverage region outside frozen source')
        regions.append((start, end))
    return regions


def _coverage_view(source, document, review_kind):
    parts = ['<h4 class="minor-heading">Scored coverage</h4>',
             '<p class="coverage-note">Complete regions for ' + _escape(review_kind.replace('_', ' ')) +
             '; spans outside these regions are shown in the source but left unclassified.</p><ul class="note-list">']
    for label in ('SOFTWARE', 'VERSION'):
        regions = _coverage_regions(source, document.get('coverage', []), review_kind, label)
        value = ', '.join(f'{start}:{end}' for start, end in regions) if regions else f'No complete {label} coverage'
        parts.append(f'<li><strong>{label}:</strong> {_escape(value)}</li>')
    regions = _coverage_regions(source, document.get('link_coverage', []), review_kind)
    value = ', '.join(f'{start}:{end}' for start, end in regions) if regions else 'No complete ownership coverage'
    parts.append(f'<li><strong>Ownership:</strong> {_escape(value)}</li></ul>')
    return ''.join(parts)


def _span_list(source, keys):
    if not keys:
        return '<span class="na">None</span>'
    return '<ul class="span-list">' + ''.join(
        f'<li>{_escape(source[start:end])} <span class="revision">[{start}:{end}]</span></li>'
        for start, end in sorted(keys)) + '</ul>'


def _span_audit(source, document, arm, review_kind):
    if arm['status'] in ('unsupported', 'unavailable'):
        return '<p class="coverage-note">Span comparison unavailable for this arm.</p>'
    parts = ['<h4 class="minor-heading">Exact span review</h4>',
             '<p class="coverage-note">Matched, extra and missed text offsets within complete reference coverage. '
             'These lists are a review aid; aggregate metrics remain in the score tables.</p>',
             '<div class="table-scroll span-audit"><table><thead><tr><th scope="col">Field</th>'
             '<th scope="col">Matched spans</th><th scope="col">Extra spans</th>'
             '<th scope="col">Missed spans</th></tr></thead><tbody>']
    for label in ('SOFTWARE', 'VERSION'):
        regions = _coverage_regions(source, document.get('coverage', []), review_kind, label)
        if not regions:
            parts.append(f'<tr><th scope="row">{label}</th><td colspan="3" class="na">'
                         f'No complete {label} coverage</td></tr>')
            continue
        def eligible(span):
            start, end = _validate_span(source, span)
            return span.get('label') == label and any(lo <= start < end <= hi for lo, hi in regions)
        reference = {(span['start'], span['end']) for span in document.get('reference_spans', []) if eligible(span)}
        predicted = {(span['start'], span['end']) for span in arm.get('spans', []) if eligible(span)}
        parts.append(f'<tr><th scope="row">{label}</th><td>{_span_list(source, reference & predicted)}</td>'
                     f'<td>{_span_list(source, predicted - reference)}</td>'
                     f'<td>{_span_list(source, reference - predicted)}</td></tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def _field_value(row, key):
    field = row.get(key)
    if not isinstance(field, dict):
        return 'unknown'
    if field.get('status') != 'success':
        return 'unknown (' + str(field.get('status', 'missing')) + ')'
    value = field.get('value')
    if key == 'versions':
        return ', '.join(str(item.get('text', '')) for item in value if isinstance(item, dict)) or 'none predicted'
    if isinstance(value, list):
        return ', '.join(map(str, value)) or 'none predicted'
    return str(value) if value is not None else 'unknown'


def _field_table(fields, groups, *, scored=False):
    names = {row.get('mention_id'): str(row.get('name', 'unknown')) for row in fields}
    parts = ['<h4 class="minor-heading">' + ('Native field outputs' if scored else 'Other field outputs (unscored)') + '</h4>',
             '<p class="coverage-note">' + ('Native pipeline proposals; see selected-review aggregate field scores above.' if scored
                                         else 'Native pipeline proposals; intent, sentiment and aliases have no score in this report.') + '</p>']
    if fields:
        parts.append('<div class="table-scroll field-table"><table><thead><tr>'
                     '<th scope="col">Software</th><th scope="col">Intent</th>'
                     '<th scope="col">Version</th><th scope="col">Sentiment</th>'
                     '<th scope="col">Alias group members</th></tr></thead><tbody>')
        for row in fields:
            members = []
            for group in groups:
                ids = group.get('member_mention_ids', [])
                if row.get('mention_id') in ids:
                    members.extend(names[ident] for ident in ids if ident in names)
            alias = ', '.join(dict.fromkeys(members)) or 'No group shown'
            parts.append('<tr><td>' + _escape(row.get('name', 'unknown')) + '</td><td>' +
                         _escape(_field_value(row, 'intents')) + '</td><td>' +
                         _escape(_field_value(row, 'versions')) + '</td><td>' +
                         _escape(_field_value(row, 'sentiment')) + '</td><td>' +
                         _escape(alias) + '</td></tr>')
        parts.append('</tbody></table></div>')
    if groups and not fields:
        parts.append('<p class="coverage-note">Alias groups are present without field rows; inspect the raw output.</p>')
    parts.append('<details><summary>Raw field JSON</summary><pre>' +
                 _json({'field_predictions': fields, 'alias_groups': groups}) + '</pre></details>')
    return ''.join(parts)


def _arm_output(source, document, arm, review_kind, *, open_by_default, fields_scored=False):
    parts = [f'<details class="model-output"{" open" if open_by_default else ""}>'
             f'<summary>{_escape(arm["arm_id"])} {_status(arm["status"])}</summary>',
             f'<div class="source-text">{_highlight(source, arm.get("spans", []))}</div>']
    parts.append(_span_audit(source, document, arm, review_kind))
    parts.append('<h4 class="minor-heading">Predicted version owners</h4>')
    parts.append(_edge_list(source, arm.get('version_links', [])))
    details = arm.get('link_details')
    if details:
        excluded = details.get('excluded_predictions')
        if excluded is not None:
            label = 'prediction' if excluded == 1 else 'predictions'
            parts.append(f'<p class="mask">{_escape(excluded)} excluded {label} in this passage; '
                         'masked endpoints are outside the ownership score.</p>')
        parts.append('<h4 class="minor-heading error">False positive links</h4>')
        parts.append(_offset_edges(source, details.get('false_positive_edges', [])))
        parts.append('<h4 class="minor-heading">Missed links</h4>')
        parts.append(_offset_edges(source, details.get('missed_edges', [])))
    parts.append(_notes(arm.get('notes', [])))
    if arm.get('fields') or arm.get('alias_groups'):
        parts.append(_field_table(arm.get('fields', []), arm.get('alias_groups', []), scored=fields_scored))
    parts.append('</details>')
    return ''.join(parts)


def _document(document, models, review_kind, anchor, field_scores=None):
    source = document['text']
    kinds = {model['arm_id']: model['kind'] for model in models}
    source_html = _highlight(source, document.get('reference_spans', []))
    arms = sorted(document.get('arms', []), key=lambda arm: (kinds.get(arm['arm_id']) not in ('pipeline', 'softcite'), arm['arm_id']))
    parts = [f'<article class="passage" id="{anchor}">',
             f'<div class="passage-heading"><h3>{_escape(document["document_id"])}</h3>'
             f'<span class="revision">{_escape(document["text_revision"])}</span></div>',
             _summary(document.get('summaries', [])), '<div class="passage-grid">',
             '<section class="reference-panel"><div class="panel-heading">Frozen source and ' +
             _escape('Human reviewed reference' if review_kind == 'human_reviewed' else
                     'Agent provisional reference' if review_kind == 'agent_provisional' else
                     'Reference annotations') + '</div>',
             f'<div class="source-text">{source_html}</div>',
             '<p class="legend"><span>Blue: software name</span><span>Green: version</span></p>',
             _coverage_view(source, document, review_kind),
             '<h4 class="minor-heading">Reference version owners</h4>',
             _edge_list(source, document.get('reference_links', [])), _mask(source, document), '</section>',
             '<section class="model-panel"><div class="panel-heading">Model annotations on the same text</div>']
    parts.extend(_arm_output(source, document, arm, review_kind,
                             open_by_default=kinds.get(arm['arm_id']) in ('pipeline', 'softcite'),
                             fields_scored=field_scores is not None and arm['arm_id'] in field_scores) for arm in arms)
    parts.extend(['</section></div><a class="back-link" href="#passages">Back to passages</a></article>'])
    return ''.join(parts)


def _safe_link(url, title):
    url = str(url)
    parsed = urlsplit(url)
    if parsed.scheme not in ('https', 'http') or not parsed.netloc or any(ord(char) < 32 for char in url):
        return _escape(title)
    return f'<a href="{_escape(url)}" rel="noopener noreferrer">{_escape(title)}</a>'


def _appendix(appendix):
    if not appendix:
        return ''
    parts = ['<section id="appendix"><h2>' + _escape(appendix.get('title', 'Appendix')) + '</h2>',
             '<p>' + _escape(appendix.get('summary', '')) + '</p>']
    if appendix.get('sources'):
        parts.append('<h3>Sources</h3><ul>')
        parts.extend('<li>' + _safe_link(item.get('url', ''), item.get('title', 'Source')) + '</li>'
                     for item in appendix['sources'])
        parts.append('</ul>')
    for example in appendix.get('examples', []):
        value = example.get('output')
        displayed = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        parts += ['<article class="appendix-example"><h3>' + _escape(example.get('title', 'Example')) + '</h3>',
                  '<p><strong>Kind:</strong> ' + _escape(example.get('kind', 'unspecified')) + '</p>',
                  '<p>' + _escape(example.get('note', '')) + '</p>',
                  '<h4 class="minor-heading">Input</h4><pre>' + _escape(example.get('input', '')) + '</pre>',
                  '<h4 class="minor-heading">Output</h4><pre>' + _escape(displayed) + '</pre>']
        if example.get('source_url'):
            parts.append('<p>Source: ' + _safe_link(example['source_url'], example['source_url']) + '</p>')
        parts.append('</article>')
    if appendix.get('evidence'):
        parts.append('<details><summary>Additional evidence</summary><pre>' + _json(appendix['evidence']) + '</pre></details>')
    parts.append('</section>')
    return ''.join(parts)


def _timing_table(rows):
    parts = ['<section id="timing"><div class="section-head"><h2>Full-pipeline timing</h2>',
             '<p>Document-level prediction after a synthetic warmup on the same frozen passages. Load and warmup are separate; '
             'the warmup does not guarantee every attribute head was exercised. '
             'Softcite load is a service health check, not cold model load. CPU/MPS and AMD64 service results are separate environments.</p></div>',
             '<div class="table-scroll"><table><thead><tr><th>Device</th><th>Report arm / timing arm</th><th>Scheduled</th>',
             '<th>Success</th><th>Failure</th><th>Median</th><th>P95</th><th>Successful/s</th><th>Load</th></tr></thead><tbody>']
    for row in rows:
        summary = row['summary']
        load_label = 'service health' if row.get('load_semantics') == 'service_health_check_only' else 'local initialization'
        client = str(row['device']) + ' / ' + str(row['hardware'].get('machine', 'unknown'))
        device_label = 'Resident service; client ' + client if load_label == 'service health' else client
        def timing_value(value):
            return _count(None) if value is None else _escape(f'{value:.4g}')
        parts.append('<tr><td>' + _escape(device_label) +
                     '</td><td>' + _escape(row['arm_id']) + ' / ' + _escape(row['timing_arm_id']) +
                     '</td><td>' + _count(summary.get('scheduled')) + '</td><td>' + _count(summary.get('successful')) +
                     '</td><td>' + _count(summary.get('failed')) + '</td><td>' + timing_value(summary.get('latency_median_seconds')) +
                     ' s</td><td>' + timing_value(summary.get('latency_p95_seconds')) + ' s</td><td>' +
                     timing_value(summary.get('successful_per_second')) + '</td><td>' +
                     timing_value(row.get('load_seconds')) + ' s (' + load_label + ')</td></tr>')
    parts.append('</tbody></table></div></section>')
    return ''.join(parts)


def _field_score_table(scores):
    parts = ['<section id="fields"><div class="section-head"><h2>Field scores</h2>',
             '<p>Selected-review exact occurrences. Full-class macro is N/A when any reference class is missing; '
             'observed-class macro is shown separately. Alias accuracy is conditional on reviewed endpoint pairs, not end-to-end.</p></div>',
             '<div class="table-scroll"><table><thead><tr><th>Arm</th><th>Intent classes (TP / FP / FN)</th>',
             '<th>Intent macro</th><th>Observed-class macro</th><th>Exact intent set</th>',
             '<th>Sentiment classes (TP / FP / FN)</th><th>Sentiment macro</th><th>Sentiment observed-class macro</th><th>Complete occurrence F1</th>',
             '<th>Alias pairs</th></tr></thead><tbody>']
    for arm_id, score in scores.items():
        intents, sentiment, aliases = score['intents'], score['sentiment'], score['aliases']
        def classes(values):
            return '<br>'.join(_escape(label) + ': ' + ' / '.join(_count(metric.get(key)) for key in ('tp', 'fp', 'fn'))
                             + ' · ' + _score(metric.get('f1')) + ' · Support ' +
                             _escape(metric.get('tp', 0) + metric.get('fn', 0)) for label, metric in values.items())
        parts.append('<tr><td>' + _escape(arm_id) + '</td><td>' + classes(intents['per_label']) +
                     '<div class="table-note">Missing: ' + _escape(', '.join(intents['missing_classes']) or 'none') + '</div></td><td>' +
                     _score(intents['macro']['f1']) + '</td><td>' + _score(intents['observed_macro_f1']) +
                     '</td><td>' + _score(intents['exact_set_accuracy']) + '</td><td>' + classes(sentiment['per_class']) +
                     '<div class="table-note">Missing: ' + _escape(', '.join(sentiment['missing_classes']) or 'none') + '</div></td><td>' +
                     _score(sentiment['macro_f1']) + '</td><td>' + _score(sentiment['observed_macro_f1']) +
                     '</td><td>' + _score(score['complete_occurrence']['f1']) +
                     '</td><td>' + _score(aliases['f1']) + '<div class="table-note">' +
                     _escape(aliases['reviewed_pairs']) + ' reviewed pairs; ' +
                     _escape(aliases['unreviewed_positive_predictions']) + ' unreviewed positives</div></td></tr>')
    parts.append('</tbody></table></div></section>')
    return ''.join(parts)


def render_dashboard(data: dict) -> str:
    """Render verified comparison data as one portable HTML page.

    The caller establishes artifact provenance. This layer escapes every supplied
    value and checks source offsets before inserting highlight markup.
    """
    models = data['models']
    documents = data['documents']
    title = _escape(data['title'])
    provenance = data.get('provenance', {})
    review_kind = provenance.get('review_kind')
    review_name = {'human_reviewed': 'Human reviewed', 'agent_provisional': 'Agent provisional'}.get(
        review_kind, str(provenance.get('review', 'Local comparison evidence')))
    population = provenance.get('population_documents', len(documents))
    navigation = [('provenance', 'Provenance'), ('software', 'Software names'),
                  ('version', 'Versions'), ('links', 'Version owners'), ('passages', 'Passages')]
    if data.get('field_scores') is not None:
        navigation.insert(4, ('fields', 'Field scores'))
    if data.get('timing_rows'):
        navigation.insert(4, ('timing', 'Timing'))
    if data.get('feedback'):
        navigation.append(('feedback', 'Later feedback'))
    if data.get('appendix'):
        navigation.append(('appendix', 'Appendix'))
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>{title}</title><style>{_CSS}</style></head><body>',
             '<header><div class="masthead"><div><h1>' + title + '</h1>',
             '<p>Inspect exact source text, extraction outcomes, and version ownership across the same frozen passages.</p></div>',
             '<div class="evidence-stamp"><strong>' + _escape(review_name) + '</strong>' +
             '<span>' + _escape(population) + ' frozen passages</span>' +
             'Metrics and summaries follow the supplied local artifacts.</div></div>',
             '<nav aria-label="Report sections">' + ''.join(
                 f'<a href="#{anchor}">{label}</a>' for anchor, label in navigation) + '</nav></header><main>',
             '<section id="provenance"><h2>Provenance and limits</h2><dl class="meta-list">']
    for key, value in provenance.items():
        display = value if isinstance(value, (str, int, float, bool)) or value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f'<dt>{_escape(key)}</dt><dd>{_escape(display)}</dd>')
    parts.append('</dl>')
    if data.get('limitations'):
        parts.append('<div class="limitations"><strong>Interpretation limits</strong>')
        parts.extend(f'<p>{_escape(item)}</p>' for item in data['limitations'])
        parts.append('</div>')
    parts.append('<aside class="metric-primer"><h3>How to read scores</h3>'
                 '<p>TP is an exact match to an eligible reference span or owner link. FP is an extra eligible prediction. '
                 'FN is an eligible reference missed by the model.</p>'
                 '<p>Precision is TP / (TP + FP); recall is TP / (TP + FN). F1 balances precision and recall. '
                 'N/A means the denominator is undefined, the result is unavailable or unsupported, or there is no scored coverage; '
                 '0.0% means an evaluated zero.</p></aside>')
    parts.append(_model_inventory(models))
    parts.append('</section>')
    parts.append(_metric_section(data, 'software', 'Software names', 'Exact software span detection within the stated reference coverage.'))
    parts.append(_metric_section(data, 'version', 'Version spans', 'Exact version text detection; ownership is scored separately below.'))
    parts.append(_metric_section(data, 'links', 'Version owners', 'Exact software occurrence to version ownership within eligible reviewed regions.'))
    if data.get('timing_rows'):
        parts.append(_timing_table(data['timing_rows']))
    if data.get('field_scores') is not None:
        parts.append(_field_score_table(data['field_scores']))
    parts += ['<section id="passages"><div class="section-head"><h2>Passage review</h2>',
              '<p>Each track repeats the frozen source text so the highlighted output remains anchored to its original offsets.</p></div>',
              '<nav class="passage-index" aria-label="Passage index">']
    parts.extend(f'<a href="#passage-{index}">{_escape(document["document_id"])}</a>'
                 for index, document in enumerate(documents, 1))
    parts.append('</nav>')
    parts.extend(_document(document, models, review_kind or 'agent_provisional', f'passage-{index}', data.get('field_scores'))
                 for index, document in enumerate(documents, 1))
    parts.append('</section>')
    if data.get('feedback'):
        parts.append('<section id="feedback"><h2>Later feedback</h2><p class="muted">Observations here are unscored and outside the frozen comparison population.</p>')
        for item in data['feedback']:
            parts.append('<article class="feedback"><h3>' + _escape(item.get('title', 'Observation')) + '</h3>'
                         '<p>' + _escape(item.get('body', '')) + '</p><p class="muted">'
                         + _escape(item.get('provenance', '')) + '</p></article>')
        parts.append('</section>')
    parts.append(_appendix(data.get('appendix')))
    parts.append('</main><footer>Generated from local comparison evidence. No network assets are required.</footer></body></html>')
    return ''.join(parts)
