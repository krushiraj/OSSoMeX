"""Build the review-only HTML/PDF brief from frozen cycle 003 evidence."""

from html import escape
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/openalex-progress-2026-09-30'
PDF = ROOT / 'output/pdf/OpenAlex-software-extraction-progress-Krushi-Raj-Tula.pdf'
REPORT = ROOT / 'reports/scibert-v2/focused-cycle-003/fresh30/dashboard-reviewed/report.json'
HASH = '1ff19fe04e56c8b9208865f91b2c15e9e3e50b29d724b8f166b377a824e6aa10'
CONTROL = ROOT / 'reports/scibert-v2/focused-cycle-003/corrected-reserve27-spans/report.json'
CONTROL_HASH = '068531888da8d0b96d6d572bb6031789620556ffd94b382d3132b184c553bbb8'
SCIBERT_REPORT = ROOT / 'reports/scibert-v2/softcite-scibert-001/dashboard/report.json'
SCIBERT_HASH = '40932ebe4678bc795f52faece25feb3f7ceff1ae89d82afb125611558dccdda9'
ENGINE_METADATA = ROOT / 'configs/scibert/softcite-scibert-local-metadata-001.json'
ENGINE_HASH = '6e2563533bede8bef74e7f7d6b43ad6c335c37c1084256d15ae531187bf7c7d0'
PROJECTION = ROOT / 'reports/scibert-v2/softcite-scibert-001/schema-projection-001.json'
PROJECTION_HASH = '8af5e5da5ad83c2fff29ef1eccbaed0fdf1d27aad99f1346fd21f12b9f1cc019'
SENSITIVITY = ROOT / 'reports/scibert-v2/softcite-scibert-001/schema-projection-002.json'
SENSITIVITY_HASH = '1d0526670fda31d44b96e5039c745cf955b4392a3ead36252d4fc1066939a448'
SCIBERT_ARM = 'softcite-scibert-0.8.1'
ARMS = [('full-label-006', 'OSSoMeX'), ('softcite-0.8.1', 'Softcite (CRF)'),
        (SCIBERT_ARM, 'Softcite (SciBERT)')]


def merge_comparison(original, extra, arm_id):
    reference_hash = original['provenance'].get('reference_sha256')
    if not reference_hash or reference_hash != extra['provenance'].get('reference_sha256'):
        raise ValueError('Comparison must use the same frozen references.')

    def indexed(documents):
        result = {doc['document_id']: doc for doc in documents}
        if len(result) != len(documents):
            raise ValueError('Duplicate document in comparison.')
        return result

    original_docs = indexed(original['documents'])
    extra_docs = indexed(extra['documents'])
    if original_docs.keys() != extra_docs.keys():
        raise ValueError('Comparison populations differ.')
    if arm_id in original['field_scores']:
        raise ValueError('Cannot replace a frozen comparison arm.')
    models = [model for model in extra['models'] if model['arm_id'] == arm_id]
    if len(models) != 1 or arm_id not in extra['field_scores']:
        raise ValueError('Comparison arm metadata or scores missing.')
    result = deepcopy(original)
    for doc in result['documents']:
        other = extra_docs[doc['document_id']]
        if any(doc[key] != other[key] for key in ('text', 'text_revision')):
            raise ValueError('Comparison input text differs.')
        for key in ('reference_spans', 'reference_links', 'coverage', 'link_coverage',
                    'ignored_software', 'ignored_versions'):
            if doc.get(key) != other.get(key):
                raise ValueError('Comparison embedded reference or coverage differs.')
        arms = [arm for arm in other['arms'] if arm['arm_id'] == arm_id]
        if len(arms) != 1 or any(arm['arm_id'] == arm_id for arm in doc['arms']):
            raise ValueError('Missing or duplicate document arm.')
        doc['arms'].append(deepcopy(arms[0]))
    result['models'].append(deepcopy(models[0]))
    result['field_scores'][arm_id] = deepcopy(extra['field_scores'][arm_id])
    result['timing_rows'].extend(deepcopy([
        row for row in extra['timing_rows'] if row['arm_id'] == arm_id
    ]))
    return result


def clean_name_difference(reference_spans, arms):
    def names(spans):
        return {(span['start'], span['end'], span['text']) for span in spans
                if span['label'] == 'SOFTWARE'}

    if not arms or any(arm['status'] not in ('success', 'no_mentions') for arm in arms):
        return False
    expected = names(reference_spans)
    correct = [names(arm['spans']) == expected for arm in arms]
    return any(correct) and not all(correct)


def with_name_projection(report, projection):
    if projection['reference_sha256'] != report['provenance']['reference_sha256']:
        raise ValueError('Name projection references differ.')
    docs = {doc['document_id']: doc for doc in report['documents']}
    for arm_id, arm in projection['arms'].items():
        projected = {doc['document_id']: doc for doc in arm['per_document']}
        if projected.keys() != docs.keys() or len(projected) != len(arm['per_document']):
            raise ValueError('Name projection population differs.')
        native = report['field_scores'][arm_id]['mention_detection']
        if any(native[key] != value for key, value in arm['native_name_score'].items()):
            raise ValueError('Name projection native score differs.')
        for doc_id, doc in projected.items():
            if doc['text_revision'] != docs[doc_id]['text_revision']:
                raise ValueError('Name projection text revision differs.')
            for span in doc['projected_names']:
                if span['label'] != 'SOFTWARE' or docs[doc_id]['text'][span['start']:span['end']] != span['text']:
                    raise ValueError('Name projection span does not match input.')
    result = deepcopy(report)
    result['name_projection'] = deepcopy(projection)
    return result


def name_score(report, arm_id):
    projection = report.get('name_projection', {}).get('arms', {}).get(arm_id)
    return projection['projected_name_score'] if projection else report['field_scores'][arm_id]['mention_detection']


def name_arms(report, doc):
    arms = []
    for arm_id, _ in ARMS:
        arm = deepcopy(next(a for a in doc['arms'] if a['arm_id'] == arm_id))
        projection = report.get('name_projection', {}).get('arms', {}).get(arm_id)
        if projection:
            projected = next(d for d in projection['per_document'] if d['document_id'] == doc['document_id'])
            arm['spans'] = projected['projected_names']
        arms.append(arm)
    return arms


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified():
    raw = REPORT.read_bytes()
    assert sha(raw) == HASH, 'Frozen report changed; review claims before regenerating.'
    control_raw = CONTROL.read_bytes()
    assert sha(control_raw) == CONTROL_HASH, 'Frozen control evidence changed.'
    control = json.loads(control_raw)['references']['agent_provisional']['arms']
    assert control['scibert-frozen-002']['labels']['SOFTWARE']['operational']['f1'] == 0.0
    assert pct(control['scibert-detector-005']['labels']['SOFTWARE']['operational']['f1']) == '63.8%'
    extra_raw = SCIBERT_REPORT.read_bytes()
    assert sha(extra_raw) == SCIBERT_HASH, 'Softcite SciBERT evidence changed; review before regenerating.'
    engine_raw = ENGINE_METADATA.read_bytes()
    assert sha(engine_raw) == ENGINE_HASH, 'Verified Softcite engine metadata changed.'
    engine = json.loads(engine_raw)
    assert engine['model_selection']['software'] == 'delft_BERT_SciBERT'
    assert 'Loading DeLFT model for software with architecture BERT' in engine['evidence']
    extra = json.loads(extra_raw)
    assert extra['field_scores'][SCIBERT_ARM]['failures']['failed_documents'] == 0
    projection_raw = PROJECTION.read_bytes()
    assert sha(projection_raw) == PROJECTION_HASH, 'Name projection evidence changed.'
    sensitivity_raw = SENSITIVITY.read_bytes()
    assert sha(sensitivity_raw) == SENSITIVITY_HASH, 'Name projection sensitivity changed.'
    merged = merge_comparison(json.loads(raw), extra, SCIBERT_ARM)
    return with_name_projection(merged, json.loads(projection_raw))


def p(text, style='body'):
    return {'type': 'p', 'text': text, 'style': style}


def heading(text):
    return {'type': 'heading', 'text': text}


def table(headers, rows, widths):
    return {'type': 'table', 'headers': headers, 'rows': rows, 'widths': widths}


def pct(value):
    return 'N/A' if value is None else f'{value:.1%}'


def example(report, prefix, title, quote, analysis):
    matches = [d for d in report['documents'] if d['document_id'].split(':')[-1].startswith(prefix)]
    assert len(matches) == 1
    doc = matches[0]
    assert quote in doc['text'], (title, quote)
    arms = name_arms(report, doc)
    assert clean_name_difference(doc['reference_spans'], arms), 'No clean name-extraction winner.'
    doi = doc['source_ids']['doi']
    return {'type': 'example', 'title': title, 'quote': quote, 'analysis': analysis,
            'source_url': doi, 'source_label': doi.removeprefix('https://doi.org/'),
            'document_id': doc['document_id']}


def make_pages(report):
    quality_rows = []
    timing_rows = []
    for arm, label in ARMS:
        m = name_score(report, arm)
        quality_rows.append([label, pct(m['precision']), pct(m['recall']), pct(m['f1']),
                             f"{m['tp']} / {m['fp']} / {m['fn']}"])
        timing = next(t for t in report['timing_rows'] if t['device'] == 'cpu' and t['arm_id'] == arm)
        s = timing['summary']
        assert s['successful'] == s['scheduled'] == 90 and s['failed'] == 0
        timing_rows.append([label, f"{s['latency_median_seconds'] * 1000:.1f} ms",
                            f"{s['latency_p95_seconds'] * 1000:.1f} ms",
                            f"{s['successful_per_second']:.2f}"])
    extra = name_score(report, SCIBERT_ARM)
    if extra['f1'] > report['field_scores']['full-label-006']['mention_detection']['f1']:
        conclusion = ('Softcite’s SciBERT extractor has the best name F1 in this test. '
                      'My model improves on the CRF baseline, but does not beat this stronger baseline.')
    elif extra['f1'] == report['field_scores']['full-label-006']['mention_detection']['f1']:
        conclusion = ('My model and Softcite’s SciBERT extractor tie on name F1 in this test. '
                      'Both improve on the CRF baseline here; neither result proves a general advantage.')
    else:
        mine = report['field_scores']['full-label-006']['mention_detection']
        gain = (mine['f1'] - extra['f1']) * 100
        conclusion = (f'The two encoders are close: my model is ahead by {gain:.1f} F1 points, '
                      f'finding {mine["tp"]} reference names versus {extra["tp"]}. '
                      'This is a modest lead on a small sample, not proof of a general advantage.')
    first = [
        p('OSSoMeX', 'title'),
        p('Finding software mentions in research', 'subtitle'),
        p('Krushi Raj Tula | 30 September 2026 | Early results', 'meta'),
        p('Following my earlier pilot, I fine-tuned SciBERT to find software names and describe '
          'how they are mentioned. I compared it with both the CRF and SciBERT extractors in Softcite 0.8.1.', 'lead'),
        p('OSSoMeX means Open Source Software Mention Extractor. My local model has five stages '
          'for names, version links, intent, sentiment and aliases. I trained on 161 passages from '
          '21 CC-BY papers: 11 passages reviewed by a person and 150 provisionally labelled by AI.', 'small'),
        heading('What the test shows'),
        table(['Model', 'Precision', 'Recall', 'Name F1', 'Correct / extra / missed'],
              quality_rows, [140, 70, 70, 70, 145]),
        p(conclusion),
        p('Precision asks how many extracted names are correct. Recall asks how many reference names '
          'were found. F1 balances both. Names must match exact text boundaries. For a common format, '
          'I counted Softcite’s language fields as names and left out entries it marked implicit. '
          'This mapping was checked after the run; labels stayed unchanged. Native name-only F1 was '
          '36.9% for CRF and 69.7% for SciBERT. Counting languages without filtering implicit names '
          'gives SciBERT 72.5%, so the result stays close.', 'small'),
        p('<b>Test scope:</b> the same 30 OpenAlex snippets, found using software-name queries. '
          'There are 51 reference names in 28 scorable snippets; two snippets are fully excluded from scoring. '
          'Unreviewed regions are not scored. The labels were prepared by AI before inference, not '
          'independently confirmed by a person. These are now development results, not a final blind test. '
          'No passages were removed to improve the numbers.', 'small'),
        heading('Usage and other fields'),
        p('Intent describes whether software was used, created, shared or just mentioned. '
          'My model gets 27 of 48 eligible intent labels right, counting missed names as errors. '
          'The field-mapping audit above covers names, not an aligned intent comparison.'),
        p('Version linking is not settled: this sample has no true version links, so it cannot measure '
          'version recall. My model also misses the four expressed opinions and the one checked alias pair, '
          'and adds seven false opinions. These fields need more work before I would rely on them.'),
        heading('Time in my local setup'),
        table(['CPU comparison', 'Median / snippet', 'Slow tail (p95)', 'Snippets / second'],
              timing_rows, [160, 115, 115, 105]),
        p('Three warmed runs of all 30 snippets, one request at a time; all 90 requests per row succeeded. '
          'Model loading is excluded. Timing runs are separate from the scored runs. '
          'The name-field conversion is not timed. '
          'My model runs native PyTorch on an Apple M4 Max; Softcite runs in an amd64 Linux container. '
          'Work also varies with the number of names found. '
          'These numbers describe my setup, not a fair production cost comparison.', 'small'),
    ]
    second = [
        p('Examples and next steps', 'title2'),
        p('Short excerpts from the full test inputs. Examples were chosen after scoring to show a clear '
          'name-extraction difference; they do not change the benchmark above.', 'small'),
    ]
    candidates = [
        ('1072bf', 'Explicit software use', 'by using both WEKA and Scikit-Learn'),
        ('12827f', 'Software inside a workflow',
         'by utilizing GROMACS modules which are integrated into python scripts.'),
    ]
    for prefix, title, quote in candidates:
        doc = next(d for d in report['documents'] if d['document_id'].split(':')[-1].startswith(prefix))
        arms = name_arms(report, doc)
        if not clean_name_difference(doc['reference_spans'], arms):
            continue
        expected = ', '.join(s['text'] for s in doc['reference_spans'] if s['label'] == 'SOFTWARE')
        outputs = '; '.join(
            label + ': ' + (', '.join(s['text'] for s in arm['spans'] if s['label'] == 'SOFTWARE') or 'none')
            for (_, label), arm in zip(ARMS, arms)
        )
        analysis = f'<b>Expected names:</b> {escape(expected)}.<br/>{escape(outputs)}.'
        if prefix == '1072bf':
            analysis += ' Both encoders also correctly label the two names as used.'
        if prefix == '12827f':
            analysis += ' Counting Softcite’s separate language field makes both encoders correct here.'
        second.append(example(report, prefix, title, quote, analysis))
    second.extend([
        heading('Why keep testing a fine-tuned encoder?'),
        p('<a href="https://github.com/allenai/scibert">Base SciBERT</a> is not a ready-made software '
          'extractor. Fine-tuning teaches it the names and labels needed for this task. '
          '<a href="https://github.com/softcite/software-mentions">Softcite’s SciBERT extractor</a> '
          'is already fine-tuned too. This comparison tests two trained systems; it does not isolate '
          'fine-tuning as the only reason for a difference.'),
        p('The name results give me a reason to keep testing this approach alongside Softcite. '
          'They do not show that Softcite fails generally or contradict its published benchmarks. '
          'Short, noisy snippets and provisional labels are different from a full-paper gold-standard test.'),
        heading('How this could help WP4'),
        p('My next step would be a separate linking layer: take a mention and its context, find candidates '
          'in software registries or GitHub, then return a supported match or leave it unresolved. '
          'A search result alone is not a verified identity. I have not measured repository-link accuracy yet.'),
        p('Usage intent could help distinguish software that was actually used from software only discussed. '
          'Later, reliable sentiment might add context for researchers and funders: a mention is not always '
          'an endorsement. I would not use mention counts or sentiment alone to judge funding value.'),
        heading('What I would improve next'),
        p('<b>Quality:</b> review the labels with a person, add new training papers with real versions and '
          'ambiguous names, and test on papers kept separate from training. '
          '<b>Speed:</b> profile the five stages, batch work and avoid repeated text processing. '
          'Check that outputs stay the same, then measure again on comparable hardware.'),
        p('Feedback I would value: which extraction fields, Softcite setup and representative papers '
          'would make the next comparison most useful for OpenAlex?', 'callout'),
    ])
    return [first, second]


def html_block(b):
    kind = b['type']
    if kind == 'p':
        return f'<p class="{b["style"]}">{b["text"]}</p>'
    if kind == 'heading':
        return f'<h2>{escape(b["text"])}</h2>'
    if kind == 'table':
        head = ''.join(f'<th scope="col">{escape(x)}</th>' for x in b['headers'])
        rows = ''.join('<tr>' + ''.join(f'<td>{escape(x)}</td>' for x in row) + '</tr>' for row in b['rows'])
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    return (f'<section class="example"><h3>{escape(b["title"])}</h3>'
            f'<blockquote>{escape(b["quote"])}</blockquote><p>{b["analysis"]}</p>'
            f'<p class="source">Source: <a href="{escape(b["source_url"])}">{escape(b["source_label"])}</a></p></section>')


CSS = '''
:root{--ink:#16343c;--muted:#52676e;--accent:#136e79;--paper:#ffffff;--line:#cddcde;--wash:#edf5f6}
*{box-sizing:border-box}body{margin:0;background:#e8eff1;color:var(--ink);font:15px/1.48 Arial,Helvetica,sans-serif}
main{max-width:900px;margin:30px auto}.page{background:var(--paper);padding:42px 48px;margin:0 0 24px;border-top:7px solid var(--accent)}
p{margin:0 0 12px}.title{font:700 38px/1.1 Georgia,serif;letter-spacing:-.7px;margin-bottom:8px}.title2{font:700 30px/1.15 Georgia,serif}
.subtitle{font-size:19px;margin-bottom:6px}.meta,.small{font-size:12px;line-height:1.5;color:var(--muted)}.meta{margin-bottom:20px}.lead{font-size:17px;line-height:1.45}
h2{font:700 19px/1.25 Arial,Helvetica,sans-serif;margin:20px 0 9px;color:var(--accent)}
table{width:100%;border-collapse:collapse;margin:8px 0 10px;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line)}
th{background:var(--wash);font-weight:700}td:not(:first-child),th:not(:first-child){white-space:nowrap}.table-wrap{overflow-x:auto}
.example{margin:13px 0;padding:0 0 12px;border-bottom:1px solid var(--line)}h3{font-size:15px;margin:0 0 6px}blockquote{margin:0 0 7px;padding-left:12px;border-left:3px solid var(--accent);color:var(--accent);font:italic 16px/1.4 Georgia,serif}
.example p{margin-bottom:5px}.source,.footnote{font-size:11px;line-height:1.45;color:var(--muted)}.callout{border-left:4px solid var(--accent);padding:10px 14px;background:var(--wash)}
a{color:#126c79;text-underline-offset:2px}a:focus-visible{outline:3px solid #d17c29;outline-offset:4px}footer{margin-top:17px;color:var(--muted);font-size:11px}
@media(max-width:650px){main{margin:0}.page{padding:25px 20px;margin-bottom:15px}.title{font-size:30px}.title2{font-size:25px}table{min-width:580px}}
@media print{body{background:white;font-size:10pt}main{margin:0;max-width:none}.page{padding:0;border-top:4px solid var(--accent);break-after:page;margin:0}.page:last-child{break-after:auto}a{color:inherit}h2{break-after:avoid}.example,table{break-inside:avoid}.table-wrap{overflow:visible}footer{display:none}@page{size:A4;margin:15mm}}
'''


def render_pdf(pages):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    ink = colors.HexColor('#16343c')
    accent = colors.HexColor('#136e79')
    muted = colors.HexColor('#52676e')
    base = ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=13.5,
                          textColor=ink, spaceAfter=6.5, alignment=TA_LEFT)
    styles = {'body': base}
    for name, settings in {
        'title': dict(fontName='Times-Bold', fontSize=25, leading=28, spaceAfter=5),
        'title2': dict(fontName='Times-Bold', fontSize=21, leading=24, spaceAfter=8),
        'subtitle': dict(fontSize=12.7, leading=16, spaceAfter=5),
        'meta': dict(fontSize=8.1, leading=10, textColor=muted, spaceAfter=13),
        'lead': dict(fontSize=10.6, leading=14, spaceAfter=9),
        'heading': dict(fontName='Helvetica-Bold', fontSize=11.5, leading=14,
                        textColor=accent, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'small': dict(fontSize=8.8, leading=11.5, textColor=muted, spaceAfter=7),
        'footnote': dict(fontSize=7.4, leading=9.2, textColor=muted, spaceBefore=3),
        'example_title': dict(fontName='Helvetica-Bold', fontSize=9.6, leading=12, spaceAfter=3),
        'quote': dict(fontName='Times-Italic', fontSize=10.2, leading=12.5, textColor=accent,
                      leftIndent=8, spaceAfter=4),
        'source': dict(fontSize=7.2, leading=9, textColor=muted, spaceAfter=6),
        'callout': dict(fontName='Helvetica-Bold', fontSize=9.3, leading=12.5, textColor=accent,
                        spaceBefore=5, spaceAfter=7),
        'cell': dict(fontSize=8.6, leading=10.8, spaceAfter=0),
        'th': dict(fontName='Helvetica-Bold', fontSize=8.2, leading=10.5, spaceAfter=0),
    }.items():
        styles[name] = ParagraphStyle(name, parent=base, **settings)
    story = []
    width = A4[0] - 80
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for b in page:
            if b['type'] in ('p', 'heading'):
                story.append(Paragraph(b['text'], styles['heading' if b['type'] == 'heading' else b['style']]))
            elif b['type'] == 'table':
                data = [[Paragraph(escape(s), styles['th']) for s in b['headers']]]
                data += [[Paragraph(escape(s), styles['cell']) for s in row] for row in b['rows']]
                scale = width / sum(b['widths'])
                t = Table(data, colWidths=[w * scale for w in b['widths']], hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf5f6')),
                    ('LINEBELOW', (0, 0), (-1, -1), .45, colors.HexColor('#cddcde')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story += [t, Spacer(1, 6)]
            else:
                story.append(KeepTogether([
                    Paragraph(escape(b['title']), styles['example_title']),
                    Paragraph(escape(b['quote']), styles['quote']),
                    Paragraph(b['analysis'], styles['body']),
                    Paragraph(f'Source: <a href="{b["source_url"]}">{escape(b["source_label"])}</a>', styles['source']),
                ]))
    def furniture(canvas, doc):
        canvas.setStrokeColor(accent)
        canvas.setLineWidth(3)
        canvas.line(40, A4[1] - 27, A4[0] - 40, A4[1] - 27)
        canvas.setFillColor(muted)
        canvas.setFont('Helvetica', 7.4)
        canvas.drawString(40, 23, 'Krushi Raj Tula | OpenAlex software-graph POC | 30 September 2026')
        canvas.drawRightString(A4[0] - 40, 23, f'{doc.page} / {len(pages)}')
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=40, leftMargin=40,
                            topMargin=41, bottomMargin=36, title='OSSoMeX: finding software mentions in research',
                            author='Krushi Raj Tula', subject='Preliminary comparison with Softcite CRF and SciBERT')
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def main():
    from pypdf import PdfReader
    report = read_verified()
    pages = make_pages(report)
    OUT.mkdir(parents=True, exist_ok=True)
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Software mention extraction | Krushi Raj Tula</title><style>' + CSS +
            '</style></head><body><main>' + ''.join(
                '<article class="page">' + ''.join(html_block(b) for b in page) +
                f'<footer>Krushi Raj Tula | Research POC | {i + 1} / {len(pages)}</footer></article>'
                for i, page in enumerate(pages)) + '</main></body></html>')
    (OUT / 'OpenAlex-software-extraction-progress.html').write_text(html, encoding='utf-8')
    (OUT / 'report-content.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2) + '\n')
    render_pdf(pages)
    pdf = PdfReader(PDF)
    assert len(pdf.pages) == len(pages), f'Expected {len(pages)} pages, got {len(pdf.pages)}; inspect layout.'
    text = '\n'.join(page.extract_text() for page in pdf.pages)
    for required in ('OSSoMeX', 'Softcite (SciBERT)', 'Softcite (CRF)', '74.2%', '36.9%',
                     '33 / 5 / 18', '32 / 5 / 19', '12 / 1 / 39', '7.07', '3.60', 'WP4'):
        assert required in text, required
    for arm, _ in ARMS:
        assert pct(name_score(report, arm)['f1']) in text
    for private in ('/Users/', '127.0.0.1', 'localhost', 'github.com/krushiraj/openalex-sw-mentions',
                    'github.com/krushiraj/OSSoMeX'):
        assert private not in html and private not in text, private
    assert not re.search(r'\b(?:005|006|007|our|we)\b', text, flags=re.IGNORECASE)
    assert 'selection trade' not in text.lower()
    assert '<script' not in html and 'http://' not in html
    links = []
    for page in pdf.pages:
        for ann in page.get('/Annots', []):
            uri = ann.get_object().get('/A', {}).get('/URI')
            if uri:
                links.append(str(uri))
    source_links = sorted(set(links))
    expected_links = {b['source_url'] for page in pages for b in page if b['type'] == 'example'}
    expected_links.update({'https://github.com/allenai/scibert',
                           'https://github.com/softcite/software-mentions'})
    assert set(source_links) == expected_links
    assert 'engineering team' not in text and 'join the team' not in text
    receipt = {'report_sha256': HASH, 'control_report_sha256': CONTROL_HASH,
               'softcite_scibert_report_sha256': SCIBERT_HASH,
               'softcite_scibert_engine_metadata_sha256': ENGINE_HASH,
               'name_projection_sha256': PROJECTION_HASH,
               'name_projection_sensitivity_sha256': SENSITIVITY_HASH,
               'html_sha256': sha(html.encode()),
               'pdf_sha256': sha(PDF.read_bytes()), 'pdf_pages': len(pdf.pages),
               'pdf_bytes': PDF.stat().st_size, 'source_link_count': len(source_links),
               'source_links': source_links, 'word_count_pdf': len(text.split()),
               'claims_status': 'provisional_diagnostic_not_general_superiority',
               'publication_status': 'local_review_only_not_sent',
               'population_documents': len(report['documents']),
               'reference_sha256': report['provenance']['reference_sha256'],
               'comparison_labels': dict(ARMS),
               'example_selection': 'posthoc_clear_name_difference_aggregate_population_unchanged',
               'examples': [{k: b[k] for k in ('title', 'document_id', 'source_url', 'quote')}
                            for page in pages for b in page if b['type'] == 'example']}
    (OUT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
