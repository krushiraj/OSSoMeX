"""Build the review-only HTML/PDF brief from frozen cycle 003 evidence."""

from html import escape
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/openalex-progress-2026-09-30'
PDF = ROOT / 'output/pdf/OpenAlex-software-extraction-progress-Krushi-Raj-Tula.pdf'
REPORT = ROOT / 'reports/scibert-v2/focused-cycle-003/fresh30/dashboard-reviewed/report.json'
HASH = '1ff19fe04e56c8b9208865f91b2c15e9e3e50b29d724b8f166b377a824e6aa10'
CONTROL = ROOT / 'reports/scibert-v2/focused-cycle-003/corrected-reserve27-spans/report.json'
CONTROL_HASH = '068531888da8d0b96d6d572bb6031789620556ffd94b382d3132b184c553bbb8'


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
    return json.loads(raw)


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
    doi = doc['source_ids']['doi']
    return {'type': 'example', 'title': title, 'quote': quote, 'analysis': analysis,
            'source_url': doi, 'source_label': doi.removeprefix('https://doi.org/'),
            'document_id': doc['document_id']}


def make_pages(report):
    rows = []
    for arm, label in [('full-label-006', 'SciBERT 006'), ('softcite-0.8.1', 'Softcite 0.8.1')]:
        s = report['field_scores'][arm]
        m = s['mention_detection']
        rows.append([label, pct(m['precision']), pct(m['recall']), pct(m['f1']),
                     f"{m['tp']} / {m['fp']} / {m['fn']}"])
    assert rows[0][3] == '74.2%' and rows[1][3] == '36.9%'
    time_rows = []
    for device in ('cpu', 'mps'):
        for arm, label in [('full-label-006', '006'), ('softcite-0.8.1', 'Softcite service')]:
            r = next(t for t in report['timing_rows'] if t['device'] == device and t['arm_id'] == arm)
            s = r['summary']
            assert s['successful'] == s['scheduled'] == 90 and s['failed'] == 0
            time_rows.append([label, device.upper() + ' pass',
                              f"{s['latency_median_seconds'] * 1000:.1f} ms",
                              f"{s['latency_p95_seconds'] * 1000:.1f} ms",
                              f"{s['successful_per_second']:.2f}"])
    first = [
        p('Software mention extraction', 'title'),
        p('Selected POC: SciBERT 006 compared with Softcite', 'subtitle'),
        p('Krushi Raj Tula | 30 September 2026 | Research POC', 'meta'),
        p('<b>006 is the selected model for this POC.</b> It recovers 33 of 51 reviewed software mentions, '
          'versus 12 for the installed Softcite configuration, at similar precision. '
          'The strongest result is improved recall on these OpenAlex snippets.', 'lead'),
        p('006 uses five separately fine-tuned SciBERT stages: names/versions, version ownership, '
          'author intent, sentiment and aliases. Training: 161 passages from 21 CC-BY papers '
          '(11 human-reviewed, 150 agent-provisional). Scores are uncalibrated.', 'small'),
        heading('Quality on identical inputs'),
        table(['Model', 'Precision', 'Recall', 'Name F1', 'TP / FP / FN'], rows,
              [130, 80, 80, 80, 125]),
        p('Precision = correct / scored predictions; recall = correct / reference mentions; '
          'F1 balances both. Exact boundaries matter. 006: 33/38 scored predictions are correct; '
          'Softcite: 12/14. Five additional 006 predictions and three Softcite predictions fall outside '
          'reviewed software coverage and are excluded, not counted correct.', 'small'),
        p('<b>Scope:</b> 30 software-query-selected OpenAlex snippets; 28 with scorable regions and '
          'two fully masked. Agents labelled the text before inference; no human-adjudicated gold test. '
          'Inputs include non-English, OCR and metadata-like fragments. All 30 remain in operational counts. '
          'We now select 006 using these results, so this sample is exposed, not an untouched final test.'),
        table(['Other fields', '006', 'Softcite', 'Support / interpretation'], [
            ['Intent exact-set accuracy', '56.25%', '16.67%', '48 eligible mentions; missed names count'],
            ['Author-use F1', '50.0%', '0.0%', 'Only two positive uses'],
            ['False version-owner links', '3', '1', 'No true versions; recall unmeasured'],
            ['Expressed sentiments recovered', '0 / 4', 'N/A', '006 also adds seven false opinions'],
            ['Reviewed alias pairs recovered', '0 / 1', 'N/A', 'One positive; no negative pairs'],
            ['Complete-occurrence F1', '45.8%', 'N/A', '48 refs: name, versions, intent, sentiment; no aliases'],
        ], [165, 60, 65, 205]),
        p('N/A means unsupported by the compared output, not zero quality. Created/shared intent and '
          'mixed sentiment have no positive references. Both systems avoid false names on eight checked negatives.', 'small'),
        p('<b>Selection trade-off:</b> 007 loses four correct names (68.2% name F1) and adds more false '
          'version links; it remains experimental. 005 has a slightly higher complete-row F1 (47.5%). '
          '006 is the name-extraction choice, not a winner in every field. On the separate earlier 27 '
          'snippets, 006/Softcite name F1 is 63.8%/59.6%; the margin depends on the sample.'),
        heading('Local full-pipeline timing'),
        table(['System', 'Comparison', 'Median', 'p95', 'Snippets/s'], time_rows, [145, 90, 80, 85, 95]),
        p('Three warmed, batch-one repeats; 90/90 successful requests per row; excludes load/warmup and publication. '
          '006 has higher aggregate throughput; Softcite has lower median latency. Apple M4 Max; '
          'SciBERT runs native CPU/MPS, Softcite runs an amd64 container using Wapiti + LinkBERT. '
          'The pass label does not mean Softcite uses MPS. Softcite returns no mentions on 22/30 inputs. '
          'These are separate timing runs, not timed copies of the scored predictions. '
          'No equal-runtime cost or full-paper throughput claim follows.', 'small'),
    ]
    second = [
        p('Cases where each system does better', 'title2'),
        p('Illustrative cases selected after evaluation, not a second benchmark. Quotes show short source fragments; '
          'outputs below come from the full captured snippets, not from rerunning these shortened quotes.', 'small'),
        example(report, '1072bf', 'Softcite misses explicit author use',
                'by using both WEKA and Scikit-Learn',
                '006 finds both names and labels both used. Softcite returns no mentions. '
                'This is a recall gap on this input; it does not identify the internal cause of Softcite\'s miss.'),
        example(report, '8ab60e', '006 recovers names in a library description',
                'IPython [16] supports parallelized NumPy operations.',
                '006 finds 5 of 6 reviewed names in the full snippet; Softcite finds none. '
                '006 still misses one repeated NumPy mention. These are name-recall gains, not perfect full-field outputs.'),
        example(report, '36ce86', '006 separates a software name from a section heading',
                'II.MATERYAL VE METOD 2.1.GROMACS GROMACS',
                'In this Turkish fragment, 006 finds three GROMACS mentions without inventing versions. '
                'Softcite instead labels MATERYAL VE METOD as software and 2.1.GROMACS GROMACS as its version. '
                '006 still misses the expanded name and alias relation.'),
        example(report, '868e13', 'Softcite finds repeated LaTeX and SPSS names',
                'Create plots and LaTeX tables that look like SPSS output',
                'Softcite finds five of seven reference mentions; 006 finds three. '
                'Softcite gets both LaTeX and all three SPSS occurrences. 006 gets R, knitr and one LaTeX. '
                'Softcite also mistakes code for software. This source is a package-description fragment.'),
        example(report, '03d232', 'Softcite gets this boundary right',
                'image processing software (ImageJ®,',
                'The reference excludes the trademark glyph. Softcite extracts ImageJ; 006 extracts ImageJ® '
                'and fails exact matching. Token-boundary handling needs work.'),
        example(report, 'f5b911', 'Our version linking still makes a serious error',
                '2 (Pedregosa et al., 2011).',
                '006 finds both scikit-learn mentions but assigns citation number 2 as their version and '
                'positive sentiment to both. The reviewed versions are absent and sentiment not expressed: '
                'the nearby praise describes another tool. Softcite misses both names. '
                'Name recall alone cannot establish trustworthy citation objects.'),
        p('Source fragments above are short, attributed illustrations. The separate local review file '
          'contains all 30 full inputs, provisional expected outputs, masks and model outputs. '
          'Review files remain outside Git and are not email attachments until sharing rights are checked.', 'small'),
    ]
    third = [
        p('Interpretation and next experiments', 'title2'),
        heading('Published F1 above 0.8 and lower snippet recall can coexist'),
        p('The OpenAlex proposal cites strong published extraction results. Our installed Softcite misses '
          '39 of 51 reference mentions on this real-source snippet sample, including explicit author use. '
          'That is a substantial recall gap for this workflow; it does not invalidate a full-document gold benchmark.'),
        p('The cited <a href="https://aclanthology.org/2025.sdp-1.13/">SOMD2025 shared-task paper</a> '
          'reports winning composite NER/relation scores of 0.89 in Phase I and 0.63 on out-of-distribution '
          'Phase II data. Those are not our exact-name micro-F1 and are not Softcite-specific results. '
          'Our narrower context, noisy text, label policy and provisional references also differ. '
          'They are plausible contributors to the gap, not isolated causes.'),
        p('<a href="https://github.com/softcite/software-mentions">Softcite supports CRF and fine-tuned '
          'transformer extraction, including SciBERT</a>, plus PDF structure-aware processing. '
          'This benchmark used Wapiti extraction with LinkBERT context heads, not its SciBERT '
          'extractor. Recheck the team\'s preferred configuration on representative documents '
          'before making broader superiority claims.'),
        heading('Why test a fine-tuned encoder?'),
        p('<a href="https://github.com/allenai/scibert">Base SciBERT</a> provides scientific-text '
          'representations, not a ready software-name head. Task supervision lets us adapt name boundaries, '
          'version ownership and usage distinctions to the target schema. 006\'s observed advantages here '
          'are 21 more correct names, 41.2 percentage points more recall and 37.2 points more name F1, '
          'with similar precision. These are pipeline-level gains; this comparison does not isolate '
          'encoder fine-tuning from architecture, data or deployment differences.'),
        p('On the separate earlier 27 snippets, the matched frozen-encoder/trained-head control scores '
          '0.0% exact-name F1 versus 63.8% for 006\'s detector under the corrected same scorer. '
          'This fixed-recipe experiment supports fine-tuning, but does not test an optimized frozen-feature '
          'baseline. The pipeline also exposes local version/alias and '
          'sentiment outputs; extra fields are useful capabilities, not evidence of reliable accuracy.', 'small'),
        heading('WP4 resolution layer and optional funder context'),
        p('Add a resolver after extraction: <b>mention + paper context → registry candidates → evidence-based '
          'ranking → resolved ID or abstention</b>. GitHub search can retrieve candidates; its first result '
          'does not establish identity. Check explicit URLs, package ecosystem, software DOI, authors and '
          'repository metadata. Prefer a cached WP1 registry; include non-GitHub projects, forks, renamed '
          'repositories and ambiguous names such as STAR. Preserve the extracted mention and resolution '
          'provenance separately. Measure end-to-end link precision/recall and retrieval coverage.'),
        p('For funders, resolved usage by field and time could show adoption and dependence. '
          'Software-directed sentiment might add context about documented strengths or pain points, '
          'but it is experimental: 006 misses all four expressed opinions and invents seven. '
          'Do not rank funding priority from sentiment or mention counts alone. Coverage bias, criticality '
          'and maintenance need also matter; human judgment remains essential.'),
        heading('Next steps for quality and timing'),
        p('<b>Quality:</b> review the exported labels; keep these diagnostics out of training. Add new '
          'licensed training papers with repeated names, trademark boundaries, real versions and citation/section '
          'number negatives. Test one stage at a time and multiple seeds on development data. '
          'Calibrate confidence there, then use another untouched test. Compare registry matching and '
          'the preferred Softcite configuration as well as 006.'),
        p('<b>Timing:</b> the CLI already loads models once and uses evaluation/inference mode. '
          'Profile the five stages, remove repeated candidate construction/tokenization, batch attribute '
          'candidates and detector windows, and reduce device transfers. Gate each change on exact output '
          'agreement or a disclosed quality trade-off. Shared-encoder distillation comes later because '
          'it changes the model. No optimization speedup is claimed yet; measure cold load, warm median/p95, '
          'memory and cost on comparable native deployments.'),
        p('Evidence: cycle 003 report SHA-256 begins 1ff19fe04e56; selected pipeline manifest begins '
          'fe92420fff8a. Base: allenai/scibert_scivocab_cased; detector-005-wordpiece, linker-004, '
          'intent-003, sentiment-003, alias-003. Selection changes no weights. '
          'Registry links, calibrated confidence and funding-use validity remain unproven. '
          'Full excerpts and weights remain local pending sharing checks.', 'footnote'),
    ]
    return [first, second, third]


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
    ink = colors.HexColor('#16343c')
    accent = colors.HexColor('#136e79')
    muted = colors.HexColor('#52676e')
    base = ParagraphStyle('body', fontName='Helvetica', fontSize=9.15, leading=12.1,
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
        'small': dict(fontSize=8, leading=10.3, textColor=muted, spaceAfter=7),
        'footnote': dict(fontSize=7.4, leading=9.2, textColor=muted, spaceBefore=3),
        'example_title': dict(fontName='Helvetica-Bold', fontSize=9.6, leading=12, spaceAfter=3),
        'quote': dict(fontName='Times-Italic', fontSize=10.2, leading=12.5, textColor=accent,
                      leftIndent=8, spaceAfter=4),
        'source': dict(fontSize=7.2, leading=9, textColor=muted, spaceAfter=6),
        'callout': dict(fontName='Helvetica-Bold', fontSize=9.3, leading=12.5, textColor=accent,
                        spaceBefore=5, spaceAfter=7),
        'cell': dict(fontSize=8, leading=10, spaceAfter=0),
        'th': dict(fontName='Helvetica-Bold', fontSize=7.8, leading=10, spaceAfter=0),
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
                            topMargin=41, bottomMargin=36, title='Software mention extraction: OpenAlex progress',
                            author='Krushi Raj Tula', subject='Preliminary SciBERT and Softcite comparison')
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def main():
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
    for required in ('74.2%', '68.2%', '36.9%', '33 / 5 / 18', '12 / 2 / 39', '7.07', '3.60',
                     '18.55', '3.85', 'WP4', '0.89', '0.63', '2 (Pedregosa'):
        assert required in text, required
    for private in ('/Users/', '127.0.0.1', 'localhost', 'github.com/krushiraj/openalex-sw-mentions'):
        assert private not in html and private not in text, private
    assert '<script' not in html and 'http://' not in html
    links = []
    for page in pdf.pages:
        for ann in page.get('/Annots', []):
            uri = ann.get_object().get('/A', {}).get('/URI')
            if uri:
                links.append(str(uri))
    source_links = sorted(set(links))
    expected_links = {b['source_url'] for page in pages for b in page if b['type'] == 'example'}
    expected_links.update({'https://github.com/allenai/scibert', 'https://aclanthology.org/2025.sdp-1.13/',
                           'https://github.com/softcite/software-mentions'})
    assert set(source_links) == expected_links and len(source_links) == 9
    assert 'engineering team' not in text and 'join the team' not in text
    receipt = {'report_sha256': HASH, 'control_report_sha256': CONTROL_HASH,
               'html_sha256': sha(html.encode()),
               'pdf_sha256': sha(PDF.read_bytes()), 'pdf_pages': len(pdf.pages),
               'pdf_bytes': PDF.stat().st_size, 'source_link_count': len(source_links),
               'source_links': source_links, 'word_count_pdf': len(text.split()),
               'claims_status': 'provisional_diagnostic_not_general_superiority',
               'publication_status': 'local_review_only_not_sent',
               'examples': [{k: b[k] for k in ('title', 'document_id', 'source_url', 'quote')}
                            for page in pages for b in page if b['type'] == 'example']}
    (OUT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
