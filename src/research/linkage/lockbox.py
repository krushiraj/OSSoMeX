"""Fresh, never-trained-on lockbox: Methods text of recent open-access articles for end-to-end linkage."""

import json
import random
import re
from pathlib import Path
from urllib.parse import urlencode

from ..data.acquire import AcquisitionError, fetch_public
from ..data.jats import normalize_doi, safe_xml
from ..data.manifest import digest, json_bytes, write_jsonl, write_once
from ..registry.normalize import form_exact
from .gold import _context

EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
METHOD_TITLE = re.compile(r'method|materials|statistic|data analys|software|computational|experimental procedure|'
                          r'bioinformatic|image analys|data processing', re.I)
LICENSE = re.compile(r'creativecommons\.org/(licenses/(by|by-sa)/(2\.0|3\.0|4\.0)|publicdomain/zero/1\.0)/?(?![\w.-])')
ID_PATTERNS = [re.compile(r'PMC\d{5,}'), re.compile(r'10\.\d{4,9}/[^\s"\'<>]+', re.I)]


def _text(node):
    return re.sub(r'\s+', ' ', ''.join(node.itertext())).strip()


def _methods_sections(node):
    for sec in node.findall('sec'):
        title = sec.find('title')
        if 'method' in (sec.get('sec-type') or '') or (title is not None and METHOD_TITLE.search(_text(title))):
            yield sec
        else:
            yield from _methods_sections(sec)


def methods_document(xml_bytes):
    root = safe_xml(xml_bytes)
    meta, body = root.find('./front/article-meta'), root.find('body')
    if meta is None or body is None:
        raise ValueError('MISSING_ARTICLE_BODY')
    ids = {}
    for node in meta.findall('article-id'):
        kind, value = node.get('pub-id-type'), _text(node)
        if kind == 'pmcid':
            ids['pmcid'] = 'PMC' + value.removeprefix('PMC')
        elif kind == 'doi':
            ids['doi'] = normalize_doi(value)
    licence = None
    for node in meta.findall('./permissions/license'):
        content = ' '.join([*node.itertext(), *(v for n in node.iter() for v in n.attrib.values())])
        match = LICENSE.search(content)
        if match:
            licence = 'CC0-1.0' if match[1].startswith('publicdomain') else f'CC-{match[2].upper()}-{match[3]}'
            break
    if not licence:
        raise ValueError('TEXT_LICENSE_UNSUPPORTED')
    paragraphs = [_text(p) for sec in _methods_sections(body) for p in sec.iter('p')]
    paragraphs = [p for p in dict.fromkeys(paragraphs) if p]
    if not paragraphs:
        raise ValueError('NO_METHODS_SECTION')
    title = meta.find('./title-group/article-title')
    return {'text': '\n\n'.join(paragraphs), 'source_ids': ids, 'title': _text(title) if title is not None else None,
            'text_license': licence, 'supplied_text_scope': 'methods_sections'}


def sample_candidates(rows, n, per_journal, seed):
    unique = sorted({r['pmcid']: r for r in rows if r.get('pmcid')}.values(), key=lambda r: r['pmcid'])
    random.Random(seed).shuffle(unique)
    picked, journals = [], {}
    for row in unique:
        journal = row.get('journalTitle') or ''
        if journals.get(journal, 0) < per_journal:
            picked.append(row)
            journals[journal] = journals.get(journal, 0) + 1
        if len(picked) == n:
            break
    return picked


def paper_link_metrics(gold, predicted):
    pairs = set(predicted)
    groups = {}
    for g in gold:
        if g['entity_ids']:
            groups.setdefault(g['paper_id'], []).append(set(g['entity_ids']))
    correct = sum(any(e in ids for ids in groups.get(p, [])) for p, e in pairs)
    recalled = sum(any((p, e) in pairs for e in ids) for p, gs in groups.items() for ids in gs)
    total = sum(len(gs) for gs in groups.values())
    precision = correct / len(pairs) if pairs else None
    recall = recalled / total if total else None
    return {'predicted_links': len(pairs), 'correct_links': correct, 'gold_links': total, 'recalled_links': recalled,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision and recall else None}


def exposed_identifiers(paths):
    """Every PMCID/DOI string found in prior corpora, training and evaluation artifacts."""
    found = set()
    for path in paths:
        for file in Path(path).rglob('*.json*'):
            text = file.read_text(encoding='utf-8', errors='ignore')
            for pattern in ID_PATTERNS:
                found.update(m.rstrip('.,;)').lower() for m in pattern.findall(text))
    return found


def build_lockbox(config: dict, output: Path) -> dict:
    output = Path(output)
    policy = {'format': 'json', 'user_agent': config['user_agent']}
    rows, cursor = [], '*'
    for page in range(1, config.get('max_pages', 5) + 1):
        url = f'{EPMC}/search?' + urlencode({'query': config['query'], 'format': 'json', 'resultType': 'lite',
                                              'pageSize': 1000, 'cursorMark': cursor})
        destination = output / 'raw/search' / f'page-{page:02d}.json'
        fetch_public(url, destination, policy)
        body = json.loads(destination.read_bytes())
        rows += body['resultList']['result']
        if body.get('nextCursorMark') in (None, cursor) or not body['resultList']['result']:
            break
        cursor = body['nextCursorMark']
    exposed = exposed_identifiers(config['exposure_paths'])
    fresh = [r for r in rows if r.get('pmcid', '').lower() not in exposed and (r.get('doi') or '').lower() not in exposed]
    documents, rejected = [], []
    for row in sample_candidates(fresh, config['oversample'], config['per_journal'], config['seed']):
        destination = output / 'raw/fulltext' / f"{row['pmcid']}.xml"
        try:
            fetch_public(f"{EPMC}/{row['pmcid']}/fullTextXML", destination, {'user_agent': config['user_agent']})
            doc = methods_document(destination.read_bytes())
        except (AcquisitionError, ValueError) as exc:
            rejected.append({'pmcid': row['pmcid'], 'reason': str(exc)})
            continue
        if len(doc['text']) < config.get('min_chars', 800):
            rejected.append({'pmcid': row['pmcid'], 'reason': 'METHODS_TOO_SHORT'})
            continue
        documents.append({'document_id': f"lockbox_{row['pmcid']}", 'journal': row.get('journalTitle'),
                          'first_publication_date': row.get('firstPublicationDate'), **doc})
        if len(documents) == config['n']:
            break
    write_jsonl(output / 'documents.jsonl', documents)
    summary = {'status': 'success' if len(documents) == config['n'] else 'short', 'config': config,
               'candidates': len(rows), 'fresh_candidates': len(fresh), 'exposed_identifiers': len(exposed),
               'documents': len(documents), 'rejected': rejected,
               'documents_sha256': digest((output / 'documents.jsonl').read_bytes())}
    write_once(output / 'summary.json', json_bytes(summary))
    return summary


def tasks_from_prediction(document, prediction):
    """Group a document's predicted software occurrences into resolver tasks (one per distinct name)."""
    # Linkage needs detector occurrences only; attribute stages may be partial on long documents.
    detector = prediction.get('stage_status', {}).get('detector', {}).get('status')
    if (detector or prediction.get('status')) != 'success' or prediction.get('document_id') != document['document_id']:
        raise ValueError(f"PREDICTION_FAILED: {document['document_id']}")
    groups = {}
    for occurrence in prediction['occurrences']:
        span = occurrence['name_span']
        name = document['text'][span['start']:span['end']]
        groups.setdefault(form_exact(name), []).append((name, span))
    tasks = []
    for key, items in groups.items():
        forms = [name for name, _ in items]
        tasks.append({'task_id': f"{document['document_id']}#{key}", 'paper_id': document['document_id'],
                      'name': max(dict.fromkeys(forms), key=forms.count), 'surface_forms': sorted(set(forms)),
                      'mention_count': len(items), 'domain': document.get('journal') or '',
                      'contexts': [_context(document['text'], s['start'], s['end']) for _, s in items[:3]],
                      'urls': [], 'versions': [], 'publishers': [], 'languages': []})
    return tasks


def tasks_from_gold(document, rows):
    text, tasks = document['text'], []
    for row in rows:
        spans = [(m.start(), m.end()) for form in row['surface_forms']
                 for m in re.finditer(re.escape(form), text)]
        if not spans:
            raise ValueError(f"SURFACE_FORM_NOT_FOUND: {row['paper_id']} {row['name']}")
        spans.sort()
        tasks.append({'task_id': f"{row['paper_id']}#{form_exact(row['name'])}", 'paper_id': row['paper_id'],
                      'name': row['name'], 'surface_forms': row['surface_forms'], 'mention_count': len(spans),
                      'domain': document.get('journal') or '', 'contexts': [_context(text, a, b) for a, b in spans[:3]],
                      'urls': [], 'versions': [], 'publishers': [], 'languages': []})
    return tasks


def extraction_recall(gold, tasks):
    """Share of gold software (per paper) whose any surface form the extractor produced, ignoring case."""
    predicted = {}
    for t in tasks:
        predicted.setdefault(t['paper_id'], set()).update(form_exact(f) for f in t['surface_forms'])
    gold_forms = {}
    for g in gold:
        gold_forms.setdefault(g['paper_id'], []).append({form_exact(f) for f in g['surface_forms']})
    found = sum(bool(forms & predicted.get(p, set())) for p, groups in gold_forms.items() for forms in groups)
    total = sum(len(groups) for groups in gold_forms.values())
    unmatched = sum(not any({form_exact(f) for f in t['surface_forms']} & forms
                            for forms in gold_forms.get(t['paper_id'], [])) for t in tasks)
    return {'gold_software': total, 'found': found, 'recall': found / total if total else None,
            'predicted_names': len(tasks), 'unmatched_predicted_names': unmatched}


def _paper_bootstrap(gold, predicted, samples=2000, seed=0):
    papers = sorted({g['paper_id'] for g in gold})
    rng, values = random.Random(seed), {'precision': [], 'recall': []}
    for _ in range(samples):
        picks = rng.choices(papers, k=len(papers))
        weights = {p: picks.count(p) for p in set(picks)}
        pred = [(f'{p}~{i}', e) for p, e in set(predicted) for i in range(weights.get(p, 0))]
        g = [{**row, 'paper_id': f"{row['paper_id']}~{i}"} for row in gold for i in range(weights.get(row['paper_id'], 0))]
        m = paper_link_metrics(g, pred)
        for name in values:
            if m[name] is not None:
                values[name].append(m[name])
    return {k: [sorted(v)[int(0.025 * (len(v) - 1))], sorted(v)[int(0.975 * (len(v) - 1))]] if v else None
            for k, v in values.items()}


def evaluate_lockbox(run: Path, registry: Path, gold_path: Path, checkpoints) -> dict:
    from ..data.manifest import read_jsonl
    from .gold import load_index
    from .resolve import ContextResolver, FastTextMatch
    run = Path(run)
    documents = {d['document_id']: d for d in read_jsonl(run / 'documents.jsonl')}
    gold = read_jsonl(Path(gold_path))
    index = load_index(registry)
    arms = {'fast-text-match': FastTextMatch(index), 'context': ContextResolver(index)}
    gold_tasks = [t for doc_id, doc in documents.items()
                  for t in tasks_from_gold(doc, [g for g in gold if g['paper_id'] == doc_id])]
    report = {'documents': len(documents), 'gold_software': len(gold),
              'gold_linked': sum(bool(g['entity_ids']) for g in gold), 'arms': {}}
    sources = {'gold-mentions': gold_tasks}
    for checkpoint in checkpoints:
        predictions = {p['document_id']: p for p in read_jsonl(run / f'ossomex-{checkpoint}.jsonl')}
        tasks = [t for doc_id, doc in documents.items() for t in tasks_from_prediction(doc, predictions[doc_id])]
        sources[f'ossomex-{checkpoint}'] = tasks
        report[f'extraction_ossomex-{checkpoint}'] = extraction_recall(gold, tasks)
    for source, tasks in sources.items():
        for arm, resolver in arms.items():
            pairs = [(t['paper_id'], r['entity_id']) for t in tasks for r in [resolver.resolve(t)] if r['entity_id']]
            report['arms'][f'{source}+{arm}'] = {**paper_link_metrics(gold, pairs),
                                                 'ci95': _paper_bootstrap(gold, pairs)}
    return report
