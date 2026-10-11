"""Paper-to-software link gold tasks: one task per distinct software name in a paper."""

import random
import re
import zipfile
from pathlib import Path, PurePosixPath

from ..data.manifest import digest, json_bytes, read_jsonl, write_jsonl, write_once
from ..data.tei import read_tei
from ..registry.normalize import form_exact, form_loose

ATTRIBUTES = {'url': 'urls', 'publisher': 'publishers', 'version': 'versions', 'language': 'languages'}
CONTEXT_CHARS = 250
MAX_CONTEXTS = 3
MAX_CANDIDATES = 12


def _tokens(text):
    return {t for t in re.findall(r'[a-z0-9]+', form_exact(text)) if len(t) >= 2}


def _context(text, start, end):
    left = text[max(0, start - CONTEXT_CHARS):start]
    right = text[end:end + CONTEXT_CHARS]
    return re.sub(r'\s+', ' ', f'{left}[[{text[start:end]}]]{right}').strip()


def paper_tasks(paper_id, domain, xml_bytes):
    doc = read_tei(xml_bytes)
    text = doc['text']
    attached = {}
    for span in doc['spans']:
        field = ATTRIBUTES.get(span['type'])
        for target in (span['attributes'].get('corresp') or '').split() if field else []:
            attached.setdefault(target.lstrip('#'), []).append((field, span['text'].strip()))
    groups = {}
    for span in doc['spans']:
        subtype = span['attributes'].get('subtype')
        if span['type'] != 'software' or subtype == 'implicit' or not span['text'].strip():
            continue
        key = form_exact(span['text'])
        group = groups.setdefault(key, {'spans': [], 'subtypes': set(), **{f: [] for f in ATTRIBUTES.values()}})
        group['spans'].append(span)
        if subtype:
            group['subtypes'].add(subtype)
        for field, value in attached.get(span['source_id'], []):
            if value not in group[field]:
                group[field].append(value)
    tasks = []
    for key, group in groups.items():
        forms = [s['text'].strip() for s in group['spans']]
        tasks.append({
            'task_id': f'{paper_id}#{key}', 'paper_id': paper_id, 'domain': domain,
            'name': max(dict.fromkeys(forms), key=forms.count), 'surface_forms': sorted(set(forms)),
            'mention_count': len(forms), 'subtypes': sorted(group['subtypes']),
            'contexts': [_context(text, s['start'], s['end']) for s in group['spans'][:MAX_CONTEXTS]],
            **{field: group[field] for field in ATTRIBUTES.values()}})
    return tasks


def _summary(entity, match, form):
    return {'entity_id': entity['entity_id'], 'match': match, 'matched_form': form, 'names': entity['names'][:3],
            'aliases': entity['aliases'][:3], 'description': (entity['description'] or '')[:200],
            'package_ids': entity['package_ids'], 'repo_keys': entity['repo_keys'][:3], 'sources': entity['sources']}


class CandidateIndex:
    def __init__(self, entities, forms):
        self.entities = {e['entity_id']: e for e in entities}
        self.exact, self.loose, self.by_token = {}, {}, {}
        for row in forms:
            ids = [e['entity_id'] for e in row['entities']]
            (self.exact if row['key_type'] == 'exact' else self.loose)[row['form']] = ids
            if row['key_type'] == 'exact':
                for token in _tokens(row['form']):
                    self.by_token.setdefault(token, set()).add(row['form'])
        self.haystack = {e['entity_id']: ' '.join([*e['names'], *e['aliases'], e['description'] or '', *e['repo_keys'],
                                                   *(p for ps in e['package_ids'].values() for p in ps)]).casefold()
                         for e in entities}

    def candidates(self, surface_forms):
        found = {}

        def add(ids, match, form):
            for entity_id in sorted(ids):
                found.setdefault(entity_id, (match, form))
        for form in surface_forms:
            add(self.exact.get(form_exact(form), []), 'exact', form)
        for form in surface_forms:
            add(self.loose.get(form_loose(form) or '', []), 'loose', form)
        for form in surface_forms:
            query = _tokens(form)
            if not query or max(len(t) for t in query) < 3:
                continue
            pool = set.intersection(*(self.by_token.get(t, set()) for t in query))
            for other in sorted(pool):
                if other != form_exact(form):
                    add(self.exact[other], 'token', other)
        order = {'exact': 0, 'loose': 1, 'token': 2}
        ranked = sorted(found.items(), key=lambda kv: (order[kv[1][0]], kv[0]))[:MAX_CANDIDATES]
        return [_summary(self.entities[e], match, form) for e, (match, form) in ranked]

    def search(self, query, limit=15):
        needle = query.casefold().strip()
        hits = []
        for entity_id, haystack in self.haystack.items():
            if needle in haystack:
                names = [n.casefold() for n in self.entities[entity_id]['names']]
                rank = 0 if needle in names else 1 if any(needle in n for n in names) else 2
                hits.append((rank, entity_id))
        return [_summary(self.entities[e], 'search', query) for _, e in sorted(hits)[:limit]]


def sample_papers(names, per_domain, seed, exclude):
    domains = {}
    for name in sorted(n for n in names if n.endswith('.xml')):
        stem = PurePosixPath(name).name.split('.')[0]
        if name not in exclude and stem not in exclude:
            domains.setdefault(PurePosixPath(name).parts[1], []).append(name)
    rng = random.Random(seed)
    picked = []
    for domain in sorted(domains):
        picked += rng.sample(domains[domain], min(per_domain, len(domains[domain])))
    return sorted(picked)


def load_index(registry_run: Path):
    run = Path(registry_run)
    return CandidateIndex(read_jsonl(run / 'entities.jsonl'), read_jsonl(run / 'surface_forms.jsonl'))


def build_sofair_tasks(zip_path: Path, registry_run: Path, output: Path, per_domain=4, seed=42, exclude=()):
    zip_path, output = Path(zip_path), Path(output)
    index = load_index(registry_run)
    tasks = []
    with zipfile.ZipFile(zip_path) as archive:
        papers = sample_papers(archive.namelist(), per_domain, seed, set(exclude))
        for name in papers:
            parts = PurePosixPath(name).parts
            for task in paper_tasks('sofair:' + '/'.join(parts[1:]), parts[1], archive.read(name)):
                task['candidates'] = index.candidates(task['surface_forms'])
                tasks.append(task)
    write_jsonl(output / 'tasks.jsonl', tasks)
    summary = {'status': 'success', 'source': 'SoFAIR v2 (CC BY 4.0, doi:10.21954/ou.rd.30374830.v2)',
               'archive_sha256': digest(zip_path.read_bytes()), 'registry_run': str(registry_run),
               'per_domain': per_domain, 'seed': seed, 'excluded': sorted(exclude), 'papers': len(papers),
               'tasks': len(tasks), 'paper_ids': papers,
               'development_exposed': 'SoFAIR papers were in OSSoMeX detector training; resolution-only use'}
    write_once(output / 'summary.json', json_bytes(summary))
    return summary
