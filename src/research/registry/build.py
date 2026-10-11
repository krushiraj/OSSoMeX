"""Merge source records into entities by shared identifiers and index their surface forms."""

import json
import re
from pathlib import Path

from ..data.manifest import json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from .normalize import form_exact, form_loose
from .sources import LICENCES, SOURCE_PRIORITY, parse_biotools, parse_ecosystems, parse_wikidata

PLACEHOLDER = re.compile(r'\b(reserved|placeholder|security holding package|dummy package)\b', re.I)
LIST_FIELDS = ('aliases', 'homepages', 'repo_keys', 'rrids', 'publication_dois', 'topics', 'classes')


def _priority(key):
    return (SOURCE_PRIORITY[key.split(':', 1)[0]], key)


class _Components:
    def __init__(self, records):
        self.parent = {r['key']: r['key'] for r in records}
        self.packages = {r['key']: {(e, p) for e, ps in r['package_ids'].items() for p in ps} for r in records}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def conflict(self, a, b):
        left, right = self.packages[self.find(a)], self.packages[self.find(b)]
        for ecosystem in sorted({e for e, _ in left} & {e for e, _ in right}):
            if {p for e, p in left if e == ecosystem} != {p for e, p in right if e == ecosystem}:
                return f'distinct_{ecosystem}_packages'
        return None

    def union(self, a, b):
        ra, rb = sorted((self.find(a), self.find(b)), key=_priority)
        if ra != rb:
            self.parent[rb] = ra
            self.packages[ra] |= self.packages.pop(rb)


def merge(records):
    """Strong edges (package ID, RRID) always merge; repository edges refuse package conflicts."""
    components = _Components(records)
    strong, repos = {}, {}
    for r in records:
        for edge in [f'pkg:{e}:{p}' for e, ps in r['package_ids'].items() for p in ps] + [f'rrid:{x}' for x in r['rrids']]:
            strong.setdefault(edge, []).append(r['key'])
        for repo in r['repo_keys']:
            repos.setdefault(f'repo:{repo}', []).append(r['key'])
    for edge in sorted(strong):
        first, *rest = sorted(strong[edge], key=_priority)
        for key in rest:
            components.union(first, key)
    conflicts = []
    for edge in sorted(repos):
        first, *rest = sorted(repos[edge], key=_priority)
        for key in rest:
            if components.find(first) == components.find(key):
                continue
            reason = components.conflict(first, key)
            if reason:
                conflicts.append({'edge': edge, 'left': first, 'right': key, 'reason': reason})
            else:
                components.union(first, key)
    groups = {}
    for r in records:
        groups.setdefault(components.find(r['key']), []).append(r)
    return [_entity(sorted(members, key=lambda r: _priority(r['key']))) for members in groups.values()], conflicts


def _entity(members):
    names = list(dict.fromkeys(r['name'] for r in members if r['name']))
    entity = {'entity_id': members[0]['key'], 'members': sorted(r['key'] for r in members),
              'sources': sorted({r['source'] for r in members}), 'names': names,
              'description': next((r['description'] for r in members if r['description']), None),
              'package_ids': {}, 'czi': [{**r['czi'], 'record': r['key']} for r in members if 'czi' in r]}
    for field in LIST_FIELDS:
        entity[field] = list(dict.fromkeys(v for r in members for v in r[field]
                                           if field != 'aliases' or v not in names))
    for r in members:
        for ecosystem, ids in r['package_ids'].items():
            entity['package_ids'][ecosystem] = list(dict.fromkeys(entity['package_ids'].get(ecosystem, []) + ids))
    entity['package_ids'] = dict(sorted(entity['package_ids'].items()))
    return entity


def _entity_forms(entity):
    yield from ((n, 'name') for n in entity['names'])
    yield from ((a, 'alias') for a in entity['aliases'])
    yield from ((p, 'package') for ids in entity['package_ids'].values() for p in ids)
    yield from ((c['czi_name'], 'czi') for c in entity['czi'])


def surface_forms(entities):
    index = {}
    for entity in entities:
        for text, provenance in _entity_forms(entity):
            for key_type, form in (('exact', form_exact(text)), ('loose', form_loose(text))):
                if form:
                    index.setdefault((key_type, form), {}).setdefault(entity['entity_id'], set()).add(provenance)
    return [{'form': form, 'key_type': key_type,
             'entities': [{'entity_id': e, 'provenance': sorted(p)} for e, p in sorted(found.items())]}
            for (key_type, form), found in sorted(index.items())]


def czi_links(entities, forms):
    """CZI name->package links are untrusted; flag placeholders and names claimed by another entity."""
    exact = {f['form']: f['entities'] for f in forms if f['key_type'] == 'exact'}
    rows = []
    for entity in entities:
        for link in entity['czi']:
            others = [e['entity_id'] for e in exact.get(form_exact(link['czi_name']), [])
                      if e['entity_id'] != entity['entity_id'] and set(e['provenance']) & {'name', 'alias', 'package'}]
            flags = []
            if not entity['description'] or PLACEHOLDER.search(entity['description']):
                flags.append('placeholder_description')
            if others:
                flags.append('name_collision')
            rows.append({**link, 'entity_id': entity['entity_id'], 'description': entity['description'],
                         'same_name_entities': others, 'flags': flags,
                         'trust': 'low_trust' if flags else 'unaudited'})
    return sorted(rows, key=lambda r: r['record'])


def coverage(forms, names):
    found = {(f['key_type'], f['form']): len(f['entities']) for f in forms}
    exact = [n for n in names if ('exact', form_exact(n)) in found]
    loose = [n for n in names if form_loose(n) and ('loose', form_loose(n)) in found]
    return {'names': len(names), 'exact': len(exact), 'loose': len(loose),
            'ambiguous_exact': sum(1 for n in exact if found[('exact', form_exact(n))] > 1),
            'missing_examples': sorted(n for n in names if n not in exact and n not in loose)[:25]}


def _load_records(run):
    manifest = json.loads((run / 'pull-manifest.json').read_bytes())
    parsers = {'ecosystems': lambda d: [parse_ecosystems(r) for r in d],
               'biotools': lambda d: [parse_biotools(t) for t in d['list']],
               'wikidata': lambda d: parse_wikidata(d['results']['bindings'])}
    records = []
    for source in sorted(manifest['sources']):
        for page in manifest['sources'][source]['files']:
            if page.get('role') == 'list':
                continue
            data = json.loads(verified_path(run, page).read_bytes())
            records += parsers[source](data)
    return records, manifest


def build_run(run: Path) -> dict:
    run = Path(run)
    records, manifest = _load_records(run)
    entities, conflicts = merge(records)
    entities.sort(key=lambda e: e['entity_id'])
    forms = surface_forms(entities)
    links = czi_links(entities, forms)
    exact = [f for f in forms if f['key_type'] == 'exact']
    stats = {
        'status': 'success', 'pull_manifest_sha256': manifest.get('sha256'),
        'records': {s: sum(1 for r in records if r['source'] == s) for s in sorted(manifest['sources'])},
        'entities': len(entities), 'multi_source_entities': sum(1 for e in entities if len(e['sources']) > 1),
        'refused_merges': len(conflicts), 'exact_forms': len(exact),
        'ambiguous_exact_forms': sum(1 for f in exact if len(f['entities']) > 1),
        'czi_links': {'total': len(links), 'low_trust': sum(r['trust'] == 'low_trust' for r in links),
                      'unaudited': sum(r['trust'] == 'unaudited' for r in links)},
        'licences': {s: LICENCES[s] for s in sorted(manifest['sources'])}}
    write_jsonl(run / 'records.jsonl', records)
    write_jsonl(run / 'entities.jsonl', entities)
    write_jsonl(run / 'surface_forms.jsonl', forms)
    write_jsonl(run / 'merge-conflicts.jsonl', conflicts)
    write_jsonl(run / 'czi_links.jsonl', links)
    write_once(run / 'stats.json', json_bytes(stats))
    return stats


def read_names(path: Path):
    names = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('{'):
            line = json.loads(line).get('name') or ''
        if line:
            names.append(line)
    return list(dict.fromkeys(names))


def coverage_run(run: Path, names_path: Path) -> dict:
    return {'status': 'success', **coverage(read_jsonl(Path(run) / 'surface_forms.jsonl'), read_names(names_path))}
