"""Work-level identity, leakage checks, and deterministic quota assignment."""

from copy import deepcopy
import random
import re
from urllib.parse import unquote

from ..contracts import validate_document
from .manifest import digest, json_bytes

SOURCE_ORDER = ['sofair', 'somesci', 'ecosystems', 'openalex']


class SplitError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(report['code'])


def identifiers(doc: dict) -> set[str]:
    result = set()
    for key, value in doc.get('source_ids', {}).items():
        if not value:
            continue
        value = unquote(str(value)).strip().lower()
        if key == 'doi':
            value = re.sub(r'^(?:https?://(?:dx\.)?doi.org/|doi:)', '', value)
        elif key in ('pmcid', 'openalex', 'pmid'):
            value = value.rstrip('/').rsplit('/', 1)[-1]
        result.add(f'{key}:{value}')
    return result


def group_works(documents: list[dict]) -> dict:
    docs = sorted([validate_document(d) for d in documents], key=lambda d: d['document_id'])
    if len({d['document_id'] for d in docs}) != len(docs):
        raise ValueError('DUPLICATE_DOCUMENT_ID')
    parents = list(range(len(docs)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, doc in enumerate(docs):
        keys = identifiers(doc) | {'text:' + digest(re.sub(r'\s+', ' ', doc['text']).strip().encode())}
        for key in sorted(keys):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    grouped = {}
    for i, doc in enumerate(docs):
        grouped.setdefault(root(i), []).append(doc)
    groups = []
    for members in grouped.values():
        sources = {d.get('source', (d.get('metadata') or {}).get('source')) for d in members}
        source = min(sources, key=lambda s: (SOURCE_ORDER.index(s) if s in SOURCE_ORDER else 99, s or ''))
        group_id = 'work:' + digest(json_bytes(sorted(d['document_id'] for d in members)))[:24]
        groups.append({'work_group_id': group_id, 'source': source, 'documents': members,
                       'unresolved_overlaps': []})
    groups.sort(key=lambda g: g['work_group_id'])
    overlaps = []
    tokens = {d['document_id']: re.findall(r'\w+', d['text'].lower()) for d in docs}
    shingles = {k: set(tuple(t[i:i+5]) for i in range(len(t)-4)) for k, t in tokens.items()}
    doc_group = {d['document_id']: g for g in groups for d in g['documents']}
    for i, a in enumerate(docs):
        aid = a['document_id']; sa = shingles[aid]
        for b in docs[i+1:]:
            bid = b['document_id']; sb = shingles[bid]
            if doc_group[aid] is doc_group[bid] or not sa or not sb:
                continue
            ratio = len(sa & sb) / len(sa | sb)
            excerpt = min(len(sa), len(sb)) >= 6 and (sa <= sb or sb <= sa)
            if ratio >= .85 or excerpt:
                issue = {'left': aid, 'right': bid, 'jaccard': ratio,
                         'reason': 'near_duplicate' if ratio >= .85 else 'excerpt_overlap'}
                overlaps.append(issue)
                doc_group[aid]['unresolved_overlaps'].append(issue)
                doc_group[bid]['unresolved_overlaps'].append(issue)
    return {'groups': groups, 'unresolved_overlaps': overlaps}


def check_training_manifest(manifest: dict, heldout: dict) -> list[dict]:
    issues = []
    for d in manifest.get('documents', []):
        codes = []
        if d.get('work_group_id') in heldout.get('work_group_ids', []):
            codes.append('HELDOUT_WORK_GROUP')
        if d.get('text_revision') in heldout.get('text_revisions', []):
            codes.append('HELDOUT_TEXT')
        if identifiers(d) & set(heldout.get('source_ids', [])):
            codes.append('HELDOUT_IDENTIFIER')
        if d.get('native_split') in ('test', 'dev', 'devel', 'validation', 'unknown'):
            codes.append('NATIVE_HELDOUT')
        if d.get('historical') or d.get('split') not in (None, 'train'):
            codes.append('HELDOUT_DOCUMENT')
        if 'synthetic' in str(d.get('source', (d.get('metadata') or {}).get('source', ''))).lower():
            codes.append('SYNTHETIC_NOT_RESEARCH')
        issues.extend({'document_id': d['document_id'], 'code': c} for c in codes)
    return issues


def assign_splits(groups: list[dict], quotas: dict, seed: int) -> dict:
    if any(g.get('unresolved_overlaps') for g in groups):
        raise SplitError({'code': 'UNRESOLVED_OVERLAPS'})
    rng = random.Random(seed)
    selected, shortages, excluded = [], [], []
    for source, quota in sorted(quotas.items()):
        pool = []
        for g in sorted(groups, key=lambda g: g['work_group_id']):
            if g['source'] != source:
                continue
            if any(d.get('historical') for d in g['documents']):
                excluded.append({'work_group_id': g['work_group_id'], 'reason': 'historical'})
                continue
            if any(d.get('native_split') == 'unknown' for d in g['documents']):
                excluded.append({'work_group_id':g['work_group_id'], 'reason':'native_split_unknown'})
                continue
            eligible = [d for d in g['documents'] if d.get('fulltext_eligible') is True
                        and 'synthetic' not in str(d.get('source', '')).lower()]
            if eligible:
                representative = sorted(eligible, key=lambda d: (d.get('source') != source, d['document_id']))[0]
                pool.append((g, representative))
        rng.shuffle(pool)
        used = set()
        # Reserve held-outs first, preferring native partitions over unrestricted candidates.
        for role in ('test', 'dev', 'train'):
            count = quota.get(role, 0)
            if type(count) is not int or count < 0:
                raise ValueError('INVALID_QUOTA')
            candidates = []
            for g, d in pool:
                native = {x.get('native_split') for x in g['documents']}
                if role != 'train' and any(x.get('development_exposed') for x in g['documents']):
                    continue
                if g['work_group_id'] in used:
                    continue
                if role == 'train' and native & {'test','dev','devel','validation'}:
                    continue
                if role == 'dev' and 'test' in native:
                    continue
                preferred = ('test' in native if role == 'test' else bool(native & {'dev','devel','validation'}) if role == 'dev' else True)
                candidates.append((not preferred, g, d))
            # Round-robin metadata strata within each native priority tier.
            buckets = {}
            for priority, g, d in candidates:
                metadata = d.get('metadata') or {}
                stratum = str(metadata.get('domain') or metadata.get('publication_year') or 'unknown')
                buckets.setdefault((priority, stratum), []).append((g, d))
            ordered = []
            for priority in (False, True):
                active = [buckets[k][:] for k in sorted(buckets) if k[0] == priority]
                while any(active):
                    for bucket in active:
                        if bucket:
                            ordered.append(bucket.pop(0))
            if len(ordered) < count:
                shortages.append({'source': source, 'split': role, 'required': count,
                                  'available': len(ordered), 'shortage': count-len(ordered)})
            for g, d in ordered[:count]:
                used.add(g['work_group_id'])
                selected.append({**deepcopy(d), 'work_group_id': g['work_group_id'], 'split': role})
    if shortages:
        raise SplitError({'code': 'QUOTA_SHORTAGE', 'shortages': shortages})
    heldout_groups = [g for g in groups if any(d.get('historical') or d.get('native_split') in ('test','dev','devel','validation') for d in g['documents'])
                      or g['work_group_id'] in {d['work_group_id'] for d in selected if d['split'] != 'train'}]
    heldout_docs = [d for g in heldout_groups for d in g['documents']]
    heldout = {'work_group_ids': sorted(g['work_group_id'] for g in heldout_groups),
               'text_revisions': sorted({d['text_revision'] for d in heldout_docs}),
               'source_ids': sorted(set().union(*(identifiers(d) for d in heldout_docs)))}
    result = {'documents': sorted(selected, key=lambda d: (d['split'], d['document_id'])),
              'heldout': heldout, 'exclusions': excluded, 'seed': seed, 'quotas': quotas}
    result['split_digest'] = digest(json_bytes(result))
    return result
