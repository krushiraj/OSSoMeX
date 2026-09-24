"""Explicit software-name relations within one owned annotation task."""

from copy import deepcopy

from ..contracts import occurrence_id
from ..data.manifest import digest, json_bytes


def alias_relation_id(document_id: str, revision: str, members: list[str]) -> str:
    return 'alias:' + digest(json_bytes([document_id, revision, sorted(members)]))


def local_group_id(document_id: str, revision: str, members: list[str]) -> str:
    return 'local-software:' + digest(json_bytes([document_id, revision, sorted(members)]))


def _span(value, lower: int, upper: int) -> tuple[int, int] | None:
    if (not isinstance(value, dict) or type(value.get('start')) is not int
            or type(value.get('end')) is not int):
        return None
    start, end = value['start'], value['end']
    return (start, end) if lower <= start < end <= upper else None


def _endpoint(task: dict, occurrence: dict) -> tuple[int, int]:
    if not isinstance(occurrence, dict):
        raise ValueError('ALIAS_ENDPOINT_INVALID')
    doc_id, revision = task['document_id'], task['text_revision']
    if occurrence.get('document_id') != doc_id or occurrence.get('text_revision') != revision:
        raise ValueError('ALIAS_ENDPOINT_INVALID')
    context = task['context_span']
    owned = task['annotation_region']
    source = task['text']
    base = task['offset_base']
    name_span = _span(occurrence.get('name_span'), owned['start'], owned['end'])
    context_span = _span(occurrence.get('context_span'), context['start'], context['end'])
    if name_span is None or context_span is None:
        raise ValueError('ALIAS_ENDPOINT_INVALID')
    start, end = name_span
    cs, ce = context_span
    if (not cs <= start < end <= ce
            or occurrence.get('mention_id') != occurrence_id(doc_id, revision, start, end)
            or not isinstance(occurrence.get('name'), str)
            or occurrence['name'] != source[start-base:end-base]
            or occurrence.get('context_sentence') != source[cs-base:ce-base]):
        raise ValueError('ALIAS_ENDPOINT_INVALID')
    return name_span


def validate_alias_annotations(task: dict, occurrences: list[dict], layer: dict | None) -> dict | None:
    if layer is None:
        return None
    if not isinstance(layer, dict) or layer.get('schema_version') != '1.0':
        raise ValueError('ALIAS_SCHEMA_UNSUPPORTED')
    relations = layer.get('relations')
    if not isinstance(relations, list):
        raise ValueError('INVALID_ALIAS_RECORD')

    by_id = {occurrence.get('mention_id'): occurrence for occurrence in occurrences
             if isinstance(occurrence, dict)}
    seen_pairs = set()
    context = task['context_span']
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError('INVALID_ALIAS_RECORD')
        members = relation.get('member_mention_ids')
        if (any(not isinstance(relation.get(key), str) or not relation[key].strip()
                for key in ('relation_id', 'document_id', 'text_revision', 'relation_type', 'decision'))
                or not isinstance(members, list) or any(not isinstance(mid, str) or not mid.strip()
                                                          for mid in members)
                or 'preferred_mention_id' not in relation
                or (relation['preferred_mention_id'] is not None
                    and not isinstance(relation['preferred_mention_id'], str))
                or not isinstance(relation.get('evidence_spans'), list)
                or not isinstance(relation.get('review'), dict)):
            raise ValueError('INVALID_ALIAS_RECORD')
        if (relation['relation_type'] not in ('abbreviation', 'explicit_alternative_name')
                or relation['decision'] not in ('alias', 'not_alias', 'unresolved')):
            raise ValueError('INVALID_ALIAS_RECORD')
        if (relation['document_id'] != task['document_id']
                or relation['text_revision'] != task['text_revision']):
            raise ValueError('ALIAS_IDENTITY_MISMATCH')
        if len(members) != 2 or members[0] == members[1] or members != sorted(members):
            raise ValueError('INVALID_ALIAS_RECORD')
        if any(mid not in by_id for mid in members):
            raise ValueError('ALIAS_ENDPOINT_INVALID')
        spans = [_endpoint(task, by_id[mid]) for mid in members]
        if spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]:
            raise ValueError('ALIAS_ENDPOINT_INVALID')
        if relation['relation_id'] != alias_relation_id(task['document_id'], task['text_revision'], members):
            raise ValueError('ALIAS_ID_MISMATCH')
        pair = tuple(members)
        if pair in seen_pairs:
            raise ValueError('DUPLICATE_ALIAS_PAIR')
        seen_pairs.add(pair)
        preferred = relation['preferred_mention_id']
        if (relation['decision'] == 'alias' and preferred not in members
                or relation['decision'] != 'alias' and preferred is not None):
            raise ValueError('ALIAS_PREFERENCE_INVALID')
        evidence = relation['evidence_spans']
        evidence_spans = [_span(span, context['start'], context['end']) for span in evidence]
        if (not evidence_spans or any(span is None for span in evidence_spans)
                or not any(span[0] <= min(a[0] for a in spans)
                           and span[1] >= max(a[1] for a in spans) for span in evidence_spans)):
            raise ValueError('ALIAS_EVIDENCE_INVALID')
        review = relation['review']
        reasons = review.get('reasons')
        if (not isinstance(review.get('status'), str) or not review['status'].strip()
                or not isinstance(reasons, list)
                or any(not isinstance(reason, str) or not reason.strip() for reason in reasons)
                or (review['status'] == 'human_reviewed'
                    and any(not isinstance(review.get(key), str) or not review[key].strip()
                            for key in ('reviewer', 'decision_id', 'recorded_at_utc')))):
            raise ValueError('ALIAS_REVIEW_INVALID')

    result = deepcopy(layer)
    build_alias_groups(occurrences, result)
    return result


def build_alias_groups(occurrences: list[dict], layer: dict | None) -> list[dict]:
    if layer is None:
        return []
    relations = layer['relations']
    by_id = {occurrence['mention_id']: occurrence for occurrence in occurrences}
    adjacency: dict[str, set[str]] = {}
    for relation in relations:
        if relation['decision'] != 'alias':
            continue
        left, right = relation['member_mention_ids']
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    groups = []
    visited = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        component = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        if len(component) < 2:
            continue
        contributing = [relation for relation in relations
                        if relation['decision'] == 'alias'
                        and set(relation['member_mention_ids']) <= component]
        if any(relation['decision'] == 'not_alias'
               and set(relation['member_mention_ids']) <= component for relation in relations):
            raise ValueError('ALIAS_CONTRADICTION')
        members = sorted(component)
        source_order = sorted(members, key=lambda mid: (by_id[mid]['name_span']['start'],
                                                         by_id[mid]['name_span']['end'], mid))
        preferences = {by_id[relation['preferred_mention_id']]['name'] for relation in contributing}
        statuses = {relation['review']['status'] for relation in contributing}
        quality = ('human_reviewed' if statuses == {'human_reviewed'} else
                   'synthetic_fixture' if statuses == {'synthetic_fixture'} else 'provisional')
        first = by_id[members[0]]
        group = {'local_entity_id': local_group_id(first['document_id'], first['text_revision'], members),
                 'document_id': first['document_id'], 'text_revision': first['text_revision'],
                 'member_mention_ids': members, 'preferred_name': next(iter(preferences))
                 if len(preferences) == 1 else None,
                 'names': [by_id[mid]['name'] for mid in source_order], 'quality': quality}
        if len(preferences) > 1:
            group['review_reasons'] = ['alias_preference_conflict']
        groups.append(group)
    return sorted(groups, key=lambda group: group['local_entity_id'])
