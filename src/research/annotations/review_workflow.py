"""Pure projection of scoped human review and conservative legacy provenance."""

from __future__ import annotations

from copy import deepcopy
import re

from ..contracts import FIELDS, INTENT_BITS, occurrence_id
from ..data.manifest import digest, json_bytes
from .aliases import alias_relation_id, build_alias_groups


WORKFLOW_VERSION = '1.0'
HASH = re.compile(r'^[0-9a-f]{64}$')


def required_fields(task: dict) -> set[str]:
    mapping = {'software': ('software',), 'version': ('versions',),
               'version_links': ('versions',),
               'intents': ('created', 'used', 'shared'), 'sentiment': ('sentiment',)}
    return {field for key in task['requested_fields'] for field in mapping.get(key, ())}


def value_hash(value) -> str:
    return digest(json_bytes(value))


def field_fingerprint(task: dict, occurrence: dict, field: str) -> str:
    if field not in FIELDS:
        raise ValueError('WORKFLOW_FIELD_INVALID')
    known = occurrence['known'][field]
    identity = [task['task_id'], task['document_id'], task['text_revision'],
                occurrence['mention_id'], occurrence['name_span']]
    if field == 'software':
        value = {'name': occurrence['name'], 'name_span': occurrence['name_span']}
    elif field == 'versions':
        value = {'version_status': occurrence['version_status'],
                 'version_links': occurrence['version_links']}
    elif field in INTENT_BITS:
        value = {'positive': field in (occurrence['intents'] or []),
                 'evidence': occurrence['evidence']['intents']}
    else:
        value = {'sentiment': occurrence['sentiment'],
                 'evidence': occurrence['evidence']['sentiment']}
    return value_hash([identity, field, known, value])


def _content_without_stamps(annotation: dict) -> dict:
    content = {key: deepcopy(annotation.get(key)) for key in
               ('status', 'occurrences', 'covered_regions', 'unresolved_regions',
                'alias_annotations') if key in annotation}
    for occurrence in content.get('occurrences', []):
        occurrence.pop('review', None)
        occurrence.pop('field_reviews', None)
    content['occurrences'] = sorted(content.get('occurrences', []),
                                    key=lambda occurrence: occurrence['mention_id'])
    for region in content.get('covered_regions', []):
        region.pop('human_reviewed_fields', None)
    layer = content.get('alias_annotations')
    if isinstance(layer, dict) and isinstance(layer.get('relations'), list):
        for relation in layer['relations']:
            relation.pop('review', None)
        layer['relations'].sort(key=lambda relation: relation['relation_id'])
    workflow = annotation.get('review_workflow')
    content['source_issues'] = deepcopy(workflow.get('source_issues', [])) if isinstance(workflow, dict) else []
    for issue in content['source_issues']:
        for key in ('decision_id', 'operation_id', 'reviewer', 'recorded_at_utc'):
            issue.pop(key, None)
    content['source_issues'].sort(key=lambda issue: issue['issue_id'])
    return content


def content_fingerprint(task: dict, annotation: dict) -> str:
    identity = [task['task_id'], task['document_id'], task['text_revision']]
    return value_hash([identity, _content_without_stamps(annotation)])


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _span(span, lower: int, upper: int) -> bool:
    return (isinstance(span, dict) and type(span.get('start')) is int
            and type(span.get('end')) is int
            and lower <= span['start'] < span['end'] <= upper)


def _history_index(item: dict, history: list[dict]) -> dict[str, dict]:
    if not isinstance(history, list):
        raise ValueError('WORKFLOW_HISTORY_INVALID')
    records = {}
    for row in history:
        if not isinstance(row, dict) or not isinstance(row.get('payload'), dict) \
                or not isinstance(row.get('result'), dict) or not _nonblank(row.get('recorded_at')):
            raise ValueError('WORKFLOW_HISTORY_INVALID')
        payload = row['payload']
        ident = payload.get('decision_id')
        if (not _nonblank(ident) or ident in records
                or payload.get('task_id') != item['task']['task_id']
                or payload.get('document_id') != item['task']['document_id']
                or payload.get('text_revision') != item['task']['text_revision']
                or row['result'].get('decision_id') != ident):
            raise ValueError('WORKFLOW_HISTORY_IDENTITY_MISMATCH')
        recorded = row['result'].get('recorded_at_utc')
        if recorded is not None and recorded != row['recorded_at']:
            raise ValueError('WORKFLOW_HISTORY_TIME_MISMATCH')
        records[ident] = row
    return records


def _matching_row(records: dict[str, dict], ident: str, reviewer: str, recorded_at: str) -> dict:
    row = records.get(ident)
    if (row is None or row['payload'].get('reviewer') != reviewer
            or row['recorded_at'] != recorded_at):
        raise ValueError('WORKFLOW_REVIEW_REFERENCE_MISMATCH')
    return row


def _operation(row: dict, operation_id: str) -> dict:
    payload = row['payload']
    if payload.get('action') != 'apply_review_batch' or not isinstance(payload.get('value'), dict):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    operations = payload['value'].get('operations')
    if not isinstance(operations, list):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    generated = row['result'].get('approval_operations', [])
    if not isinstance(generated, list):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    if generated and (payload.get('actor_kind') != 'human'
                      or payload['value'].get('completion') != 'approve'
                      or payload['value'].get('proposals_revealed') is not True
                      or any(not isinstance(op, dict) or op.get('action') not in
                             ('accept_fields', 'accept_alias', 'reject_alias') for op in generated)):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    operations = operations + generated
    ids = [op.get('operation_id') for op in operations if isinstance(op, dict)]
    if any(not isinstance(ident, str) for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    matches = [operation for operation in operations if isinstance(operation, dict)
               and operation.get('operation_id') == operation_id]
    if len(matches) != 1:
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    return matches[0]


def _validate_field_stamp(item: dict, records: dict[str, dict], mid: str,
                          field: str, stamp: dict, occurrences: dict[str, dict]) -> None:
    if field not in FIELDS or mid not in occurrences or not isinstance(stamp, dict):
        raise ValueError('WORKFLOW_FIELD_REVIEW_INVALID')
    if (stamp.get('state') not in ('confirmed', 'unresolved')
            or not isinstance(stamp.get('value_hash'), str)
            or not HASH.fullmatch(stamp['value_hash'])
            or any(not _nonblank(stamp.get(key)) for key in
                   ('decision_id', 'operation_id', 'reviewer', 'recorded_at_utc'))):
        raise ValueError('WORKFLOW_FIELD_REVIEW_INVALID')
    row = _matching_row(records, stamp['decision_id'], stamp['reviewer'], stamp['recorded_at_utc'])
    if row['payload'].get('actor_kind') != 'human':
        raise ValueError('WORKFLOW_REVIEW_REFERENCE_MISMATCH')
    operation = _operation(row, stamp['operation_id'])
    if (operation.get('action') not in ('upsert_occurrence', 'accept_fields')
            or not isinstance(operation.get('fields'), list)
            or field not in operation['fields']):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    occurrence = occurrences[mid]
    target = operation.get('target_name_span')
    value = operation.get('value')
    if target != occurrence['name_span'] and not (
            isinstance(value, dict) and value.get('name_span') == occurrence['name_span']
            and value.get('mention_id', mid) == mid):
        raise ValueError('WORKFLOW_REVIEW_SCOPE_MISMATCH')
    if operation['action'] == 'upsert_occurrence' and isinstance(value, dict):
        try:
            if 'mention_id' not in value:
                value = {**value, 'mention_id': occurrence_id(item['task']['document_id'],
                    item['task']['text_revision'], value['name_span']['start'], value['name_span']['end'])}
            expected_hash = field_fingerprint(item['task'], value, field)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('WORKFLOW_REVIEW_HASH_MISMATCH') from exc
        if expected_hash != stamp['value_hash']:
            raise ValueError('WORKFLOW_REVIEW_HASH_MISMATCH')
    historical = row['result'].get('field_hashes')
    if operation['action'] == 'accept_fields' and not isinstance(historical, dict):
        raise ValueError('WORKFLOW_REVIEW_HASH_MISMATCH')
    if isinstance(historical, dict):
        scoped = historical.get(mid)
        stored = scoped.get(field) if isinstance(scoped, dict) else None
        if stored != stamp['value_hash']:
            raise ValueError('WORKFLOW_REVIEW_HASH_MISMATCH')
    if (stamp['state'] == 'confirmed' and not occurrence['known'][field]
            and stamp['value_hash'] == field_fingerprint(item['task'], occurrence, field)):
        raise ValueError('WORKFLOW_CONFIRMED_UNKNOWN')


def _validate_source_issue(task: dict, issue: dict, records: dict[str, dict]) -> None:
    if (not isinstance(issue, dict)
            or any(not _nonblank(issue.get(key)) for key in ('issue_id', 'message'))
            or issue.get('code') not in ('broken_passage', 'boundary_fragment')
            or issue.get('task_id') != task['task_id']
            or issue.get('text_revision') != task['text_revision']
            or not _span(issue.get('span'), task['context_span']['start'], task['context_span']['end'])):
        raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')
    if 'document_id' in issue and issue['document_id'] != task['document_id']:
        raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')
    if 'source_report_sha256' in issue and (not isinstance(issue['source_report_sha256'], str)
                                           or not HASH.fullmatch(issue['source_report_sha256'])):
        raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')
    if any(not _nonblank(issue.get(key)) for key in
           ('decision_id', 'operation_id', 'reviewer', 'recorded_at_utc')):
        raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')
    row = _matching_row(records, issue['decision_id'], issue['reviewer'], issue['recorded_at_utc'])
    operation = _operation(row, issue['operation_id'])
    value = operation.get('value')
    source_fields = ('issue_id', 'code', 'message', 'span', 'task_id', 'text_revision',
                     'document_id', 'source_report_sha256')
    if operation.get('action') != 'record_source_issue' or not isinstance(value, dict) \
            or any(issue.get(key) != value.get(key) for key in source_fields):
        raise ValueError('WORKFLOW_SOURCE_ISSUE_REFERENCE_MISMATCH')


def validate_workflow(item: dict, history: list[dict]) -> None:
    if not isinstance(item, dict) or not isinstance(item.get('task'), dict) \
            or not isinstance(item.get('annotation'), dict):
        raise ValueError('WORKFLOW_ITEM_INVALID')
    records = _history_index(item, history)
    annotation = item['annotation']
    workflow = annotation.get('review_workflow', ...)
    if workflow is ...:
        return
    if not isinstance(workflow, dict) or workflow.get('schema_version') != WORKFLOW_VERSION:
        raise ValueError('WORKFLOW_SCHEMA_UNSUPPORTED')
    if not isinstance(workflow.get('field_reviews'), dict) \
            or not isinstance(workflow.get('source_issues'), list) \
            or ('approval' not in workflow):
        raise ValueError('WORKFLOW_METADATA_INVALID')
    occurrences = {occurrence['mention_id']: occurrence for occurrence in annotation.get('occurrences', [])}
    for mid, reviews in workflow['field_reviews'].items():
        if not isinstance(reviews, dict):
            raise ValueError('WORKFLOW_FIELD_REVIEW_INVALID')
        for field, stamp in reviews.items():
            _validate_field_stamp(item, records, mid, field, stamp, occurrences)
    issue_ids = set()
    for issue in workflow['source_issues']:
        _validate_source_issue(item['task'], issue, records)
        if issue['issue_id'] in issue_ids:
            raise ValueError('WORKFLOW_SOURCE_ISSUE_DUPLICATE')
        issue_ids.add(issue['issue_id'])
    approval = workflow['approval']
    if approval is not None:
        if (not isinstance(approval, dict)
                or any(not _nonblank(approval.get(key)) for key in
                       ('decision_id', 'reviewer', 'recorded_at_utc'))
                or type(approval.get('annotation_revision')) is not int
                or approval['annotation_revision'] < 1
                or not isinstance(approval.get('content_hash'), str)
                or not HASH.fullmatch(approval['content_hash'])):
            raise ValueError('WORKFLOW_APPROVAL_INVALID')
        row = _matching_row(records, approval['decision_id'], approval['reviewer'], approval['recorded_at_utc'])
        payload = row['payload']
        if (payload.get('actor_kind') != 'human'
                or payload.get('action') != 'apply_review_batch'
                or not isinstance(payload.get('value'), dict)
                or payload['value'].get('completion') != 'approve'
                or row['result'].get('annotation_revision') != approval['annotation_revision']):
            raise ValueError('WORKFLOW_APPROVAL_REFERENCE_MISMATCH')
        historical = row['result'].get('content_hash')
        if historical is not None and historical != approval['content_hash']:
            raise ValueError('WORKFLOW_APPROVAL_HASH_MISMATCH')


def _legacy_accepted(item: dict, records: dict[str, dict], occurrence: dict) -> bool:
    review = occurrence.get('review')
    if not isinstance(review, dict) or review.get('status') != 'human_reviewed':
        return False
    if any(not _nonblank(review.get(key)) for key in ('decision_id', 'reviewer', 'recorded_at_utc')):
        return False
    try:
        row = _matching_row(records, review['decision_id'], review['reviewer'], review['recorded_at_utc'])
    except ValueError:
        return False
    payload = row['payload']
    return (payload.get('action') == 'accept_occurrence'
            and payload.get('target_name_span') == occurrence['name_span'])


def _legacy_name_audit(item: dict, records: dict[str, dict]) -> bool:
    annotation = item['annotation']
    audit = annotation.get('human_passage_review')
    if not isinstance(audit, dict) or audit.get('fields') != ['software']:
        return False
    if any(not _nonblank(audit.get(key)) for key in ('decision_id', 'reviewer', 'recorded_at_utc')):
        return False
    try:
        row = _matching_row(records, audit['decision_id'], audit['reviewer'], audit['recorded_at_utc'])
    except ValueError:
        return False
    if row['payload'].get('action') != 'accept_passage':
        return False
    seen = False
    for later in records.values():
        if later is row:
            seen = True
            continue
        if seen:
            payload = later['payload']
            if payload.get('action') in ('upsert_occurrence', 'remove_occurrence'):
                return False
            if payload.get('action') == 'apply_review_batch' and isinstance(payload.get('value'), dict):
                operations = payload['value'].get('operations')
                if isinstance(operations, list) and any(isinstance(operation, dict)
                        and (operation.get('action') == 'remove_occurrence'
                             or operation.get('action') == 'upsert_occurrence'
                             and (not isinstance(operation.get('fields'), list)
                                  or 'software' in operation['fields']))
                        for operation in operations):
                    return False
    return True


def _field_state(item: dict, occurrence: dict, field: str, records: dict[str, dict]) -> str:
    reviews = (item['annotation'].get('review_workflow') or {}).get('field_reviews', {})
    stamp = reviews.get(occurrence['mention_id'], {}).get(field)
    if stamp is not None and stamp['value_hash'] == field_fingerprint(item['task'], occurrence, field):
        if stamp['state'] == 'unresolved' or occurrence['known'][field]:
            return stamp['state']
    if not occurrence['known'][field]:
        if (field == 'versions' and occurrence['version_status'] == 'ambiguous') \
                or (isinstance(occurrence.get('review'), dict)
                    and occurrence['review'].get('status') == 'unresolved'):
            return 'unresolved'
        return 'missing'
    return 'confirmed' if _legacy_accepted(item, records, occurrence) else 'proposed'


def _relation_confirmed(relation: dict, records: dict[str, dict]) -> bool:
    review = relation.get('review')
    if not isinstance(review, dict) or review.get('status') != 'human_reviewed' \
            or any(not _nonblank(review.get(key)) for key in
                   ('decision_id', 'reviewer', 'recorded_at_utc')):
        return False
    try:
        row = _matching_row(records, review['decision_id'], review['reviewer'], review['recorded_at_utc'])
    except ValueError:
        return False
    payload = row['payload']
    action = payload.get('action')
    if action == 'apply_review_batch':
        if payload.get('actor_kind') != 'human':
            return False
        try:
            operation = _operation(row, review.get('operation_id'))
        except ValueError:
            return False
        action = operation.get('action')
        target = operation.get('target_relation_id')
        if action == 'upsert_alias':
            value = operation.get('value')
            fields = ('member_mention_ids', 'relation_type', 'decision',
                      'preferred_mention_id', 'evidence_spans')
            if not isinstance(value, dict) or any(key not in value for key in fields):
                return False
            members = value['member_mention_ids']
            if not isinstance(members, list) or any(not isinstance(mid, str) for mid in members):
                return False
            expected_id = alias_relation_id(relation['document_id'], relation['text_revision'], members)
            if expected_id != relation['relation_id'] or target not in (None, relation['relation_id']):
                return False
            return relation.get('member_mention_ids') == sorted(members) and all(
                relation.get(key) == value[key] for key in fields[1:])
    else:
        target = payload.get('target_relation_id')
    return action in ('accept_alias', 'reject_alias') and target == relation['relation_id']


def _unit_questions(mid: str, fields: dict[str, dict], requested: set[str], occurrence: dict) -> list[dict]:
    units = [('software', ('software',)), ('versions', ('versions',)),
             ('intents', INTENT_BITS), ('sentiment', ('sentiment',))]
    questions = []
    for unit, members in units:
        relevant = [field for field in members if field in requested]
        if not relevant:
            continue
        states = {fields[field]['state'] for field in relevant}
        if unit == 'intents' and any(field in (occurrence.get('intents') or [])
                                     for field in relevant) and not occurrence['evidence']['intents']:
            states.add('missing')
        if unit == 'sentiment' and occurrence['sentiment'] in ('positive', 'negative', 'mixed') \
                and not occurrence['evidence']['sentiment']:
            states.add('missing')
        if 'unresolved' in states or 'missing' in states:
            code = 'unresolved' if 'unresolved' in states else 'missing'
        elif 'proposed' in states:
            code = 'confirm_proposal'
        else:
            continue
        questions.append({'target_id': mid, 'field': unit, 'code': code,
                          'message': {'unresolved': 'Resolve this field.',
                                      'missing': 'Supply a supported decision.',
                                      'confirm_proposal': 'Confirm the proposed value.'}[code]})
    return questions


def project_review(item: dict, history: list[dict]) -> dict:
    validate_workflow(item, history)
    task, annotation = item['task'], item['annotation']
    records = _history_index(item, history)
    requested = required_fields(task)
    occurrences = annotation.get('occurrences', [])
    layer = annotation.get('alias_annotations') or {'schema_version': '1.0', 'relations': []}
    relations = layer['relations']
    groups = build_alias_groups(occurrences, layer) if relations else []
    fields = {}
    questions = []
    needs_decisions = proposals_to_confirm = 0
    for occurrence in occurrences:
        mid = occurrence['mention_id']
        fields[mid] = {field: {'state': _field_state(item, occurrence, field, records)} for field in FIELDS}
        units = _unit_questions(mid, fields[mid], requested, occurrence)
        questions.extend(units)
        needs_decisions += sum(q['code'] in ('missing', 'unresolved') for q in units)
        proposals_to_confirm += sum(q['code'] == 'confirm_proposal' for q in units)
    relation_states = {}
    for relation in relations:
        rid = relation['relation_id']
        confirmed = _relation_confirmed(relation, records)
        state = ('unresolved' if relation['decision'] == 'unresolved' else
                 'confirmed' if confirmed else 'proposed')
        relation_states[rid] = {'state': state}
        if state == 'unresolved':
            needs_decisions += 1
            questions.append({'target_id': rid, 'field': 'aliases', 'code': 'unresolved',
                              'message': 'Resolve this name relation.'})
        elif state == 'proposed':
            proposals_to_confirm += 1
            questions.append({'target_id': rid, 'field': 'aliases', 'code': 'confirm_proposal',
                              'message': 'Confirm the proposed name relation.'})
    for group in groups:
        if group.get('preferred_name') is None:
            needs_decisions += 1
            questions.append({'target_id': group['local_entity_id'], 'field': 'aliases',
                              'code': 'preference_conflict', 'message': 'Choose one preferred name.'})
    workflow = annotation.get('review_workflow') or {}
    issues = deepcopy(workflow.get('source_issues', []))
    approval = workflow.get('approval')
    approved = (isinstance(approval, dict)
                and approval['content_hash'] == content_fingerprint(task, annotation)
                and approval['annotation_revision'] == item['annotation_revision'])
    name_audit = 'confirmed' if approved or _legacy_name_audit(item, records) else 'pending'
    blockers = [q for q in questions if q['code'] in ('missing', 'unresolved', 'preference_conflict')]
    blockers.extend({'target_id': issue['issue_id'], 'field': 'source', 'code': issue['code'],
                     'message': issue['message']} for issue in issues)
    status = ('source_issue' if issues else 'approved' if approved and not blockers else
              'in_progress' if item.get('status') != 'unreviewed' or name_audit == 'confirmed'
              or any(value['state'] == 'confirmed' for row in fields.values() for value in row.values())
              else 'pending')
    return {'schema_version': WORKFLOW_VERSION,
            'counts': {'mentions': len(occurrences),
                       'version_links': sum(len(occurrence['version_links']) for occurrence in occurrences),
                       'alias_groups': len(groups)},
            'needs_decisions': needs_decisions, 'proposals_to_confirm': proposals_to_confirm,
            'source_issues': issues, 'name_audit': name_audit, 'workflow_status': status,
            'partial_source_coverage': annotation.get('status') != 'complete'
                or bool(annotation.get('unresolved_regions')),
            'can_approve': not blockers, 'approval_blockers': blockers,
            'fields': fields, 'relations': relation_states, 'questions': questions}
