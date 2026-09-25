"""Ordered, field-scoped review batches without database side effects."""

from copy import deepcopy

from ..contracts import FIELDS, INTENT_BITS
from .alias_decisions import ALIAS_ACTIONS
from .review_mutations import reduce_mutation
from .validation import validate_task_occurrence
from .review_workflow import content_fingerprint, field_fingerprint, project_review, required_fields, validate_workflow, value_hash


REASONS = {
    'upsert_occurrence': {'identify_name', 'link_version', 'set_intent', 'set_sentiment', 'attach_evidence', 'accept_proposal', 'wrong_span', 'insufficient_evidence', 'ambiguous_referent'},
    'accept_fields': {'accept_proposal'},
    'remove_occurrence': {'wrong_span', 'not_software'},
    'upsert_alias': {'link_alias', 'accept_proposal', 'wrong_software_link', 'insufficient_evidence', 'ambiguous_referent', 'attach_evidence'},
    'accept_alias': {'accept_proposal', 'link_alias'},
    'reject_alias': {'wrong_software_link', 'not_software', 'accept_proposal'},
    'unresolve_alias': {'insufficient_evidence', 'ambiguous_referent'},
    'remove_alias': {'wrong_software_link', 'not_software'},
    'record_source_issue': {'broken_passage', 'wrong_span'},
    'record_note': {'note'},
}
PROVENANCE_KEYS = {'review_workflow', 'field_reviews', 'approval', 'approval_operations',
                   'decision_id', 'reviewer', 'recorded_at_utc', 'actor_kind', 'base_annotation_revision', 'annotation_revision'}


def _reject_provenance(value, operation=False):
    if isinstance(value, dict):
        forbidden = PROVENANCE_KEYS | ({'review'} if operation else {'operation_id'})
        if forbidden & value.keys():
            raise ValueError('BATCH_PROVENANCE_FORBIDDEN')
        for key, child in value.items():
            if key != 'review':
                _reject_provenance(child)
    elif isinstance(value, list):
        for child in value:
            _reject_provenance(child)


def _valid_span(value):
    return (isinstance(value, dict) and type(value.get('start')) is int
            and type(value.get('end')) is int and 0 <= value['start'] < value['end'])


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def validate_operation(operation, task):
    if not isinstance(operation, dict) or not isinstance(operation.get('operation_id'), str) or not operation['operation_id'].strip():
        raise ValueError('BATCH_OPERATION_INVALID')
    _reject_provenance(operation, operation=True)
    action = operation.get('action')
    if not isinstance(action, str) or action not in REASONS:
        raise ValueError('BATCH_ACTION_UNSUPPORTED')
    reason = operation.get('reason_code')
    if not isinstance(reason, str) or reason not in REASONS[action] | {'other'}:
        raise ValueError('BATCH_REASON_INVALID')
    note = operation.get('note')
    if reason == 'other' or action == 'record_note':
        if not isinstance(note, str) or not note.strip():
            raise ValueError('BATCH_NOTE_REQUIRED')
    elif note is not None and not isinstance(note, str):
        raise ValueError('BATCH_NOTE_INVALID')
    if action in ('upsert_occurrence', 'accept_fields'):
        fields = operation.get('fields')
        if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or f not in FIELDS for f in fields) or len(set(fields)) != len(fields):
            raise ValueError('BATCH_FIELDS_INVALID')
    if action in ('upsert_occurrence', 'accept_fields', 'remove_occurrence'):
        target = operation.get('target_name_span')
        if not (action == 'upsert_occurrence' and target is None) and not _valid_span(target):
            raise ValueError('BATCH_TARGET_INVALID')
    if action in ALIAS_ACTIONS:
        target = operation.get('target_relation_id')
        if not (action == 'upsert_alias' and target is None) and not _nonblank(target):
            raise ValueError('BATCH_TARGET_INVALID')
    value = operation.get('value')
    if action == 'upsert_occurrence':
        try:
            validate_task_occurrence(task, value)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('INVALID_OCCURRENCE:' + str(exc)) from exc
    elif action == 'upsert_alias':
        if not isinstance(value, dict):
            raise ValueError('INVALID_ALIAS_RECORD')
        members = value.get('member_mention_ids')
        if (not isinstance(members, list) or len(members) != 2 or any(not _nonblank(mid) for mid in members)
                or members[0] == members[1] or value.get('relation_type') not in ('abbreviation', 'explicit_alternative_name')
                or value.get('decision') not in ('alias', 'not_alias', 'unresolved')
                or 'preferred_mention_id' not in value
                or value['preferred_mention_id'] is not None and not _nonblank(value['preferred_mention_id'])
                or not isinstance(value.get('evidence_spans'), list) or not value['evidence_spans']
                or any(not _valid_span(span) for span in value['evidence_spans'])):
            raise ValueError('INVALID_ALIAS_RECORD')
    elif action == 'record_source_issue':
        if (not isinstance(value, dict) or any(not _nonblank(value.get(key)) for key in
                ('issue_id', 'message', 'task_id', 'text_revision'))
                or value.get('code') not in ('broken_passage', 'boundary_fragment')
                or not _valid_span(value.get('span'))):
            raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')


def _scope_values(occurrence, fields):
    if occurrence is None:
        return None
    values = {}
    for field in fields:
        if field == 'software':
            value = {key: deepcopy(occurrence[key]) for key in ('name', 'name_span')}
        elif field == 'versions':
            value = {key: deepcopy(occurrence[key]) for key in ('version_status', 'version_links')}
        elif field in INTENT_BITS:
            value = {'positive': field in (occurrence['intents'] or []), 'evidence': deepcopy(occurrence['evidence']['intents'])}
        else:
            value = {'sentiment': occurrence['sentiment'], 'evidence': deepcopy(occurrence['evidence']['sentiment'])}
        values[field] = {'known': occurrence['known'][field], 'value': value}
    return values


def _assert_scope(before, after, fields):
    if before is None:
        if 'software' not in fields or any(after['known'][f] for f in FIELDS if f not in fields):
            raise ValueError('BATCH_OUT_OF_SCOPE')
        return
    for field in FIELDS:
        if field in fields:
            continue
        old, new = _scope_values(before, [field]), _scope_values(after, [field])
        if field in INTENT_BITS and set(fields) & set(INTENT_BITS):
            old[field]['value'].pop('evidence')
            new[field]['value'].pop('evidence')
        if old != new:
            raise ValueError('BATCH_OUT_OF_SCOPE')
    owned = {'review', 'name', 'name_span', 'mention_id', 'known', 'version_status', 'version_links', 'intents', 'sentiment', 'evidence'}
    if any(before.get(k) != after.get(k) for k in (before.keys() | after.keys()) - owned):
        raise ValueError('BATCH_OUT_OF_SCOPE')


def result_metadata(annotation, operation_results):
    workflow = annotation['review_workflow']
    return {'operation_results': operation_results,
            'field_hashes': {mid: {field: stamp['value_hash'] for field, stamp in reviews.items()}
                             for mid, reviews in workflow['field_reviews'].items()},
            'approval_operations': [row['operation'] for row in operation_results if row.get('generated')],
            'content_hash': workflow['approval']['content_hash'] if workflow['approval'] else None}


def apply_batch(item: dict, decision: dict, history: list[dict], recorded_at: str) -> tuple[dict, str, list[dict]]:
    if {'approval_operations', 'field_hashes', 'review_workflow', 'field_reviews', 'approval',
        'operation_results', 'recorded_at_utc', 'annotation_revision'} & decision.keys():
        raise ValueError('BATCH_PROVENANCE_FORBIDDEN')
    value = decision.get('value')
    if not isinstance(value, dict) or value.get('schema_version') != '1.0' or value.get('completion') not in ('save', 'approve') or type(value.get('proposals_revealed')) is not bool:
        raise ValueError('BATCH_ENVELOPE_INVALID')
    if set(value) - {'schema_version', 'completion', 'proposals_revealed', 'operations'}:
        raise ValueError('BATCH_PROVENANCE_FORBIDDEN')
    operations = value.get('operations')
    if not isinstance(operations, list) or len(operations) > 200:
        raise ValueError('BATCH_OPERATIONS_INVALID')
    for operation in operations:
        validate_operation(operation, item['task'])
    ids = [op['operation_id'] for op in operations]
    if len(ids) != len(set(ids)):
        raise ValueError('BATCH_DUPLICATE_OPERATION_ID')
    actor = decision.get('actor_kind')
    if actor not in ('human', 'system') or actor == 'system' and (
            value['completion'] != 'save' or any(op['action'] not in ('record_note', 'record_source_issue') for op in operations)):
        raise ValueError('BATCH_ACTOR_FORBIDDEN')
    if value['completion'] == 'approve' and not value['proposals_revealed']:
        raise ValueError('BATCH_REVEAL_REQUIRED')
    validate_workflow(item, history)
    task = item['task']
    annotation = deepcopy(item['annotation'])
    annotation.setdefault('review_workflow', {'schema_version': '1.0', 'field_reviews': {}, 'source_issues': [], 'approval': None})
    annotation['review_workflow']['approval'] = None
    provenance = {'status': 'pending', 'reviewer': decision['reviewer'], 'decision_id': decision['decision_id'],
                  'recorded_at_utc': recorded_at, 'reasons': []}
    results = []

    def apply(operation, generated=False):
        nonlocal annotation
        action = operation['action']
        stamp = {**provenance, 'operation_id': operation['operation_id']}
        existing = next((o for o in annotation['occurrences'] if o['name_span'] == operation.get('target_name_span')), None)
        before = after = None
        fields = operation.get('fields', [])
        if action in ('upsert_occurrence', 'accept_fields'):
            before = _scope_values(existing, fields)
            if action == 'upsert_occurrence':
                annotation = reduce_mutation(task, annotation, operation, stamp)
                target = operation['value']['name_span']
                updated = next(o for o in annotation['occurrences'] if o['name_span'] == target)
                _assert_scope(existing, updated, fields)
            else:
                if existing is None:
                    raise ValueError('UNKNOWN_OCCURRENCE')
                updated = existing
                if any(not updated['known'][field] for field in fields):
                    raise ValueError('BATCH_CANNOT_CONFIRM_UNKNOWN')
            reviews = annotation['review_workflow']['field_reviews'].setdefault(updated['mention_id'], {})
            for field in fields:
                reviews[field] = {key: stamp[key] for key in ('decision_id', 'operation_id', 'reviewer', 'recorded_at_utc')}
                reviews[field].update(state='confirmed' if updated['known'][field] else 'unresolved',
                                      value_hash=field_fingerprint(task, updated, field))
            confirmed = all(updated['known'][f] and f in reviews and reviews[f]['state'] == 'confirmed'
                            and reviews[f]['value_hash'] == field_fingerprint(task, updated, f) for f in required_fields(task))
            updated['review'] = {**stamp, 'status': 'human_reviewed' if confirmed else
                                'unresolved' if any(not updated['known'][f] for f in fields) else 'pending'}
            after = _scope_values(updated, fields)
        elif action in ALIAS_ACTIONS or action == 'remove_occurrence':
            if action in ALIAS_ACTIONS:
                before = deepcopy(next((r for r in annotation.get('alias_annotations', {}).get('relations', [])
                                        if r['relation_id'] == operation.get('target_relation_id')), None))
                stamp['status'] = 'human_reviewed'
            else:
                before = deepcopy(existing)
            annotation = reduce_mutation(task, annotation, operation, stamp)
            if action in ALIAS_ACTIONS:
                from .aliases import alias_relation_id
                target = (alias_relation_id(task['document_id'], task['text_revision'], operation['value']['member_mention_ids'])
                          if action == 'upsert_alias' else operation.get('target_relation_id'))
                after = deepcopy(next((r for r in annotation.get('alias_annotations', {}).get('relations', []) if r['relation_id'] == target), None))
        elif action == 'record_source_issue':
            issue = operation.get('value')
            if not isinstance(issue, dict):
                raise ValueError('WORKFLOW_SOURCE_ISSUE_INVALID')
            after = {**deepcopy(issue), **{key: stamp[key] for key in ('decision_id', 'operation_id', 'reviewer', 'recorded_at_utc')}}
            annotation['review_workflow']['source_issues'].append(after)
        results.append({'operation_id': operation['operation_id'], 'action': action,
                        'action_text': action.replace('_', ' ').capitalize(), 'before': before, 'after': after,
                        'operation': deepcopy(operation), 'generated': generated})

    def pending_history():
        result = {'decision_id': decision['decision_id'], 'task_id': task['task_id'],
                  'recorded_at_utc': recorded_at, 'annotation_revision': item['annotation_revision'] + 1,
                  **result_metadata(annotation, results)}
        return history + [{'payload': decision, 'result': result, 'recorded_at': recorded_at}]

    for operation in operations:
        apply(operation)
    preview = {**item, 'annotation': annotation}
    projection = project_review(preview, pending_history())
    if value['completion'] == 'approve':
        if projection['source_issues']:
            raise ValueError('SOURCE_ISSUE_BLOCKS_APPROVAL')
        if not projection['can_approve']:
            raise ValueError('BATCH_APPROVAL_NOT_READY')
        generated = []
        for occurrence in annotation['occurrences']:
            fields = sorted(required_fields(task))
            if fields:
                generated.append({'operation_id': 'approval:' + value_hash([decision['decision_id'], occurrence['mention_id']]),
                    'action': 'accept_fields', 'target_name_span': deepcopy(occurrence['name_span']), 'fields': fields,
                    'reason_code': 'accept_proposal'})
        for relation in annotation.get('alias_annotations', {}).get('relations', []):
            generated.append({'operation_id': 'approval:' + value_hash([decision['decision_id'], relation['relation_id']]),
                'action': 'accept_alias' if relation['decision'] == 'alias' else 'reject_alias',
                'target_relation_id': relation['relation_id'], 'reason_code': 'accept_proposal'})
        for operation in generated:
            validate_operation(operation, task)
            if operation['operation_id'] in ids:
                raise ValueError('BATCH_DUPLICATE_OPERATION_ID')
            apply(operation, generated=True)
        annotation = reduce_mutation(task, annotation, {'action': 'accept_passage'}, provenance)
        annotation['review_workflow']['approval'] = {key: provenance[key] for key in ('decision_id', 'reviewer', 'recorded_at_utc')}
        annotation['review_workflow']['approval'].update(annotation_revision=item['annotation_revision'] + 1,
                                                       content_hash=content_fingerprint(task, annotation))
    annotation['annotation_revision'] = item['annotation_revision'] + 1
    validate_workflow({**item, 'annotation': annotation, 'annotation_revision': annotation['annotation_revision']}, pending_history())
    return annotation, 'reviewed' if value['completion'] == 'approve' else 'in_progress', results
