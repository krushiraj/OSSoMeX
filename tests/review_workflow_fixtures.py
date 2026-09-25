from copy import deepcopy

from alias_fixtures import alias_item


def workflow_item():
    result = alias_item()
    result.update(annotation_revision=1, status='unreviewed')
    return result


def batch(item, operations, completion='save', ident='batch-1'):
    task = item['task']
    return {'decision_id': ident, 'task_id': task['task_id'],
            'document_id': task['document_id'], 'text_revision': task['text_revision'],
            'base_annotation_revision': item['annotation_revision'],
            'reviewer': 'Krushi', 'actor_kind': 'human',
            'action': 'apply_review_batch', 'reason': 'Reviewer saved scoped decisions.',
            'value': {'schema_version': '1.0', 'completion': completion,
                      'proposals_revealed': True, 'operations': deepcopy(operations)}}


def field_operation(occurrence, fields, ident='op-1'):
    return {'operation_id': ident, 'action': 'upsert_occurrence',
            'target_name_span': deepcopy(occurrence['name_span']),
            'value': deepcopy(occurrence), 'fields': list(fields),
            'reason_code': 'accept_proposal'}
