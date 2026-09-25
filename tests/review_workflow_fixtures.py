from copy import deepcopy

from alias_fixtures import alias_item
from research.annotations.tasks import make_region_task
from research.contracts import FIELDS, validate_occurrence


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


def selection_item(variant=None):
    if variant in ('alias', 'blind', 'negative', 'partial'):
        item = alias_item('design-negative' if variant == 'negative' else 'design-explicit-icekat')
        item['task']['whole_passage_audit'] = variant == 'blind'
        if variant == 'partial':
            occurrence = item['annotation']['occurrences'][0]
            occurrence['known']['shared'] = False
            item['annotation']['status'] = 'partial'
        return item
    text = ('We used excellent scikit-learn v0.17.' if variant in ('complete_proposal', 'complete_proposal_ui') else
            'Alpha (A) also called Beta.' if variant == 'multi' else
            '😀 Cafe\u0301 uses NumPy and NumPy v2.' if variant == 'unicode' else
            'ContextTool. We used scikit-learn v0.17.' if variant == 'context' else
            'We used scikit-learn v0.17.')
    start = 13 if variant == 'context' else 0
    task = make_region_task({'document_id': 'sélection-😀' if variant == 'unicode' else 'selection-demo', 'text': text,
                            'source': 'synthetic-contract-fixture', 'split': 'demo'},
                           {'policy_version': 'scibert-poc-2.1', 'policy_hash': 'a' * 64,
                            'alias_schema_version': '1.0'},
                           {'start': start, 'end': len(text)}, {'start': 0, 'end': len(text)},
                           region_kind='sentence' if variant == 'context' else 'region')
    task['whole_passage_audit'] = False
    item = {'task': task, 'annotation': {'occurrences': [], 'covered_regions': [],
            'unresolved_regions': [task['annotation_region']], 'status': 'partial',
            'annotation_revision': 1, 'review_status': 'synthetic_fixture'}}
    if variant in ('complete_proposal', 'complete_proposal_ui'):
        span = lambda value: {'start': text.index(value), 'end': text.index(value) + len(value)}
        name_span = span('scikit-learn')
        occurrence = {'schema_version': '2.0', 'document_id': task['document_id'],
                      'text_revision': task['text_revision'],
                      'mention_id': f"{task['document_id']}|{task['text_revision']}|{name_span['start']}:{name_span['end']}",
                      'name': 'scikit-learn', 'name_span': name_span,
                      'context_sentence': text, 'context_span': {'start': 0, 'end': len(text)},
                      'context_kind': 'sentence', 'version_status': 'explicit',
                      'version_links': [{'text': 'v0.17', 'span': span('v0.17'), 'status': 'explicit_local'}],
                      'intents': ['used'], 'sentiment': 'positive', 'known': dict.fromkeys(FIELDS, True),
                      'evidence': {'intents': [span(text)], 'sentiment': [span('excellent')]},
                      'review': {'status': 'pending', 'reasons': []}}
        validate_occurrence(occurrence, text)
        item['annotation']['occurrences'] = [occurrence]
    return item
