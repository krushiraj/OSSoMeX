from copy import deepcopy

import pytest

from alias_fixtures import alias_item
from review_workflow_fixtures import batch, field_operation, workflow_item
from research.annotations.review_workflow import (
    content_fingerprint, field_fingerprint, project_review, validate_workflow,
)


def _source_issue(item, code='broken_passage'):
    issue = {'issue_id': 'issue-1', 'code': code, 'message': 'Broken source text',
             'span': deepcopy(item['task']['annotation_region']),
             'task_id': item['task']['task_id'],
             'text_revision': item['task']['text_revision'],
             'decision_id': 'source-1', 'operation_id': 'source-op',
             'reviewer': 'Krushi', 'recorded_at_utc': '2026-09-25T00:00:00+00:00'}
    operation = {'operation_id': 'source-op', 'action': 'record_source_issue',
                 'value': deepcopy(issue), 'reason_code': code}
    payload = batch(item, [operation], ident='source-1')
    payload['actor_kind'] = 'system'
    history = [{'payload': payload,
                'result': {'decision_id': 'source-1', 'recorded_at_utc': issue['recorded_at_utc']},
                'recorded_at': issue['recorded_at_utc']}]
    item['annotation']['review_workflow'] = {'schema_version': '1.0',
                                             'field_reviews': {}, 'source_issues': [issue],
                                             'approval': None}
    return history


def test_partial_intent_is_one_question_not_false_negatives():
    item = alias_item()
    item.update(annotation_revision=1, status='unreviewed')
    occurrence = item['annotation']['occurrences'][0]
    occurrence['known'].update(created=False, used=True, shared=False)
    occurrence['intents'] = ['used']
    before = deepcopy(item)
    result = project_review(item, [])
    questions = [q for q in result['questions']
                 if q['target_id'] == occurrence['mention_id'] and q['field'] == 'intents']
    assert len(questions) == 1
    assert not result['can_approve']
    assert item == before


def test_complete_proposals_are_confirmable_but_not_already_confirmed():
    item = workflow_item()
    result = project_review(item, [])
    mid = item['annotation']['occurrences'][0]['mention_id']
    assert result['fields'][mid]['created']['state'] == 'proposed'
    assert result['proposals_to_confirm'] == 9
    assert result['needs_decisions'] == 0
    assert result['can_approve']
    assert result['counts'] == {'mentions': 2, 'version_links': 0, 'alias_groups': 1}


def test_healthy_empty_legacy_task_can_approve_before_name_audit():
    item = workflow_item()
    item['annotation']['occurrences'] = []
    item['annotation']['alias_annotations']['relations'] = []
    item['annotation']['status'] = 'partial'
    item['annotation']['covered_regions'] = []
    item['annotation']['unresolved_regions'] = [item['task']['annotation_region']]
    result = project_review(item, [])
    assert result['counts'] == {'mentions': 0, 'version_links': 0, 'alias_groups': 0}
    assert result['name_audit'] == 'pending'
    assert result['partial_source_coverage']
    assert result['can_approve']


def test_requested_field_mapping_excludes_unrequested_unknowns():
    item = workflow_item()
    item['task']['requested_fields'] = ['version_links']
    item['annotation']['occurrences'][0]['known']['sentiment'] = False
    item['annotation']['occurrences'][0]['sentiment'] = None
    result = project_review(item, [])
    assert result['can_approve']
    assert all(q['field'] != 'sentiment' for q in result['questions'])


def test_three_name_alias_component_counts_one_group():
    item = workflow_item()
    first, second = item['annotation']['occurrences']
    third = deepcopy(second)
    third['mention_id'] = 'third'
    third['name_span'] = {'start': 74, 'end': 75}
    item['annotation']['occurrences'].append(third)
    relation = deepcopy(item['annotation']['alias_annotations']['relations'][0])
    relation['relation_id'] = 'second-pair'
    relation['member_mention_ids'] = [second['mention_id'], third['mention_id']]
    item['annotation']['alias_annotations']['relations'].append(relation)
    assert project_review(item, [])['counts']['alias_groups'] == 1


@pytest.mark.parametrize('patch', [
    {'schema_version': None}, {'schema_version': '2.0'},
    {'source_issues': None}, {'approval': {'decision_id': 'x'}},
])
def test_invalid_optional_metadata_is_rejected(patch):
    item = workflow_item()
    item['annotation']['review_workflow'] = patch
    with pytest.raises(ValueError):
        validate_workflow(item, [])


def test_field_hash_is_scoped_to_value_and_relevant_evidence():
    item = workflow_item()
    task, occurrence = item['task'], item['annotation']['occurrences'][0]
    baseline = field_fingerprint(task, occurrence, 'created')
    changed = deepcopy(occurrence)
    changed['review'] = {'status': 'human_reviewed'}
    assert field_fingerprint(task, changed, 'created') == baseline
    changed['evidence']['intents'] = []
    assert field_fingerprint(task, changed, 'created') != baseline
    assert field_fingerprint(task, changed, 'sentiment') == field_fingerprint(task, occurrence, 'sentiment')


def test_content_hash_ignores_review_stamps_but_tracks_source_issues():
    item = workflow_item()
    baseline = content_fingerprint(item['task'], item['annotation'])
    changed = deepcopy(item['annotation'])
    changed['annotation_revision'] = 9
    changed['occurrences'][0]['review'] = {'status': 'human_reviewed'}
    changed['covered_regions'][0]['human_reviewed_fields'] = ['software']
    assert content_fingerprint(item['task'], changed) == baseline
    _source_issue(item)
    changed['review_workflow'] = deepcopy(item['annotation']['review_workflow'])
    assert content_fingerprint(item['task'], changed) != baseline
    issue_hash = content_fingerprint(item['task'], changed)
    changed['review_workflow']['source_issues'][0]['recorded_at_utc'] = '2026-09-25T01:00:00+00:00'
    assert content_fingerprint(item['task'], changed) == issue_hash


def test_human_field_confirmation_requires_matching_batch_scope_and_hash():
    item = workflow_item()
    occurrence = item['annotation']['occurrences'][0]
    operation = field_operation(occurrence, ['created'])
    payload = batch(item, [operation])
    stamp = {'state': 'confirmed', 'decision_id': payload['decision_id'],
             'operation_id': operation['operation_id'],
             'reviewer': payload['reviewer'], 'recorded_at_utc': '2026-09-25T00:00:00+00:00',
             'value_hash': field_fingerprint(item['task'], occurrence, 'created')}
    item['annotation']['review_workflow'] = {'schema_version': '1.0',
        'field_reviews': {occurrence['mention_id']: {'created': stamp}},
        'source_issues': [], 'approval': None}
    history = [{'payload': payload, 'result': {'decision_id': payload['decision_id'],
                'annotation_revision': 2, 'recorded_at_utc': stamp['recorded_at_utc']},
                'recorded_at': stamp['recorded_at_utc']}]
    validate_workflow(item, history)
    assert project_review(item, history)['fields'][occurrence['mention_id']]['created']['state'] == 'confirmed'
    occurrence['evidence']['intents'] = []
    assert project_review(item, history)['fields'][occurrence['mention_id']]['created']['state'] == 'proposed'
    item['annotation']['review_workflow']['field_reviews'][occurrence['mention_id']]['used'] = deepcopy(stamp)
    with pytest.raises(ValueError, match='WORKFLOW_REVIEW_SCOPE_MISMATCH'):
        validate_workflow(item, history)


def test_missing_evidence_and_source_issue_block_approval():
    item = workflow_item()
    item['annotation']['occurrences'][0]['evidence']['intents'] = []
    result = project_review(item, [])
    assert not result['can_approve']
    assert any(q['field'] == 'intents' for q in result['questions'])
    item['annotation']['occurrences'][0]['evidence']['intents'] = [item['task']['context_span']]
    history = _source_issue(item, 'boundary_fragment')
    result = project_review(item, history)
    assert result['workflow_status'] == 'source_issue'
    assert not result['can_approve']


def test_source_issue_requires_provenance_and_rejects_bool_offset():
    item = workflow_item()
    history = _source_issue(item)
    issue = item['annotation']['review_workflow']['source_issues'][0]
    del issue['operation_id']
    with pytest.raises(ValueError, match='WORKFLOW_SOURCE_ISSUE_INVALID'):
        validate_workflow(item, history)
    issue['operation_id'] = 'source-op'
    issue['span']['start'] = True
    with pytest.raises(ValueError, match='WORKFLOW_SOURCE_ISSUE_INVALID'):
        validate_workflow(item, history)


def test_current_approval_invalidates_after_content_change():
    item = workflow_item()
    payload = batch(item, [], completion='approve', ident='approval-1')
    recorded = '2026-09-25T00:00:00+00:00'
    history = [{'payload': payload, 'result': {'decision_id': 'approval-1',
                'annotation_revision': 2, 'recorded_at_utc': recorded},
                'recorded_at': recorded}]
    item['annotation_revision'] = 2
    item['annotation']['review_workflow'] = {'schema_version': '1.0',
        'field_reviews': {}, 'source_issues': [],
        'approval': {'decision_id': 'approval-1', 'reviewer': 'Krushi',
                     'recorded_at_utc': recorded, 'annotation_revision': 2,
                     'content_hash': content_fingerprint(item['task'], item['annotation'])}}
    assert project_review(item, history)['workflow_status'] == 'approved'
    item['annotation']['occurrences'][0]['sentiment'] = 'positive'
    assert project_review(item, history)['workflow_status'] != 'approved'


def test_legacy_name_audit_expires_after_structural_batch_edit():
    item = workflow_item()
    task = item['task']
    recorded = '2026-09-25T00:00:00+00:00'
    audit = {'fields': ['software'], 'reviewer': 'Krushi',
             'decision_id': 'passage-1', 'recorded_at_utc': recorded}
    item['annotation']['human_passage_review'] = audit
    passage = batch(item, [], ident='passage-1')
    passage['action'] = 'accept_passage'
    occurrence = item['annotation']['occurrences'][0]
    edit = batch(item, [field_operation(occurrence, ['software'])], ident='edit-1')
    history = [{'payload': passage, 'result': {'decision_id': 'passage-1'}, 'recorded_at': recorded},
               {'payload': edit, 'result': {'decision_id': 'edit-1'},
                'recorded_at': '2026-09-25T00:01:00+00:00'}]
    assert project_review(item, history)['name_audit'] == 'pending'


def test_batch_alias_acceptance_confirms_only_its_relation():
    item = workflow_item()
    relation = item['annotation']['alias_annotations']['relations'][0]
    operation = {'operation_id': 'alias-op', 'action': 'accept_alias',
                 'target_relation_id': relation['relation_id'], 'reason_code': 'accept_proposal'}
    payload = batch(item, [operation], ident='alias-decision')
    recorded = '2026-09-25T00:00:00+00:00'
    relation['review'] = {'status': 'human_reviewed', 'reasons': [],
                          'decision_id': 'alias-decision', 'operation_id': 'alias-op',
                          'reviewer': 'Krushi', 'recorded_at_utc': recorded}
    history = [{'payload': payload, 'result': {'decision_id': 'alias-decision',
                'recorded_at_utc': recorded}, 'recorded_at': recorded}]
    result = project_review(item, history)
    assert result['relations'][relation['relation_id']]['state'] == 'confirmed'
    assert result['name_audit'] == 'pending'


def test_confirmed_stamp_cannot_confirm_an_unknown_mask():
    item = workflow_item()
    occurrence = item['annotation']['occurrences'][0]
    occurrence['known']['sentiment'] = False
    occurrence['sentiment'] = None
    operation = field_operation(occurrence, ['sentiment'])
    payload = batch(item, [operation])
    recorded = '2026-09-25T00:00:00+00:00'
    item['annotation']['review_workflow'] = {'schema_version': '1.0',
        'field_reviews': {occurrence['mention_id']: {'sentiment': {
            'state': 'confirmed', 'value_hash': field_fingerprint(item['task'], occurrence, 'sentiment'),
            'decision_id': 'batch-1', 'operation_id': 'op-1', 'reviewer': 'Krushi',
            'recorded_at_utc': recorded}}}, 'source_issues': [], 'approval': None}
    history = [{'payload': payload, 'result': {'decision_id': 'batch-1'}, 'recorded_at': recorded}]
    result = project_review(item, history)
    assert result['fields'][occurrence['mention_id']]['sentiment']['state'] == 'missing'
    assert not result['can_approve']


def test_accept_fields_stamp_requires_logged_result_hash():
    item = workflow_item()
    occurrence = item['annotation']['occurrences'][0]
    operation = {'operation_id': 'accept-op', 'action': 'accept_fields',
                 'target_name_span': deepcopy(occurrence['name_span']),
                 'fields': ['created'], 'reason_code': 'accept_proposal'}
    payload = batch(item, [operation])
    recorded = '2026-09-25T00:00:00+00:00'
    value_hash = field_fingerprint(item['task'], occurrence, 'created')
    item['annotation']['review_workflow'] = {'schema_version': '1.0',
        'field_reviews': {occurrence['mention_id']: {'created': {
            'state': 'confirmed', 'value_hash': value_hash, 'decision_id': 'batch-1',
            'operation_id': 'accept-op', 'reviewer': 'Krushi', 'recorded_at_utc': recorded}}},
        'source_issues': [], 'approval': None}
    history = [{'payload': payload, 'result': {'decision_id': 'batch-1'}, 'recorded_at': recorded}]
    with pytest.raises(ValueError, match='WORKFLOW_REVIEW_HASH_MISMATCH'):
        validate_workflow(item, history)
    history[0]['result']['field_hashes'] = {occurrence['mention_id']: {'created': value_hash}}
    assert project_review(item, history)['fields'][occurrence['mention_id']]['created']['state'] == 'confirmed'


@pytest.mark.parametrize('changed', ['code', 'span', 'message', 'source_report_sha256'])
def test_source_issue_must_match_logged_content(changed):
    item = workflow_item()
    history = _source_issue(item)
    issue = item['annotation']['review_workflow']['source_issues'][0]
    if changed == 'source_report_sha256':
        issue[changed] = 'a' * 64
    elif changed == 'span':
        issue[changed] = {'start': issue['span']['start'] + 1, 'end': issue['span']['end']}
    else:
        issue[changed] = 'boundary_fragment' if changed == 'code' else 'Different source text'
    with pytest.raises(ValueError, match='WORKFLOW_SOURCE_ISSUE_REFERENCE_MISMATCH'):
        validate_workflow(item, history)


def test_legacy_alias_upsert_stays_proposed():
    item = workflow_item()
    relation = item['annotation']['alias_annotations']['relations'][0]
    recorded = '2026-09-25T00:00:00+00:00'
    payload = batch(item, [], ident='legacy-alias')
    payload.update(action='upsert_alias', target_relation_id=relation['relation_id'],
                   value={key: deepcopy(relation[key]) for key in
                          ('member_mention_ids', 'relation_type', 'decision',
                           'preferred_mention_id', 'evidence_spans')})
    relation['review'] = {'status': 'human_reviewed', 'reasons': [],
                          'decision_id': 'legacy-alias', 'reviewer': 'Krushi',
                          'recorded_at_utc': recorded}
    history = [{'payload': payload, 'result': {'decision_id': 'legacy-alias'}, 'recorded_at': recorded}]
    assert project_review(item, history)['relations'][relation['relation_id']]['state'] == 'proposed'


def test_batch_alias_upsert_confirms_only_matching_relation_content():
    item = workflow_item()
    relation = item['annotation']['alias_annotations']['relations'][0]
    value = {key: deepcopy(relation[key]) for key in
             ('member_mention_ids', 'relation_type', 'decision',
              'preferred_mention_id', 'evidence_spans')}
    value['member_mention_ids'].reverse()
    operation = {'operation_id': 'alias-upsert', 'action': 'upsert_alias',
                 'value': value, 'reason_code': 'link_alias'}
    payload = batch(item, [operation], ident='alias-batch')
    recorded = '2026-09-25T00:00:00+00:00'
    relation['review'] = {'status': 'human_reviewed', 'reasons': [],
                          'decision_id': 'alias-batch', 'operation_id': 'alias-upsert',
                          'reviewer': 'Krushi', 'recorded_at_utc': recorded}
    history = [{'payload': payload, 'result': {'decision_id': 'alias-batch'}, 'recorded_at': recorded}]
    assert project_review(item, history)['relations'][relation['relation_id']]['state'] == 'confirmed'
    relation['evidence_spans'].append(deepcopy(relation['evidence_spans'][0]))
    assert project_review(item, history)['relations'][relation['relation_id']]['state'] == 'proposed'


def test_legacy_upsert_does_not_confirm_every_field_or_name_audit():
    item = workflow_item()
    occurrence = item['annotation']['occurrences'][0]
    occurrence['review'] = {'status': 'human_reviewed', 'reviewer': 'Krushi',
                            'decision_id': 'legacy-upsert',
                            'recorded_at_utc': '2026-09-25T00:00:00+00:00', 'reasons': []}
    payload = batch(item, [], ident='legacy-upsert')
    payload['action'] = 'upsert_occurrence'
    payload['target_name_span'] = deepcopy(occurrence['name_span'])
    history = [{'payload': payload, 'result': {'decision_id': 'legacy-upsert'},
                'recorded_at': occurrence['review']['recorded_at_utc']}]
    result = project_review(item, history)
    assert result['fields'][occurrence['mention_id']]['created']['state'] == 'proposed'
    assert result['name_audit'] == 'pending'
