from copy import deepcopy
import sqlite3

import pytest

from research.annotations import review_store as store
from research.annotations.review_workflow import project_review
from research.contracts import FIELDS
from review_workflow_fixtures import workflow_item, batch, field_operation
from test_review_store import decision


@pytest.fixture
def saved(tmp_path):
    item = workflow_item()
    c = store.open_store(tmp_path / 'review.sqlite')
    store.import_items(c, [item], 'demo')
    yield c, item
    c.close()


def current(c, item):
    return store.get_item(c, item['task']['task_id'])


def summary(c, item):
    return project_review(current(c, item), store.decision_history(c, item['task']['task_id']))


def test_batch_retry_field_scope_and_restart(saved, tmp_path):
    c, item = saved
    occurrence = item['annotation']['occurrences'][0]
    request = batch(item, [field_operation(occurrence, ['software'])])
    result = store.apply_decision(c, request)
    assert store.apply_decision(c, request) == result
    assert result['annotation_revision'] == 2
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0] == 1
    annotation = current(c, item)['annotation']
    assert set(annotation['review_workflow']['field_reviews'][occurrence['mention_id']]) == {'software'}
    assert 'human_passage_review' not in annotation
    assert annotation['occurrences'][0]['review']['status'] != 'human_reviewed'
    assert result['operation_results'][0]['before'] == result['operation_results'][0]['after']
    with pytest.raises(store.ReviewError, match='DECISION_ID_CONFLICT'):
        store.apply_decision(c, {**request, 'reason': 'different'})
    with pytest.raises(store.ReviewError, match='STALE_REVISION'):
        store.apply_decision(c, {**request, 'decision_id': 'stale'})
    reopened = store.open_store(tmp_path / 'review.sqlite')
    assert store.apply_decision(reopened, request) == result
    assert summary(reopened, item)['fields'][occurrence['mention_id']]['software']['state'] == 'confirmed'
    reopened.close()
    for sql in ('DELETE FROM decisions', "UPDATE decisions SET recorded_at='forged'"):
        with pytest.raises(sqlite3.DatabaseError, match='append-only'):
            c.execute(sql)


def test_failed_second_operation_and_log_insert_commit_nothing(saved):
    c, item = saved
    before = current(c, item)
    operation = field_operation(item['annotation']['occurrences'][0], ['software'])
    bad = {'operation_id': 'op-2', 'action': 'remove_alias',
           'target_relation_id': 'missing', 'reason_code': 'wrong_software_link'}
    with pytest.raises(store.ReviewError, match='UNKNOWN_ALIAS_RELATION'):
        store.apply_decision(c, batch(item, [operation, bad]))
    assert current(c, item) == before
    c.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON decisions BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    with pytest.raises(sqlite3.DatabaseError, match='disk failure'):
        store.apply_decision(c, batch(item, [operation]))
    assert current(c, item) == before
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0] == 0


@pytest.mark.parametrize('change,code', [
    ({'fields': []}, 'FIELDS'), ({'fields': ['software', 'software']}, 'FIELDS'),
    ({'fields': ['intents']}, 'FIELDS'), ({'reviewer': 'forged'}, 'PROVENANCE'),
    ({'base_annotation_revision': 1}, 'PROVENANCE'),
    ({'review': {'status': 'human_reviewed'}}, 'PROVENANCE'),
    ({'action': 'accept_passage'}, 'ACTION'), ({'reason_code': 'link_alias'}, 'REASON'),
    ({'reason_code': 'other'}, 'NOTE'),
])
def test_malformed_operations_are_atomic(saved, change, code):
    c, item = saved
    operation = {**field_operation(item['annotation']['occurrences'][0], ['software']), **change}
    with pytest.raises(store.ReviewError, match=code):
        store.apply_decision(c, batch(item, [operation]))
    assert current(c, item)['annotation_revision'] == 1


def test_duplicate_operations_and_unsupported_envelopes(saved):
    c, item = saved
    op = field_operation(item['annotation']['occurrences'][0], ['software'])
    for operations in ([op, op], [None], {}, [op] * 201):
        with pytest.raises(store.ReviewError):
            store.apply_decision(c, batch(item, operations))
    for patch in ({'schema_version': '2'}, {'completion': 'wrong'}, {'proposals_revealed': 'yes'}):
        request = batch(item, [op]); request['value'].update(patch)
        with pytest.raises(store.ReviewError):
            store.apply_decision(c, request)


def test_scope_and_forged_metadata_rejected_but_review_stamp_discarded(saved):
    c, item = saved
    op = field_operation(item['annotation']['occurrences'][0], ['software'])
    op['value']['sentiment'] = None; op['value']['known']['sentiment'] = False
    with pytest.raises(store.ReviewError, match='OUT_OF_SCOPE'):
        store.apply_decision(c, batch(item, [op]))
    op = field_operation(item['annotation']['occurrences'][0], ['software'])
    op['value']['review_workflow'] = {}
    with pytest.raises(store.ReviewError, match='PROVENANCE'):
        store.apply_decision(c, batch(item, [op]))
    del op['value']['review_workflow']
    op['value']['review'] = {'status': 'human_reviewed', 'reviewer': 'forged'}
    store.apply_decision(c, batch(item, [op]))
    assert current(c, item)['annotation']['occurrences'][0]['review']['reviewer'] == 'Krushi'


def test_shared_evidence_invalidates_other_intents_without_confirming_them(saved):
    c, item = saved
    o = item['annotation']['occurrences'][0]
    store.apply_decision(c, batch(item, [field_operation(o, ['created', 'used', 'shared'])]))
    now = current(c, item)
    op = field_operation(now['annotation']['occurrences'][0], ['used'])
    op['value']['evidence']['intents'] = [deepcopy(o['name_span'])]
    store.apply_decision(c, batch(now, [op], ident='evidence'))
    fields = summary(c, item)['fields'][o['mention_id']]
    assert fields['used']['state'] == 'confirmed'
    assert fields['created']['state'] == fields['shared']['state'] == 'proposed'


def test_note_only_does_not_change_labels_and_system_cannot_review(saved):
    c, item = saved
    op = {'operation_id': 'note', 'action': 'record_note', 'note': 'Check source.', 'reason_code': 'note'}
    request = batch(item, [op]); request['actor_kind'] = 'system'
    store.apply_decision(c, request)
    now = current(c, item)
    assert now['annotation']['occurrences'] == item['annotation']['occurrences']
    assert summary(c, item)['name_audit'] == 'pending'
    request = batch(now, [field_operation(item['annotation']['occurrences'][0], ['software'])], ident='bad')
    request['actor_kind'] = 'system'
    with pytest.raises(store.ReviewError, match='ACTOR'):
        store.apply_decision(c, request)
    request = batch(now, [], completion='approve', ident='bad'); request['actor_kind'] = 'system'
    with pytest.raises(store.ReviewError, match='ACTOR'):
        store.apply_decision(c, request)


def test_source_issue_blocks_legacy_and_batch_approval(saved):
    c, item = saved
    task = item['task']
    op = {'operation_id': 'issue', 'action': 'record_source_issue', 'reason_code': 'broken_passage',
          'value': {'issue_id': 'broken', 'code': 'broken_passage', 'message': 'Missing source text.',
                    'span': task['context_span'], 'task_id': task['task_id'], 'text_revision': task['text_revision']}}
    store.apply_decision(c, batch(item, [op]))
    now = current(c, item)
    assert summary(c, item)['workflow_status'] == 'source_issue'
    for request in (batch(now, [], completion='approve', ident='approve'), decision(task, 'legacy', 2)):
        with pytest.raises(store.ReviewError, match='SOURCE_ISSUE'):
            store.apply_decision(c, request)
    assert current(c, item) == now


def test_approval_readiness_reveal_partial_coverage_and_legacy_invalidation(saved):
    c, item = saved
    request = batch(item, [], completion='approve'); request['value']['proposals_revealed'] = False
    with pytest.raises(store.ReviewError, match='REVEAL'):
        store.apply_decision(c, request)
    request['value']['proposals_revealed'] = True
    store.apply_decision(c, request)
    assert summary(c, item)['workflow_status'] == 'approved'
    assert summary(c, item)['proposals_to_confirm'] == 0
    assert all(field['state'] == 'confirmed' for row in summary(c, item)['fields'].values() for field in row.values())
    assert all(relation['state'] == 'confirmed' for relation in summary(c, item)['relations'].values())
    now = current(c, item)
    store.apply_decision(c, decision(item['task'], 'edit', 2, action='mark_field_unresolved',
        field='sentiment', target_name_span=now['annotation']['occurrences'][0]['name_span']))
    assert summary(c, item)['workflow_status'] != 'approved'
    assert current(c, item)['annotation']['status'] == 'partial'
    with pytest.raises(store.ReviewError, match='NOT_READY'):
        store.apply_decision(c, batch(current(c, item), [], completion='approve', ident='again'))


def test_relation_removal_must_precede_member_edit(saved):
    c, item = saved
    o = item['annotation']['occurrences'][0]
    rid = item['annotation']['alias_annotations']['relations'][0]['relation_id']
    remove = {'operation_id': 'member', 'action': 'remove_occurrence', 'target_name_span': o['name_span'], 'reason_code': 'not_software'}
    relation = {'operation_id': 'relation', 'action': 'remove_alias', 'target_relation_id': rid, 'reason_code': 'wrong_software_link'}
    with pytest.raises(store.ReviewError, match='ALIAS_MEMBER_IN_USE'):
        store.apply_decision(c, batch(item, [remove, relation]))
    store.apply_decision(c, batch(item, [relation, remove]))
    assert len(current(c, item)['annotation']['occurrences']) == 1


def test_accept_fields_uses_actual_legacy_history(saved):
    c, item = saved
    o = item['annotation']['occurrences'][0]
    store.apply_decision(c, decision(item['task'], 'legacy', action='accept_occurrence', target_name_span=o['name_span']))
    assert summary(c, item)['fields'][o['mention_id']]['sentiment']['state'] == 'confirmed'
    op = {'operation_id': 'field', 'action': 'accept_fields', 'fields': ['software'],
          'target_name_span': o['name_span'], 'reason_code': 'accept_proposal'}
    store.apply_decision(c, batch(current(c, item), [op]))
    assert summary(c, item)['fields'][o['mention_id']]['software']['state'] == 'confirmed'
    assert len(store.decision_history(c, item['task']['task_id'])) == 2


def test_empty_approval_logs_generated_confirmations_and_retries_exactly(saved):
    c, item = saved
    request = batch(item, [], completion='approve')
    result = store.apply_decision(c, request)
    assert store.apply_decision(c, request) == result
    history = store.decision_history(c, item['task']['task_id'])
    assert len(history) == 1 and history[0]['payload'] == request
    assert len(result['approval_operations']) == 3
    assert len({o['operation_id'] for o in result['approval_operations']}) == 3
    assert result['annotation_revision'] == 2
    assert summary(c, item)['proposals_to_confirm'] == 0


def test_partial_name_only_approval_does_not_invent_coverage_or_alias_negatives(tmp_path):
    item = workflow_item(); item['task']['requested_fields'] = ['software', 'aliases']
    item['annotation']['alias_annotations']['relations'] = []
    item['annotation']['covered_regions'][0].update(fields=dict.fromkeys(FIELDS, False), provenance={'source': 'agent'})
    for occurrence in item['annotation']['occurrences']:
        occurrence.update(version_status='unannotated', version_links=[], intents=None, sentiment=None,
                          evidence={'intents': [], 'sentiment': []}, known={field: field == 'software' for field in FIELDS})
    c = store.open_store(tmp_path / 'partial.sqlite')
    try:
        store.import_items(c, [item], 'demo')
        store.apply_decision(c, batch(item, [], completion='approve'))
        annotation = current(c, item)['annotation']
        assert annotation['alias_annotations']['relations'] == []
        assert annotation['covered_regions'][0]['provenance'] == {'source': 'agent'}
        assert annotation['covered_regions'][0]['fields'] == {field: field == 'software' for field in FIELDS}
        assert summary(c, item)['partial_source_coverage'] is True
        assert summary(c, item)['workflow_status'] == 'approved'
    finally:
        c.close()


def test_new_name_can_leave_other_fields_unknown_and_unknown_cannot_be_accepted(saved):
    c, item = saved
    o = deepcopy(item['annotation']['occurrences'][0])
    rid = item['annotation']['alias_annotations']['relations'][0]['relation_id']
    ops = [{'operation_id': 'remove-relation', 'action': 'remove_alias', 'target_relation_id': rid,
            'reason_code': 'wrong_software_link'},
           {'operation_id': 'remove', 'action': 'remove_occurrence', 'target_name_span': o['name_span'],
            'reason_code': 'wrong_span'}]
    o.update(version_links=[], version_status='unannotated', intents=None, sentiment=None,
             evidence={'intents': [], 'sentiment': []}, known={f: f == 'software' for f in FIELDS})
    o.pop('mention_id')
    op = field_operation(o, ['software'], ident='new'); op['target_name_span'] = None
    store.apply_decision(c, batch(item, [*ops, op]))
    before = current(c, item)
    accept = {'operation_id': 'accept', 'action': 'accept_fields', 'target_name_span': o['name_span'],
              'fields': ['versions'], 'reason_code': 'accept_proposal'}
    with pytest.raises(store.ReviewError, match='CANNOT_CONFIRM_UNKNOWN'):
        store.apply_decision(c, batch(before, [accept], ident='unknown'))
    assert current(c, item) == before


@pytest.mark.parametrize('patch', [
    {'span': {'start': -1, 'end': 2}}, {'text_revision': 'wrong'}, {'source_report_sha256': 'bad'},
    {'code': 'other'}, {'message': ''},
])
def test_source_issue_validation_is_atomic(saved, patch):
    c, item = saved
    task = item['task']
    issue = {'issue_id': 'broken', 'code': 'boundary_fragment', 'message': 'Cut text.',
             'span': task['context_span'], 'task_id': task['task_id'], 'text_revision': task['text_revision'], **patch}
    op = {'operation_id': 'issue', 'action': 'record_source_issue', 'value': issue, 'reason_code': 'broken_passage'}
    with pytest.raises(store.ReviewError, match='SOURCE_ISSUE'):
        store.apply_decision(c, batch(item, [op]))
    assert current(c, item)['annotation_revision'] == 1


def test_duplicate_issue_and_bad_pending_metadata_never_commit(saved):
    c, item = saved
    task = item['task']
    issue = {'issue_id': 'broken', 'code': 'broken_passage', 'message': 'Cut text.',
             'span': task['context_span'], 'task_id': task['task_id'], 'text_revision': task['text_revision']}
    op = {'operation_id': 'issue', 'action': 'record_source_issue', 'value': issue, 'reason_code': 'broken_passage'}
    with pytest.raises(store.ReviewError, match='SOURCE_ISSUE_DUPLICATE'):
        store.apply_decision(c, batch(item, [op, {**op, 'operation_id': 'second'}]))
    request = batch(item, [], completion='approve'); request['approval_operations'] = []
    with pytest.raises(store.ReviewError, match='PROVENANCE'):
        store.apply_decision(c, request)
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0] == 0


def test_scope_disallows_unknown_field_mutation_and_preserves_extensions(saved):
    c, item = saved
    o = deepcopy(item['annotation']['occurrences'][0])
    o['extension'] = {'source': 'preserve'}
    store.apply_decision(c, decision(item['task'], 'legacy-add', action='upsert_occurrence',
                                   target_name_span=o['name_span'], value=o))
    now = current(c, item)
    replacement = deepcopy(now['annotation']['occurrences'][0]); del replacement['extension']
    store.apply_decision(c, decision(item['task'], 'legacy-edit', 2, action='upsert_occurrence',
                                   target_name_span=o['name_span'], value=replacement))
    now = current(c, item)
    assert now['annotation']['occurrences'][0]['extension'] == {'source': 'preserve'}
    op = field_operation(now['annotation']['occurrences'][0], ['software'])
    op['value']['extension'] = {'source': 'changed'}
    with pytest.raises(store.ReviewError, match='OUT_OF_SCOPE'):
        store.apply_decision(c, batch(now, [op]))


def test_pure_reducer_does_not_change_inputs(saved):
    from research.annotations.review_mutations import reduce_mutation
    _, item = saved
    before = deepcopy(item)
    o = item['annotation']['occurrences'][0]
    provenance = {'status': 'human_reviewed', 'reviewer': 'Krushi', 'decision_id': 'pure', 'recorded_at_utc': 'now', 'reasons': []}
    result = reduce_mutation(item['task'], item['annotation'], {'action': 'accept_occurrence', 'target_name_span': o['name_span']}, provenance)
    result['occurrences'][0]['review']['reasons'].append('changed')
    assert item == before and provenance['reasons'] == []


def test_generated_confirmation_provenance_rejects_tampered_log(saved):
    from research.annotations.review_workflow import validate_workflow
    c, item = saved
    store.apply_decision(c, batch(item, [], completion='approve'))
    saved_item = current(c, item)
    history = store.decision_history(c, item['task']['task_id'])
    for key, value in [('completion', 'save'), ('proposals_revealed', False)]:
        forged = deepcopy(history); forged[0]['payload']['value'][key] = value
        with pytest.raises(ValueError):
            validate_workflow(saved_item, forged)
    forged = deepcopy(history)
    forged[0]['result']['approval_operations'].append(deepcopy(forged[0]['result']['approval_operations'][0]))
    with pytest.raises(ValueError):
        validate_workflow(saved_item, forged)
    forged = deepcopy(history)
    mid = saved_item['annotation']['occurrences'][0]['mention_id']
    forged[0]['result']['field_hashes'][mid]['software'] = '0' * 64
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        validate_workflow(saved_item, forged)


def test_source_issue_rejects_caller_operation_provenance(saved):
    c, item = saved
    task = item['task']
    issue = {'issue_id': 'broken', 'code': 'broken_passage', 'message': 'Cut text.',
             'span': task['context_span'], 'task_id': task['task_id'], 'text_revision': task['text_revision'],
             'operation_id': 'forged'}
    with pytest.raises(store.ReviewError, match='PROVENANCE'):
        store.apply_decision(c, batch(item, [{'operation_id': 'issue', 'action': 'record_source_issue',
                                             'value': issue, 'reason_code': 'broken_passage'}]))
