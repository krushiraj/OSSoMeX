import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.annotations import review_store as store
from research.annotations.snapshots import read_review_snapshot, restore_review_snapshot
from research.annotations.review_workflow import project_review
from research.data.manifest import digest, json_bytes, read_jsonl
from review_workflow_fixtures import workflow_item, batch, field_operation
from test_alias_snapshots import raw_rows
from test_review_store import decision


def workflow_snapshot(tmp_path, completion='save', source_issue=False):
    item = workflow_item()
    c = store.open_store(tmp_path / 'source.sqlite')
    store.import_items(c, [item], 'demo')
    operations = [field_operation(item['annotation']['occurrences'][0], ['software'])]
    if source_issue:
        operations = [{'operation_id': 'issue', 'action': 'record_source_issue',
                       'reason_code': 'broken_passage', 'value': {
                           'issue_id': 'source-1', 'code': 'broken_passage',
                           'message': 'Imported source is truncated.',
                           'task_id': item['task']['task_id'],
                           'text_revision': item['task']['text_revision'],
                           'span': item['task']['annotation_region'],
                           'source_report_sha256': 'a' * 64}}]
    request = batch(item, operations, completion=completion)
    if source_issue:
        request['actor_kind'] = 'system'
    saved = store.apply_decision(c, request)
    before = store.get_item(c, item['task']['task_id'])
    rows = raw_rows(c)
    manifest = store.export_reference(c, tmp_path / 'snapshot')
    c.close()
    return before, request, saved, rows, manifest


@pytest.mark.parametrize('completion,source_issue', [('save', False), ('approve', False), ('save', True)])
def test_scoped_batch_snapshot_roundtrip(tmp_path, completion, source_issue):
    before, request, saved, rows, manifest = workflow_snapshot(tmp_path, completion, source_issue)
    assert manifest['snapshot_schema_version'] == '1.1'
    assert manifest['capabilities'] == {'aliases': True, 'exact_store_restore': True, 'review_workflow': True}
    assert manifest['review_workflow_schema_version'] == '1.0'
    restore_review_snapshot(tmp_path / 'snapshot', tmp_path / 'restored.sqlite')
    c = store.open_store(tmp_path / 'restored.sqlite')
    try:
        assert raw_rows(c) == rows
        current = store.get_item(c, before['task']['task_id'])
        assert current == before
        assert store.apply_decision(c, request) == saved
        projection = project_review(current, store.decision_history(c, before['task']['task_id']))
        assert projection['workflow_status'] == ('source_issue' if source_issue else
                                                'approved' if completion == 'approve' else 'in_progress')
        if source_issue:
            assert projection['can_approve'] is False
            with pytest.raises(ValueError, match='SOURCE_ISSUE_BLOCKS_APPROVAL'):
                store.apply_decision(c, decision(current['task'], 'legacy', 2))
    finally:
        c.close()


def rewrite_sidecar(bundle, name, rows, manifest):
    path = bundle / name
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    next(row for row in manifest['files'] if row['path'] == name)['sha256'] = digest(path.read_bytes())


@pytest.mark.parametrize('corruption', [
    'null', 'version', 'field_reference', 'operation_reference', 'approval_reference',
    'issue_reference', 'batch_version', 'missing_workflow', 'hash', 'capability',
    'workflow_version', 'null_workflow_version', 'missing_workflow_version', 'extra_capability', 'downgrade'])
def test_workflow_snapshot_rejects_corruption_before_creating_destination(tmp_path, corruption):
    workflow_snapshot(tmp_path, completion='approve' if corruption == 'approval_reference' else 'save',
                      source_issue=corruption == 'issue_reference')
    bundle = tmp_path / 'snapshot'
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    if corruption in ('capability', 'extra_capability', 'downgrade', 'workflow_version',
                      'null_workflow_version', 'missing_workflow_version'):
        if corruption == 'capability': manifest['capabilities'].pop('review_workflow', None)
        elif corruption == 'extra_capability': manifest['capabilities']['arbitrary'] = True
        elif corruption == 'downgrade':
            manifest['snapshot_schema_version'] = '1.0'
            manifest['capabilities'].pop('review_workflow', None)
            manifest.pop('review_workflow_schema_version', None)
        elif corruption == 'missing_workflow_version': manifest.pop('review_workflow_schema_version', None)
        else: manifest['review_workflow_schema_version'] = None if corruption == 'null_workflow_version' else '99'
    elif corruption == 'hash':
        (bundle / 'items.jsonl').write_text('{}\n')
    elif corruption == 'batch_version':
        rows = read_jsonl(bundle / 'decision-records.jsonl')
        payload = json.loads(rows[0]['payload']); payload['value']['schema_version'] = '99'
        rows[0]['payload'] = json.dumps(payload)
        rewrite_sidecar(bundle, 'decision-records.jsonl', rows, manifest)
        exported = read_jsonl(bundle / 'decisions.jsonl'); exported[0]['value']['schema_version'] = '99'
        rewrite_sidecar(bundle, 'decisions.jsonl', exported, manifest)
    else:
        rows = read_jsonl(bundle / 'store-items.jsonl')
        annotation = json.loads(rows[0]['annotation']); workflow = annotation['review_workflow']
        if corruption == 'null': annotation['review_workflow'] = None
        elif corruption == 'version': workflow['schema_version'] = '99'
        elif corruption == 'missing_workflow': annotation.pop('review_workflow')
        elif corruption == 'approval_reference': workflow['approval']['decision_id'] = 'missing'
        elif corruption == 'issue_reference': workflow['source_issues'][0]['message'] = 'Forged description'
        else:
            stamp = next(iter(workflow['field_reviews'].values()))['software']
            stamp['decision_id' if corruption == 'field_reference' else 'operation_id'] = 'missing'
        rows[0]['annotation'] = json.dumps(annotation)
        rewrite_sidecar(bundle, 'store-items.jsonl', rows, manifest)
        items = read_jsonl(bundle / 'items.jsonl'); items[0]['annotation'] = annotation
        rewrite_sidecar(bundle, 'items.jsonl', items, manifest)
    (bundle / 'manifest.json').write_bytes(json_bytes(manifest))
    destination = tmp_path / 'bad.sqlite'
    with pytest.raises(ValueError): restore_review_snapshot(bundle, destination)
    assert not destination.exists()


@pytest.mark.parametrize('corruption', ['null', 'version', 'field_reference', 'batch_version', 'result_revision'])
def test_export_validates_workflow_before_writing_any_files(tmp_path, corruption):
    before, request, saved, rows, manifest = workflow_snapshot(tmp_path)
    c = store.open_store(tmp_path / 'source.sqlite')
    try:
        annotation = before['annotation']
        if corruption == 'null': annotation['review_workflow'] = None
        elif corruption == 'version': annotation['review_workflow']['schema_version'] = '99'
        elif corruption == 'field_reference':
            next(iter(annotation['review_workflow']['field_reviews'].values()))['software']['decision_id'] = 'missing'
        elif corruption == 'batch_version':
            c.execute('DROP TRIGGER decisions_no_update')
            request['value']['schema_version'] = '99'
            c.execute('UPDATE decisions SET payload=?', (json.dumps(request),))
        else:
            c.execute('DROP TRIGGER decisions_no_update')
            saved['annotation_revision'] = 400
            c.execute('UPDATE decisions SET result=?', (json.dumps(saved),))
        c.execute('UPDATE items SET annotation=?', (json.dumps(annotation),))
        with pytest.raises(ValueError): store.export_reference(c, tmp_path / 'bad-export')
        assert not (tmp_path / 'bad-export').exists()
    finally:
        c.close()


@pytest.mark.parametrize('relative,tasks,decisions', [
    ('ecosystems-pilot-002/exports/krushi-reviewed-001', 10, 22),
    ('icekat-alias-001/exports/krushi-reviewed-001', 1, 4),
    ('icekat-alias-001/exports/provisional-001', 1, 0),
    ('ecosystems-expansion-001/exports/provisional-001', 24, 0),
])
def test_retained_legacy_snapshots_restore_and_reexport_losslessly(tmp_path, relative, tasks, decisions):
    source = Path(__file__).resolve().parents[1] / 'annotations/scibert-v2' / relative
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
    manifest, data = read_review_snapshot(source)
    assert manifest['snapshot_schema_version'] == '1.0'
    assert restore_review_snapshot(source, tmp_path / 'restored.sqlite') == {
        'role': 'train', 'task_count': tasks, 'decision_count': decisions}
    c = store.open_store(tmp_path / 'restored.sqlite')
    try:
        assert [dict(r) for r in c.execute('SELECT * FROM items ORDER BY task_id')] == data['store-items.jsonl']
        assert raw_rows(c)['decisions'] == data['decision-records.jsonl']
        exported = store.export_reference(c, tmp_path / 'export')
        assert exported['snapshot_schema_version'] == '1.0'
        assert exported['capabilities'] == {'aliases': True, 'exact_store_restore': True}
        assert 'review_workflow_schema_version' not in exported
        for name in data:
            assert read_jsonl(tmp_path / 'export' / name) == data[name]
    finally:
        c.close()
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()} == before


def test_legacy_edit_invalidates_workflow_approval_and_survives_restore(tmp_path):
    before, request, saved, rows, manifest = workflow_snapshot(tmp_path, completion='approve')
    c = store.open_store(tmp_path / 'source.sqlite')
    try:
        occurrence = before['annotation']['occurrences'][0]
        store.apply_decision(c, decision(before['task'], 'legacy-edit', 2,
                            action='mark_field_unresolved', target_name_span=occurrence['name_span'],
                            field='sentiment'))
        current = store.get_item(c, before['task']['task_id'])
        assert current['annotation']['review_workflow']['approval'] is None
        projection = project_review(current, store.decision_history(c, before['task']['task_id']))
        assert projection['fields'][occurrence['mention_id']]['sentiment']['state'] == 'unresolved'
        assert projection['workflow_status'] == 'in_progress'
        store.export_reference(c, tmp_path / 'legacy-edited')
    finally:
        c.close()
    restore_review_snapshot(tmp_path / 'legacy-edited', tmp_path / 'restored.sqlite')
    c = store.open_store(tmp_path / 'restored.sqlite')
    try: assert store.get_item(c, before['task']['task_id']) == current
    finally: c.close()


@pytest.mark.parametrize('workflow', [None, {'schema_version': '99'}])
def test_import_rejects_explicit_invalid_workflow_atomically(tmp_path, workflow):
    item = workflow_item(); item['annotation']['review_workflow'] = workflow
    c = store.open_store(tmp_path / 'import.sqlite')
    try:
        with pytest.raises(ValueError, match='WORKFLOW_SCHEMA_UNSUPPORTED'):
            store.import_items(c, [item], 'demo')
        assert raw_rows(c) == {'items': [], 'decisions': [], 'metadata': []}
    finally:
        c.close()


MALFORMED_ACTIONS = [
    ('upsert_occurrence', 'value', 'missing'),
    ('upsert_occurrence', 'value', None),
    ('upsert_occurrence', 'value', []),
    ('upsert_occurrence', 'value', {}),
    ('upsert_occurrence', 'target_name_span', []),
    ('upsert_occurrence', 'target_name_span', {'start': True, 'end': 2}),
    ('accept_fields', 'target_name_span', 'missing'),
    ('accept_fields', 'target_name_span', None),
    ('accept_fields', 'target_name_span', []),
    ('remove_occurrence', 'target_name_span', 'missing'),
    ('remove_occurrence', 'target_name_span', {'start': 4, 'end': 4}),
    ('upsert_alias', 'value', 'missing'),
    ('upsert_alias', 'value', None),
    ('upsert_alias', 'value', []),
    ('upsert_alias', 'value', {}),
    ('upsert_alias', 'target_relation_id', []),
    ('accept_alias', 'target_relation_id', 'missing'),
    ('accept_alias', 'target_relation_id', None),
    ('reject_alias', 'target_relation_id', 1),
    ('unresolve_alias', 'target_relation_id', []),
    ('remove_alias', 'target_relation_id', ''),
    ('record_source_issue', 'value', 'missing'),
    ('record_source_issue', 'value', None),
    ('record_source_issue', 'value', []),
    ('record_source_issue', 'value', {}),
    ('record_note', 'note', 'missing'),
    ('record_note', 'note', []),
]


def malformed_operation(item, action, key, bad):
    operation = {'operation_id': 'op', 'action': action, 'reason_code': 'other', 'note': 'Review.'}
    if action in ('upsert_occurrence', 'accept_fields', 'remove_occurrence'):
        operation['target_name_span'] = item['annotation']['occurrences'][0]['name_span']
        operation['fields'] = ['software']
    if action == 'upsert_occurrence': operation['value'] = item['annotation']['occurrences'][0]
    if action.endswith('_alias'):
        relation = item['annotation']['alias_annotations']['relations'][0]
        operation['target_relation_id'] = relation['relation_id']
        if action == 'upsert_alias': operation['value'] = relation
    if bad == 'missing': operation.pop(key, None)
    else: operation[key] = bad
    return deepcopy(operation)


@pytest.mark.parametrize('action,key,bad', MALFORMED_ACTIONS)
@pytest.mark.parametrize('boundary', ['live', 'export', 'restore'])
def test_unreferenced_batch_operation_shapes_rejected_before_writes(tmp_path, action, key, bad, boundary):
    item = workflow_item(); c = store.open_store(tmp_path / 'source.sqlite')
    try:
        store.import_items(c, [item], 'demo')
        operation = malformed_operation(item, action, key, bad)
        if boundary == 'live':
            before = raw_rows(c)
            with pytest.raises(ValueError): store.apply_decision(c, batch(item, [operation]))
            assert raw_rows(c) == before
            return
        request = batch(item, [{'operation_id': 'op', 'action': 'record_note', 'reason_code': 'note', 'note': 'Review.'}])
        result = store.apply_decision(c, request)
        bundle = tmp_path / 'snapshot'; manifest = store.export_reference(c, bundle)
        request['value']['operations'] = [operation]
        result['operation_results'][0].update(operation=operation, action=action)
        if boundary == 'export':
            c.execute('DROP TRIGGER decisions_no_update')
            c.execute('UPDATE decisions SET payload=?,result=?', (json.dumps(request), json.dumps(result)))
            with pytest.raises(ValueError): store.export_reference(c, tmp_path / 'bad-export')
            assert not (tmp_path / 'bad-export').exists()
        else:
            records = read_jsonl(bundle / 'decision-records.jsonl')
            records[0].update(payload=json.dumps(request), result=json.dumps(result))
            rewrite_sidecar(bundle, 'decision-records.jsonl', records, manifest)
            rewrite_sidecar(bundle, 'decisions.jsonl', [{**request, 'recorded_at_utc': result['recorded_at_utc']}], manifest)
            (bundle / 'manifest.json').write_bytes(json_bytes(manifest))
            with pytest.raises(ValueError): restore_review_snapshot(bundle, tmp_path / 'bad.sqlite')
            assert not (tmp_path / 'bad.sqlite').exists()
    finally:
        c.close()


@pytest.mark.parametrize('action,key,bad', [
    ('accept_fields', 'target_name_span', None),
    ('accept_alias', 'target_relation_id', None),
    ('reject_alias', 'target_relation_id', []),
])
def test_generated_acceptance_requires_targets_even_without_current_stamps(tmp_path, action, key, bad):
    item = workflow_item(); c = store.open_store(tmp_path / 'source.sqlite')
    try:
        store.import_items(c, [item], 'demo')
        request = batch(item, [{'operation_id': 'op', 'action': 'record_note', 'reason_code': 'note', 'note': 'Review.'}])
        result = store.apply_decision(c, request)
        operation = malformed_operation(item, action, key, bad)
        request['value'].update(completion='approve', operations=[])
        result['approval_operations'] = [operation]
        result['operation_results'][0].update(operation=operation, action=action, generated=True)
        c.execute('DROP TRIGGER decisions_no_update')
        c.execute('UPDATE decisions SET payload=?,result=?', (json.dumps(request), json.dumps(result)))
        with pytest.raises(ValueError): store.export_reference(c, tmp_path / 'bad-export')
        assert not (tmp_path / 'bad-export').exists()
    finally:
        c.close()
