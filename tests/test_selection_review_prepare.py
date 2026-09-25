import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from research.annotations import review_store as store
from research.annotations.snapshots import read_review_snapshot, restore_review_snapshot
from research.data.manifest import digest, write_jsonl
from review_workflow_fixtures import batch, workflow_item
from test_review_store import decision


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_selection_review.py'


@pytest.fixture
def prepare_module():
    spec = importlib.util.spec_from_file_location('prepare_selection_review', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    c = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    try:
        c.execute('BEGIN')
        return {table: [dict(row) for row in c.execute(f'SELECT * FROM {table} ORDER BY {key}')]
                for table, key in [('items', 'task_id'), ('decisions', 'rowid'), ('metadata', 'rowid')]}
    finally:
        c.close()


@pytest.fixture
def inputs(tmp_path):
    fixture = workflow_item()
    source = tmp_path / 'source' / 'review.sqlite'
    c = store.open_store(source)
    store.import_items(c, [fixture], 'demo')
    store.apply_decision(c, decision(fixture['task'], action='accept_alias',
                        target_relation_id=fixture['annotation']['alias_annotations']['relations'][0]['relation_id']))
    c.close()
    report = tmp_path / 'source-report.md'
    report.write_text('Synthetic fixture with a known boundary issue.\n')
    task = fixture['task']
    issue = {'issue_id': 'fixture-boundary-1', 'code': 'boundary_fragment',
             'message': 'Synthetic boundary issue', 'span': task['annotation_region'],
             **{key: task[key] for key in ('task_id', 'document_id', 'text_revision')},
             'source_report_sha256': digest(report.read_bytes())}
    issues = tmp_path / 'issues.jsonl'
    write_jsonl(issues, [issue])
    return source, tmp_path / 'preview', issues, report


def test_preview_preserves_history_labels_and_blocks_approval(inputs, prepare_module, tmp_path):
    source, destination, issues, report = inputs
    before = read_rows(source)
    result = prepare_module.prepare_preview(*inputs)
    assert read_rows(source) == before
    restored = read_rows(destination / 'review.sqlite')
    assert restored['decisions'][:-1] == before['decisions']
    assert restored['metadata'] == before['metadata']
    for old, new in zip(before['items'], restored['items']):
        assert all(new[key] == old[key] for key in ('task_id', 'original_hash', 'task'))
        old_annotation, new_annotation = json.loads(old['annotation']), json.loads(new['annotation'])
        assert {key: value for key, value in new_annotation.items()
                if key not in ('review_workflow', 'annotation_revision')} == {
                    key: value for key, value in old_annotation.items() if key != 'annotation_revision'}
        assert new_annotation['review_workflow']['field_reviews'] == {}
        assert new_annotation['review_workflow']['approval'] is None
        issue = new_annotation['review_workflow']['source_issues'][0]
        assert all(issue[key] == value for key, value in json.loads(issues.read_text()).items())
    payload = json.loads(restored['decisions'][-1]['payload'])
    assert payload['actor_kind'] == 'system'
    assert payload['value']['completion'] == 'save'
    assert payload['value']['proposals_revealed'] is False
    assert [op['action'] for op in payload['value']['operations']] == ['record_source_issue']
    assert result == json.loads((destination / 'preservation.json').read_bytes())
    assert result['source_unchanged'] is True
    assert result['baseline_table_sha256'] == result['source_after_table_sha256']
    assert result['decision_prefix_preserved'] is True
    assert result['source_report_sha256'] == digest(report.read_bytes())
    assert result['issues_sha256'] == digest(issues.read_bytes())
    assert (destination / 'source-issues.jsonl').read_bytes() == issues.read_bytes()
    assert (destination / 'source-report.md').read_bytes() == report.read_bytes()
    _, baseline = read_review_snapshot(destination / 'baseline')
    assert baseline['store-items.jsonl'] == before['items']
    assert baseline['decision-records.jsonl'] == before['decisions']
    restore_review_snapshot(destination / 'prepared', tmp_path / 'restored.sqlite')
    assert read_rows(tmp_path / 'restored.sqlite') == restored
    c = store.open_store(destination / 'review.sqlite')
    current = store.get_item(c, before['items'][0]['task_id'])
    try:
        for request in [decision(current['task'], 'legacy', current['annotation_revision']),
                        batch(current, [], completion='approve', ident='approve')]:
            with pytest.raises(ValueError, match='SOURCE_ISSUE_BLOCKS_APPROVAL'):
                store.apply_decision(c, request)
        forbidden = batch(current, [], completion='approve', ident='system-approve')
        forbidden['actor_kind'] = 'system'
        with pytest.raises(ValueError, match='BATCH_ACTOR_FORBIDDEN'):
            store.apply_decision(c, forbidden)
    finally:
        c.close()
    with pytest.raises(FileExistsError):
        prepare_module.prepare_preview(*inputs)
    assert read_rows(destination / 'review.sqlite') == restored


@pytest.mark.parametrize('corruption', ['revision', 'document', 'task', 'span', 'hash', 'missing_hash',
                                      'duplicate', 'code', 'message', 'provenance', 'empty', 'non_object'])
def test_invalid_issues_fail_before_creating_outputs(inputs, prepare_module, corruption):
    source, destination, issues, _ = inputs
    before = read_rows(source)
    issue = json.loads(issues.read_text())
    records = [issue]
    if corruption == 'revision': issue['text_revision'] = 'sha256:' + '0' * 64
    elif corruption == 'document': issue['document_id'] = 'wrong-document'
    elif corruption == 'task': issue['task_id'] = 'missing-task'
    elif corruption == 'span': issue['span'] = {'start': 0, 'end': 99999}
    elif corruption == 'hash': issue['source_report_sha256'] = '0' * 64
    elif corruption == 'missing_hash': issue.pop('source_report_sha256')
    elif corruption == 'duplicate': records.append(issue)
    elif corruption == 'code': issue['code'] = 'invented'
    elif corruption == 'message': issue['message'] = ''
    elif corruption == 'provenance': issue['actor_kind'] = 'human'
    elif corruption == 'empty': records = []
    elif corruption == 'non_object': records = [[]]
    issues.write_text(''.join(json.dumps(row) + '\n' for row in records))
    with pytest.raises(ValueError):
        prepare_module.prepare_preview(*inputs)
    assert not destination.exists()
    assert read_rows(source) == before


@pytest.mark.parametrize('relationship', ['same', 'ancestor', 'below_file', 'symlink', 'dangling'])
def test_rejects_destination_overlap_or_existing_paths(inputs, prepare_module, relationship):
    source, destination, issues, report = inputs
    if relationship == 'same': destination = source
    elif relationship == 'ancestor': destination = source.parent
    elif relationship == 'below_file': destination = source / 'preview'
    elif relationship == 'symlink': destination.symlink_to(source.parent, target_is_directory=True)
    elif relationship == 'dangling': destination.symlink_to(destination.parent / 'missing')
    before = read_rows(source)
    with pytest.raises((ValueError, FileExistsError)):
        prepare_module.prepare_preview(source, destination, issues, report)
    assert read_rows(source) == before


def test_cli_requires_explicit_paths_and_refuses_second_run(inputs):
    source, destination, issues, report = inputs
    args = [sys.executable, str(SCRIPT), '--source-store', str(source), '--destination', str(destination),
            '--issues', str(issues), '--report', str(report)]
    first = subprocess.run(args, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)['source_unchanged'] is True
    before = read_rows(destination / 'review.sqlite')
    second = subprocess.run(args, capture_output=True, text=True)
    assert second.returncode != 0
    assert read_rows(destination / 'review.sqlite') == before
    missing = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert missing.returncode != 0


def test_captures_uncheckpointed_wal_and_reports_concurrent_divergence(inputs, prepare_module, monkeypatch):
    source, destination, _, _ = inputs
    writer = store.open_store(source)
    writer.execute('PRAGMA wal_autocheckpoint=0')
    current = store.get_item(writer, read_rows(source)['items'][0]['task_id'])
    store.apply_decision(writer, batch(current, [{'operation_id': 'note', 'action': 'record_note',
                        'reason_code': 'note', 'note': 'Latest committed source edit'}], ident='wal-note'))
    before = read_rows(source)
    assert Path(str(source) + '-wal').stat().st_size > 0
    original_export = prepare_module.export_reference

    def concurrent_export(c, path):
        if path.name == 'baseline':
            live = store.get_item(writer, current['task']['task_id'])
            store.apply_decision(writer, batch(live, [{'operation_id': 'later', 'action': 'record_note',
                                'reason_code': 'note', 'note': 'Concurrent human edit'}], ident='later-note'))
        return original_export(c, path)

    monkeypatch.setattr(prepare_module, 'export_reference', concurrent_export)
    try:
        result = prepare_module.prepare_preview(*inputs)
    finally:
        writer.close()
    _, baseline = read_review_snapshot(destination / 'baseline')
    assert baseline['decision-records.jsonl'] == before['decisions']
    assert result['source_unchanged'] is False
    assert result['ready_for_review'] is False
    assert result['baseline_table_sha256'] != result['source_after_table_sha256']
    assert len(read_rows(source)['decisions']) == len(before['decisions']) + 1


def test_later_invalid_issue_and_already_imported_issue_leave_no_destination(inputs, prepare_module):
    source, destination, issues, report = inputs
    original = issues.read_bytes()
    valid = json.loads(original)
    invalid = {**valid, 'issue_id': 'later-invalid', 'text_revision': 'wrong'}
    issues.write_text(json.dumps(valid) + '\n' + json.dumps(invalid) + '\n')
    with pytest.raises(ValueError):
        prepare_module.prepare_preview(*inputs)
    assert not destination.exists()
    issues.write_bytes(original)
    prepare_module.prepare_preview(*inputs)
    second_destination = destination.parent / 'second-preview'
    with pytest.raises(ValueError, match='SOURCE_ISSUE_ALREADY_IMPORTED'):
        prepare_module.prepare_preview(destination / 'review.sqlite', second_destination, issues, report)
    assert not second_destination.exists()


def test_frozen_policy_and_prompt_assets_survive_preparation(tmp_path, prepare_module):
    from alias_fixtures import alias_item
    from research.cli import main
    from research.annotations.snapshots import read_policy_provenance
    from research.data.bundles import materialize_bundle
    from research.data.manifest import read_jsonl

    fixture = alias_item()
    materialize_bundle({'role': 'train', 'documents': [{
        'document_id': fixture['task']['document_id'], 'text': fixture['task']['text'],
        'source': 'ecosystems', 'work_group_id': 'fixture', 'split': 'train',
        'public': True, 'access_basis': 'fixture', 'text_license': 'CC-BY-4.0'}],
        'heldout': {}}, tmp_path / 'train')
    main(['annotate', 'prepare', '--bundle', str(tmp_path / 'train'), '--policy',
          str(SCRIPT.parents[1] / 'annotations/scibert-v2/policy-2.1.md'),
          '--output', str(tmp_path / 'tasks')])
    manifest = json.loads((tmp_path / 'tasks/manifest.json').read_bytes())
    task = read_jsonl(tmp_path / 'tasks/tasks.jsonl')[0]
    fixture['task'] = task
    fixture['annotation'].update({key: task[key] for key in ('task_id', 'document_id', 'text_revision', 'policy_version')})
    fixture['annotation']['annotator']['prompt_hash'] = manifest['prompt_sources']['annotate']['sha256']
    source = tmp_path / 'frozen.sqlite'
    c = store.open_store(source)
    store.import_items(c, [fixture], 'train', read_policy_provenance(tmp_path / 'tasks', manifest))
    c.close()
    report = tmp_path / 'report.md'
    report.write_text('Fixture boundary report.\n')
    issues = tmp_path / 'issues.jsonl'
    write_jsonl(issues, [{'issue_id': 'frozen-boundary', 'code': 'boundary_fragment', 'message': 'Fixture issue.',
                         'span': task['annotation_region'], 'source_report_sha256': digest(report.read_bytes()),
                         **{key: task[key] for key in ('task_id', 'document_id', 'text_revision')}}])
    before = read_rows(source)
    destination = tmp_path / 'preview'
    prepare_module.prepare_preview(source, destination, issues, report)
    assert read_rows(source) == before
    assert read_rows(destination / 'review.sqlite')['metadata'] == before['metadata']
    for record in [manifest['policy_source'], *manifest['prompt_sources'].values()]:
        expected = (tmp_path / 'tasks' / record['path']).read_bytes()
        assert (destination / 'baseline' / record['path']).read_bytes() == expected
        assert (destination / 'prepared' / record['path']).read_bytes() == expected
