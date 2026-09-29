from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.contracts import text_revision


def parent(name='paper'):
    text = 'We used NumPy.\n\nToolX was difficult to configure but easy to use.\n\nA control passage.'
    return {'document_id': name, 'text': text, 'text_revision': text_revision(text),
            'split': 'train', 'source': 'ecosystems', 'work_group_id': name,
            'public': True, 'text_license': 'CC-BY-4.0',
            'access_basis': {'article_url': 'https://example.org/paper'},
            'paragraphs': [{'start': 0, 'end': 14, 'kind': 'p'}, {'start': 16, 'end': 65, 'kind': 'p'},
                           {'start': 67, 'end': len(text), 'kind': 'p'}]}


def policy(**extra):
    return {'seed': 42, 'max_passages': 72, 'random_per_parent': 3,
            'signal_per_parent': 3, 'cues': ['difficult', 'also known as'], **extra}


def test_topup_excludes_owned_text_and_never_creates_labels():
    from research.data.training_topup import select_regions
    doc = parent()
    item = {'task': {'document_id': 'paper', 'text_revision': doc['text_revision'],
                     'task_id': 'reviewed', 'annotation_region': {'start': 0, 'end': 14}},
            'annotation': {'review_status': 'human_reviewed'}}
    original = deepcopy(item)
    rows = select_regions([doc], [item], policy(random_per_parent=0))
    assert len(rows) == 1
    assert rows[0]['annotation_region'] == {'start': 16, 'end': 65}
    assert rows[0]['selection_reason'] == 'training_gap_candidate'
    assert 'annotation' not in rows[0] and 'sentiment' not in rows[0]
    assert item == original


def test_topup_caps_are_deterministic_and_regions_disjoint():
    from research.data.training_topup import select_regions
    documents = [parent('a'), parent('b')]
    first = select_regions(documents, [], policy(max_passages=4))
    assert first == select_regions(documents, [], policy(max_passages=4))
    assert len(first) == 4
    assert len({(r['document_id'], r['annotation_region']['start']) for r in first}) == 4


@pytest.mark.parametrize('change', ['revision', 'split', 'license', 'duplicate'])
def test_topup_refuses_invalid_training_sources(change):
    from research.data.training_topup import select_regions
    doc = parent()
    if change == 'revision':
        doc['text_revision'] = 'sha256:' + '0' * 64
    if change == 'split':
        doc['split'] = 'dev'
    if change == 'license':
        doc['text_license'] = 'unknown'
    with pytest.raises(ValueError):
        select_regions([doc, deepcopy(doc)] if change == 'duplicate' else [doc], [], policy())


@pytest.mark.parametrize('config', [policy(max_passages=73), policy(random_per_parent=4),
                                  policy(signal_per_parent=4), policy(seed=7)])
def test_topup_refuses_scope_expansion(config):
    from research.data.training_topup import select_regions
    with pytest.raises(ValueError):
        select_regions([parent()], [], config)


def test_topup_rejects_stale_or_duplicate_ownership():
    from research.data.training_topup import select_regions
    doc = parent()
    task = {'task_id': 'owned', 'document_id': 'paper',
            'text_revision': doc['text_revision'], 'annotation_region': {'start': 0, 'end': 14}}
    with pytest.raises(ValueError, match='ownership'):
        select_regions([doc], [{'task': task}, {'task': task}], policy())
    task['text_revision'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError, match='revision'):
        select_regions([doc], [{'task': task}], policy())


def test_topup_preparation_refuses_output_before_loading(tmp_path):
    from research.data.training_topup import prepare_topup
    with pytest.raises(FileExistsError):
        prepare_topup({}, tmp_path)


def test_topup_can_target_existing_parents_without_accepting_unknown_ids():
    from research.data.training_topup import select_regions
    docs = [parent('a'), parent('b')]
    rows = select_regions(docs, [], policy(document_ids=['b']))
    assert {r['document_id'] for r in rows} == {'b'}
    for ids in ([], ['unknown'], ['b', 'b'], 'b'):
        with pytest.raises(ValueError, match='parent selection'):
            select_regions(docs, [], policy(document_ids=ids))


def test_exposure_configuration_retains_exclusions_and_reserves_supplement():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / 'configs/scibert/exposures-001.json').read_bytes())
    new = json.loads((root / 'configs/scibert/exposures-002.json').read_bytes())
    assert new['inputs'][:-1] == old['inputs']
    extra = new['inputs'][-1]
    assert extra == {'kind': 'bundle_manifest', 'role': 'train_reserved',
                     'path': 'data/scibert-v2/supplemental-001/bundle/manifest.json',
                     'sha256': '01bd0dd99db1e18ce0b4ab64718e915f150e25e45e7b7a2f0ac422d1dca58240'}


def topup_sources(tmp_path, role='train_reserved'):
    from research.annotations import review_store
    from research.annotations.policies import load_policy
    from research.annotations.tasks import make_region_task
    from research.contracts import FIELDS
    from research.data.bundles import materialize_bundle
    from research.data.manifest import digest
    from research.data.training_topup import POLICY
    from research.training.data import prepare_data
    from test_detector_data import exposure_spec
    doc = parent()
    task = make_region_task(doc, {'policy_version': 'scibert-poc-2.0', 'policy_hash': 'a' * 64}, {'start': 0, 'end': 14},
                            {'start': 0, 'end': 14}, region_kind='sentence')
    item = {'task': task, 'annotation': {'occurrences': [], 'covered_regions': [
        {'start': 0, 'end': 14, 'fields': {f: f == 'software' for f in FIELDS},
         'status': 'partial', 'provenance': {'kind': 'agent_provisional'}}],
        'unresolved_regions': [], 'status': 'partial', 'review_status': 'agent_provisional',
        'annotation_revision': 1}}
    bundle = tmp_path / 'parents'
    materialize_bundle({'role': 'train', 'documents': [doc]}, bundle)
    conn = review_store.open_store(tmp_path / 'review.sqlite')
    try:
        review_store.import_items(conn, [item], 'train')
        review_store.export_reference(conn, tmp_path / 'snapshot')
    finally:
        conn.close()
    pinned = lambda path: {'path': str(path), 'sha256': digest(path.read_bytes())}
    data = tmp_path / 'training'
    prepare_data({'parents': [pinned(bundle / 'manifest.json')],
                  'snapshots': [pinned(tmp_path / 'snapshot/manifest.json')],
                  'companions': [], 'attribution': [], 'exclusions': []}, data)
    return {'training_data': pinned(data / 'manifest.json'),
            'exposures': exposure_spec(tmp_path, [doc], role),
            'policy_sha256': load_policy(POLICY)['policy_hash'], 'selection': policy()}


def test_preparation_preserves_snapshot_and_pins_every_output(tmp_path):
    from research.data.manifest import digest
    from research.data.training_topup import prepare_topup
    config = topup_sources(tmp_path)
    before = {p: digest(p.read_bytes()) for p in (tmp_path / 'snapshot').rglob('*') if p.is_file()}
    output = tmp_path / 'tasks'
    manifest = prepare_topup(config, output)
    assert manifest['task_count'] == 2
    assert manifest['role'] == 'train' and manifest['status'] == 'ready_for_annotation'
    assert before == {p: digest(p.read_bytes()) for p in before}
    for row in manifest['files']:
        assert digest((output / row['path']).read_bytes()) == row['sha256']
    tasks = [json.loads(line) for line in (output / 'tasks.jsonl').read_text().splitlines()]
    assert all(t['annotation_region']['start'] >= 16 and 'annotation' not in t for t in tasks)
    with pytest.raises(FileExistsError):
        prepare_topup(config, output)


@pytest.mark.parametrize('role', ['diagnostic', 'heldout_reserved'])
def test_preparation_blocks_comparison_or_heldout_exposure(tmp_path, role):
    from research.data.training_topup import prepare_topup
    config = topup_sources(tmp_path, role)
    with pytest.raises(ValueError, match='exposure overlap'):
        prepare_topup(config, tmp_path / 'tasks')
    assert not (tmp_path / 'tasks').exists()


@pytest.mark.parametrize('key', ['training_data', 'exposures', 'policy_sha256'])
def test_preparation_refuses_changed_manifest_or_policy(tmp_path, key):
    from research.data.training_topup import prepare_topup
    config = topup_sources(tmp_path)
    if key == 'policy_sha256':
        config[key] = '0' * 64
    else:
        config[key]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='changed topup'):
        prepare_topup(config, tmp_path / 'tasks')
    assert not (tmp_path / 'tasks').exists()
