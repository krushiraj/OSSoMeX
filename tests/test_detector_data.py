from copy import deepcopy
import json

import pytest

from research.data.manifest import digest, json_bytes, write_jsonl


def example():
    parent = {'document_id': 'd', 'text': 'We used NumPy.', 'text_revision': 'sha256:a',
              'split': 'train', 'source': 'ecosystems', 'public': True,
              'text_license': 'CC-BY-4.0', 'access_basis': {'article_url': 'https://example.org'}}
    task = {**parent, 'task_id': 't', 'offset_base': 0,
            'context_span': {'start': 0, 'end': 14}, 'annotation_region': {'start': 0, 'end': 14}}
    annotation = {'occurrences': [{'name': 'NumPy', 'name_span': {'start': 8, 'end': 13},
                                 'known': {'software': True, 'versions': True}, 'version_links': []}],
                  'covered_regions': [{'start': 0, 'end': 14, 'fields': {'software': True, 'versions': False}}]}
    return parent, {'task': task, 'annotation': annotation, 'status': 'reviewed'}


def test_select_preserves_partial_coverage_and_excludes_unknown_only():
    from research.training.data import select_supervision
    parent, item = example()
    unknown = deepcopy(item)
    unknown['task']['task_id'] = 'unknown'
    unknown['annotation'] = {'occurrences': [], 'covered_regions': []}
    result = select_supervision([parent], [item, unknown])
    assert result['items'] == [item]
    assert result['excluded'] == [{'task_id': 'unknown', 'reason': 'no_detector_supervision'}]
    assert result['summary']['software_spans'] == 1
    assert result['summary']['version_spans'] == 0
    assert result['items'][0]['annotation']['covered_regions'][0]['fields']['versions'] is False


@pytest.mark.parametrize('change', ['text', 'revision', 'split', 'license', 'ownership', 'duplicate'])
def test_select_rejects_invalid_or_duplicate_supervision(change):
    from research.training.data import select_supervision
    parent, item = example()
    if change == 'text': item['task']['text'] = 'wrong'
    if change == 'revision': item['task']['text_revision'] = 'wrong'
    if change == 'split': parent['split'] = 'test'
    if change == 'license': parent['text_license'] = 'unknown'
    if change == 'ownership': item['task']['annotation_region']['start'] = 10
    with pytest.raises(ValueError):
        select_supervision([parent], [item, deepcopy(item)] if change == 'duplicate' else [item])


def preparation_sources(tmp_path):
    from research.annotations import review_store
    from research.annotations.tasks import make_tasks
    from research.annotations.validation import validate_task_occurrence
    from research.contracts import FIELDS, text_revision
    from research.data.bundles import materialize_bundle

    config = {kind: [] for kind in ('parents', 'snapshots', 'companions', 'attribution', 'exclusions')}
    documents = []
    for ordinal, source in enumerate(('ecosystems', 'europepmc')):
        root = tmp_path / str(ordinal)
        root.mkdir()
        text = ('We used NumPy for matrix arithmetic.' if ordinal == 0
                else 'An independent article applies pandas to tabular records.')
        name = 'NumPy' if ordinal == 0 else 'pandas'
        document = {'document_id': f'provider-{ordinal}', 'text': text, 'text_revision': text_revision(text),
                    'split': 'train', 'source': source, 'public': True, 'text_license': 'CC-BY-4.0',
                    'access_basis': {'article_url': f'https://example.org/{ordinal}'},
                    'source_ids': {'doi': f'10.1000/parent-{ordinal}', source: f'record-{ordinal}'}}
        task = make_tasks(document, {'policy_version': 'scibert-poc-2.0', 'policy_hash': 'a' * 64})[0]
        occurrence = validate_task_occurrence(task, {
            'schema_version': '2.0', 'document_id': document['document_id'], 'text_revision': document['text_revision'],
            'name': name, 'name_span': {'start': text.index(name), 'end': text.index(name) + len(name)},
            'context_sentence': text, 'context_span': task['context_span'], 'context_kind': 'paragraph',
            'version_links': [], 'version_status': 'unannotated', 'intents': [], 'sentiment': None,
            'known': {field: field == 'software' for field in FIELDS},
            'evidence': {'intents': [], 'sentiment': []},
            'review': {'status': 'agent_provisional', 'reasons': []}})
        item = {'task': task, 'annotation': {
            'occurrences': [occurrence], 'covered_regions': [{**task['annotation_region'],
            'fields': {field: field == 'software' for field in FIELDS}, 'status': 'partial',
            'provenance': {'kind': 'agent_provisional'}}], 'unresolved_regions': [],
            'status': 'partial', 'review_status': 'agent_provisional', 'annotation_revision': 1}}
        parent_path, snapshot_path = root / 'parents', root / 'snapshot'
        materialize_bundle({'role': 'train', 'documents': [document]}, parent_path)
        connection = review_store.open_store(root / 'synthetic-review.sqlite')
        try:
            review_store.import_items(connection, [item], 'train')
            review_store.export_reference(connection, snapshot_path)
        finally:
            connection.close()
        for kind, path in [('parents', parent_path), ('snapshots', snapshot_path)]:
            manifest = path / 'manifest.json'
            config[kind].append({'path': str(manifest), 'sha256': digest(manifest.read_bytes())})
        documents.append(document)
    return config, documents


def exposure_spec(tmp_path, rows, role='diagnostic'):
    from research.data.exposure import build_exposures

    source = tmp_path / 'reservation-source'
    source.mkdir()
    row_name = 'inputs.jsonl' if role == 'diagnostic' else 'documents.jsonl'
    write_jsonl(source / row_name, [row for row in rows if row.get('text')])
    if role == 'diagnostic':
        write_jsonl(source / 'identities.jsonl', [row for row in rows if not row.get('text')])
    elif any(not row.get('text') for row in rows):
        # Non-diagnostic D1 source bundles use documents.jsonl for identity rows too.
        (source / row_name).unlink()
        write_jsonl(source / row_name, rows)
    source_manifest = source / 'manifest.json'
    source_manifest.write_bytes(json_bytes({'role': 'train', 'files': [
        {'path': path.name, 'sha256': digest(path.read_bytes())} for path in sorted(source.glob('*.jsonl'))]}))
    output = tmp_path / 'exposures'
    build_exposures({'inputs': [{'kind': 'bundle_manifest', 'role': role,
                     'path': str(source_manifest), 'sha256': digest(source_manifest.read_bytes())}]}, output)
    manifest = output / 'manifest.json'
    return {'path': str(manifest), 'sha256': digest(manifest.read_bytes())}


@pytest.mark.parametrize('role', ['diagnostic', 'native_excluded', 'heldout_reserved', 'historical_exposed'])
@pytest.mark.parametrize('match', ['identity', 'text'])
def test_prepare_blocks_real_d1_restrictive_reservations(tmp_path, role, match):
    from research.training.data import prepare_data

    config, documents = preparation_sources(tmp_path)
    reservation = ({'source_ids': {'doi': 'https://doi.org/10.1000/PARENT-0', 'openalex': 'W999'},
                    'reason': 'no_snippets'} if match == 'identity' else
                   {**documents[0], 'document_id': 'different-provider', 'source_ids': {}})
    config['exposure_bundles'] = [exposure_spec(tmp_path, [reservation], role)]
    with pytest.raises(ValueError, match='exposure.*overlap'):
        prepare_data(config, tmp_path / 'training')
    assert not (tmp_path / 'training' / 'manifest.json').exists()


@pytest.mark.parametrize('text_bearing', [False, True])
def test_prepare_combined_train_reservations_preserves_auditable_provenance(tmp_path, text_bearing):
    from research.training.data import load_training_data, prepare_data

    config, documents = preparation_sources(tmp_path)
    reservations = documents if text_bearing else [{'source_ids': row['source_ids']} for row in documents]
    spec = exposure_spec(tmp_path, reservations, 'train_reserved')
    config['exposure_bundles'] = [spec]
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    provenance = manifest['exposure_bundles'][0]
    assert provenance['path'] == spec['path'] and provenance['sha256'] == spec['sha256']
    frozen = output / provenance['frozen_path']
    assert digest(frozen.read_bytes()) == spec['sha256']
    assert {path.name for path in frozen.parent.iterdir()} == {
        'manifest.json', 'config.json', 'documents.jsonl', 'identities.jsonl', 'issues.jsonl'}
    assert manifest['forbidden_documents_checked'] == 0
    assert json.loads((output / 'forbidden.json').read_bytes()) == {
        'work_group_ids': [], 'text_revisions': [], 'source_ids': []}
    _, loaded, items = load_training_data(output)
    assert {row['source'] for row in loaded} == {'ecosystems', 'europepmc'}
    assert len(items) == 2


@pytest.mark.parametrize('specs', [None, {}, [None], [{}], [{'path': '', 'sha256': 'a' * 64}],
                                  [{'path': 'missing/manifest.json', 'sha256': 'a' * 64}],
                                  [{'path': 'manifest.json', 'sha256': 'sha256:' + 'a' * 64}],
                                  [{'path': 'manifest.json', 'sha256': 'A' * 64}]])
def test_prepare_rejects_malformed_or_missing_d1_specs(tmp_path, specs):
    from research.training.data import prepare_data

    config, _ = preparation_sources(tmp_path)
    config['exposure_bundles'] = specs
    with pytest.raises(ValueError):
        prepare_data(config, tmp_path / 'training')


def test_prepare_rejects_duplicate_d1_manifest_paths(tmp_path):
    from research.training.data import prepare_data

    config, documents = preparation_sources(tmp_path)
    spec = exposure_spec(tmp_path, documents, 'train_reserved')
    config['exposure_bundles'] = [spec, {**spec, 'path': str(tmp_path / 'exposures' / '.' / 'manifest.json')}]
    with pytest.raises(ValueError, match='duplicate'):
        prepare_data(config, tmp_path / 'training')


@pytest.mark.parametrize('target', ['manifest', 'companion', 'original_manifest', 'original_companion'])
@pytest.mark.parametrize('phase', ['prepare', 'reload'])
def test_d1_stale_evidence_cannot_prepare_or_reload_training(tmp_path, target, phase):
    from research.training.data import load_training_data, prepare_data

    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    output = tmp_path / 'training'
    if phase == 'reload':
        prepare_data(config, output)
    path = {'manifest': tmp_path / 'exposures' / 'manifest.json',
            'companion': tmp_path / 'exposures' / 'identities.jsonl',
            'original_manifest': tmp_path / 'reservation-source' / 'manifest.json',
            'original_companion': tmp_path / 'reservation-source' / 'documents.jsonl'}[target]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        (prepare_data(config, output) if phase == 'prepare' else load_training_data(output))
    if phase == 'prepare':
        assert not (output / 'manifest.json').exists()


def test_prepare_binds_config_pin_to_manifest_actually_loaded(tmp_path, monkeypatch):
    import research.training.data as module
    from research.data.exposure import load_exposures

    config, documents = preparation_sources(tmp_path)
    spec = exposure_spec(tmp_path, documents, 'train_reserved')
    config['exposure_bundles'] = [spec]
    def replace_before_loading(bundle):
        path = bundle / 'manifest.json'
        path.write_bytes(path.read_bytes() + b'\n')
        return load_exposures(bundle)
    monkeypatch.setattr(module, 'load_exposures', replace_before_loading, raising=False)
    with pytest.raises(ValueError):
        module.prepare_data(config, tmp_path / 'training')
    assert not (tmp_path / 'training' / 'manifest.json').exists()


def test_prepare_rechecks_d1_sources_after_supervision_before_publication(tmp_path, monkeypatch):
    import research.training.data as module

    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    original = module.select_supervision
    def mutate_source(*args):
        result = original(*args)
        path = tmp_path / 'reservation-source' / 'documents.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        return result
    monkeypatch.setattr(module, 'select_supervision', mutate_source)
    with pytest.raises(ValueError):
        module.prepare_data(config, tmp_path / 'training')
    assert not (tmp_path / 'training' / 'manifest.json').exists()


@pytest.mark.parametrize('target', ['frozen_manifest', 'frozen_companion', 'pin', 'missing_provenance',
                                  'source', 'config', 'unsafe_path', 'missing_companion_record'])
def test_reload_rejects_tampered_exposure_provenance(tmp_path, target):
    from research.training.data import load_training_data, prepare_data

    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    if target.startswith('frozen_'):
        path = output / manifest['exposure_bundles'][0]['frozen_path']
        if target == 'frozen_companion':
            path = path.parent / 'identities.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        # Updating the outer file hash must not bypass the independently pinned D1 evidence.
        next(row for row in manifest['files'] if row['path'] == str(path.relative_to(output)))['sha256'] = digest(path.read_bytes())
    elif target == 'pin':
        manifest['exposure_bundles'][0]['sha256'] = '0' * 64
    elif target == 'missing_provenance':
        del manifest['exposure_bundles']
    elif target == 'source':
        next(row for row in manifest['sources'] if row['kind'] == 'exposure_bundles')['sha256'] = '0' * 64
    elif target == 'unsafe_path':
        manifest['exposure_bundles'][0]['frozen_path'] = '../exposures/manifest.json'
    elif target == 'missing_companion_record':
        manifest['files'] = [row for row in manifest['files']
                             if row['path'] != 'sources/exposure_bundles-0/identities.jsonl']
    else:
        path = output / 'config.json'
        saved = json.loads(path.read_bytes())
        saved['exposure_bundles'][0]['sha256'] = '0' * 64
        path.write_bytes(json_bytes(saved))
        next(row for row in manifest['files'] if row['path'] == 'config.json')['sha256'] = digest(path.read_bytes())
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):
        load_training_data(output)


def test_legacy_preparation_and_reload_omit_new_exposure_fields(tmp_path):
    from research.training.data import load_training_data, prepare_data

    config, _ = preparation_sources(tmp_path)
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    assert 'exposure_bundles' not in manifest
    assert all(row['kind'] != 'exposure_bundles' for row in manifest['sources'])
    assert not any('exposure_bundles' in row['path'] for row in manifest['files'])
    assert (output / 'config.json').read_bytes() == json_bytes(config)
    assert len(load_training_data(output)[2]) == 2


def test_legacy_loader_still_accepts_bundle_without_optional_source_metadata(tmp_path):
    from research.training.data import load_training_data, prepare_data

    config, _ = preparation_sources(tmp_path)
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    del manifest['sources']
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    assert len(load_training_data(output)[2]) == 2


@pytest.mark.parametrize('cause', ['text_exclusion', 'duplicate_parent'])
@pytest.mark.parametrize('use_d1', [False, True])
def test_prepare_retains_legacy_exclusion_and_duplicate_checks(tmp_path, cause, use_d1):
    from research.training.data import prepare_data

    config, documents = preparation_sources(tmp_path)
    if use_d1:
        config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    if cause == 'duplicate_parent':
        config['parents'].append(config['parents'][0])
    else:
        path = tmp_path / 'old-exclusions.jsonl'
        write_jsonl(path, [{**documents[0], 'document_id': 'old-provider', 'source_ids': {},
                           'text_revision': 'sha256:old-extraction'}])
        config['exclusions'] = [{'path': str(path), 'sha256': digest(path.read_bytes())}]
    with pytest.raises(ValueError, match='identity or text overlap'):
        prepare_data(config, tmp_path / 'training')


def test_prepare_rechecks_sources_after_copying_before_terminal_manifest(tmp_path, monkeypatch):
    import research.training.data as module

    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    original = module.atomic_write_jsonl_new
    def mutate_after_write(path, rows):
        original(path, rows)
        if path.name == 'exclusions.jsonl':
            source = tmp_path / 'exposures' / 'manifest.json'
            source.write_bytes(source.read_bytes() + b'\n')
    monkeypatch.setattr(module, 'atomic_write_jsonl_new', mutate_after_write)
    with pytest.raises(ValueError):
        module.prepare_data(config, tmp_path / 'training')
    assert not (tmp_path / 'training' / 'manifest.json').exists()


def test_reload_rechecks_training_parents_against_pinned_reservations(tmp_path):
    from research.training.data import load_training_data, prepare_data

    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, [
        {'source_ids': {'doi': '10.1000/diagnostic-only'}, 'reason': 'no_snippets'}])]
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    documents[0]['source_ids']['doi'] = '10.1000/diagnostic-only'
    path = output / 'documents.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in documents))
    next(row for row in manifest['files'] if row['path'] == 'documents.jsonl')['sha256'] = digest(path.read_bytes())
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError, match='exposure.*overlap'):
        load_training_data(output)


def test_relative_exposure_preparation_reloads_from_different_working_directory(tmp_path, monkeypatch):
    from research.training.data import load_training_data, prepare_data

    config, documents = preparation_sources(tmp_path)
    absolute_spec = exposure_spec(tmp_path, documents, 'train_reserved')
    config['exposure_bundles'] = [{**absolute_spec, 'path': 'exposures/manifest.json'}]
    caller_config = deepcopy(config)
    originals = {path: path.read_bytes() for root in (tmp_path / 'exposures', tmp_path / 'reservation-source')
                 for path in root.iterdir()}
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'training'
    manifest = prepare_data(config, output)
    monkeypatch.chdir(tmp_path.parent)
    _, loaded, items = load_training_data(output)
    assert len(loaded) == 2 and len(items) == 2
    assert config == caller_config
    assert json.loads((output / 'config.json').read_bytes()) == {**caller_config, 'exposure_bundles': [absolute_spec]}
    assert manifest['exposure_bundles'][0] == {**absolute_spec,
        'frozen_path': 'sources/exposure_bundles-0/manifest.json'}
    assert next(row for row in manifest['sources'] if row['kind'] == 'exposure_bundles') == {
        'kind': 'exposure_bundles', **absolute_spec}
    assert next(row for row in manifest['files'] if row['path'] == 'config.json')['sha256'] == digest(
        (output / 'config.json').read_bytes())
    assert {path: path.read_bytes() for path in originals} == originals
