from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.contracts import text_revision
from research.data import exposure
from research.data.manifest import digest, json_bytes, read_jsonl
from research.data.splits import SplitError, assign_splits, group_works


ROLES = ('train_reserved', 'diagnostic', 'native_excluded', 'heldout_reserved', 'historical_exposed')
PURPOSES = ('new_acquisition', 'training', 'heldout')


def document(name, text=None, **extra):
    text = text or f'{name} has its own paper content.'
    return {'document_id': name, 'text': text, 'text_revision': text_revision(text),
            'source': 'ecosystems', 'source_ids': {}, 'fulltext_eligible': True, **extra}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return digest(path.read_bytes())


def input_record(path, rows, role='diagnostic', kind='documents_jsonl'):
    return {'kind': kind, 'role': role, 'path': str(path), 'sha256': jsonl(path, rows)}


def make_exposures(tmp_path, rows, role='diagnostic', kind='documents_jsonl'):
    spec = input_record(tmp_path / 'source.jsonl', rows, role, kind)
    output = tmp_path / 'exposures'
    report = exposure.build_exposures({'inputs': [spec]}, output)
    return exposure.load_exposures(output), report


def bundle_record(tmp_path, role, *, identities=None):
    root = tmp_path / 'source-bundle'
    filename = 'inputs.jsonl' if role == 'diagnostic' else 'documents.jsonl'
    files = [{'path': filename, 'sha256': jsonl(root / filename, [
        document('parent', source_ids={'doi': '10.1000/parent'})])}]
    if identities is not None:
        files.append({'path': 'identities.jsonl', 'sha256': jsonl(root / 'identities.jsonl', identities)})
    manifest = root / 'manifest.json'
    manifest.write_bytes(json_bytes({'files': files, 'role': 'diagnostic' if role == 'diagnostic' else 'train'}))
    return {'kind': 'bundle_manifest', 'role': role, 'path': str(manifest),
            'sha256': digest(manifest.read_bytes())}


@pytest.mark.parametrize('role', ROLES)
def test_exposure_roles_differ_by_purpose(tmp_path, role):
    exposures, _ = make_exposures(tmp_path, [document('parent', source_ids={'doi': '10.1000/parent'})], role)
    candidate = document('alias', source_ids={'doi': 'https://doi.org/10.1000/PARENT'})
    for purpose in PURPOSES:
        reasons = exposure.exposure_reasons(candidate, exposures, purpose=purpose)
        if role == 'train_reserved' and purpose == 'training':
            assert reasons == []
        else:
            assert reasons and {reason['role'] for reason in reasons} == {role}


@pytest.mark.parametrize(('source_ids', 'alias'), [
    ({'doi': 'doi:10.1000/PARENT'}, {'doi': 'https://dx.doi.org/10.1000%2fparent'}),
    ({'pmcid': 'PMC125'}, {'pmcid': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC125/'}),
    ({'openalex': 'W125'}, {'openalex': 'https://openalex.org/w125/'}),
])
def test_identifier_aliases_cannot_bypass_exposure(tmp_path, source_ids, alias):
    exposures, _ = make_exposures(tmp_path, [document('parent', source_ids=source_ids)])
    reasons = exposure.exposure_reasons(document('alias', source_ids=alias), exposures, purpose='training')
    assert any(reason['reason'] == 'identifier_overlap' for reason in reasons)


@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('variant', ['exact', 'whitespace', 'near', 'excerpt', 'parent'])
def test_text_exposures_are_role_aware(tmp_path, role, variant):
    text = ' '.join(f'word{i}' for i in range(100))
    candidate_text = {'exact': text, 'whitespace': text.replace(' ', '  '),
                      'near': text + ' ending',
                      'excerpt': ' '.join(f'word{i}' for i in range(10, 30)),
                      'parent': 'prefix words before ' + text + ' words after'}[variant]
    exposures, _ = make_exposures(tmp_path, [document('parent', text)], role)
    candidate = document('different-id', candidate_text)
    for purpose in PURPOSES:
        reasons = exposure.exposure_reasons(candidate, exposures, purpose=purpose)
        assert bool(reasons) == (role != 'train_reserved' or purpose != 'training')
        assert all(reason['reason'] == 'text_overlap' for reason in reasons)


def test_document_identity_and_identity_only_reservations(tmp_path):
    exposures, report = make_exposures(tmp_path, [
        {'source_ids': {'doi': '10.1000/identity-only'}, 'reason': 'no_snippets'},
        {'document_id': 'known-native-id'},
    ], 'heldout_reserved')
    assert report['document_count'] == 0 and report['identity_count'] == 2
    assert not exposures['documents']
    for candidate in (document('alias', source_ids={'doi': '10.1000/identity-only'}),
                      document('known-native-id', 'A new extraction.')):
        assert all(exposure.exposure_reasons(candidate, exposures, purpose=purpose) for purpose in PURPOSES)


def test_legacy_metadata_and_every_source_association_participate(tmp_path):
    legacy = document('legacy', metadata={'work_id': 'https://openalex.org/W101', 'doi': 'https://doi.org/10.1000/LEGACY'})
    multi = document('multi', source_ids={'openalex': 'W201', 'doi': None}, source_associations=[
        {'source_ids': {'openalex': 'W201', 'doi': None}},
        {'source_ids': {'openalex': 'W202', 'doi': '10.1000/second'}},
        {'work_id': 'https://openalex.org/W203', 'doi': '10.1000/third'},
    ])
    exposures, _ = make_exposures(tmp_path, [legacy, multi], kind='diagnostic_inputs')
    assert exposures['documents'][0]['source_ids'] == {'openalex': 'https://openalex.org/W101', 'doi': 'https://doi.org/10.1000/LEGACY'}
    for key, value in [('openalex', 'W101'), ('doi', '10.1000/legacy'), ('openalex', 'W201'),
                       ('openalex', 'W202'), ('doi', '10.1000/second'), ('openalex', 'W203'), ('doi', '10.1000/third')]:
        assert exposure.exposure_reasons(document('other', source_ids={key: value}), exposures, purpose='training')
    # No cross-provider identity is invented for a work whose DOI is unknown.
    assert exposure.exposure_reasons(document('other', source_ids={'doi': '10.1000/unknown'}), exposures, purpose='training') == []
    assert any(issue['code'] == 'MISSING_DOI' for issue in exposures['issues'])


def test_diagnostic_bundle_reserves_all_o2_identity_rows(tmp_path):
    spec = bundle_record(tmp_path, 'diagnostic', identities=[
        {'source_ids': {'openalex': 'W301', 'doi': None}, 'reason': 'no_snippets'},
        {'source_ids': {'openalex': 'W302', 'doi': '10.1000/blank'},
         'source_associations': [{'source_ids': {'openalex': 'W303', 'doi': '10.1000/association'}}]},
    ])
    exposure.build_exposures({'inputs': [spec]}, tmp_path / 'exposures')
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    assert len(exposures['identities']) == 2
    for source_ids in ({'openalex': 'W301'}, {'doi': '10.1000/blank'}, {'openalex': 'W303'}):
        assert exposure.exposure_reasons(document('other', source_ids=source_ids), exposures, purpose='training')


def test_overlapping_full_and_dev_inputs_deduplicate_without_losing_sources(tmp_path):
    row = document('legacy', metadata={'work_id': 'W111', 'doi': '10.1000/shared'})
    config = {'inputs': [input_record(tmp_path / 'full.jsonl', [row]),
                         input_record(tmp_path / 'dev.jsonl', [row])]}
    report = exposure.build_exposures(config, tmp_path / 'exposures')
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    assert report['document_count'] == 1
    assert len(exposures['documents'][0]['exposure_sources']) == 2
    assert len(exposures['manifest']['inputs']) == 2


def test_conflicting_roles_are_visible_and_use_restrictive_decision(tmp_path):
    config = {'inputs': [input_record(tmp_path / f'{role}.jsonl', [
        document(role, source_ids={'doi': '10.1000/conflict'})], role) for role in ('train_reserved', 'diagnostic')]}
    exposure.build_exposures(config, tmp_path / 'exposures')
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    assert any(issue['code'] == 'EXPOSURE_ROLE_CONFLICT' and set(issue['roles']) == {'train_reserved', 'diagnostic'}
               for issue in exposures['issues'])
    reasons = exposure.exposure_reasons(document('alias', source_ids={'doi': '10.1000/conflict'}), exposures, purpose='training')
    assert reasons and {row['role'] for row in reasons} == {'diagnostic'}


def test_known_parent_aliases_propagate_restrictive_role_without_merging_snippet_parents(tmp_path):
    config = {'inputs': [
        input_record(tmp_path / 'training.jsonl', [document('train', source_ids={
            'doi': '10.1000/known', 'openalex': 'W401'})], 'train_reserved'),
        input_record(tmp_path / 'diagnostic.jsonl', [document('diagnostic', source_ids={
            'openalex': 'W401', 'doi': None})]),
    ]}
    exposure.build_exposures(config, tmp_path / 'exposures')
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    candidate = document('different-provider', source_ids={'doi': '10.1000/known'})
    assert exposure.exposure_reasons(candidate, exposures, purpose='training')
    # Same snippet text does not establish an identity link between its different parents.
    rows = [document('shared', source_ids={'openalex': 'W501', 'doi': None}, source_associations=[
        {'source_ids': {'openalex': 'W501', 'doi': None}},
        {'source_ids': {'openalex': 'W502', 'doi': '10.1000/separate'}},
    ])]
    config = {'inputs': [input_record(tmp_path / 'shared.jsonl', rows, 'train_reserved'),
                         input_record(tmp_path / 'single.jsonl', [{'source_ids': {'openalex': 'W501'}}])]}
    exposure.build_exposures(config, tmp_path / 'separate')
    separate = exposure.load_exposures(tmp_path / 'separate')
    assert exposure.exposure_reasons(document('other', source_ids={'doi': '10.1000/separate'}), separate, purpose='training') == []


@pytest.mark.parametrize('identity_format', ['source_ids', 'legacy_metadata'])
@pytest.mark.parametrize('later_parent', ['W401', 'W402'])
def test_duplicate_rows_retain_parent_alias_groups_without_merging_unrelated_parents(
        tmp_path, identity_format, later_parent):
    later_identity = ({'source_ids': {'openalex': later_parent, 'doi': '10.1000/known'}}
                      if identity_format == 'source_ids' else {
                          'metadata': {'work_id': f'https://openalex.org/{later_parent}',
                                       'doi': 'https://doi.org/10.1000/KNOWN'}})
    config = {'inputs': [
        input_record(tmp_path / 'first.jsonl', [document('train',
            source_ids={'openalex': 'W401'})], 'train_reserved'),
        input_record(tmp_path / 'later.jsonl', [document('train', **later_identity)], 'train_reserved'),
        input_record(tmp_path / 'diagnostic.jsonl', [{'source_ids': {'openalex': 'W401'}}]),
    ]}
    output = tmp_path / 'exposures'
    report = exposure.build_exposures(config, output)
    loaded = exposure.load_exposures(output)
    assert report['document_count'] == 1
    assert len(loaded['documents'][0]['exposure_sources']) == 2
    candidate = document('different-provider', source_ids={'doi': '10.1000/known'})
    reasons = exposure.exposure_reasons(candidate, loaded, purpose='training')
    if later_parent == 'W401':
        assert reasons and {(row['role'], row['reason']) for row in reasons} == {
            ('diagnostic', 'identifier_overlap')}
    else:
        assert reasons == []


def test_relative_source_paths_remain_bound_to_build_directory(tmp_path, monkeypatch):
    spec = input_record(tmp_path / 'source.jsonl', [document('parent')])
    spec['path'] = 'source.jsonl'
    monkeypatch.chdir(tmp_path)
    exposure.build_exposures({'inputs': [spec]}, tmp_path / 'exposures')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    assert exposure.exposure_reasons(document('parent'), exposures, purpose='training')
    exposure.verify_exposure_sources(exposures)


@pytest.mark.parametrize('change', ['missing', 'changed'])
def test_source_changes_block_build_and_load(tmp_path, change):
    source = tmp_path / 'source.jsonl'
    spec = input_record(source, [document('parent')])
    exposure.build_exposures({'inputs': [spec]}, tmp_path / 'exposures')
    if change == 'missing':
        source.unlink()
    else:
        source.write_text('{}\n')
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.load_exposures(tmp_path / 'exposures')
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.build_exposures({'inputs': [spec]}, tmp_path / 'another-output')
    assert not (tmp_path / 'another-output').exists()


@pytest.mark.parametrize('role', ['train_reserved', 'diagnostic'])
def test_bundle_companion_hashes_are_required_and_verified(tmp_path, role):
    spec = bundle_record(tmp_path, role, identities=[] if role == 'diagnostic' else None)
    exposure.build_exposures({'inputs': [spec]}, tmp_path / 'exposures')
    filename = 'identities.jsonl' if role == 'diagnostic' else 'documents.jsonl'
    (tmp_path / 'source-bundle' / filename).write_text('{}\n')
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.load_exposures(tmp_path / 'exposures')
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.build_exposures({'inputs': [spec]}, tmp_path / 'invalid')


def test_diagnostic_manifest_without_identity_companion_fails_closed(tmp_path):
    spec = bundle_record(tmp_path, 'diagnostic')
    with pytest.raises(ValueError, match='MISSING_EXPOSURE_COMPANION'):
        exposure.build_exposures({'inputs': [spec]}, tmp_path / 'invalid')


@pytest.mark.parametrize('target', ['source', 'source_manifest', 'source_companion', 'manifest', 'documents', 'identities', 'issues'])
def test_before_publish_recheck_catches_changes_after_load(tmp_path, target):
    spec = bundle_record(tmp_path, 'diagnostic', identities=[])
    direct = input_record(tmp_path / 'direct.jsonl', [document('direct')])
    output = tmp_path / 'exposures'
    exposure.build_exposures({'inputs': [spec, direct]}, output)
    exposures = exposure.load_exposures(output)
    exposure.verify_exposure_sources(exposures)
    paths = {'source': Path(direct['path']), 'source_manifest': Path(spec['path']),
             'source_companion': Path(spec['path']).parent / 'identities.jsonl',
             **{name: output / f'{name}.jsonl' for name in ('documents', 'identities', 'issues')},
             'manifest': output / 'manifest.json'}
    paths[target].write_bytes(paths[target].read_bytes() + b' ')
    with pytest.raises(ValueError):
        exposure.verify_exposure_sources(exposures)


def test_cached_indexes_do_not_retokenize_exposure_corpus(tmp_path, monkeypatch):
    import research.data.ecosystems as ecosystems

    original = ecosystems.shingles
    seen = []

    def tracked(text):
        seen.append(text)
        return original(text)

    monkeypatch.setattr(ecosystems, 'shingles', tracked)
    corpus_text = 'Frozen native document with enough words for shingles.'
    exposures, _ = make_exposures(tmp_path, [document('parent', corpus_text)], 'native_excluded')
    for purpose in PURPOSES:
        for index in range(3):
            exposure.exposure_reasons(document(f'candidate-{index}'), exposures, purpose=purpose)
    exposure.verify_exposure_sources(exposures)
    assert seen.count(corpus_text) == 1


def test_split_flags_preserve_allocations_and_feed_real_assignment(tmp_path):
    config = {'inputs': [input_record(tmp_path / f'{role}.jsonl', [
        {'source_ids': {'doi': f'10.1000/{role}'}}], role) for role in ROLES]}
    output = tmp_path / 'exposures'
    exposure.build_exposures(config, output)
    exposures = exposure.load_exposures(output)
    documents = [document(role, source_ids={'doi': f'10.1000/{role}'}, native_split=None,
                          split='train', historical=False) for role in ROLES]
    original = deepcopy(documents)
    updated = exposure.apply_split_exposures(documents, exposures)
    assert documents == original
    assert all(row['split'] == 'train' and row['native_split'] is None for row in updated)
    assert all(row['development_exposed'] and row['exposure_reasons'] for row in updated)
    assert updated[0]['historical'] is False
    assert all(row['historical'] for row in updated[1:])
    groups = group_works(updated)['groups']
    assigned = assign_splits(groups, {'ecosystems': {'train': 1, 'dev': 0, 'test': 0}}, 42)
    assert [row['document_id'] for row in assigned['documents']] == ['train_reserved']
    for quota in ({'train': 0, 'dev': 1, 'test': 0}, {'train': 0, 'dev': 0, 'test': 1}, {'train': 2}):
        with pytest.raises(SplitError, match='QUOTA_SHORTAGE'):
            assign_splits(groups, {'ecosystems': quota}, 42)


def test_immutable_bundle_and_invalid_purpose(tmp_path):
    exposures, _ = make_exposures(tmp_path, [document('parent')])
    before = {p.name: p.read_bytes() for p in (tmp_path / 'exposures').iterdir()}
    with pytest.raises(FileExistsError):
        exposure.build_exposures({'inputs': []}, tmp_path / 'exposures')
    assert before == {p.name: p.read_bytes() for p in (tmp_path / 'exposures').iterdir()}
    with pytest.raises(ValueError, match='INVALID_EXPOSURE_PURPOSE'):
        exposure.exposure_reasons(document('candidate'), exposures, purpose='unknown')


def test_effective_config_retains_and_rechecks_original_config_source(tmp_path):
    spec = input_record(tmp_path / 'source.jsonl', [document('parent')])
    original = tmp_path / 'config.json'
    original.write_bytes(json_bytes({'inputs': [spec]}))
    config = {'inputs': [spec], 'config_source': {
        'path': str(original), 'sha256': digest(original.read_bytes()),
        'bytes_utf8': original.read_text()}}
    output = tmp_path / 'exposures'
    exposure.build_exposures(config, output)
    exposures = exposure.load_exposures(output)
    assert json.loads((output / 'config.json').read_bytes()) == config
    original.write_text('{}')
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.verify_exposure_sources(exposures)


def test_sources_are_rechecked_before_manifest_publication(tmp_path, monkeypatch):
    source = tmp_path / 'source.jsonl'
    spec = input_record(source, [document('parent')])
    original = exposure.atomic_write_new

    def change_source(path, payload):
        original(path, payload)
        if path.name == 'issues.jsonl':
            source.write_text('{}\n')

    monkeypatch.setattr(exposure, 'atomic_write_new', change_source)
    with pytest.raises(ValueError, match='STALE_EXPOSURE_SOURCE'):
        exposure.build_exposures({'inputs': [spec]}, tmp_path / 'exposures')
    assert not (tmp_path / 'exposures' / 'manifest.json').exists()


@pytest.mark.parametrize('malformation', ['role', 'kind', 'sha256', 'empty_identity'])
def test_invalid_sources_cannot_silently_remove_reservations(tmp_path, malformation):
    spec = input_record(tmp_path / 'source.jsonl', [{}] if malformation == 'empty_identity' else [document('parent')])
    if malformation != 'empty_identity':
        spec[malformation] = 'invalid'
    with pytest.raises(ValueError):
        exposure.build_exposures({'inputs': [spec]}, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_all_eight_verified_seed_inputs_are_retained(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/scibert/exposures-001.json').read_bytes())
    config['inputs'] = [{**row, 'path': str(root / row['path'])} for row in config['inputs']]
    report = exposure.build_exposures(config, tmp_path / 'exposures')
    exposures = exposure.load_exposures(tmp_path / 'exposures')
    assert len(exposures['manifest']['inputs']) == 8
    parents = [row for row in exposures['documents'] if row['role'] == 'train_reserved']
    assert {row['source_ids']['doi'].lower() for row in parents} == {
        '10.1021/acsnano.0c10632', '10.1021/acsomega.1c00991', '10.1111/aos.14122',
        '10.1186/s13568-016-0260-6', '10.1186/s12859-020-3513-y',
        '10.1038/s42003-021-02669-y', '10.1007/s00401-020-02226-7', '10.1186/s40478-018-0574-5'}
    assert len(parents) == 8
    probe = read_jsonl(root / 'runs/scibert-v2/openalex-parallel-probe-20260927-001/inputs.jsonl')
    known_ids = {key for row in exposures['documents'] for key in row['identity_keys']}
    assert len(probe) == 8
    assert all('openalex:' + row['source_ids']['openalex'].rsplit('/', 1)[-1].lower() in known_ids for row in probe)
    assert len([row for row in exposures['documents'] if row['role'] == 'native_excluded']) == 1852
    assert len([row for row in exposures['documents'] if row['role'] == 'historical_exposed']) == 16
    assert len([row for row in exposures['documents'] if row['role'] == 'diagnostic']) == 1488
    assert report['heldout_freeze_complete'] is False
