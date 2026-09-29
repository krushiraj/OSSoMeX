import json

import pytest

from research.contracts import text_revision
from research.data.exposure import build_exposures, load_exposures
from research.data.manifest import digest, json_bytes


def doc(name, text=None, **extra):
    text = text or f'Unique passage {name}.'
    return {'document_id': name, 'text': text, 'text_revision': text_revision(text),
            'source_ids': {'openalex': 'W' + name}, **extra}


def history(tmp_path, rows):
    source = tmp_path / 'history.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    build_exposures({'inputs': [{'path': str(source), 'sha256': digest(source.read_bytes()),
                                'kind': 'documents_jsonl', 'role': 'train_reserved'}]}, tmp_path / 'exposures')
    return load_exposures(tmp_path / 'exposures')


def select(*args, **kwargs):
    from research.data import focused_evaluation
    return focused_evaluation.select_evaluation(*args, **kwargs)


def test_sampling_excludes_paper_aliases_and_exposed_text(tmp_path):
    old = doc('1', 'One previously seen sentence with many distinctive words here.',
              source_ids={'doi': '10.1000/old'})
    exposures = history(tmp_path, [old])
    candidates = [doc('2', source_ids={'doi': 'https://doi.org/10.1000/OLD'}),
                  doc('3', old['text'].replace(' ', '  ')), doc('4'), doc('5'), doc('6')]
    result = select(candidates, exposures, seed=42, evaluation_count=2, development_count=1)
    assert {r['document_id'] for r in result['evaluation'] + result['development']} == {'4', '5', '6'}
    assert {r['document_id'] for r in result['excluded']} == {'2', '3'}
    assert result == select(list(reversed(candidates)), exposures, seed=42, evaluation_count=2, development_count=1)


def test_transitive_aliases_cannot_cross_partitions(tmp_path):
    exposures = history(tmp_path, [doc('9')])
    candidates = [doc('1', source_ids={'doi': '10.1000/a'}),
                  doc('2', source_ids={'doi': '10.1000/a', 'openalex': 'W2'}),
                  doc('3', source_ids={'openalex': 'W2'}), doc('4'), doc('5')]
    result = select(candidates, exposures, seed=42, evaluation_count=2, development_count=1)
    ids = {r['document_id'] for r in result['evaluation'] + result['development']}
    assert len(ids & {'1', '2', '3'}) == 1
    assert {'4', '5'} <= ids


def test_association_only_identity_and_shortages_are_explicit(tmp_path):
    exposures = history(tmp_path, [doc('9', source_ids={'doi': '10.1000/old'})])
    candidate = doc('1', source_associations=[{'source_ids': {'doi': '10.1000/old'}}])
    result = select([candidate, doc('2')], exposures, seed=42, evaluation_count=2, development_count=1)
    assert len(result['evaluation']) == 1 and result['development'] == []
    assert result['shortages'] == {'evaluation': 1, 'development': 1}
    assert result['excluded'][0]['document_id'] == '1'


def test_near_duplicate_papers_only_supply_one_snippet(tmp_path):
    exposures = history(tmp_path, [doc('9')])
    text = ' '.join(f'word{i}' for i in range(60))
    result = select([doc('1', text), doc('2', text + ' appendix'), doc('3')], exposures,
                    seed=42, evaluation_count=2, development_count=0)
    assert len(result['evaluation']) == 2
    assert '3' in {r['document_id'] for r in result['evaluation']}


@pytest.mark.parametrize('kwargs', [{'seed': True}, {'evaluation_count': -1}, {'development_count': 1.5}])
def test_invalid_sampling_options_fail(tmp_path, kwargs):
    with pytest.raises(ValueError):
        select([doc('1')], history(tmp_path, [doc('9')]), **{ 'seed': 42, 'evaluation_count': 1,
                                                         'development_count': 0, **kwargs})


def collection(tmp_path):
    root = tmp_path / 'collection'
    root.mkdir()
    rows = [doc('1'), doc('2'), doc('3')]
    payload = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (root / 'inputs.jsonl').write_bytes(payload)
    (root / 'manifest.json').write_bytes(json_bytes({'status': 'completed', 'files': [
        {'path': 'inputs.jsonl', 'sha256': digest(payload)}]}))
    return root


def test_freeze_records_population_provenance_and_refuses_overwrite(tmp_path):
    from research.data import focused_evaluation as module
    assert hasattr(module, 'freeze_evaluation'), 'immutable evaluation publication missing'
    source = collection(tmp_path)
    history(tmp_path, [doc('9')])
    target = tmp_path / 'frozen'
    result = module.freeze_evaluation([source], tmp_path / 'exposures', target,
                                     seed=42, evaluation_count=2, development_count=1)
    assert result['counts']['evaluation'] == 2
    assert result['counts']['development'] == 1
    assert result['human_reviewed'] is False
    assert len((target / 'inputs.jsonl').read_text().splitlines()) == 2
    assert result['source_manifests'][0]['sha256'] == digest((source / 'manifest.json').read_bytes())
    for item in result['files']:
        assert digest((target / item['path']).read_bytes()) == item['sha256']
    with pytest.raises(FileExistsError):
        module.freeze_evaluation([source], tmp_path / 'exposures', target,
                                 seed=42, evaluation_count=2, development_count=1)


def test_changed_collection_rejected_without_publishing(tmp_path):
    from research.data import focused_evaluation as module
    assert hasattr(module, 'freeze_evaluation'), 'source verification missing'
    source = collection(tmp_path)
    history(tmp_path, [doc('9')])
    (source / 'inputs.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash|changed'):
        module.freeze_evaluation([source], tmp_path / 'exposures', tmp_path / 'frozen',
                                 seed=42, evaluation_count=2, development_count=1)
    assert not (tmp_path / 'frozen').exists()
