import json
from pathlib import Path

import pytest

from research.cli import main


def test_detector_cli_has_explicit_partial_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['detector', '--help'])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert 'prepare' in output and 'train' in output and 'predict' in output


def test_prepare_real_frozen_sources_and_reject_changed_bundle(tmp_path):
    from research.training.data import load_training_data
    config_path = Path('configs/scibert/detector-data-001.json')
    if not Path('data/scibert-v2/corpus-001/documents.jsonl').exists():
        pytest.skip('local licensed acquisition artifacts unavailable')
    destination = tmp_path / 'data'
    assert main(['detector', 'prepare', '--config', str(config_path), '--output', str(destination)]) == 0
    manifest, documents, items = load_training_data(destination)
    assert manifest['summary'] == json.loads(config_path.read_bytes())['expected_summary']
    assert len(documents) == 8 and len(items) == 33
    assert len((destination / 'exclusions.jsonl').read_text().splitlines()) == 2
    assert manifest['forbidden_documents_checked'] == 1868
    with pytest.raises(FileExistsError):
        main(['detector', 'prepare', '--config', str(config_path), '--output', str(destination)])
    (destination / 'items.jsonl').write_text('')
    with pytest.raises(ValueError, match='changed'):
        load_training_data(destination)


def test_predictions_publish_atomically_and_refuse_overwrite(tmp_path, monkeypatch):
    import os
    from research.training.cli import write_predictions
    output = tmp_path / 'predictions.jsonl'
    real_sync = os.fsync

    def fail_sync(fd):
        assert not output.exists()
        raise OSError('simulated full disk')

    monkeypatch.setattr(os, 'fsync', fail_sync)
    with pytest.raises(OSError, match='full disk'):
        write_predictions(output, [{'document_id': 'a'}, {'document_id': 'b'}])
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(os, 'fsync', real_sync)
    write_predictions(output, [{'document_id': 'a'}, {'document_id': 'b'}])
    assert len(output.read_text().splitlines()) == 2
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_predictions(output, [{'document_id': 'wrong'}])
    assert output.read_bytes() == before
