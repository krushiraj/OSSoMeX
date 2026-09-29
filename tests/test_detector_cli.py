import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.cli import main
from research.training import cli as detector_cli


def test_render_prediction_pretty_preserves_unicode_and_values():
    from research.training.json_output import render_prediction

    assert render_prediction({'name': 'NumPy 🧪'}) == '{\n  "name": "NumPy 🧪"\n}'
    result = {'name': 'NumPy 🧪', 'scores': [0.25, 0.75], 'available': False, 'sentiment': None}
    assert json.loads(render_prediction(result)) == result


def test_render_prediction_jsonl_is_one_line():
    from research.training.json_output import render_prediction

    result = {'name': 'NumPy', 'text': 'first\r\nsecond'}
    rendered = render_prediction(result, 'jsonl')
    assert '\n' not in rendered
    assert json.loads(rendered) == result


@pytest.mark.parametrize('output_format', ['pretty', 'jsonl'])
def test_render_prediction_rejects_nonfinite_scores(output_format):
    from research.training.json_output import render_prediction

    with pytest.raises(ValueError):
        render_prediction({'score': float('nan')}, output_format)


def test_render_prediction_rejects_unknown_format():
    from research.training.json_output import render_prediction

    with pytest.raises(ValueError, match='format'):
        render_prediction({'name': 'NumPy'}, 'xml')


@pytest.mark.parametrize('output_format', [None, 'jsonl'])
def test_predict_stdout_formats_two_unicode_results(output_format, tmp_path, monkeypatch, capsys):
    class FakeDetector:
        def __init__(self, checkpoint, device):
            pass

        def predict(self, document):
            return {'document_id': document['document_id'], 'name': document['text'], 'status': 'no_mentions'}

    input_path = tmp_path / 'input.jsonl'
    input_path.write_text('{"document_id": "a", "text": "NumPy 🧪"}\n'
                          '{"document_id": "b", "text": "Python"}\n')
    monkeypatch.setattr('research.training.predict.Detector', FakeDetector)
    args = ['detector', 'predict', '--model', 'unused', '--input', str(input_path)]
    if output_format is not None:
        args.extend(['--format', output_format])
    assert main(args) == 0
    captured = capsys.readouterr()
    if output_format == 'jsonl':
        assert len(captured.out.splitlines()) == 2
        rows = [json.loads(line) for line in captured.out.splitlines()]
    else:
        assert captured.out == ('{\n  "document_id": "a",\n  "name": "NumPy 🧪",\n'
                                '  "status": "no_mentions"\n}\n\n'
                                '{\n  "document_id": "b",\n  "name": "Python",\n'
                                '  "status": "no_mentions"\n}\n')
        decoder = json.JSONDecoder()
        first, end = decoder.raw_decode(captured.out)
        second, _ = decoder.raw_decode(captured.out[end:].lstrip())
        rows = [first, second]
    assert rows == [{'document_id': 'a', 'name': 'NumPy 🧪', 'status': 'no_mentions'},
                    {'document_id': 'b', 'name': 'Python', 'status': 'no_mentions'}]
    assert captured.err == ''


@pytest.mark.parametrize('command', ['predict', 'interactive'])
def test_prediction_commands_reject_unknown_format(command):
    args = ['detector', command, '--model', 'unused', '--format', 'xml']
    if command == 'predict':
        args.extend(['--text', 'NumPy'])
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_stdin_preserves_crlf():
    document = detector_cli.read_stdin_document(BytesIO(b'We used NumPy.\r\n\r\n'), 'x')
    assert document == {'document_id': 'x', 'text': 'We used NumPy.\r\n\r\n'}


@pytest.mark.parametrize('payload', [b'\xff', b'', b' \t\r\n', '\u2003\n'.encode('utf-8')])
def test_invalid_stdin_does_not_load_model(payload, monkeypatch):
    def unexpected_detector(*args, **kwargs):
        pytest.fail('invalid stdin must be rejected before loading the model')

    monkeypatch.setattr('research.training.predict.Detector', unexpected_detector)
    monkeypatch.setattr('sys.stdin', SimpleNamespace(buffer=BytesIO(payload)))
    with pytest.raises(ValueError):
        main(['detector', 'predict', '--model', 'unused', '--stdin'])


@pytest.mark.parametrize('sources', [
    [],
    ['--stdin', '--text', 'NumPy'],
    ['--stdin', '--input', 'unused.jsonl'],
    ['--text', 'NumPy', '--input', 'unused.jsonl'],
])
def test_predict_requires_exactly_one_source(sources):
    with pytest.raises(SystemExit) as exc:
        main(['detector', 'predict', '--model', 'unused', *sources])
    assert exc.value.code == 2


def test_existing_stdin_output_prevents_read_and_model_load(tmp_path, monkeypatch):
    class UnreadableStream:
        def read(self, *args):
            pytest.fail('an output collision must be rejected before reading stdin')

    def unexpected_detector(*args, **kwargs):
        pytest.fail('an output collision must be rejected before loading the model')

    output = tmp_path / 'predictions.jsonl'
    output.write_bytes(b'existing output\n')
    monkeypatch.setattr('sys.stdin', SimpleNamespace(buffer=UnreadableStream()))
    monkeypatch.setattr('research.training.predict.Detector', unexpected_detector)
    with pytest.raises(FileExistsError):
        main(['detector', 'predict', '--model', 'unused', '--stdin', '--output', str(output)])
    assert output.read_bytes() == b'existing output\n'


@pytest.mark.parametrize('source', ['stdin', 'text', 'input'])
@pytest.mark.parametrize(('status', 'exit_code'), [('no_mentions', 0), ('failure', 1)])
@pytest.mark.parametrize('to_file', [False, True])
def test_predict_preserves_input_and_prediction_envelope(source, status, exit_code, to_file,
                                                       tmp_path, monkeypatch, capsys):
    import hashlib

    text = '  We used NumPy with caf\u00e9 \U0001f40d.\r\n\r\n'
    result = {
        'schema_version': 'detector-prediction-1', 'document_id': 'custom-id',
        'text_revision': 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'checkpoint_sha256': 'fixture-checkpoint',
        'capabilities': {'software_spans': True, 'version_spans': True, 'version_linking': False,
                         'aliases': False, 'intent': False, 'sentiment': False, 'full_contract': False},
        'scores_calibrated': False, 'quality_evaluated': False, 'review_required': True,
        'review_reasons': ['experimental_small_data', 'uncalibrated_scores'],
        'offset_unit': 'unicode_codepoint_half_open',
        'chunks': [{'window': 0, 'status': 'success' if status == 'no_mentions' else 'failure'}],
        'spans': [], 'status': status,
    }
    if status == 'failure':
        result['chunks'][0]['error'] = 'simulated inference failure'
        result['review_reasons'].append('incomplete_document')
    received = []

    class FakeDetector:
        def __init__(self, checkpoint, device):
            assert checkpoint == Path('unused')
            assert device == 'cpu'

        def predict(self, document):
            received.append(document)
            return result

    class CountingStream(BytesIO):
        reads = 0

        def read(self, size=-1):
            self.reads += 1
            return super().read(size)

    stream = CountingStream(text.encode('utf-8'))
    monkeypatch.setattr('sys.stdin', SimpleNamespace(buffer=stream))
    monkeypatch.setattr('research.training.predict.Detector', FakeDetector)
    if source == 'stdin':
        sources = ['--stdin']
    elif source == 'text':
        sources = ['--text', text]
    else:
        input_path = tmp_path / 'input.jsonl'
        input_path.write_text(json.dumps({'document_id': 'custom-id', 'text': text}) + '\n')
        sources = ['--input', str(input_path)]
    output = tmp_path / 'predictions.jsonl'
    args = ['detector', 'predict', '--model', 'unused', '--device', 'cpu',
            '--document-id', 'custom-id', *sources]
    if to_file:
        args.extend(['--output', str(output)])
    else:
        args.extend(['--format', 'jsonl'])

    assert main(args) == exit_code

    captured = capsys.readouterr()
    if to_file:
        rows = output.read_text().splitlines()
        assert json.loads(captured.out) == {'output': str(output), 'documents': 1}
    else:
        rows = captured.out.splitlines()
        assert not output.exists()
    assert len(rows) == 1
    assert json.loads(rows[0]) == result
    assert received == [{'document_id': 'custom-id', 'text': text}]
    assert stream.reads == (1 if source == 'stdin' else 0)
    assert captured.err == ''


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
