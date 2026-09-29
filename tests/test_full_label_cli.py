import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cli_pretty_jsonl_partial_exit_and_immutable_output(monkeypatch, capsys, tmp_path):
    from research.cli import main
    from research.training import full_label
    class Pipeline:
        def __init__(self, *args):
            pass
        def predict(self, document):
            return {'status': 'partial', 'text': document['text'], 'pipeline_complete': False}
    monkeypatch.setattr(full_label, 'FullLabelPipeline', Pipeline)
    assert main(['full-label', 'predict', '--model', 'model', '--text', '工具']) == 1
    assert capsys.readouterr().out.startswith('{\n  ')
    assert main(['full-label', 'predict', '--model', 'model', '--text', '工具', '--format', 'jsonl']) == 1
    assert len(capsys.readouterr().out.splitlines()) == 1
    path = tmp_path / 'out.jsonl'
    assert main(['full-label', 'predict', '--model', 'model', '--text', '工具', '--output', str(path)]) == 1
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(['full-label', 'predict', '--model', 'model', '--text', 'other', '--output', str(path)])
    assert path.read_bytes() == before and json.loads(before)['text'] == '工具'


def test_cli_strict_stdin(monkeypatch):
    from research.cli import main
    monkeypatch.setattr('sys.stdin', SimpleNamespace(buffer=io.BytesIO(b'\xff')))
    with pytest.raises(UnicodeDecodeError):
        main(['full-label', 'predict', '--model', 'model', '--stdin'])


def test_interactive_loads_once_two_multiline_submissions_and_recovers():
    from research.training.full_label_cli import run_full_label_interactive
    from research.training import interactive
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    class Session:
        def __init__(self):
            self.drafts = iter(['first\n工具', 'second\npassage'])
            self.default_buffer = SimpleNamespace(reset=lambda: None)
        def prompt(self, *args, **kwargs):
            try:
                return next(self.drafts)
            except StopIteration:
                raise EOFError
    instances = []
    class Pipeline:
        def __init__(self, *args):
            instances.append(self)
            self.documents = []
        def predict(self, document):
            self.documents.append(document)
            if len(self.documents) == 1:
                raise RuntimeError('head failed')
            return {'status': 'success', 'text': document['text']}
        def failure_result(self, document, error):
            return {'status': 'failure', 'document_id': document['document_id']}
    stdout, stderr = io.StringIO(), io.StringIO()
    assert run_full_label_interactive(Path('model'), 'cpu', 'bracketed', pipeline_factory=Pipeline,
        prompt_session=Session(), stdin=Terminal(), stdout=stdout, stderr=stderr) == 0
    assert len(instances) == 1 and len(instances[0].documents) == 2
    decoder = json.JSONDecoder()
    content = stdout.getvalue().lstrip()
    first, index = decoder.raw_decode(content)
    second = json.loads(content[index:])
    assert first['status'] == 'failure' and second['text'] == 'second\npassage'
    assert 'Enter: submit' in stderr.getvalue() and '\n\n{' in stdout.getvalue()
