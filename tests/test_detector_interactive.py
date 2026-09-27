"""Resident CLI behavior through real prompt input and injected inference."""

import hashlib
import importlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from research.cli import main


class Terminal(io.StringIO):
    def isatty(self):
        return True


class StubDetector:
    def __init__(self, model, device):
        self.identity = 'stub-checkpoint'
        self.manifest = {'capabilities': {'software_spans': True, 'version_spans': True,
                                        'version_linking': False, 'aliases': False,
                                        'intent': False, 'sentiment': False, 'full_contract': False}}
        self.documents = []

    def predict(self, document):
        self.documents.append(document)
        return {'schema_version': 'detector-prediction-1', **document,
                'text_revision': 'sha256:' + hashlib.sha256(document['text'].encode()).hexdigest(),
                'checkpoint_sha256': self.identity, 'capabilities': self.manifest['capabilities'],
                'scores_calibrated': False, 'quality_evaluated': False, 'review_required': True,
                'review_reasons': ['experimental_small_data', 'uncalibrated_scores'],
                'offset_unit': 'unicode_codepoint_half_open', 'chunks': [], 'spans': [],
                'status': 'no_mentions'}


def api():
    return importlib.import_module('research.training.interactive')


def run_keys(keys, *, detector_class=StubDetector, tmp_path=None, monkeypatch=None):
    # Playwright's session fixture owns the main thread's event loop during the full suite.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_keys, keys, detector_class=detector_class,
                               tmp_path=tmp_path, monkeypatch=monkeypatch).result(timeout=15)


def _run_keys(keys, *, detector_class, tmp_path, monkeypatch):
    interactive = api()
    loaded = []

    def factory(model, device):
        detector = detector_class(model, device)
        loaded.append(detector)
        return detector

    stdout, stderr = io.StringIO(), io.StringIO()
    if tmp_path is not None:
        monkeypatch.chdir(tmp_path)
    with create_pipe_input() as pipe:
        session = interactive.build_prompt_session(input=pipe, output=DummyOutput())
        pipe.send_text(keys)
        status = interactive.run_interactive(Path('checkpoint'), 'cpu', 'bracketed',
                                            detector_factory=factory, prompt_session=session,
                                            stdin=Terminal(), stdout=stdout, stderr=stderr)
        assert session.history.get_strings() == []
    if tmp_path is not None:
        assert list(tmp_path.iterdir()) == []
    return status, loaded, [json.loads(line) for line in stdout.getvalue().splitlines()], stderr.getvalue()


def test_bracketed_paste_is_one_unchanged_submission(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    preserved = [*sorted((root / 'checkpoints/scibert-detector-002').rglob('*')),
                 *(root / 'annotations/scibert-v2' / name / 'review.sqlite' for name in
                   ('ecosystems-pilot-002', 'icekat-alias-001', 'ecosystems-expansion-001'))]

    def fingerprints():
        result = {}
        for path in preserved:
            if path.is_file():
                with path.open('rb') as stream:
                    result[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return result

    before = fingerprints()
    status, loaded, rows, _ = run_keys('\x1b[200~NumPy\r\n\r\n🧪 Python\n\x1b[201~\r\x04',
                                       tmp_path=tmp_path, monkeypatch=monkeypatch)
    received_texts = [document['text'] for document in loaded[0].documents]
    assert received_texts == ['NumPy\r\n\r\n🧪 Python\n']
    assert rows[0]['text_revision'] == 'sha256:' + hashlib.sha256(received_texts[0].encode()).hexdigest()
    assert status == 0
    assert fingerprints() == before


def test_two_submissions_load_detector_once():
    status, loaded, rows, _ = run_keys('NumPy\rPython\r\x04')
    assert len(loaded) == 1
    assert [r['document_id'] for r in rows] == ['interactive-1', 'interactive-2']
    assert [r['text'] for r in rows] == ['NumPy', 'Python']
    assert status == 0


@pytest.mark.parametrize(('keys', 'texts'), [
    ('NumPy\x1b\rPython\r\x04', ['NumPy\nPython']),
    ('\r  \rNumPy\r\x04', ['NumPy']),
    ('discard me\x03NumPy\r\x04', ['NumPy']),
    ('discard me\x04', []),
    ('NumPy\r\x1b[A\r\x04', ['NumPy']),
])
def test_prompt_keys_discard_or_submit_exact_drafts(keys, texts):
    status, loaded, rows, stderr = run_keys(keys)
    assert [d['text'] for d in loaded[0].documents] == texts
    assert [r['text'] for r in rows] == texts
    assert status == 0
    assert 'Enter' in stderr and 'Ctrl+C' in stderr and 'Ctrl+D' in stderr


def test_load_failure_precedes_prompt_and_does_not_echo_exception():
    interactive = api()

    def fail(model, device):
        raise ValueError('private load detail')

    class UnreachablePrompt:
        def prompt(self, *args, **kwargs):
            pytest.fail('prompted before successful checkpoint load')

    stdout, stderr = io.StringIO(), io.StringIO()
    status = interactive.run_interactive(Path('missing'), 'cpu', 'bracketed', detector_factory=fail,
                                        prompt_session=UnreachablePrompt(), stdin=Terminal(),
                                        stdout=stdout, stderr=stderr)
    assert status == 1
    assert stdout.getvalue() == ''
    assert 'load' in stderr.getvalue().lower()
    assert 'private load detail' not in stderr.getvalue()


@pytest.mark.parametrize('raises', [False, True])
def test_inference_failure_emits_json_and_stops(raises):
    class Failure(StubDetector):
        def predict(self, document):
            result = super().predict(document)
            if raises:
                raise RuntimeError('private input may appear in errors')
            return {**result, 'status': 'failure', 'review_reasons': ['incomplete_document']}

    status, loaded, rows, stderr = run_keys('NumPy\rPython\r\x04', detector_class=Failure)
    assert status == 1
    assert len(loaded[0].documents) == 1
    assert len(rows) == 1 and rows[0]['status'] == 'failure'
    assert rows[0]['schema_version'] == 'detector-prediction-1'
    assert rows[0]['document_id'] == 'interactive-1'
    assert rows[0]['checkpoint_sha256'] == 'stub-checkpoint'
    assert 'private input' not in json.dumps(rows) + stderr


def test_interrupt_during_inference_returns_130_without_success():
    class Interrupted(StubDetector):
        def predict(self, document):
            raise KeyboardInterrupt

    status, loaded, rows, stderr = run_keys('NumPy\r\x04', detector_class=Interrupted)
    assert status == 130
    assert rows == []
    assert 'interrupt' in stderr.lower()


def test_model_diagnostics_cannot_contaminate_json_stdout(capsys):
    class Noisy(StubDetector):
        def __init__(self, *args):
            print('loading model')
            super().__init__(*args)

        def predict(self, document):
            print('model diagnostic')
            return super().predict(document)

    status, _, rows, stderr = run_keys('NumPy\r\x04', detector_class=Noisy)
    assert status == 0 and len(rows) == 1
    assert capsys.readouterr().out == ''
    assert 'loading model' in stderr and 'model diagnostic' in stderr


@pytest.mark.parametrize(('source', 'expected'), [
    ('NumPy\r\n\r\n🧪 Python\r\n:submit\r\ndiscarded', ['NumPy\n\n🧪 Python\n']),
    (':submit\n  \n:submit\nNumPy\n:submit\n', ['NumPy\n']),
    ('unsent\n', []),
])
def test_line_mode_terminator_normalization_and_eof(source, expected):
    interactive = api()
    detector = StubDetector(Path('checkpoint'), 'cpu')
    stdout, stderr = io.StringIO(), io.StringIO()
    status = interactive.run_interactive(Path('checkpoint'), 'cpu', 'lines',
                                        detector_factory=lambda *args: detector,
                                        stdin=Terminal(source), stdout=stdout, stderr=stderr)
    assert status == 0
    assert [d['text'] for d in detector.documents] == expected
    assert len(stdout.getvalue().splitlines()) == len(expected)
    assert ':submit' in stderr.getvalue() and 'normaliz' in stderr.getvalue()


def test_non_tty_rejected_before_loading():
    interactive = api()
    stdout, stderr = io.StringIO(), io.StringIO()
    status = interactive.run_interactive(Path('missing'), 'cpu', 'bracketed',
                                        detector_factory=lambda *args: pytest.fail('loaded detector'),
                                        stdin=io.StringIO('NumPy'), stdout=stdout, stderr=stderr)
    assert status == 2
    assert '--stdin' in stderr.getvalue()
    assert stdout.getvalue() == ''


def test_cli_registers_interactive_with_explicit_input_mode(monkeypatch):
    interactive = api()
    received = []
    monkeypatch.setattr(interactive, 'run_interactive', lambda *args: received.append(args) or 130)
    assert main(['detector', 'interactive', '--model', 'local', '--device', 'cpu', '--input-mode', 'lines']) == 130
    assert received == [(Path('local'), 'cpu', 'lines')]
