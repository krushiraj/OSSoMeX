import io
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_full_label_prediction import pipeline_factory
from research.training.json_output import render_prediction


def test_table_preserves_mentions_multiple_versions_and_alias_members(pipeline_factory):
    pipeline, document = pipeline_factory('We used NumPy 1.24 and 1.26 and ToolX.',
        ('NumPy', 'ToolX'), ('1.24', '1.26'), values={'linker': [[5], [5], [-5], [-5]], 'alias': [[5]]})
    result = pipeline.predict(document)
    before = deepcopy(result)
    output = render_prediction(result, 'table')
    lines = [line for line in output.splitlines() if '|' in line]
    assert 'Software' in lines[0] and 'Aliases' in lines[0] and 'Sentiment' in lines[0]
    assert len(lines) == 3
    assert all(value in lines[1] for value in ('NumPy', '1.24', '1.26', 'created', 'used', 'ToolX'))
    assert all(value in lines[2] for value in ('ToolX', 'mentioned', 'NumPy'))
    assert 'checkpoint_hashes' not in output and 'model_input_context' not in output
    assert result == before


def test_table_does_not_collapse_repeated_literal_names(pipeline_factory):
    pipeline, document = pipeline_factory('A and B.', ('A', 'B'), ())
    result = pipeline.predict(document)
    # The renderer must key aliases/rows by occurrence ID, not by literal name.
    result['field_predictions'][1]['name'] = 'A'
    output = render_prediction(result, 'table')
    assert sum(line.split('|')[0].strip() == 'A' for line in output.splitlines() if '|' in line) == 2


def test_table_partial_fields_are_unknown_not_defaults(pipeline_factory):
    pipeline, document = pipeline_factory(missing=('linker', 'intent', 'sentiment', 'alias'))
    output = render_prediction(pipeline.predict(document), 'table')
    assert 'NumPy' in output and 'ToolX' in output and output.count('unknown') >= 8
    assert 'mentioned' not in output and 'not expressed' not in output
    assert 'partial' in output and 'unavailable' in output


def test_low_confidence_alias_marks_both_endpoints_for_review(pipeline_factory):
    pipeline, document = pipeline_factory(values={'alias': [[.1]]})
    output = render_prediction(pipeline.predict(document), 'table')
    assert 'NumPy !' in output and 'ToolX !' in output and '! Review:' in output


def test_low_confidence_version_detection_marks_only_its_linked_software(pipeline_factory):
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    version = next(row for row in result['detector_diagnostics']['spans'] if row['text'] == '1.24')
    version['score'] = .6
    result['review_reasons'].append('low_confidence:detector')
    output = render_prediction(result, 'table')
    assert 'NumPy !' in output and 'ToolX !' not in output and '! Review:' in output


def test_table_distinguishes_failed_detection_from_no_mentions(pipeline_factory):
    pipeline, document = pipeline_factory('word.', (), ())
    assert 'No software mentions' in render_prediction(pipeline.predict(document), 'table')
    pipeline, document = pipeline_factory(detector_status='failure')
    failed = render_prediction(pipeline.predict(document), 'table')
    assert 'failed' in failed.lower() and 'No software mentions' not in failed


def test_table_renders_exception_envelope_without_detector_diagnostics(pipeline_factory):
    pipeline, document = pipeline_factory()
    result = pipeline.failure_result(document, RuntimeError('inference failed'))
    assert result['detector_diagnostics'] is None
    output = render_prediction(result, 'table')
    assert 'failed' in output.lower() and 'No software mentions' not in output


def test_color_only_adds_ansi_and_input_control_characters_are_escaped(pipeline_factory):
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    result['field_predictions'][0]['name'] = '工具\x1b[2J\r\nX'
    result['document_id'] = '\x1b]52;c;secret\x07'
    plain = render_prediction(result, 'table')
    colored = render_prediction(result, 'table', color=True)
    assert '\x1b' not in plain and '\r' not in plain and '\x07' not in plain
    assert '工具' in plain and '\\x1b' in plain
    assert '\x1b[' in colored and '\x1b[2J' not in colored and '\x1b]52' not in colored
    assert re.sub(r'\x1b\[[0-9;]*m', '', colored) == plain


def test_table_wraps_long_unicode_names_without_dropping_text(pipeline_factory):
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    result['field_predictions'][0]['name'] = '工具' * 18
    output = render_prediction(result, 'table', width=80)
    rows = [line for line in output.splitlines()[2:] if '|' in line]
    name_cells = ''.join(line.split('|')[0].strip() for line in rows)
    assert '工具' * 18 in name_cells
    assert all(len(line) + sum(c in '工具' for c in line) <= 80 for line in rows)


@pytest.mark.parametrize('no_color,term,terminal,want_color', [
    (None, 'xterm-256color', True, True), ('1', 'xterm-256color', True, False),
    (None, 'dumb', True, False), (None, 'xterm-256color', False, False),
])
def test_cli_table_colors_follow_terminal_and_jsonl_file_is_unchanged(
        pipeline_factory, monkeypatch, tmp_path, no_color, term, terminal, want_color):
    from research.cli import main
    from research.training import full_label
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    class Output(io.StringIO):
        def isatty(self):
            return terminal
    stdout = Output()
    monkeypatch.setattr('sys.stdout', stdout)
    monkeypatch.setenv('TERM', term)
    if no_color is None:
        monkeypatch.delenv('NO_COLOR', raising=False)
    else:
        monkeypatch.setenv('NO_COLOR', no_color)
    monkeypatch.setattr(full_label, 'FullLabelPipeline', lambda *args: SimpleNamespace(predict=lambda doc: result))
    args = ['full-label', 'predict', '--model', 'unused', '--text', document['text'], '--format', 'table']
    assert main(args) == 0
    assert ('\x1b[' in stdout.getvalue()) is want_color
    assert 'NumPy' in stdout.getvalue() and 'checkpoint_sha256' not in stdout.getvalue()
    path = tmp_path / 'predictions.jsonl'
    assert main([*args, '--output', str(path)]) == 0
    assert len(path.read_text().splitlines()) == 1 and json.loads(path.read_text()) == result


def test_interactive_two_tables_keep_multiline_inputs_and_prompts_separate(pipeline_factory, monkeypatch):
    from research.training.full_label_cli import run_full_label_interactive
    pipeline, document = pipeline_factory()
    result = pipeline.predict(document)
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    class Session:
        default_buffer = SimpleNamespace(reset=lambda: None)
        drafts = iter(['We used\nNumPy.', 'Second\npassage.'])
        def prompt(self, *args, **kwargs):
            try:
                return next(self.drafts)
            except StopIteration:
                raise EOFError
    received = []
    class Pipeline:
        def __init__(self, *args):
            pass
        def predict(self, doc):
            received.append(doc['text'])
            return result
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    stdout, stderr = Terminal(), io.StringIO()
    assert run_full_label_interactive(Path('unused'), 'cpu', 'bracketed', output_format='table',
        pipeline_factory=Pipeline, prompt_session=Session(), stdin=Terminal(), stdout=stdout, stderr=stderr) == 0
    assert received == ['We used\nNumPy.', 'Second\npassage.']
    assert stdout.getvalue().count('Software') == 2 and '\x1b[' in stdout.getvalue()
    assert 'Enter: submit' not in stdout.getvalue() and 'Enter: submit' in stderr.getvalue()
