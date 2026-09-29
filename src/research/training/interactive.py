"""Resident, local inference with ephemeral terminal drafts."""

from contextlib import redirect_stdout
import hashlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TextIO

from .json_output import render_prediction

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


def build_prompt_session(*, input=None, output=None) -> 'PromptSession':
    from prompt_toolkit import PromptSession
    from prompt_toolkit.filters import Condition, is_true
    from prompt_toolkit.history import DummyHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output.defaults import create_output

    bindings = KeyBindings()
    preceding_cr = False

    @bindings.add('enter')
    def submit(event):
        nonlocal preceding_cr
        preceding_cr = not is_true(session.multiline)
        event.current_buffer.validate_and_handle()

    @bindings.add('c-j', filter=Condition(lambda: not is_true(session.multiline)))
    def linefeed(event):
        nonlocal preceding_cr
        # CR resets the prompt; a first-key LF in the next prompt completes that CRLF.
        completes_crlf = preceding_cr and not event.previous_key_sequence
        preceding_cr = False
        if not completes_crlf:
            event.current_buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def newline(event):
        nonlocal preceding_cr
        preceding_cr = False
        event.current_buffer.insert_text('\n')

    @bindings.add('<bracketed-paste>')
    def paste(event):
        nonlocal preceding_cr
        preceding_cr = False
        # The upstream binding rewrites CRLF and CR; offsets require the delivered text.
        event.current_buffer.insert_text(event.data)

    @bindings.add('c-c')
    def cancel(event):
        nonlocal preceding_cr
        preceding_cr = False
        event.current_buffer.reset()
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add('c-d')
    def finish(event):
        nonlocal preceding_cr
        preceding_cr = False
        event.current_buffer.reset()
        event.app.exit(exception=EOFError())

    session = PromptSession(multiline=True, key_bindings=bindings, history=DummyHistory(),
                            auto_suggest=None, enable_history_search=False,
                            enable_open_in_editor=False, enable_system_prompt=False,
                            input=input, output=output if output is not None else create_output(stdout=sys.stderr))
    return session


def read_draft(session, input_mode: str, *, stream: TextIO) -> str | None:
    if input_mode not in ('bracketed', 'lines'):
        raise ValueError('unknown input mode')
    lines = []
    while True:
        try:
            text = session.prompt('text> ', multiline=input_mode == 'bracketed')
        except EOFError:
            return None
        finally:
            session.default_buffer.reset()
        if input_mode == 'bracketed':
            return text
        line = text.replace('\r\n', '\n').replace('\r', '\n')
        if line == ':submit':
            return ''.join(lines)
        lines.append(line + '\n')


def _failure_result(detector, document):
    return {'schema_version': 'detector-prediction-1', 'document_id': document['document_id'],
            'text_revision': 'sha256:' + hashlib.sha256(document['text'].encode()).hexdigest(),
            'checkpoint_sha256': detector.identity, 'capabilities': detector.manifest['capabilities'],
            'scores_calibrated': False, 'quality_evaluated': False, 'review_required': True,
            'review_reasons': ['experimental_small_data', 'uncalibrated_scores', 'incomplete_document'],
            'offset_unit': 'unicode_codepoint_half_open', 'chunks': [], 'spans': [], 'status': 'failure'}


def run_interactive(model: Path, device: str, input_mode: str, *, output_format='pretty', detector_factory=None,
                    prompt_session=None, stdin=None, stdout=None, stderr=None) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    if not stdin.isatty():
        print('Interactive input requires a terminal; use detector predict --stdin for piped UTF-8 text.',
              file=stderr, flush=True)
        return 2
    if input_mode not in ('bracketed', 'lines'):
        raise ValueError('unknown input mode')
    try:
        with redirect_stdout(stderr):
            if detector_factory is None:
                from .predict import Detector
                detector_factory = Detector
            detector = detector_factory(model, device)
    except KeyboardInterrupt:
        print('Checkpoint load interrupted.', file=stderr, flush=True)
        return 130
    except Exception as exc:
        print(f'Checkpoint load failed ({type(exc).__name__}).', file=stderr, flush=True)
        return 1

    if prompt_session is None:
        from prompt_toolkit.input.defaults import create_input
        from prompt_toolkit.output.defaults import create_output
        prompt_session = build_prompt_session(input=create_input(stdin=stdin),
                                              output=create_output(stdout=stderr))
    if input_mode == 'bracketed':
        print('Enter: submit; Escape then Enter: newline; Ctrl+C: clear; Ctrl+D: exit. '
              'Pasted text stays one draft. Terminal/OS line endings may change; use --stdin for exact UTF-8.',
              file=stderr, flush=True)
    else:
        print('Line mode: a line containing exactly :submit submits; Ctrl+C clears; Ctrl+D exits. '
              'Line endings are normalized to LF; EOF discards an unfinished draft.', file=stderr, flush=True)
    submission = 0
    while True:
        try:
            text = read_draft(prompt_session, input_mode, stream=stdin)
        except KeyboardInterrupt:
            print('Draft cleared.', file=stderr, flush=True)
            continue
        if text is None:
            return 0
        if not text.strip():
            print('Enter nonblank text before submitting.', file=stderr, flush=True)
            continue
        submission += 1
        document = {'document_id': f'interactive-{submission}', 'text': text}
        try:
            with redirect_stdout(stderr):
                result = detector.predict(document)
        except KeyboardInterrupt:
            print('Inference interrupted; no result emitted.', file=stderr, flush=True)
            return 130
        except Exception as exc:
            result = _failure_result(detector, document)
            print(f'Inference failed ({type(exc).__name__}).', file=stderr, flush=True)
        if submission > 1 and output_format == 'pretty':
            print(file=stdout)
        print(render_prediction(result, output_format), file=stdout, flush=True)
        if result['status'] == 'failure':
            print('Inference failed; restart the command before another request.', file=stderr, flush=True)
            return 1
