"""Exercise the actual terminal renderer with JSON on a separate stdout pipe."""

import json
import os
import pty
import select
import subprocess
import sys
import termios
import time


def test_terminal_prompts_stay_on_stderr_and_paste_stays_one_request(tmp_path):
    script = '''
import json, sys
from pathlib import Path
from research.training.interactive import run_interactive
class Detector:
    def __init__(self, model, device):
        print("DETECTOR_LOADED", file=sys.stderr, flush=True)
    def predict(self, document):
        return {"schema_version": "detector-prediction-1", "status": "no_mentions", **document}
raise SystemExit(run_interactive(Path("stub"), "cpu", "bracketed", detector_factory=Detector))
'''
    master, slave = pty.openpty()
    process = subprocess.Popen([sys.executable, '-B', '-c', script], stdin=slave,
                               stdout=subprocess.PIPE, stderr=slave, cwd=tmp_path,
                               env={**os.environ, 'TERM': 'xterm-256color', 'PROMPT_TOOLKIT_NO_CPR': '1'})
    os.close(slave)
    terminal = bytearray()
    transcript = bytearray()

    def wait_for_terminal(marker):
        deadline = time.monotonic() + 10
        while marker not in terminal:
            assert time.monotonic() < deadline, terminal.decode(errors='replace')
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b''
                assert chunk, terminal.decode(errors='replace')
                terminal.extend(chunk)
                transcript.extend(chunk)
            assert process.poll() is None, terminal.decode(errors='replace')

    try:
        wait_for_terminal(b'\x1b[?2004h')
        os.write(master, '\x1b[200~NumPy\n\n🧪 Python\n\x1b[201~'.encode())
        assert not select.select([process.stdout], [], [], 0.2)[0], 'paste submitted before Enter'
        os.write(master, b'\r')
        assert select.select([process.stdout], [], [], 10)[0]
        first = process.stdout.readline()
        terminal.clear()
        wait_for_terminal(b'\x1b[?2004h')
        os.write(master, b'Python\r')
        assert select.select([process.stdout], [], [], 10)[0]
        second = process.stdout.readline()
        terminal.clear()
        wait_for_terminal(b'\x1b[?2004h')
        os.write(master, b'discard this\x04')
        remaining, _ = process.communicate(timeout=10)
        assert process.returncode == 0
        assert remaining == b''
        assert b'\x1b' not in first + second
        rows = [json.loads(first), json.loads(second)]
        assert [r['text'] for r in rows] == ['NumPy\n\n🧪 Python\n', 'Python']
        assert [r['document_id'] for r in rows] == ['interactive-1', 'interactive-2']
        assert transcript.count(b'DETECTOR_LOADED') == 1
        assert list(tmp_path.iterdir()) == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)


def test_piped_cli_reports_stdin_guidance_without_model_load(tmp_path):
    result = subprocess.run([sys.executable, '-B', '-m', 'research', 'detector', 'interactive',
                             '--model', 'missing'], input=b'NumPy', capture_output=True,
                            cwd=tmp_path, timeout=15)
    assert result.returncode == 2
    assert result.stdout == b''
    assert b'--stdin' in result.stderr
    assert b'Traceback' not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_line_mode_one_ctrl_d_discards_nonempty_current_line(tmp_path):
    script = '''
from pathlib import Path
from research.training.interactive import run_interactive
class Detector:
    def __init__(self, model, device):
        pass
    def predict(self, document):
        raise AssertionError("discarded draft must never reach inference")
raise SystemExit(run_interactive(Path("stub"), "cpu", "lines", detector_factory=Detector))
'''
    master, slave = pty.openpty()
    original_terminal = termios.tcgetattr(slave)
    process = subprocess.Popen([sys.executable, '-B', '-c', script], stdin=slave,
                               stdout=subprocess.PIPE, stderr=slave, cwd=tmp_path,
                               env={**os.environ, 'TERM': 'xterm-256color', 'PROMPT_TOOLKIT_NO_CPR': '1'})
    terminal = bytearray()

    def wait_for_terminal(marker):
        deadline = time.monotonic() + 5
        while marker not in terminal:
            assert time.monotonic() < deadline, terminal.decode(errors='replace')
            if select.select([master], [], [], 0.1)[0]:
                terminal.extend(os.read(master, 65536))
            assert process.poll() is None, terminal.decode(errors='replace')

    try:
        wait_for_terminal(b'text>')
        os.write(master, b'completed line\rpending draft')
        wait_for_terminal(b'pending draft')
        os.write(master, b'\x04')
        assert select.select([process.stdout], [], [], 3)[0], 'one Ctrl+D did not exit line mode'
        output, _ = process.communicate(timeout=3)
        assert process.returncode == 0
        assert output == b''
        restored_terminal = termios.tcgetattr(slave)
        # macOS may set PENDIN while re-enabling canonical input; it is kernel state.
        restored_terminal[3] &= ~getattr(termios, 'PENDIN', 0)
        original_terminal[3] &= ~getattr(termios, 'PENDIN', 0)
        assert restored_terminal == original_terminal
        assert list(tmp_path.iterdir()) == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)
