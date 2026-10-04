# Terminal walkthrough

`terminal-demo.gif` renders real output captured from a pseudo-terminal running the interactive CLI. It is a terminal recording, not a screen capture of a desktop application. The frame around the terminal is added by the renderer. Model responses are not edited.

- `terminal-demo.cast`: original terminal output events and timestamps in asciicast v2 format.
- `terminal-demo.txt`: readable terminal snapshots after each response.
- `terminal-demo.json`: exact command, inputs, model-manifest hash and exit status.
- `terminal-demo.gif`: looping README animation; idle gaps capped at six seconds.

Four synthetic inputs demonstrate names and versions, multiple mentions and no mentions. They are a usage walkthrough, not an evaluation of generalization or speed. The recorder clears the screen between samples while leaving the same model process loaded.

## Re-record

Install the model environment and download the complete bundle as described in the root README. Create a separate environment for the recording tools:

```sh
python3.11 -m venv .venv-demo
.venv-demo/bin/python -m pip install pillow pyte pexpect
.venv-demo/bin/python scripts/record_terminal_demo.py \
  --model model-assets/checkpoints/scibert-full-label-006 \
  --font /System/Library/Fonts/Monaco.ttf
```

Run from the repository root. The example font path is for macOS; on Linux supply an installed monospace TrueType font. The recorder requires a Unix pseudo-terminal. `--python` selects the model environment, `--device` selects CPU or MPS, and `--output` selects the destination. Existing `terminal-demo` files there are replaced. Recording dependencies do not belong in the model runtime environment.

The committed recording used Python 3.11 for model inference, checkpoint 006 on CPU, and the local `checkpoints/` model layout. The renderer used Pillow 12.3.0, pyte 0.8.2 and pexpect 4.9.0. Refer to the metadata file for the model hash and command. The generated terminal includes the CLI's own experimental-output notices.
