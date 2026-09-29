"""Additive detector commands; legacy five-field output remains untouched."""

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import BinaryIO

from ..data.manifest import read_jsonl
from .json_output import render_prediction


def read_stdin_document(stream: BinaryIO, document_id: str) -> dict:
    text = stream.read().decode('utf-8', errors='strict')
    if not text.strip():
        raise ValueError('nonempty stdin text required')
    return {'document_id': document_id, 'text': text}


def write_predictions(output, results):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.predictions-', delete=False) as stream:
            temporary = Path(stream.name)
            for result in results:
                stream.write((json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes complete bytes without overwriting a concurrent writer.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def register(subparsers):
    parser = subparsers.add_parser('detector', help='Experimental software/version span detector (partial contract)')
    commands = parser.add_subparsers(dest='detector_command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze verified real training inputs')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--output', required=True)
    prepare.set_defaults(func=cmd_prepare)
    train = commands.add_parser('train', help='Train a fixed-recipe local SciBERT detector')
    train.add_argument('--data', required=True)
    train.add_argument('--config', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    train.set_defaults(func=cmd_train)
    predict = commands.add_parser('predict', help='Emit exact spans, not full five-field records')
    predict.add_argument('--model', required=True)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--input', help='JSONL objects with document_id and text')
    source.add_argument('--stdin', action='store_true', help='One document from strict UTF-8 stdin')
    predict.add_argument('--document-id', default='input')
    predict.add_argument('--output', help='New immutable JSONL output; otherwise stdout')
    predict.add_argument('--format', choices=('pretty', 'jsonl'), default='pretty')
    predict.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    predict.set_defaults(func=cmd_predict)
    interactive = commands.add_parser('interactive', help='Keep a local detector loaded for multiline terminal input')
    interactive.add_argument('--model', required=True)
    interactive.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    interactive.add_argument('--input-mode', choices=('bracketed', 'lines'), default='bracketed')
    interactive.add_argument('--format', choices=('pretty', 'jsonl'), default='pretty')
    interactive.set_defaults(func=cmd_interactive)


def cmd_interactive(args):
    from .interactive import run_interactive
    return run_interactive(Path(args.model), args.device, args.input_mode, output_format=args.format)


def cmd_prepare(args):
    from .data import prepare_data
    result = prepare_data(json.loads(Path(args.config).read_bytes()), Path(args.output))
    print(json.dumps({'output': args.output, 'summary': result['summary']}, indent=2))
    return 0


def cmd_train(args):
    from .runner import fit_detector
    result = fit_detector(Path(args.data), json.loads(Path(args.config).read_bytes()), Path(args.output), args.device)
    print(json.dumps({'checkpoint': args.output, 'status': result['status'], 'capabilities': result['capabilities'],
                      'optimizer_steps': result['training']['optimizer_steps']}, indent=2))
    return 0


def cmd_predict(args):
    from .predict import Detector
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.stdin:
        documents = [read_stdin_document(sys.stdin.buffer, args.document_id)]
    else:
        documents = read_jsonl(Path(args.input)) if args.input else [{'document_id': args.document_id, 'text': args.text}]
    if not documents or len({d.get('document_id') for d in documents}) != len(documents):
        raise ValueError('nonempty input with unique document IDs required')
    detector = Detector(Path(args.model), args.device)
    results = [detector.predict(document) for document in documents]
    if args.output:
        write_predictions(Path(args.output), results)
        print(json.dumps({'output': args.output, 'documents': len(results)}))
    else:
        for index, result in enumerate(results):
            if index and args.format == 'pretty':
                print()
            print(render_prediction(result, args.format))
    return int(any(r['status'] == 'failure' for r in results))
