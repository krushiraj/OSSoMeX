"""Additive experimental five-stage commands using the shared terminal and JSON output."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from ..data.manifest import read_jsonl
from .cli import read_stdin_document, write_predictions
from .json_output import render_prediction


def register(subparsers):
    parser = subparsers.add_parser('full-label', help='Experimental software fields and local alias proposals')
    commands = parser.add_subparsers(dest='full_label_command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze verified real training inputs')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--output', required=True)
    prepare.set_defaults(func=cmd_prepare)
    train = commands.add_parser('train', help='Train one fixed-recipe local stage')
    train.add_argument('--stage', choices=('detector', 'linker', 'intent', 'sentiment', 'alias'), required=True)
    for key in ('data', 'config', 'output'):
        train.add_argument('--' + key, required=True)
    train.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='mps')
    train.set_defaults(func=cmd_train)
    predict = commands.add_parser('predict', help='Emit a diagnostic envelope and complete public rows')
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--input', help='JSONL documents with document_id and text')
    source.add_argument('--stdin', action='store_true', help='One document from strict UTF-8 stdin')
    predict.add_argument('--document-id', default='input')
    predict.add_argument('--output', help='New immutable JSONL output')
    predict.set_defaults(func=cmd_predict)
    interactive = commands.add_parser('interactive', help='Keep verified stages resident for multiline input')
    interactive.add_argument('--input-mode', choices=('bracketed', 'lines'), default='bracketed')
    interactive.set_defaults(func=cmd_interactive)
    for command in (predict, interactive):
        command.add_argument('--model', required=True)
        command.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
        command.add_argument('--format', choices=('pretty', 'jsonl'), default='pretty')


def cmd_prepare(args):
    from .data import prepare_data
    result = prepare_data(json.loads(Path(args.config).read_bytes()), Path(args.output))
    print(json.dumps({'output': args.output, 'summary': result['summary']}, indent=2))
    return 0


def cmd_train(args):
    from .full_label_train import fit_stage
    result = fit_stage(args.stage, Path(args.data), json.loads(Path(args.config).read_bytes()), Path(args.output), args.device)
    print(json.dumps({'output': args.output, 'status': result['status'],
                      'support': result.get('support'), 'training': result.get('training')}, indent=2))
    return int(result['status'] != 'trained_experimental')


def cmd_predict(args):
    from .full_label import FullLabelPipeline
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.stdin:
        documents = [read_stdin_document(sys.stdin.buffer, args.document_id)]
    else:
        documents = read_jsonl(Path(args.input)) if args.input else [{'document_id': args.document_id, 'text': args.text}]
    if not documents or len({document.get('document_id') for document in documents}) != len(documents):
        raise ValueError('nonempty input with unique document IDs required')
    with redirect_stdout(sys.stderr):
        pipeline = FullLabelPipeline(Path(args.model), args.device)
        results = [pipeline.predict(document) for document in documents]
    if args.output:
        write_predictions(Path(args.output), results)
        print(json.dumps({'output': args.output, 'documents': len(results)}))
    else:
        for index, result in enumerate(results):
            if index and args.format == 'pretty':
                print()
            print(render_prediction(result, args.format))
    return int(any(result['status'] in ('partial', 'failure') for result in results))


def run_full_label_interactive(model, device, input_mode, *, pipeline_factory=None, **kwargs):
    from .full_label import FullLabelPipeline
    from .interactive import run_interactive
    return run_interactive(model, device, input_mode, detector_factory=pipeline_factory or FullLabelPipeline,
        failure_factory=lambda pipeline, document, error: pipeline.failure_result(document, error),
        stop_on_failure=False, command_name='full-label', **kwargs)


def cmd_interactive(args):
    return run_full_label_interactive(Path(args.model), args.device, args.input_mode, output_format=args.format)
