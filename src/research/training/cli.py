"""Additive detector commands; legacy five-field output remains untouched."""

import json
from pathlib import Path

from ..data.manifest import read_jsonl, write_jsonl


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
    predict.add_argument('--document-id', default='input')
    predict.add_argument('--output', help='New immutable JSONL output; otherwise stdout')
    predict.add_argument('--device', choices=('auto', 'cpu', 'mps'), default='auto')
    predict.set_defaults(func=cmd_predict)


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
    documents = read_jsonl(Path(args.input)) if args.input else [{'document_id': args.document_id, 'text': args.text}]
    if not documents or len({d.get('document_id') for d in documents}) != len(documents):
        raise ValueError('nonempty input with unique document IDs required')
    detector = Detector(Path(args.model), args.device)
    results = [detector.predict(document) for document in documents]
    if args.output:
        write_jsonl(Path(args.output), results)
        print(json.dumps({'output': args.output, 'documents': len(results)}))
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    return int(any(r['status'] == 'failure' for r in results))
