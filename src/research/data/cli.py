import json
from pathlib import Path

from .acquire import acquire_sources, audit_sources
from .manifest import json_bytes, read_jsonl, write_once
from .corpus import apply_access_decision, build_corpus, load_corpus
from .splits import SplitError, assign_splits, group_works, identifiers
from .bundles import materialize_bundle


def register(subparsers):
    parser = subparsers.add_parser('data')
    commands = parser.add_subparsers(dest='data_command', required=True)
    for name, argument in [('acquire', 'config'), ('audit', 'manifest'), ('build', 'manifest')]:
        p = commands.add_parser(name)
        p.add_argument('--' + argument, type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.set_defaults(func=run)
    p = commands.add_parser('split')
    for argument in ('corpus','config','output','private-output'):
        p.add_argument('--'+argument, type=Path, required=True)
    p.set_defaults(func=run)


def run(args):
    if args.data_command == 'split':
        config = json.loads(args.config.read_bytes())
        docs = load_corpus(args.corpus)
        history = [d for p in config['historical_documents'] for d in read_jsonl(Path(p))]
        history_ids = {d['document_id'].replace('sofair_', '').replace('somesci_', '').upper() for d in history}
        history_keys = set().union(*(identifiers(d) for d in history))
        history_text = {d['text'] for d in history}
        decisions = {d['document_id']:d for d in read_jsonl(Path(config['text_access_decisions']))} if config.get('text_access_decisions') else {}
        for i, d in enumerate(docs):
            d['historical'] = d['source_record_id'].upper() in history_ids or bool(identifiers(d)&history_keys) or d['text'] in history_text
            if d['document_id'] in decisions:
                docs[i] = apply_access_decision(d, decisions[d['document_id']])
        grouped = group_works(docs)
        # No unresolved pair can be implicitly waived by a CLI run.
        if config.get('overlap_decisions'):
            raise ValueError('OVERLAP_ADJUDICATION_NOT_IMPLEMENTED')
        try:
            result = assign_splits(grouped['groups'], config['quotas'], config['seed'])
        except SplitError as exc:
            result = {'status':'blocked', **exc.report, 'unresolved_overlaps':grouped['unresolved_overlaps']}
            write_once(args.output/'blocked.json', json_bytes(result))
        else:
            if args.output.exists() or args.private_output.exists():
                raise FileExistsError('split outputs must be new')
            for role in ('train','dev','test'):
                destination = args.private_output if role == 'test' else args.output/role
                materialize_bundle({**result,'role':role}, destination)
            summary = {k:v for k,v in result.items() if k != 'documents'}
            summary['status'] = 'ready'
            write_once(args.output/'manifest.json', json_bytes(summary))
            result = summary
    elif args.data_command == 'build':
        result = build_corpus(args.manifest, args.output)
    else:
        result = (acquire_sources(args.config, args.output) if args.data_command == 'acquire'
                  else audit_sources(args.manifest, args.output))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('success', 'ready', 'built_pending_eligibility') else 2
