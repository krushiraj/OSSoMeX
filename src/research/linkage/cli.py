import json
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser('linkage', help='paper-to-software link gold sets and evaluation')
    commands = parser.add_subparsers(dest='linkage_command', required=True)
    p = commands.add_parser('gold-tasks')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-domain', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude', default='', help='comma-separated paper IDs to leave out')
    p = commands.add_parser('resolve')
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--arm', choices=['fast-text-match', 'context'], required=True)
    p.add_argument('--threshold', type=float, default=2.2)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True, nargs='+')
    p = commands.add_parser('lockbox')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('end-to-end')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--checkpoints', nargs='+', default=['006', '012'])
    for command in commands.choices.values():
        command.set_defaults(func=run)


def run(args) -> int:
    if args.linkage_command == 'resolve':
        return _resolve(args)
    if args.linkage_command == 'evaluate':
        return _evaluate(args)
    if args.linkage_command == 'end-to-end':
        from .lockbox import evaluate_lockbox
        print(json.dumps({'status': 'success', **evaluate_lockbox(args.run, args.registry, args.gold,
                                                                 args.checkpoints)}, indent=2))
        return 0
    if args.linkage_command == 'lockbox':
        from .lockbox import build_lockbox
        result = build_lockbox(json.loads(args.config.read_bytes()), args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'rejected')}, indent=2))
        return 0 if result['status'] == 'success' else 2
    from .gold import build_sofair_tasks
    exclude = {x.strip() for x in args.exclude.split(',') if x.strip()}
    result = build_sofair_tasks(args.archive, args.registry, args.output, args.per_domain, args.seed, exclude)
    print(json.dumps({k: v for k, v in result.items() if k != 'paper_ids'}, indent=2))
    return 0 if result['status'] == 'success' else 2


def _resolve(args) -> int:
    from ..data.manifest import read_jsonl, write_jsonl
    from .gold import load_index
    from .resolve import ContextResolver, FastTextMatch
    index = load_index(args.registry)
    resolver = FastTextMatch(index) if args.arm == 'fast-text-match' else ContextResolver(index, args.threshold)
    predictions = [{**resolver.resolve(t), 'arm': args.arm} for t in read_jsonl(args.tasks)]
    write_jsonl(args.output, predictions)
    print(json.dumps({'status': 'success', 'arm': args.arm, 'tasks': len(predictions),
                      'linked': sum(p['entity_id'] is not None for p in predictions)}, indent=2))
    return 0


def _evaluate(args) -> int:
    from ..data.manifest import read_jsonl
    from .evaluate import bootstrap, link_metrics
    from .lockbox import _paper_bootstrap, paper_link_metrics
    papers = {t['task_id']: t['paper_id'] for t in read_jsonl(args.tasks)}
    gold = [{**g, 'paper_id': papers[g['task_id']]} for g in read_jsonl(args.gold)]
    result = {'status': 'success', 'gold_tasks': len(gold), 'arms': {}}
    for path in args.predictions:
        predictions = read_jsonl(path)
        arm = predictions[0].get('arm', path.stem) if predictions else path.stem
        result['arms'][arm] = {**link_metrics(gold, predictions), 'ci95': bootstrap(gold, predictions)}
        pairs = [(papers[p['task_id']], p['entity_id']) for p in predictions if p['entity_id']]
        result['arms'][arm]['paper_level'] = {**paper_link_metrics(gold, pairs), 'ci95': _paper_bootstrap(gold, pairs)}
    print(json.dumps(result, indent=2))
    return 0
