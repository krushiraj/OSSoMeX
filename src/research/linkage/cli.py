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
    for command in commands.choices.values():
        command.set_defaults(func=run)


def run(args) -> int:
    from .gold import build_sofair_tasks
    exclude = {x.strip() for x in args.exclude.split(',') if x.strip()}
    result = build_sofair_tasks(args.archive, args.registry, args.output, args.per_domain, args.seed, exclude)
    print(json.dumps({k: v for k, v in result.items() if k != 'paper_ids'}, indent=2))
    return 0 if result['status'] == 'success' else 2
