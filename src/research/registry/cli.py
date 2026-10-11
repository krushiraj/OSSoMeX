import json
from pathlib import Path


def register(subparsers):
    parser = subparsers.add_parser('registry', help='software entity registry for mention resolution')
    commands = parser.add_subparsers(dest='registry_command', required=True)
    p = commands.add_parser('pull')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('build')
    p.add_argument('--run', type=Path, required=True)
    p = commands.add_parser('audit-sample')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--top', type=int, default=200)
    p.add_argument('--random', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('audit-report')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p = commands.add_parser('coverage')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--names', type=Path, required=True)
    for command in commands.choices.values():
        command.set_defaults(func=run)


def run(args) -> int:
    if args.registry_command == 'pull':
        from .pull import pull
        result = pull(json.loads(args.config.read_bytes()), args.output)
        result = {'status': result['status'], 'pages': {s: v['pages'] for s, v in result['sources'].items()}}
    elif args.registry_command == 'build':
        from .build import build_run
        result = build_run(args.run)
    elif args.registry_command == 'audit-sample':
        from .audit import sample_run
        config = json.loads(args.config.read_bytes())
        result = sample_run(args.run, args.top, args.random, args.seed,
                            policy={'user_agent': config['user_agent']}, delay=config.get('delay_seconds', 0.2))
    elif args.registry_command == 'audit-report':
        from .audit import report_run
        result = report_run(args.run, args.labels)
    else:
        from .build import coverage_run
        result = coverage_run(args.run, args.names)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'success' else 2
