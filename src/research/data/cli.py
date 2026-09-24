import json
from pathlib import Path

from .acquire import acquire_sources, audit_sources


def register(subparsers):
    parser = subparsers.add_parser('data')
    commands = parser.add_subparsers(dest='data_command', required=True)
    for name, argument in [('acquire', 'config'), ('audit', 'manifest')]:
        p = commands.add_parser(name)
        p.add_argument('--' + argument, type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.set_defaults(func=run)


def run(args):
    result = (acquire_sources(args.config, args.output) if args.data_command == 'acquire'
              else audit_sources(args.manifest, args.output))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('success', 'ready') else 2
