import json
from pathlib import Path

from .acquire import acquire_sources, audit_sources
from .manifest import digest, json_bytes, read_jsonl, write_once
from .corpus import apply_access_decision, build_corpus, load_corpus
from .splits import SplitError, assign_splits, group_works, identifiers
from .bundles import load_bundle, materialize_bundle


def register(subparsers):
    parser = subparsers.add_parser('data')
    commands = parser.add_subparsers(dest='data_command', required=True)
    for name, argument in [('acquire', 'config'), ('audit', 'manifest'), ('build', 'manifest'), ('ecosystems','config')]:
        p = commands.add_parser(name)
        p.add_argument('--' + argument, type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.set_defaults(func=run)
    p = commands.add_parser('split')
    for argument in ('corpus','config','output','private-output'):
        p.add_argument('--'+argument, type=Path, required=True)
    p.set_defaults(func=run)
    for name in ('openalex-snippets', 'exposures', 'supplemental'):
        p = commands.add_parser(name)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'exposures':
            p.add_argument('--diagnostic', type=Path)
        if name == 'supplemental':
            p.add_argument('--exposures', type=Path, required=True)
        p.set_defaults(func=run_workflow)
    for name in ('supplemental-tasks', 'supplemental-readiness'):
        p = commands.add_parser(name)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'supplemental-readiness':
            p.add_argument('--snapshot', type=Path)
        p.set_defaults(func=run_workflow)


def _effective_config(path):
    payload = path.read_bytes()
    config = json.loads(payload)
    if not isinstance(config, dict) or 'config_source' in config:
        raise ValueError('workflow config must be an object without reserved config_source')
    return {**config, 'config_source': {'path': str(path.resolve()),
                                      'sha256': digest(payload), 'bytes_utf8': payload.decode('utf-8')}}


def run_workflow(args):
    try:
        result = _dispatch_workflow(args)
    except (ValueError, OSError) as exc:
        result = {'status': 'failed', 'error_type': type(exc).__name__, 'reason': str(exc)}
    print(json.dumps(result, indent=2))
    complete = {'complete', 'completed', 'ready', 'ready_for_annotation', 'reported'}
    return 0 if result['status'] in complete else 2


def _dispatch_workflow(args):
    if args.data_command == 'openalex-snippets':
        from .openalex_snippets import collect_snippets
        result = collect_snippets(_effective_config(args.config), args.output)
    elif args.data_command == 'exposures':
        from .exposure import build_exposures
        config = _effective_config(args.config)
        if args.diagnostic is not None:
            manifest = (args.diagnostic / 'manifest.json').resolve()
            config['inputs'] = [*config.get('inputs', []), {
                'kind': 'bundle_manifest', 'path': str(manifest),
                'sha256': digest(manifest.read_bytes()), 'role': 'diagnostic'}]
        result = build_exposures(config, args.output)
    elif args.data_command == 'supplemental-tasks':
        from ..annotations.supplemental import prepare_supplemental_tasks
        result = prepare_supplemental_tasks(args.bundle, args.output)
    elif args.data_command == 'supplemental':
        from .supplemental import collect_supplemental
        config = _effective_config(args.config)
        exposures = args.exposures.resolve()
        config.update(exposures=str(exposures),
                      exposures_manifest_sha256=digest((exposures / 'manifest.json').read_bytes()))
        result = collect_supplemental(config, args.output)
    elif args.data_command == 'supplemental-readiness':
        from .supplemental_readiness import assess_readiness
        result = assess_readiness(args.bundle, args.snapshot, args.output)
    else:
        raise ValueError('unknown data workflow command')
    return result


def run(args):
    if args.data_command == 'split':
        from .exposure import apply_split_exposures, load_exposures, verify_exposure_sources
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
        for path in config.get('training_reservations',[]):
            _,reserved = load_bundle(Path(path),roles=('train',))
            for d in reserved:
                docs.append({**d,'document_id':'exposure:'+d['document_id']+'|'+d['text_revision'],
                             'development_exposed':True,'fulltext_eligible':False})
        exposure_specs = config.get('exposure_bundles', [])
        if not isinstance(exposure_specs, list):
            raise ValueError('exposure_bundles must be a list of pinned bundles')
        exposures, exposure_provenance = [], []
        for spec in exposure_specs:
            if (not isinstance(spec, dict) or not isinstance(spec.get('path'), str)
                    or not spec['path'].strip() or not isinstance(spec.get('sha256'), str)
                    or len(spec['sha256']) != 64 or set(spec['sha256']) - set('0123456789abcdef')):
                raise ValueError('exposure bundle requires path and SHA-256')
            bundle = Path(spec['path']).resolve()
            if digest((bundle / 'manifest.json').read_bytes()) != spec['sha256']:
                raise ValueError('exposure bundle manifest hash mismatch')
            loaded = load_exposures(bundle)
            if loaded['_manifest_sha256'] != spec['sha256']:
                raise ValueError('exposure bundle manifest hash mismatch')
            exposures.append(loaded)
            exposure_provenance.append({'path': str(bundle), 'sha256': spec['sha256']})
            docs = apply_split_exposures(docs, loaded)
        grouped = group_works(docs)
        # No unresolved pair can be implicitly waived by a CLI run.
        if config.get('overlap_decisions'):
            raise ValueError('OVERLAP_ADJUDICATION_NOT_IMPLEMENTED')
        try:
            result = assign_splits(grouped['groups'], config['quotas'], config['seed'])
        except SplitError as exc:
            result = {'status':'blocked', **exc.report, 'unresolved_overlaps':grouped['unresolved_overlaps']}
            if exposures:
                result['exposure_bundles'] = exposure_provenance
            for loaded in exposures:
                verify_exposure_sources(loaded)
            write_once(args.output/'blocked.json', json_bytes(result))
        else:
            if args.output.exists() or args.private_output.exists():
                raise FileExistsError('split outputs must be new')
            for role in ('train','dev','test'):
                destination = args.private_output if role == 'test' else args.output/role
                for loaded in exposures:
                    verify_exposure_sources(loaded)
                materialize_bundle({**result,'role':role}, destination)
            summary = {k:v for k,v in result.items() if k != 'documents'}
            summary['status'] = 'ready'
            if exposures:
                summary['exposure_bundles'] = exposure_provenance
            for loaded in exposures:
                verify_exposure_sources(loaded)
            write_once(args.output/'manifest.json', json_bytes(summary))
            result = summary
    elif args.data_command == 'ecosystems':
        from .ecosystems import collect_pilot
        result = collect_pilot(json.loads(args.config.read_bytes()),args.output)
        result = {k:v for k,v in result.items() if k!='files'}
    elif args.data_command == 'build':
        result = build_corpus(args.manifest, args.output)
    else:
        result = (acquire_sources(args.config, args.output) if args.data_command == 'acquire'
                  else audit_sources(args.manifest, args.output))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('success', 'ready', 'built_pending_eligibility','ready_for_annotation') else 2
