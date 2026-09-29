"""Local comparison commands; tokenizers never acquire remote files."""

import json
from pathlib import Path
import time

from ..data.artifacts import atomic_write_new
from .backends import REPO_ROOT, make_backend, resolve_arm, sha256
from .report import build_report
from .runner import (code_identity, freeze_configuration, freeze_prompts, json_bytes, load_backend,
                     prepare_arms, publish, run_comparison, verify_files, verify_source, verify_tokenizer)
from .windows import freeze_windows


def register(subparsers) -> None:
    parser = subparsers.add_parser('comparison', help='Reproducible local span comparison diagnostics')
    commands = parser.add_subparsers(dest='comparison_command', required=True)
    for name, func in [('preflight', cmd_preflight), ('run', cmd_run), ('report', cmd_report)]:
        command = commands.add_parser(name)
        command.add_argument('--output', required=True)
        if name == 'report':
            command.add_argument('--run', required=True)
            command.add_argument('--references')
        else:
            command.add_argument('--config', required=True)
        if name == 'run':
            command.add_argument('--input', required=True)
        command.set_defaults(func=func)
    links = commands.add_parser('links', help='Score native version ownership separately from span detection')
    for key in ('run', 'references', 'output'):
        links.add_argument('--' + key, required=True)
    links.add_argument('--full-label', action='append', default=[], metavar='ARM=JSONL',
                       help='Full-label CLI predictions on the exact frozen input population')
    links.set_defaults(func=cmd_links)


def _source(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': sha256(data), 'bytes_utf8': data.decode('utf-8')}


def _config(path):
    source = _source(path)
    config = json.loads(source['bytes_utf8'])
    if not isinstance(config, dict):
        raise ValueError('comparison config must be an object')
    return config, source


def _arms(config, context):
    if not isinstance(config.get('arms'), list):
        raise ValueError('comparison config requires arms')
    return [{**arm, 'comparison_context': context} for arm in config['arms']]


def load_tokenizer(config):
    checkpoint = config.get('tokenizer_checkpoint')
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError('pinned local tokenizer_checkpoint required')
    path = (REPO_ROOT / checkpoint).resolve()
    data = (path / 'manifest.json').read_bytes()
    manifest = json.loads(data)
    if manifest.get('schema_version') != 'detector-checkpoint-1':
        raise ValueError('tokenizer checkpoint manifest required')
    files = manifest.get('files', [])
    names = [record['path'] for record in files]
    if len(set(names)) != len(names) or not {'tokenizer.json', 'tokenizer_config.json', 'config.json'} <= set(names):
        raise ValueError('incomplete pinned tokenizer files')
    identity = {'checkpoint': str(path), 'manifest_sha256': sha256(data), 'manifest': manifest,
                'files': files, 'local_files_only': True, 'trust_remote_code': False}
    verify_tokenizer(identity)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.do_lower_case:
        raise ValueError('fast cased tokenizer required')
    verify_tokenizer(identity)
    return tokenizer, identity


def cmd_preflight(args):
    config, source = _config(args.config)
    context = {'config_source': source, 'warmup_text': config.get('warmup_text', 'We used NumPy.')}
    arms, context = prepare_arms(_arms(config, context))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files, records = [], []
    freeze_configuration(output, files, arms, context)
    for arm in arms:
        backend, loaded, elapsed = load_backend(arm, make_backend, time.monotonic)
        record = {'arm_id': arm['arm_id'], **loaded, 'load_seconds': elapsed}
        try:
            freeze_prompts(output, files, arm['arm_id'], loaded['identity'])
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    record['close_error'] = str(exc) or type(exc).__name__
        records.append(record)
    publish(output, files, 'preflight.json', json_bytes(records))
    partial = any(record['status'] == 'unavailable' or record.get('close_error') for record in records)
    manifest = {'schema_version': 'comparison-preflight-1', 'status': 'partial' if partial else 'complete',
                'exit_code': int(partial), 'arms': records, 'arm_order': [arm['arm_id'] for arm in arms],
                'code': code_identity(), 'provenance': context, 'files': files}
    for arm in arms:
        resolve_arm(arm)
    verify_source(source)
    verify_files(output, files)
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    print(json.dumps({'output': args.output, 'status': manifest['status'], 'exit_code': manifest['exit_code']}))
    return manifest['exit_code']


def cmd_run(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    config, source = _config(args.config)
    input_source = _source(args.input)
    documents = [json.loads(line) for line in input_source['bytes_utf8'].splitlines()]
    if not documents:
        raise ValueError('nonempty comparison inputs required')
    tokenizer, identity = load_tokenizer(config)
    windows = freeze_windows(documents, tokenizer, max_content_tokens=config.get('max_content_tokens', 480),
                             overlap_tokens=config.get('overlap_tokens', 64))
    context = {'config_source': source, 'input_source': input_source, 'tokenizer': identity,
               'max_content_tokens': config.get('max_content_tokens', 480),
               'overlap_tokens': config.get('overlap_tokens', 64), 'warmup_text': config.get('warmup_text', 'We used NumPy.')}
    manifest = run_comparison(documents, windows, _arms(config, context), Path(args.output))
    print(json.dumps({'output': args.output, 'status': manifest['status'], 'exit_code': manifest['exit_code']}))
    return manifest['exit_code']


def cmd_report(args):
    result = build_report(Path(args.run), Path(args.output), references=Path(args.references) if args.references else None)
    print(json.dumps({'output': args.output, 'status': result['status'], 'mode': result['mode']}))
    return 0


def cmd_links(args):
    from .link_report import build_link_report
    build_link_report(args.run, args.references, args.full_label, args.output)
    print(json.dumps({'output': args.output, 'status': 'reported', 'mode': 'version_link_diagnostic'}))
    return 0
