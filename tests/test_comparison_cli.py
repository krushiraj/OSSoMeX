import argparse
import importlib
import json

import pytest


def parse(argv):
    cli = importlib.import_module('research.comparison.cli')
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers(required=True))
    return parser.parse_args(['comparison', *argv])


def test_preflight_freezes_config_and_partial_exit(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{\n "arms": [{"arm_id":"base", "backend":"base", "config":{}},'
                      '{"arm_id":"missing", "unavailable_reason":"missing", "config":{}}]}\n')
    args = parse(['preflight', '--config', str(config), '--output', str(tmp_path / 'preflight')])
    assert args.func(args) == 1
    manifest = json.loads((tmp_path / 'preflight/manifest.json').read_text())
    assert manifest['status'] == 'partial'
    assert any((tmp_path / 'preflight' / f['path']).read_bytes() == config.read_bytes() for f in manifest['files'])


def test_run_uses_frozen_windows_and_config_without_network(tmp_path, monkeypatch):
    cli = importlib.import_module('research.comparison.cli')
    from research.contracts import text_revision
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'tokenizer_checkpoint': 'local-fixture', 'max_content_tokens': 4,
                                  'overlap_tokens': 0, 'arms': [{'arm_id': 'scibert-base', 'backend': 'base', 'config': {}}]}))
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'document_id': 'd', 'text': 'Hi', 'text_revision': text_revision('Hi')}) + '\n')
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': list(range(len(text))), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}
    monkeypatch.setattr(cli, 'load_tokenizer', lambda config: (Tokenizer(), {'fixture': True}))
    args = parse(['run', '--config', str(config), '--input', str(source), '--output', str(tmp_path / 'run')])
    assert args.func(args) == 0
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert manifest['result_count'] == 1
    assert manifest['provenance']['tokenizer'] == {'fixture': True}
    args = parse(['report', '--run', str(tmp_path / 'run'), '--output', str(tmp_path / 'report')])
    assert args.func(args) == 0


def test_tokenizer_rejects_unpinned_remote_or_missing_local_checkpoint():
    cli = importlib.import_module('research.comparison.cli')
    with pytest.raises(ValueError):
        cli.load_tokenizer({'tokenizer': {'model_id': 'remote/model'}})
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.load_tokenizer({'tokenizer_checkpoint': '/does/not/exist'})


def test_local_tokenizer_is_loaded_without_network_and_has_file_identity(tmp_path, monkeypatch):
    cli = importlib.import_module('research.comparison.cli')
    from research.comparison.backends import sha256
    import sys
    from types import SimpleNamespace
    files = []
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (tmp_path / name).write_bytes(b'{}')
        files.append({'path': name, 'sha256': sha256(b'{}')})
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 'detector-checkpoint-1', 'files': files}))
    calls = []
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(is_fast=True, do_lower_case=False)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=AutoTokenizer))
    tokenizer, identity = cli.load_tokenizer({'tokenizer_checkpoint': str(tmp_path)})
    assert calls == [(tmp_path, {'use_fast': True, 'local_files_only': True, 'trust_remote_code': False})]
    assert identity['files'] == files
    (tmp_path / 'tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='artifact'):
        cli.load_tokenizer({'tokenizer_checkpoint': str(tmp_path)})
    assert len(calls) == 1
