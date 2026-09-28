import importlib
import json
from pathlib import Path

import pytest

from research.annotations.snapshots import read_policy_provenance
from research.annotations.validation import store_reply
from research.cli import main
from research.contracts import FIELDS
from research.data.manifest import digest, json_bytes, read_jsonl, write_once
from test_annotation_tasks import reply
from test_supplemental_acquisition import FakeFetch, article, config


def acquired(monkeypatch, tmp_path, sources=('europepmc',), *, shared_context=False):
    from research.data import supplemental
    cfg = config(tmp_path)
    metadata, targeted, overrides, routes = [], {}, {}, {}
    for index, source in enumerate(sources, 1):
        target = ['ImageJ', 'GROMACS'][index - 1]
        targeted[target] = [{'doi': f'10.1000/p{index}'}]
        body = ('<p>1.24 ' + ' '.join(f'Unique{index}trial{n} used {target} for analysis{n}.' for n in range(6)) + '</p>'
                if shared_context else
                ''.join(f'<p>Unique{index}trial{n} used {target} 1.24 for analysis{n}.</p>' for n in range(6)))
        original = article(index).decode()
        overrides[index] = (original.split('<body>')[0] + '<body>' + body + '</body></article>').encode()
        if source == 'ecosystems':
            project = f'https://papers.ecosyste.ms/api/v1/projects/verified-{index}'
            paper = f'https://papers.ecosyste.ms/api/v1/papers/10.1000%2Fp{index}'
            cfg['ecosystems_projects'][target] = {'project_id': str(index), 'url': project}
            routes[project] = {'id': index, 'project_url': project, 'name': target}
            routes[project + '/mentions?per_page=20&page=1'] = [{'project_url': project, 'paper_url': paper}]
            routes[paper] = {'doi': f'10.1000/p{index}'}
    fake = FakeFetch(metadata, targeted, overrides)
    def fetch(url, destination, policy):
        if url not in routes:
            return fake(url, destination, policy)
        payload = json_bytes(routes[url])
        write_once(destination, payload)
        record = {'url': url, 'sha256': digest(payload), 'retrieved_at_utc': '2026-09-27T00:00:00Z'}
        write_once(destination.with_name(destination.name + '.json'), json_bytes(record))
        return record
    monkeypatch.setattr(supplemental, 'fetch_public', fetch)
    bundle = tmp_path / 'acquisition'
    supplemental.collect_supplemental(cfg, bundle)
    return bundle


def imported(monkeypatch, tmp_path, *, sources=('europepmc',), versions=False, shared_context=False):
    module = importlib.import_module('research.annotations.supplemental')
    bundle = acquired(monkeypatch, tmp_path, sources, shared_context=shared_context)
    tasks_path = tmp_path / 'tasks'
    manifest = module.prepare_supplemental_tasks(bundle, tasks_path)
    tasks = read_jsonl(tasks_path / 'tasks.jsonl')
    for task in tasks:
        value = reply(task, 'partial')
        value['annotator']['prompt_hash'] = manifest['prompt_sources']['annotate']['sha256']
        name = 'GROMACS' if 'GROMACS' in task['text'] else 'ImageJ'
        owned_start = task['annotation_region']['start'] - task['offset_base']
        owned_end = task['annotation_region']['end'] - task['offset_base']
        lo = task['offset_base'] + task['text'].index(name, owned_start, owned_end)
        version_start = task['offset_base'] + task['text'].index('1.24')
        value['occurrences'] = [{'schema_version': '2.0', 'document_id': task['document_id'],
            'text_revision': task['text_revision'], 'name': name, 'name_span': {'start': lo, 'end': lo + len(name)},
            'context_sentence': task['text'], 'context_span': task['context_span'], 'context_kind': 'sentence',
            'version_status': 'explicit' if versions else 'unannotated',
            'version_links': [{'text': '1.24', 'span': {'start': version_start, 'end': version_start + 4},
                               'status': 'explicit_remote' if shared_context else 'explicit_local'}] if versions else [],
            'known': {field: field == 'software' or field == 'versions' and versions for field in FIELDS},
            'intents': None, 'sentiment': None, 'evidence': {'intents': [], 'sentiment': []},
            'review': {'status': 'pending', 'reasons': []}}]
        value['covered_regions'] = [{**task['annotation_region'], 'status': 'partial',
                                    'fields': {field: field == 'software' for field in FIELDS}}]
        store_reply(tmp_path / 'replies', task, value)
    destination = tmp_path / 'import'
    assert main(['annotate', 'import', '--tasks', str(tasks_path), '--replies', str(tmp_path / 'replies'),
                 '--output', str(destination)]) == 0
    return bundle, manifest, destination


def test_supplemental_preserves_random_audit_ids(monkeypatch, tmp_path):
    bundle, manifest, imported_path = imported(monkeypatch, tmp_path)
    regions = read_jsonl(bundle / 'regions.jsonl')
    policy = manifest['policy']
    expected = {'task:' + digest(json_bytes([row['document_id'], row['text_revision'],
                policy['policy_version'], policy['policy_hash'], row['annotation_region']['start'],
                row['annotation_region']['end']]))[:32]
                for row in regions if row['selection_reason'] == 'random_whole_passage'}
    assert len(expected) == 3 and len(regions) == 6
    assert set(manifest['whole_passage_audit_ids']) == expected
    items = read_jsonl(imported_path / 'items.jsonl')
    assert {i['task']['task_id'] for i in items if i['task']['whole_passage_audit']} == expected
    assert all(i['task']['supplemental_region']['whole_passage_audit'] is True for i in items)
    assert manifest['status'] == 'ready_for_annotation'
    assert read_policy_provenance(tmp_path / 'tasks', manifest)['kind'] == 'frozen_policy_sources'


def test_supplemental_preserves_masks_and_source_context(monkeypatch, tmp_path):
    bundle, manifest, destination = imported(monkeypatch, tmp_path)
    parent = read_jsonl(bundle / 'bundle/documents.jsonl')[0]
    for item in read_jsonl(destination / 'items.jsonl'):
        task, annotation = item['task'], item['annotation']
        assert task['source'] == 'europepmc'
        assert task['text_license'] == 'CC-BY-4.0'
        assert task['text_revision'] == parent['text_revision']
        assert task['text'] == parent['text'][task['context_span']['start']:task['context_span']['end']]
        assert task['access_basis'] == parent['access_basis']
        assert annotation['occurrences'][0]['known']['versions'] is False
        assert annotation['occurrences'][0]['version_status'] == 'unannotated'
        assert annotation['covered_regions'][0]['fields']['versions'] is False
    assert manifest['policy']['policy_hash'] == digest(Path('annotations/scibert-v2/policy-2.1-d17.md').read_bytes())


@pytest.mark.parametrize('corruption', ['missing_prompt', 'stale_policy'])
def test_supplemental_import_rejects_provenance_corruption(monkeypatch, tmp_path, corruption):
    imported(monkeypatch, tmp_path)
    path = tmp_path / 'tasks/manifest.json'
    manifest = json.loads(path.read_bytes())
    if corruption == 'missing_prompt':
        manifest['prompt_sources'].pop('check')
    else:
        manifest['policy']['policy_hash'] = '0' * 64
    path.write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):
        main(['annotate', 'import', '--tasks', str(tmp_path / 'tasks'), '--replies', str(tmp_path / 'replies'),
              '--output', str(tmp_path / 'invalid')])
    assert not (tmp_path / 'invalid').exists()


def test_prepare_rechecks_sources_before_manifest(monkeypatch, tmp_path):
    module = importlib.import_module('research.annotations.supplemental')
    bundle = acquired(monkeypatch, tmp_path)
    snapshot = module.snapshot_prompts
    def changed(output, policy):
        result = snapshot(output, policy)
        (tmp_path / 'reserved.jsonl').write_text('{}\n')
        return result
    monkeypatch.setattr(module, 'snapshot_prompts', changed)
    with pytest.raises(ValueError):
        module.prepare_supplemental_tasks(bundle, tmp_path / 'tasks')
    assert not (tmp_path / 'tasks/manifest.json').exists()
