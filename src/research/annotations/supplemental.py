"""Prepare D17 tasks from the acquisition's frozen passage selection."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ..data.artifacts import atomic_write_jsonl_new, atomic_write_new
from ..data.manifest import digest, json_bytes
from ..data.supplemental import load_supplemental, verify_supplemental_sources
from .policies import ROOT, load_policy, snapshot_prompts
from .tasks import check_authorization, make_region_task


POLICY_PATH = ROOT / 'annotations/scibert-v2/policy-2.1-d17.md'


def frozen_tasks(loaded: dict) -> tuple[dict, list[dict]]:
    policy = load_policy(POLICY_PATH)
    if (loaded['_bundle'] / 'annotation-policy.md').read_bytes() != POLICY_PATH.read_bytes():
        raise ValueError('D17_POLICY_REQUIRED')
    documents = {row['document_id']: row for row in loaded['documents']}
    approvals = json.loads((ROOT / 'approvals.json').read_bytes())
    tasks = []
    for region in loaded['regions']:
        task = make_region_task(documents[region['document_id']], policy,
                                region['annotation_region'], region['context_span'],
                                region_kind=region['region_kind'])
        check_authorization(task, approvals)
        task['whole_passage_audit'] = region['selection_reason'] == 'random_whole_passage'
        task['supplemental_region'] = deepcopy(region)
        tasks.append(task)
    return policy, tasks


def prepare_supplemental_tasks(bundle: Path, output: Path) -> dict:
    bundle, output = Path(bundle), Path(output)
    if os.path.lexists(output):
        raise FileExistsError(output)
    loaded = load_supplemental(bundle)
    policy, tasks = frozen_tasks(loaded)
    if not tasks:
        raise ValueError('NO_SUPPLEMENTAL_TASKS')
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl_new(output / 'tasks.jsonl', tasks)
    atomic_write_new(output / 'policy.md', POLICY_PATH.read_bytes())
    with TemporaryDirectory(prefix='supplemental-prompts-') as temporary:
        staged = Path(temporary)
        prompts = snapshot_prompts(staged, policy)
        for row in prompts.values():
            atomic_write_new(output / row['path'], (staged / row['path']).read_bytes())
    files = [{'path': 'tasks.jsonl', 'sha256': digest((output / 'tasks.jsonl').read_bytes())},
             {'path': 'policy.md', 'sha256': policy['policy_hash']}]
    files.extend({'path': row['path'], 'sha256': row['sha256']} for row in prompts.values())
    manifest = {'status': 'ready_for_annotation', 'role': 'train', 'policy': policy,
        'policy_source': {'path': 'policy.md', 'sha256': policy['policy_hash'], 'source_path': str(POLICY_PATH)},
        'prompt_sources': prompts, 'task_count': len(tasks), 'files': files,
        'source_bundle_sha256': digest((bundle / 'bundle/manifest.json').read_bytes()),
        'supplemental_bundle': str(bundle.resolve()),
        'supplemental_manifest_sha256': loaded['_manifest_sha256'],
        'whole_passage_audit_ids': [t['task_id'] for t in tasks if t['whole_passage_audit']]}
    verify_supplemental_sources(loaded)
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    return manifest
