"""Immutable POC stage fits and composed checkpoint publication."""

from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import torch
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup

from ..data.manifest import digest, json_bytes, verified_path, write_once
from .attribute_features import CONTEXT_POLICY, MARKERS, STAGES, build_attribute_features, summarize_support
from .attribute_models import AttributeModel, STAGE_LABELS, attribute_loss, build_attribute_model
from .data import load_training_data
from .decode import inference_decoder
from .features import LABELS, build_token_features
from .losses import partial_token_loss
from .runner import BASE_MODEL, BASE_REVISION, CAPABILITIES, _batch, choose_device, save_detector, validate_recipe

FIXED_RECIPE = {'seed': 42, 'epochs': 10, 'learning_rate': 5e-5, 'weight_decay': .01,
                'warmup_ratio': .1, 'max_grad_norm': 1., 'min_steps_per_epoch': 10}
CAPABILITY_STAGE = {'version_linking': 'linker', 'intent': 'intent', 'sentiment': 'sentiment', 'aliases': 'alias'}


def _recipe(config, device):
    validate_recipe(config)
    if not config.get('plumbing_test'):
        if any(config.get(key) != value for key, value in FIXED_RECIPE.items()) or (config['microbatch_size'], config['gradient_accumulation']) not in ((2, 16), (1, 32)):
            raise ValueError('fixed POC recipe required')
        if device != 'mps':
            raise ValueError('real POC fits require explicit MPS device')
    if config.get('base_model') != BASE_MODEL or config.get('base_revision') != BASE_REVISION:
        raise ValueError('pinned SciBERT base required')


def _eligible(row):
    return any(row.get('known', row.get('active_mask', [])))


def _unique_features(features):
    seen = {}
    for row in features:
        if _eligible(row):
            identity = row['feature_id']
            if identity in seen and seen[identity] != row:
                raise ValueError('conflicting duplicate feature')
            seen[identity] = row
    return [seen[key] for key in sorted(seen)]


def sample_features(features: list[dict], seed: int, draws: int) -> list[dict]:
    if type(draws) is not int or draws < 0:
        raise ValueError('nonnegative draws required')
    pools = {}
    for row in _unique_features(features):
        source, work = row['source'], row.get('work_group_id', row['document_id'])
        pools.setdefault(source, {}).setdefault(work, []).append(row)
    if not pools:
        raise ValueError('no eligible supervised features')
    for works in pools.values():
        for examples in works.values():
            examples.sort(key=lambda row: row['feature_id'])
    rng, sources = random.Random(seed), sorted(pools)
    result = []
    for _ in range(draws):
        works = pools[rng.choice(sources)]
        examples = works[rng.choice(sorted(works))]
        result.append(rng.choice(examples))
    return result


def batch_attribute_features(rows, pad_id, device):
    length = max(len(row['input_ids']) for row in rows)
    result = {}
    for key in ('input_ids', 'attention_mask', 'first_mask', 'second_mask', 'context_mask'):
        padding = pad_id if key == 'input_ids' else 0
        values = [list(row[key]) + [padding] * (length - len(row[key])) for row in rows]
        result[key] = torch.tensor(values, device=device, dtype=torch.bool if key.endswith('_mask') and key != 'attention_mask' else torch.long)
    for key in ('distance_bucket', 'order_flag', 'targets', 'known'):
        if key in rows[0]:
            result[key] = torch.tensor([row[key] for row in rows], device=device,
                                      dtype=torch.bool if key == 'known' else torch.long)
    return result


def _weight_hash(model):
    state = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        state.update(name.encode())
        state.update(value.detach().cpu().contiguous().numpy().tobytes())
    return state.hexdigest()


def _detector_support(rows):
    counts = Counter()
    works = {key: set() for key in ('SOFTWARE', 'VERSION', 'outside')}
    for row in rows:
        for active, allowed in zip(row['active_mask'], row['allowed_labels']):
            if not active:
                continue
            for key, index in [('SOFTWARE', 1), ('VERSION', 3)]:
                if allowed.count(True) == 1 and allowed[index]:
                    counts[key] += 1
                    works[key].add(row.get('work_group_id', row['document_id']))
            if allowed[0] and not all(allowed):
                counts['outside'] += 1
                works['outside'].add(row.get('work_group_id', row['document_id']))
    reasons = ['missing_supervision:' + key for key in works if counts[key] == 0]
    return {'trainable': not reasons, 'counts': dict(counts), 'works': {key: len(value) for key, value in works.items()},
            'unavailable_reasons': reasons,
            'coverage_warnings': [{'label': key, 'count': counts[key], 'works': len(value),
                'reason': 'small_or_concentrated_support'} for key, value in works.items() if counts[key] < 10 or len(value) < 3]}


def train_stage_model(model, tokenizer, features, config, stage, device='mps', on_epoch=None):
    if stage not in ('detector', *STAGES):
        raise ValueError('unknown training stage')
    _recipe(config, device)
    features = _unique_features(features)
    support = _detector_support(features) if stage == 'detector' else summarize_support({stage: features})[stage]
    if not support['trainable']:
        raise ValueError('insufficient active class support: ' + ', '.join(support['unavailable_reasons']))
    device = choose_device(device)
    torch.manual_seed(config['seed'])
    model.to(device=device, dtype=torch.float32).train()
    before = _weight_hash(model)
    micro, accumulation = config['microbatch_size'], config['gradient_accumulation']
    effective = micro * accumulation
    steps_per_epoch = max(config.get('min_steps_per_epoch', 10), math.ceil(len(features) / effective))
    total_steps = steps_per_epoch * config['epochs']
    draws = sample_features(features, config['seed'], total_steps * effective)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup = min(total_steps - 1, math.ceil(total_steps * config['warmup_ratio']))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    started, cursor, effective_steps, steps, epochs = time.monotonic(), 0, 0, 0, []
    for epoch in range(config['epochs']):
        losses = []
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                selected = draws[cursor:cursor + micro]
                cursor += micro
                if stage == 'detector':
                    batch = _batch(selected, tokenizer.pad_token_id, device)
                    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                    loss = partial_token_loss(logits, batch['allowed_labels'], batch['active_mask'])
                else:
                    batch = batch_attribute_features(selected, tokenizer.pad_token_id, device)
                    targets, known = batch.pop('targets'), batch.pop('known')
                    loss = attribute_loss(model(**batch).logits, targets, known, stage)
                if loss is None or not torch.isfinite(loss):
                    raise ValueError('nonfinite or unsupported training loss')
                losses.append(float(loss.detach().cpu()))
                (loss / accumulation).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
            if optimizer.param_groups[0]['lr'] > 0 and float(norm.detach().cpu()) > 0:
                effective_steps += 1
            optimizer.step()
            scheduler.step()
            steps += 1
        epochs.append({'epoch': epoch + 1, 'optimizer_steps': steps, 'mean_training_loss': sum(losses) / len(losses),
                       'last_gradient_norm': float(norm.detach().cpu()), 'elapsed_seconds': time.monotonic() - started})
        if on_epoch:
            on_epoch(epochs[-1])
    after = _weight_hash(model)
    if effective_steps == 0 or before == after:
        raise ValueError('untrained stage: no effective updates or changed weights')
    return {'optimizer_steps': steps, 'effective_optimizer_steps': effective_steps, 'warmup_steps': warmup,
        'weights_changed': before != after, 'initial_weights_sha256': before, 'final_weights_sha256': after,
        'epochs': epochs, 'device': device, 'dtype': 'float32', 'effective_batch_size': effective,
        'steps_per_epoch': steps_per_epoch, 'eligible_examples': len(features), 'sample_draws': len(draws),
        'source_draw_counts': dict(sorted(Counter(row['source'] for row in draws).items())),
        'work_draw_counts': dict(sorted(Counter(row.get('work_group_id', row['document_id']) for row in draws).items())),
        'feature_draw_counts': dict(sorted(Counter(row['feature_id'] for row in draws).items())),
        'task_draw_counts': dict(sorted(Counter(task for row in draws for task in
            row.get('provenance', {}).get('task_ids', [row.get('provenance', {}).get('task_id', 'unattributed')])).items())),
        'sampler': 'uniform_source_work_eligible_feature_with_replacement', 'support': support,
        'elapsed_seconds': time.monotonic() - started, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'mps_determinism': 'not_guaranteed' if device == 'mps' else 'not_applicable',
        'selection': 'fixed_final_epoch_no_dev', 'quality_evaluated': False}


def _trained(report):
    if report.get('optimizer_steps', 0) < 1 or report.get('effective_optimizer_steps', 0) < 1 or report.get('weights_changed') is not True:
        raise ValueError('untrained stage cannot be published')


def _validate_support(support, stage):
    labels = STAGE_LABELS[stage]
    if support.get('trainable') is not True or support.get('labels') != labels or set(support.get('counts', {})) != set(labels):
        raise ValueError('incompatible attribute support')
    for label in labels:
        values = support['counts'][label]
        counts = [values.get('count')] if stage == 'sentiment' else [values.get('positive'), values.get('negative')]
        if any(type(count) is not int or count < 1 for count in counts) or support.get('capabilities', {}).get(label) is not True:
            raise ValueError('insufficient active class support')


def _validate_policy(policy):
    if any(policy.get(key) != value for key, value in CONTEXT_POLICY.items()):
        raise ValueError('incompatible attribute feature policy')
    markers = policy.get('marker_token_ids', [])
    if len(markers) != 4 or any(type(token) is not int or token < 0 for token in markers) or len(set(markers)) != 4 or not policy.get('tokenizer_class'):
        raise ValueError('incompatible frozen attribute tokenizer policy')


def save_attribute_checkpoint(model, tokenizer, output, stage, config, report, provenance):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _trained(report)
    if stage not in STAGES or model.stage != stage or not report.get('support', {}).get('trainable'):
        raise ValueError('unsupported or incompatible attribute stage')
    _validate_support(report['support'], stage)
    policy = config.get('attribute_features', {})
    _validate_policy(policy)
    if policy.get('marker_token_ids') != tokenizer.convert_tokens_to_ids(MARKERS) or policy.get('tokenizer_class') != type(tokenizer).__name__:
        raise ValueError('frozen attribute tokenizer required')
    provenance = deepcopy(provenance)
    if getattr(model.config, 'plumbing_test', False) or config.get('plumbing_test'):
        provenance['purpose'] = 'plumbing_test'
    output.mkdir(parents=True, exist_ok=False)
    model.cpu().save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    manifest = {'schema_version': 'attribute-checkpoint-1', 'status': 'trained_experimental', 'stage': stage,
        'labels': STAGE_LABELS[stage], 'recipe': config, 'attribute_features': policy, 'training': report,
        'support': report['support'], 'provenance': provenance, 'scores_calibrated': False, 'quality_evaluated': False,
        'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy')},
        'files': [{'path': path.name, 'sha256': digest(path.read_bytes())} for path in sorted(output.iterdir()) if path.is_file()]}
    for record in manifest['files']:
        verified_path(output, record)
    write_once(output / 'manifest.json', json_bytes(manifest))
    return manifest


def verify_stage_checkpoint(checkpoint, stage, *, allow_plumbing=False):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_bytes())
    schema = 'detector-checkpoint-1' if stage == 'detector' else 'attribute-checkpoint-1'
    labels = LABELS if stage == 'detector' else STAGE_LABELS.get(stage)
    if not labels or manifest.get('schema_version') != schema or manifest.get('status') != 'trained_experimental' or manifest.get('labels') != labels or manifest.get('training', {}).get('optimizer_steps', 0) < 1:
        raise ValueError('invalid or untrained stage checkpoint')
    if manifest.get('provenance', {}).get('purpose') == 'plumbing_test' and not allow_plumbing:
        raise ValueError('plumbing checkpoint is not a real extractor')
    paths = [row['path'] for row in manifest.get('files', [])]
    if len(set(paths)) != len(paths) or not {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'} <= set(paths):
        raise ValueError('incomplete stage checkpoint')
    for row in manifest['files']:
        verified_path(checkpoint, row)
    if stage == 'detector':
        inference_decoder(manifest)
        if manifest.get('capabilities') != CAPABILITIES:
            raise ValueError('incompatible detector capabilities')
    else:
        _trained(manifest['training'])
        if manifest.get('stage') != stage or not manifest.get('support', {}).get('trainable'):
            raise ValueError('unsupported attribute stage checkpoint')
        _validate_support(manifest['support'], stage)
        if manifest['training'].get('support') != manifest['support']:
            raise ValueError('inconsistent attribute support')
        config = json.loads((checkpoint / 'config.json').read_bytes())
        if config.get('attribute_stage') != stage or config.get('id2label') != {str(i): label for i, label in enumerate(labels)}:
            raise ValueError('incompatible attribute model labels')
        if config.get('plumbing_test') and not allow_plumbing:
            raise ValueError('plumbing model is not a real extractor')
        if manifest.get('attribute_features') != manifest.get('recipe', {}).get('attribute_features'):
            raise ValueError('incompatible frozen attribute features')
        _validate_policy(manifest['attribute_features'])
    return manifest


class AttributeCheckpoint:
    def __init__(self, checkpoint, device='cpu', *, allow_plumbing=False):
        checkpoint = Path(checkpoint)
        stage = json.loads((checkpoint / 'manifest.json').read_bytes()).get('stage')
        self.manifest = verify_stage_checkpoint(checkpoint, stage, allow_plumbing=allow_plumbing)
        self.identity = digest((checkpoint / 'manifest.json').read_bytes())
        self.device = choose_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True, trust_remote_code=False)
        policy = self.manifest['attribute_features']
        if not self.tokenizer.is_fast or self.tokenizer.do_lower_case or type(self.tokenizer).__name__ != policy.get('tokenizer_class') or self.tokenizer.convert_tokens_to_ids(MARKERS) != policy.get('marker_token_ids'):
            raise ValueError('incompatible attribute tokenizer')
        self.model = AttributeModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True,
            attn_implementation='eager').to(self.device).eval()
        if self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            raise ValueError('attribute marker embeddings mismatch')


def _load_base(config):
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(base, do_lower_case=False, use_fast=True,
        local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.do_lower_case:
        raise ValueError('cased fast tokenizer required')
    return base, tokenizer


def fit_stage(stage: str, data: Path, config: dict, output: Path, device: str = 'mps') -> dict:
    if stage not in ('detector', *STAGES):
        raise ValueError('unknown training stage')
    output, data = Path(output), Path(data)
    if output.exists():
        raise FileExistsError(output)
    _recipe(config, device)
    manifest, documents, items = load_training_data(data)
    initial_manifest_hash = digest((data / 'manifest.json').read_bytes())
    base, tokenizer = _load_base(config)
    effective_config, rows, reports, exclusions = deepcopy(config), [], [], []
    for document in documents:
        own = [item for item in items if item['task']['document_id'] == document['document_id']]
        if stage == 'detector':
            result = build_token_features(document, [o for item in own for o in item['annotation']['occurrences']],
                [r for item in own for r in item['annotation']['covered_regions']], tokenizer, config)
            for index, row in enumerate(result['windows']):
                rows.append({**row, 'source': document['source'], 'document_id': document['document_id'],
                    'work_group_id': document.get('work_group_id', document['document_id']),
                    'feature_id': document['document_id'] + ':window:' + str(index),
                    'provenance': {'task_ids': [item['task']['task_id'] for item in own]}})
            reports.append({'document_id': document['document_id'], 'token_count': result['token_count'], 'window_count': len(result['windows']), 'exclusions': result['exclusions']})
        else:
            result = build_attribute_features(document, own, tokenizer, effective_config)
            effective_config['attribute_features'] = result['config']
            rows.extend(result['features'][stage])
            exclusions.extend(result['excluded'])
    rows = _unique_features(rows)
    support = _detector_support(rows) if stage == 'detector' else summarize_support({'features': {stage: rows}, 'excluded': exclusions})[stage]
    if not support['trainable']:
        return {'stage': stage, 'status': 'unavailable', 'support': support, 'data_manifest_sha256': initial_manifest_hash}
    load_training_data(data)
    if digest((data / 'manifest.json').read_bytes()) != initial_manifest_hash:
        raise ValueError('changed training bundle before fit')
    effective_batch = config['microbatch_size'] * config['gradient_accumulation']
    planned_steps = max(config.get('min_steps_per_epoch', 10), math.ceil(len(rows) / effective_batch)) * config['epochs']
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    print(json.dumps({'event': 'stage_preflight', 'stage': stage, 'eligible_examples': len(rows),
        'planned_optimizer_steps': planned_steps, 'planned_draws': planned_steps * effective_batch,
        'free_artifact_bytes': shutil.disk_usage(existing_parent).free, 'device': device,
        'support': support}), file=sys.stderr, flush=True)
    torch.manual_seed(config['seed'])
    if stage == 'detector':
        model = BertForTokenClassification.from_pretrained(base, num_labels=5, id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)}, classifier_dropout=.1,
            local_files_only=True, trust_remote_code=False, attn_implementation='eager')
    else:
        model = build_attribute_model({**effective_config, '_base_path': str(base)}, stage)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    report = train_stage_model(model, tokenizer, rows, effective_config, stage, device,
        on_epoch=lambda row: print(json.dumps({'event': 'stage_epoch', 'stage': stage, **row}), file=sys.stderr, flush=True))
    load_training_data(data)
    if digest((data / 'manifest.json').read_bytes()) != initial_manifest_hash:
        raise ValueError('changed training bundle before checkpoint publication')
    provenance = {'purpose': 'plumbing_test' if config.get('plumbing_test') else 'experimental_real_paper_' + stage,
        'data_manifest_sha256': initial_manifest_hash, 'data_path': str(data.resolve()), 'data_summary': manifest['summary'],
        'source_hashes': manifest.get('sources', []), 'exposure_hashes': manifest.get('exposure_bundles', []),
        'features_sha256': digest(json_bytes(rows)), 'excluded_sha256': digest(json_bytes(exclusions)),
        'features': reports, 'excluded': exclusions, 'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
        'base_files': [{'path': p.name, 'sha256': digest(p.read_bytes())} for p in sorted(base.iterdir())
                       if p.name in ('config.json', 'vocab.txt', 'pytorch_model.bin', 'model.safetensors')],
        'code_files': [{'path': p.name, 'sha256': digest(p.read_bytes())} for p in sorted(Path(__file__).parent.glob('*.py'))]}
    if stage == 'detector':
        return save_detector(model, tokenizer, output, effective_config, report, provenance)
    return save_attribute_checkpoint(model, tokenizer, output, stage, effective_config, report, provenance)


def _pipeline_capabilities(stages):
    capabilities = {**CAPABILITIES}
    for capability, stage in CAPABILITY_STAGE.items():
        capabilities[capability] = stages[stage]['status'] == 'available'
    capabilities['full_contract'] = all(capabilities[key] for key in ('software_spans', 'version_spans', 'version_linking', 'intent', 'sentiment'))
    return capabilities


def publish_detector_decoder(source: Path, output: Path, decoder: str, *, allow_plumbing=False) -> dict:
    """Copy verified weights into a distinct, inference-only checkpoint variant."""
    source, output = Path(source), Path(output)
    inference_decoder({'inference': {'decoder': decoder}})
    if os.path.lexists(output):
        raise FileExistsError(output)
    parent_hash = digest((source / 'manifest.json').read_bytes())
    manifest = verify_stage_checkpoint(source, 'detector', allow_plumbing=allow_plumbing)
    if digest((source / 'manifest.json').read_bytes()) != parent_hash:
        raise ValueError('changed detector manifest during verification')
    manifest['inference'] = {'decoder': decoder}
    manifest['inference_provenance'] = {'parent_manifest_sha256': parent_hash,
        'parent_checkpoint': str(source.resolve()), 'weights_retrained': False,
        'code_files': [{'path': name, 'sha256': digest(Path(__file__).with_name(name).read_bytes())}
                       for name in ('decode.py', 'features.py', 'predict.py')]}
    output.mkdir(parents=True, exist_ok=False)
    for record in manifest['files']:
        payload = verified_path(source, record).read_bytes()
        if digest(payload) != record['sha256']:
            raise ValueError('changed detector source before copy')
        write_once(output / record['path'], payload)
    if digest((source / 'manifest.json').read_bytes()) != parent_hash:
        raise ValueError('changed detector manifest before publication')
    write_once(output / 'manifest.json', json_bytes(manifest))
    return manifest


def publish_pipeline(detector: Path, stages: dict[str, Path], output: Path, *, allow_plumbing=False) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if any(stage not in STAGES for stage in stages):
        raise ValueError('unknown pipeline stage')
    entries = {}
    for stage in ('detector', *STAGES):
        path = detector if stage == 'detector' else stages.get(stage)
        if path is None:
            entries[stage] = {'status': 'unavailable', 'reason': 'missing_stage', 'unavailable_reasons': ['missing_stage']}
            continue
        path = Path(path).resolve()
        manifest = verify_stage_checkpoint(path, stage, allow_plumbing=allow_plumbing)
        entries[stage] = {'status': 'available', 'path': os.path.relpath(path, output.resolve()),
            'manifest_sha256': digest((path / 'manifest.json').read_bytes()), 'labels': manifest['labels'],
            'support': manifest.get('support', manifest.get('training', {}).get('support'))}
    for stage, entry in entries.items():
        if entry['status'] == 'available':
            path = (output / entry['path']).resolve()
            verify_stage_checkpoint(path, stage, allow_plumbing=allow_plumbing)
            if digest((path / 'manifest.json').read_bytes()) != entry['manifest_sha256']:
                raise ValueError('changed stage manifest before publication')
    result = {'schema_version': 'full-label-checkpoint-1', 'status': 'trained_experimental', 'stages': entries,
        'capabilities': _pipeline_capabilities(entries), 'pipeline_complete': all(row['status'] == 'available' for row in entries.values()),
        'readiness': 'full_label_ready' if all(row['status'] == 'available' for row in entries.values()) else 'partial',
        'scores_calibrated': False, 'quality_evaluated': False, 'plumbing_test': bool(allow_plumbing),
        'unavailable_stage_reasons': {stage: row['unavailable_reasons'] for stage, row in entries.items() if row['status'] == 'unavailable'}}
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'manifest.json', json_bytes(result))
    return result


def load_pipeline_manifest(checkpoint, *, allow_plumbing=False):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_bytes())
    if manifest.get('schema_version') != 'full-label-checkpoint-1' or manifest.get('status') != 'trained_experimental':
        raise ValueError('invalid full-label checkpoint')
    if manifest.get('plumbing_test') and not allow_plumbing:
        raise ValueError('plumbing bundle is not a real extractor')
    entries = manifest.get('stages', {})
    if set(entries) != {'detector', *STAGES} or entries['detector'].get('status') != 'available':
        raise ValueError('incomplete pipeline stage manifest')
    for stage, entry in entries.items():
        if entry.get('status') == 'unavailable' and stage != 'detector':
            if not entry.get('unavailable_reasons'):
                raise ValueError('missing unavailable stage reason')
            continue
        if entry.get('status') != 'available' or not isinstance(entry.get('path'), str) or Path(entry['path']).is_absolute():
            raise ValueError('invalid relative pipeline stage path')
        path = (checkpoint / entry['path']).resolve()
        if digest((path / 'manifest.json').read_bytes()) != entry.get('manifest_sha256'):
            raise ValueError('changed stage manifest hash')
        loaded = verify_stage_checkpoint(path, stage, allow_plumbing=allow_plumbing)
        if entry.get('labels') != loaded['labels'] or entry.get('support') != loaded.get('support', loaded.get('training', {}).get('support')):
            raise ValueError('incompatible pipeline stage support or labels')
    if manifest.get('capabilities') != _pipeline_capabilities(entries) or manifest.get('pipeline_complete') != all(row['status'] == 'available' for row in entries.values()):
        raise ValueError('incompatible pipeline capability map')
    return manifest
