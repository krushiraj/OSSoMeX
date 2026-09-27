"""Fixed-recipe local detector training. No dev selection or benchmark claim."""

from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import random
import resource
import time

import torch
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup

from ..data.manifest import digest, json_bytes, write_once
from .data import load_training_data
from .features import LABELS, build_token_features
from .losses import partial_token_loss

BASE_MODEL = 'allenai/scibert_scivocab_cased'
BASE_REVISION = 'ddf0be025f8e432a1870e34811997ba6725bf04a'
CAPABILITIES = {'software_spans': True, 'version_spans': True, 'version_linking': False,
                'aliases': False, 'intent': False, 'sentiment': False, 'full_contract': False}


def choose_device(name):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name not in ('cpu', 'mps') or name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('requested local device is unavailable')
    return name


def validate_recipe(config):
    minimum = config.get('min_steps_per_epoch', 1)
    if type(minimum) is not int or not 1 <= minimum <= 10:
        raise ValueError('invalid recipe: min_steps_per_epoch')
    for key, low, high in [('epochs', 1, 10), ('microbatch_size', 1, 2), ('gradient_accumulation', 1, 32)]:
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError(f'invalid recipe: {key}')
    for key, low, high in [('learning_rate', 0, .01), ('weight_decay', -1e-12, 1),
                           ('warmup_ratio', -1e-12, 1), ('max_grad_norm', 0, 10)]:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low < value <= high:
            raise ValueError(f'invalid recipe: {key}')
    if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('invalid recipe: seed')


def _batch(windows, pad_id, device):
    length = max(len(w['input_ids']) for w in windows)
    values = {k: [] for k in ('input_ids', 'attention_mask', 'allowed_labels', 'active_mask')}
    for w in windows:
        pad = length - len(w['input_ids'])
        for key, padding in [('input_ids', pad_id), ('attention_mask', 0),
                             ('allowed_labels', [True] * 5), ('active_mask', False)]:
            values[key].append(w[key] + [padding] * pad)
    return {key: torch.tensor(value, device=device, dtype=torch.bool if key in ('allowed_labels', 'active_mask') else torch.long)
            for key, value in values.items()}


def train_model(model, tokenizer, groups, config, device, on_epoch=None):
    validate_recipe(config)
    groups = {name: [w for w in windows if any(w['active_mask'])] for name, windows in groups.items()}
    groups = {name: windows for name, windows in groups.items() if windows}
    if not groups:
        raise ValueError('no eligible supervised windows')
    torch.manual_seed(config['seed'])
    rng = random.Random(config['seed'])
    device = choose_device(device)
    model.to(device).train()
    micro, accumulation = config['microbatch_size'], config['gradient_accumulation']
    effective = micro * accumulation
    steps_per_epoch = max(config.get('min_steps_per_epoch', 1), math.ceil(sum(map(len, groups.values())) / effective))
    total_steps = steps_per_epoch * config['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup_steps = min(total_steps - 1, math.ceil(total_steps * config['warmup_ratio']))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    names, counts, epochs = sorted(groups), Counter(), []
    started = time.monotonic()
    steps, effective_steps = 0, 0
    for epoch in range(config['epochs']):
        losses = []
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                batch_windows = []
                for _ in range(micro):
                    name = rng.choice(names)
                    counts[name] += 1
                    batch_windows.append(rng.choice(groups[name]))
                batch = _batch(batch_windows, tokenizer.pad_token_id, device)
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                loss = partial_token_loss(logits, batch['allowed_labels'], batch['active_mask'])
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
        entry = {'epoch': epoch + 1, 'optimizer_steps': steps, 'mean_training_loss': sum(losses) / len(losses),
                 'last_gradient_norm': float(norm.detach().cpu()), 'elapsed_seconds': time.monotonic() - started}
        epochs.append(entry)
        if on_epoch:
            on_epoch(entry)
    return {'optimizer_steps': steps, 'effective_optimizer_steps': effective_steps,
            'warmup_steps': warmup_steps, 'epochs': epochs, 'device': device, 'dtype': 'float32',
            'effective_batch_size': effective, 'steps_per_epoch': steps_per_epoch,
            'eligible_windows': sum(map(len, groups.values())), 'draw_counts': dict(counts),
            'sampler': 'single_source_uniform_document_then_eligible_window_with_replacement',
            'elapsed_seconds': time.monotonic() - started, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'selection': 'fixed_final_epoch_no_dev', 'quality_evaluated': False}


def save_detector(model, tokenizer, output, config, report, provenance):
    output = Path(output)
    if report.get('optimizer_steps', 0) < 1 or report.get('effective_optimizer_steps', 0) < 1:
        raise ValueError('untrained detector cannot be saved as ready')
    output.mkdir(parents=True, exist_ok=False)
    model.cpu().save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    manifest = {'schema_version': 'detector-checkpoint-1', 'status': 'trained_experimental',
                'labels': LABELS, 'capabilities': CAPABILITIES, 'recipe': config, 'training': report,
                'provenance': provenance, 'scores_calibrated': False, 'quality_evaluated': False,
                'runtime': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy')},
                'files': [{'path': p.name, 'sha256': digest(p.read_bytes())} for p in sorted(output.iterdir()) if p.is_file()]}
    write_once(output / 'manifest.json', json_bytes(manifest))
    return manifest


def fit_detector(data, config, output, device='auto'):
    from huggingface_hub import snapshot_download
    validate_recipe(config)
    output, data = Path(output), Path(data)
    if output.exists():
        raise FileExistsError(output)
    if config.get('base_model') != BASE_MODEL or config.get('base_revision') != BASE_REVISION:
        raise ValueError('pinned SciBERT base required')
    manifest, documents, items = load_training_data(data)
    if {d['source'] for d in documents} != {'ecosystems'}:
        raise ValueError('first-run sampler supports one ecosystems source only')
    base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(base, do_lower_case=False, use_fast=True,
                                              local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.do_lower_case:
        raise ValueError('cased fast tokenizer required')
    groups, feature_report = {}, []
    for document in documents:
        own = [i['annotation'] for i in items if i['task']['document_id'] == document['document_id']]
        features = build_token_features(document, [o for a in own for o in a['occurrences']],
                                        [r for a in own for r in a['covered_regions']], tokenizer, config)
        groups[document['document_id']] = features['windows']
        feature_report.append({'document_id': document['document_id'], 'token_count': features['token_count'],
                               'window_count': len(features['windows']), 'exclusions': features['exclusions'],
                               'active_tokens': sum(sum(w['active_mask']) for w in features['windows']),
                               'label_support': {label: sum(active and allowed.count(True) == 1 and allowed[n]
                                   for w in features['windows'] for active, allowed in zip(w['active_mask'], w['allowed_labels']))
                                   for n, label in enumerate(LABELS)}})
    torch.manual_seed(config['seed'])
    model = BertForTokenClassification.from_pretrained(base, num_labels=5, id2label=dict(enumerate(LABELS)),
                    label2id={label: n for n, label in enumerate(LABELS)}, classifier_dropout=.1,
                    local_files_only=True, trust_remote_code=False, attn_implementation='eager')
    report = train_model(model, tokenizer, groups, config, device,
                         on_epoch=lambda row: print(json.dumps(row), flush=True))
    provenance = {'purpose': 'experimental_real_paper_detector', 'data_manifest_sha256': digest((data / 'manifest.json').read_bytes()),
                  'data_path': str(data.resolve()), 'data_summary': manifest['summary'], 'features': feature_report,
                  'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
                  'base_files': [{'path': name, 'sha256': digest((base / name).read_bytes())}
                                 for name in ('config.json', 'vocab.txt', 'pytorch_model.bin')],
                  'code_files': [{'path': p.name, 'sha256': digest(p.read_bytes())}
                                 for p in sorted(Path(__file__).parent.glob('*.py'))]}
    return save_detector(model, tokenizer, output, config, report, provenance)
