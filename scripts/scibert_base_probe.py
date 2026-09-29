"""Record what the pinned base SciBERT checkpoint actually outputs offline."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, BertForPreTraining


MODEL_ID = 'allenai/scibert_scivocab_cased'
REVISION = 'ddf0be025f8e432a1870e34811997ba6725bf04a'
PROMPTS = (
    'The microscopy images were analyzed with [MASK] software.',
    'Patients with [MASK] were enrolled in the study.',
    'We compared our results with the [MASK] study.',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_head_keys(keys):
    keys = sorted(keys)
    non_encoder = [key for key in keys if not key.startswith('bert.')]
    return {
        'non_encoder_prefixes': sorted({'.'.join(key.split('.')[:2]) for key in non_encoder}),
        'task_classifier_keys': [key for key in keys if key.startswith(('classifier.', 'score.', 'token_classifier.'))],
        'software_version_label_keys': [key for key in keys if 'software' in key.lower() or 'version' in key.lower()],
    }


def rank_mask_predictions(logits, mask_index, tokens, top_k=5):
    probabilities = logits[0, mask_index].softmax(dim=-1)
    values, indices = probabilities.topk(top_k)
    return [
        {'token_id': int(index), 'token': tokens[int(index)], 'probability': float(value)}
        for value, index in zip(values, indices)
    ]


def probe(output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    base = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    config_path, weights_path = base / 'config.json', base / 'pytorch_model.bin'
    config = json.loads(config_path.read_text())
    weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    head_evidence = inspect_head_keys(weights.keys())
    head_evidence['tensor_key_count'] = len(weights)
    head_evidence['non_encoder_keys'] = sorted(key for key in weights if not key.startswith('bert.'))
    head_evidence['config_id2label'] = config.get('id2label')
    del weights
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(
        base, do_lower_case=False, use_fast=True, local_files_only=True, trust_remote_code=False,
    )
    model = BertForPreTraining.from_pretrained(
        base, local_files_only=True, trust_remote_code=False, attn_implementation='eager',
    ).eval()
    tokens = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    rows = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors='pt')
        positions = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(positions) != 1:
            raise ValueError('each probe requires exactly one mask token')
        mask_index = int(positions[0])
        with torch.inference_mode():
            prediction = model(**encoded, output_hidden_states=True)
        rows.append({
            'input': prompt,
            'mask_token_index': mask_index,
            'tokenized_input': tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()),
            'representation_shape': list(prediction.hidden_states[-1].shape),
            'masked_token_top5': rank_mask_predictions(prediction.prediction_logits, mask_index, tokens),
        })

    report = {
        'schema_version': 'base-scibert-probe-1',
        'model_id': MODEL_ID,
        'revision': REVISION,
        'local_files_only': True,
        'weights_sha256': sha256_file(weights_path),
        'config_sha256': sha256_file(config_path),
        'head_evidence': head_evidence,
        'probes': rows,
        'interpretation': 'Actual masked-token predictions and hidden states from the base pretraining checkpoint; not software, disease, citation-intent, or PICO labels.',
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x') as target:
        json.dump(report, target, indent=2, ensure_ascii=False)
        target.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/scibert-v2/base-scibert-probe-001/report.json'))
    args = parser.parse_args()
    result = probe(args.output)
    print(f"Recorded {len(result['probes'])} masked-token probes at {args.output}")
