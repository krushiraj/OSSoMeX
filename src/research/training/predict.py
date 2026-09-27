"""Local diagnostic span output, deliberately separate from the public contract."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertForTokenClassification

from ..data.manifest import digest, verified_path
from .decode import decode_bio, stitch_logits
from .features import LABELS, build_token_features
from .runner import CAPABILITIES, choose_device


class Detector:
    def __init__(self, checkpoint, device='auto', *, allow_plumbing=False):
        checkpoint = Path(checkpoint)
        manifest = json.loads((checkpoint / 'manifest.json').read_bytes())
        if (manifest.get('schema_version') != 'detector-checkpoint-1' or manifest.get('status') != 'trained_experimental'
                or manifest.get('training', {}).get('optimizer_steps', 0) < 1 or manifest.get('labels') != LABELS
                or manifest.get('capabilities') != CAPABILITIES):
            raise ValueError('invalid or untrained detector checkpoint')
        if manifest['provenance'].get('purpose') == 'plumbing_test' and not allow_plumbing:
            raise ValueError('plumbing checkpoint is not a real extractor')
        paths = [r['path'] for r in manifest['files']]
        if len(paths) != len(set(paths)) or not {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'} <= set(paths):
            raise ValueError('incomplete detector checkpoint')
        for record in manifest['files']:
            verified_path(checkpoint, record)
        self.device = choose_device(device)
        self.manifest = manifest
        self.identity = digest((checkpoint / 'manifest.json').read_bytes())
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True, trust_remote_code=False)
        self.model = BertForTokenClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
                                                               use_safetensors=True, attn_implementation='eager').to(self.device).eval()
        if self.model.config.id2label != dict(enumerate(LABELS)) or self.tokenizer.do_lower_case:
            raise ValueError('checkpoint labels or tokenizer mismatch')

    def predict(self, document):
        text, doc_id = document.get('text'), document.get('document_id')
        if not isinstance(text, str) or not text.strip() or not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError('document_id and nonempty text required')
        revision = 'sha256:' + digest(text.encode())
        if document.get('text_revision', revision) != revision:
            raise ValueError('source revision mismatch')
        features = build_token_features({'document_id': doc_id, 'text': text}, [], [], self.tokenizer, self.manifest['recipe'])
        outputs, chunks = [], []
        for index, window in enumerate(features['windows']):
            try:
                with torch.inference_mode():
                    values = self.model(input_ids=torch.tensor([window['input_ids']], device=self.device),
                                        attention_mask=torch.tensor([window['attention_mask']], device=self.device)).logits[0].cpu()
                if not torch.isfinite(values).all():
                    raise RuntimeError('nonfinite logits')
                outputs.append(values)
                chunks.append({'window': index, 'status': 'success'})
            except (RuntimeError, OSError) as exc:
                chunks.append({'window': index, 'status': 'failure', 'error': str(exc)})
        result = {'schema_version': 'detector-prediction-1', 'document_id': doc_id, 'text_revision': revision,
                  'checkpoint_sha256': self.identity, 'capabilities': CAPABILITIES, 'scores_calibrated': False,
                  'quality_evaluated': False, 'review_required': True, 'review_reasons': ['experimental_small_data', 'uncalibrated_scores'],
                  'offset_unit': 'unicode_codepoint_half_open', 'chunks': chunks, 'spans': []}
        if len(outputs) != len(features['windows']):
            return {**result, 'status': 'failure', 'review_reasons': [*result['review_reasons'], 'incomplete_document']}
        logits = stitch_logits(features['windows'], outputs, features['token_count'])
        probabilities = logits.softmax(dim=-1)
        scores, ids = probabilities.max(dim=-1)
        offsets = [None] * features['token_count']
        for window in features['windows']:
            for index, offset in zip(window['token_indices'], window['offsets']):
                if index >= 0:
                    offsets[index] = offset
        spans = [{**span, 'text': text[span['start']:span['end']]} for span in decode_bio(ids.tolist(), offsets, scores.tolist())]
        return {**result, 'status': 'success' if spans else 'no_mentions', 'spans': spans}
