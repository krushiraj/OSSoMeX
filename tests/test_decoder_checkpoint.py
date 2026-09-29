import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BertTokenizerFast

from research.training import full_label_train
from research.training.models import build_model
from research.training.predict import Detector
from research.training.runner import save_detector


@pytest.fixture
def source_checkpoint(tmp_path):
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nna\n##par\n##i\n.\n')
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    model = build_model({'vocab_size': len(tokenizer), 'hidden_size': 24, 'num_hidden_layers': 1,
                         'num_attention_heads': 2, 'intermediate_size': 32})
    checkpoint = tmp_path / 'source'
    save_detector(model, tokenizer, checkpoint, {'max_length': 5, 'overlap': 1},
                  {'optimizer_steps': 1, 'effective_optimizer_steps': 1}, {'purpose': 'plumbing_test'})
    return checkpoint


def test_decoder_variant_is_immutable_and_keeps_original_behavior(source_checkpoint, tmp_path):
    source = source_checkpoint
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    candidate = tmp_path / 'candidate'
    manifest = full_label_train.publish_detector_decoder(source, candidate, 'wordpiece-bio-v1', allow_plumbing=True)
    assert manifest['inference'] == {'decoder': 'wordpiece-bio-v1'}
    assert manifest['inference_provenance']['weights_retrained'] is False
    assert manifest['training'] == json.loads(before['manifest.json'])['training']
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert (candidate / 'model.safetensors').read_bytes() == before['model.safetensors']
    with pytest.raises(FileExistsError):
        full_label_train.publish_detector_decoder(source, candidate, 'wordpiece-bio-v1', allow_plumbing=True)

    class Emissions:
        def __call__(self, input_ids, **kwargs):
            probabilities = torch.tensor([
                [.99, .0025, .0025, .0025, .0025],
                [.422, .573, .003, .001, .001],
                [.558, .004, .437, .0005, .0005],
                [.359, .002, .637, .001, .001],
            ])
            index = torch.zeros_like(input_ids)
            for token_id, row in ((5, 1), (6, 2), (7, 3)):
                index[input_ids == token_id] = row
            return SimpleNamespace(logits=probabilities[index].log())

    outputs = []
    for path in (source, candidate):
        detector = Detector(path, 'cpu', allow_plumbing=True)
        detector.model = Emissions()
        outputs.append(detector.predict({'document_id': 'd', 'text': 'napari. napari.'}))
    assert [s['text'] for s in outputs[0]['spans']] == ['na', 'i', 'na', 'i']
    assert [s['text'] for s in outputs[1]['spans']] == ['napari', 'napari']
    assert [(s['start'], s['end']) for s in outputs[1]['spans']] == [(0, 6), (8, 14)]
    assert outputs[0]['checkpoint_sha256'] != outputs[1]['checkpoint_sha256']
    assert outputs[0]['text_revision'] == outputs[1]['text_revision']


def test_unknown_decoder_and_corrupted_parent_cannot_be_published(source_checkpoint, tmp_path):
    output = tmp_path / 'candidate'
    with pytest.raises(ValueError, match='decoder'):
        full_label_train.publish_detector_decoder(source_checkpoint, output, 'typo', allow_plumbing=True)
    assert not output.exists()
    (source_checkpoint / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        full_label_train.publish_detector_decoder(source_checkpoint, output, 'wordpiece-bio-v1', allow_plumbing=True)
    assert not output.exists()


def test_loading_unknown_decoder_fails_instead_of_silent_greedy(source_checkpoint):
    manifest = source_checkpoint / 'manifest.json'
    value = json.loads(manifest.read_text())
    value['inference'] = {'decoder': 'typo'}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='decoder'):
        Detector(source_checkpoint, 'cpu', allow_plumbing=True)
    with pytest.raises(ValueError, match='decoder'):
        full_label_train.verify_stage_checkpoint(source_checkpoint, 'detector', allow_plumbing=True)


def test_parent_manifest_change_during_verification_cannot_misstate_lineage(source_checkpoint, tmp_path, monkeypatch):
    original = full_label_train.verify_stage_checkpoint

    def changed_after_verification(*args, **kwargs):
        verified = original(*args, **kwargs)
        path = source_checkpoint / 'manifest.json'
        changed = json.loads(path.read_text())
        changed['provenance']['note'] = 'concurrent edit'
        path.write_text(json.dumps(changed))
        return verified

    monkeypatch.setattr(full_label_train, 'verify_stage_checkpoint', changed_after_verification)
    output = tmp_path / 'candidate'
    with pytest.raises(ValueError, match='changed detector manifest'):
        full_label_train.publish_detector_decoder(source_checkpoint, output, 'wordpiece-bio-v1', allow_plumbing=True)
    assert not output.exists()
