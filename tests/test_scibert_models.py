import math

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def test_detector_shape_and_safe_reload(tmp_path):
    from research.training.models import build_model
    config = {'vocab_size': 32, 'hidden_size': 24, 'num_hidden_layers': 1,
              'num_attention_heads': 2, 'intermediate_size': 32}
    model = build_model(config).eval()
    inputs = {'input_ids': torch.ones(2, 8, dtype=torch.long),
              'attention_mask': torch.ones(2, 8, dtype=torch.long)}
    with torch.inference_mode(): before = model(**inputs).logits
    assert before.shape == (2, 8, 5)
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = transformers.BertForTokenClassification.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.inference_mode(): after = restored(**inputs).logits
    assert torch.equal(before, after)


def test_decode_repairs_bio_and_keeps_repeated_mentions_and_scores():
    from research.training.decode import decode_bio
    spans = decode_bio([2, 2, 0, 4, 1], [(0, 4), (4, 5), (5, 6), (6, 9), (10, 15)],
                       [.8, .2, .9, .7, .6])
    assert spans == [{'label': 'SOFTWARE', 'start': 0, 'end': 5, 'score': math.sqrt(.16)},
                     {'label': 'VERSION', 'start': 6, 'end': 9, 'score': .7},
                     {'label': 'SOFTWARE', 'start': 10, 'end': 15, 'score': .6}]


def test_stitch_uses_maximum_context_and_first_window_on_tie():
    from research.training.decode import stitch_logits
    windows = [{'content_start': 0, 'content_end': 4, 'token_indices': [-1, 0, 1, 2, 3, -1]},
               {'content_start': 2, 'content_end': 6, 'token_indices': [-1, 2, 3, 4, 5, -1]}]
    logits = [torch.ones(6, 5), torch.ones(6, 5) * 2]
    result = stitch_logits(windows, logits, 6)
    assert result[:, 0].tolist() == [1, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError, match='missing'):
        stitch_logits(windows[:1], logits[:1], 6)
