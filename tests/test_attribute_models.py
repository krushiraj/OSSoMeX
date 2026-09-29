import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')


def tiny_config():
    return {'vocab_size': 32, 'hidden_size': 24, 'num_hidden_layers': 1,
            'num_attention_heads': 2, 'intermediate_size': 32}


def inputs():
    return {'input_ids': torch.ones(2, 8, dtype=torch.long),
            'attention_mask': torch.ones(2, 8, dtype=torch.long),
            'first_mask': torch.tensor([[False, True, True, False, False, False, False, False]] * 2),
            'second_mask': torch.tensor([[False, False, False, False, True, True, False, False]] * 2),
            'context_mask': torch.tensor([[False, True, True, True, True, True, True, False]] * 2),
            'distance_bucket': torch.tensor([129, 127]), 'order_flag': torch.tensor([1, 0])}


@pytest.mark.parametrize('stage,width', [('linker', 1), ('alias', 1), ('intent', 3), ('sentiment', 4)])
def test_head_shape_safe_reload_and_nonzero_encoder_gradient(tmp_path, stage, width):
    from research.training.attribute_models import AttributeModel, build_attribute_model, attribute_loss
    torch.manual_seed(42)
    model = build_attribute_model(tiny_config(), stage).eval()
    before = model(**inputs()).logits
    assert before.shape == (2, width)
    targets = torch.zeros(2, 1 if stage == 'sentiment' else width)
    known = torch.ones_like(targets, dtype=torch.bool)
    loss = attribute_loss(before, targets, known, stage)
    loss.backward()
    assert model.bert.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / 'model.safetensors').is_file()
    restored = AttributeModel.from_pretrained(tmp_path, local_files_only=True, use_safetensors=True).eval()
    assert torch.equal(before.detach(), restored(**inputs()).logits.detach())


@pytest.mark.parametrize('stage', ['linker', 'alias', 'intent', 'sentiment'])
def test_unknown_targets_do_not_change_loss_or_gradient(stage):
    from research.training.attribute_models import attribute_loss
    width = {'linker': 1, 'alias': 1, 'intent': 3, 'sentiment': 4}[stage]
    target_width = 1 if stage == 'sentiment' else width
    logits = torch.zeros(2, width, requires_grad=True)
    targets = torch.zeros(2, target_width)
    known = torch.zeros_like(targets, dtype=torch.bool)
    known[0, 0] = True
    loss = attribute_loss(logits, targets, known, stage)
    targets[~known] = 987
    assert torch.equal(loss, attribute_loss(logits, targets, known, stage))
    loss.backward()
    assert logits.grad[1].abs().sum() == 0
    if stage == 'intent':
        assert logits.grad[0, 1:].abs().sum() == 0
    assert attribute_loss(logits, targets, torch.zeros_like(known), stage) is None


def test_empty_pool_masks_are_rejected():
    from research.training.attribute_models import build_attribute_model
    model = build_attribute_model(tiny_config(), 'linker')
    data = inputs()
    data['second_mask'].zero_()
    with pytest.raises(ValueError, match='empty'):
        model(**data)


def test_unpinned_real_base_is_rejected_without_loading():
    from research.training.attribute_models import build_attribute_model
    with pytest.raises(ValueError, match='pinned'):
        build_attribute_model({'base_model': 'other'}, 'intent')
