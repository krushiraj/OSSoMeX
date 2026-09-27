"""Partial-label objectives: unknown is not a negative class."""

import torch


def partial_token_loss(logits, allowed_labels, active_mask):
    if logits.shape != allowed_labels.shape or logits.shape[:-1] != active_mask.shape:
        raise ValueError('loss shape mismatch')
    active = active_mask.bool()
    if not active.any():
        return None
    allowed = allowed_labels[active].bool()
    if not allowed.any(dim=-1).all():
        raise ValueError('active token has no allowed labels')
    selected = logits[active]
    return (torch.logsumexp(selected, dim=-1)
            - torch.logsumexp(selected.masked_fill(~allowed, -torch.inf), dim=-1)).mean()
