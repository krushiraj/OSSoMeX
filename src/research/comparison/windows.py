"""Freeze exact source substrings for every comparison backend."""

from bisect import bisect_right

from ..contracts import text_revision
from .contracts import validate_input


def _offsets(text, tokenizer):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    offsets = encoded['offset_mapping']
    if len(encoded['input_ids']) != len(offsets):
        raise ValueError('invalid_tokenizer_offsets: token and offset counts differ')
    previous = (0, 0)
    for offset in offsets:
        if (len(offset) != 2 or any(type(value) is not int for value in offset)
                or not 0 <= offset[0] < offset[1] <= len(text)
                or offset[0] < previous[0] or offset[1] < previous[1]):
            raise ValueError('invalid_tokenizer_offsets: expected ordered code-point boundaries')
        previous = offset
    return offsets


def freeze_windows(documents: list[dict], tokenizer, *, max_content_tokens: int = 480,
                   overlap_tokens: int = 64) -> list[dict]:
    """Retokenize every exact slice; never truncate or lose source characters."""
    if type(max_content_tokens) is not int or not 1 <= max_content_tokens <= 480:
        raise ValueError('max_content_tokens must be an integer from 1 to 480')
    if type(overlap_tokens) is not int or not 0 <= overlap_tokens < max_content_tokens:
        raise ValueError('overlap_tokens must be a nonnegative integer below capacity')
    if not isinstance(documents, list):
        raise ValueError('documents must be a list')
    documents = [validate_input(document) for document in documents]
    if len({document['document_id'] for document in documents}) != len(documents):
        raise ValueError('duplicate document_id in frozen inputs')
    windows = []
    for document in documents:
        text = document['text']
        offsets = _offsets(text, tokenizer)
        token_ends = [end for _, end in offsets]
        start, previous_end = 0, 0
        while start < len(text):
            first = bisect_right(token_ends, start)
            limit = first + max_content_tokens
            end = offsets[limit][0] if limit < len(offsets) else len(text)
            if end <= start:
                end = offsets[min(limit, len(offsets)) - 1][1]
            local = _offsets(text[start:end], tokenizer)
            while len(local) > max_content_tokens and end > start:
                candidate = start + local[max_content_tokens][0]
                end = candidate if start < candidate < end else end - 1
                local = _offsets(text[start:end], tokenizer) if end > start else []
            if end <= start:
                raise ValueError(f'cannot_fit_content_token: {document["document_id"]} at {start}')
            if end <= previous_end:
                # Retokenizing a new left edge can reduce the feasible overlap.
                start = previous_end
                continue
            window_text = text[start:end]
            windows.append({'window_id': f'{document["document_id"]}|{document["text_revision"]}|{start}:{end}',
                            'document_id': document['document_id'],
                            'text_revision': document['text_revision'], 'start': start, 'end': end,
                            'text': window_text, 'window_text_revision': text_revision(window_text),
                            'content_tokens': len(local)})
            if end == len(text):
                break
            previous_end = end
            next_start = start + local[-overlap_tokens][0] if overlap_tokens and len(local) > overlap_tokens else end
            start = next_start if next_start > start else end
    return windows
