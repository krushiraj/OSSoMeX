"""Exact offset features with one loss owner per source token/entity."""

LABELS = ['O', 'B-SOFTWARE', 'I-SOFTWARE', 'B-VERSION', 'I-VERSION']


def _span(span, base, length):
    start, end = span['start'], span['end']
    if type(start) is not int or type(end) is not int or not base <= start < end <= base + length:
        raise ValueError('invalid source span')
    return start, end


def _windows(size, capacity, overlap):
    result = []
    start = 0
    while start < size:
        end = min(size, start + capacity)
        result.append((start, end))
        if end == size:
            break
        start = end - overlap
    return result


def _owner(indices, windows):
    candidates = [(min(indices[0] - start, end - 1 - indices[-1]), -index, index)
                  for index, (start, end) in enumerate(windows)
                  if start <= indices[0] and indices[-1] < end]
    return max(candidates)[2] if candidates else None


def build_token_features(document, labels, coverage, tokenizer, config):
    if not tokenizer.is_fast:
        raise ValueError('fast tokenizer required for source offsets')
    text = document['text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('nonempty source text required')
    base = document.get('offset_base', 0)
    maximum = config.get('max_length', 482)
    capacity = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    overlap = config.get('overlap', 64)
    if not 2 <= capacity <= 510 or not 0 <= overlap < capacity:
        raise ValueError('invalid window limits')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    ids = encoded['input_ids']
    offsets = [(start + base, end + base) for start, end in encoded['offset_mapping']]
    windows = _windows(len(ids), capacity, overlap)
    allowed = [[True] * len(LABELS) for _ in ids]
    owners = [_owner([i], windows) for i in range(len(ids))]
    for region in coverage:
        rs, re = _span(region, base, len(text))
        for i, (start, end) in enumerate(offsets):
            if rs <= start < end <= re:
                for field, a, b in [('software', 1, 2), ('versions', 3, 4)]:
                    if region.get('fields', {}).get(field) is True:
                        allowed[i][a] = allowed[i][b] = False
    entities = {}
    for occurrence in labels:
        candidates = []
        if occurrence.get('known', {}).get('software') is True:
            candidates.append(('SOFTWARE', occurrence['name_span'], occurrence['name']))
        if occurrence.get('known', {}).get('versions') is True:
            candidates.extend(('VERSION', edge['span'], edge['text']) for edge in occurrence['version_links'])
        for kind, span, value in candidates:
            start, end = _span(span, base, len(text))
            if text[start-base:end-base] != value:
                raise ValueError('source span text mismatch')
            entities[(start, end, kind)] = {'span': {'start': start, 'end': end}, 'kind': kind}
    exclusions = []
    assignments = []
    masked = set()
    for (start, end, kind), entity in entities.items():
        indices = [i for i, (s, e) in enumerate(offsets) if s < end and start < e]
        reason = None
        if any(s < end and start < e and (s, e, k) != (start, end, kind) for s, e, k in entities):
            reason = 'overlapping_entity'
        elif not indices or offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
            reason = 'token_boundary'
        owner = _owner(indices, windows) if indices else None
        if reason is None and owner is None:
            reason = 'window_limit'
        if reason:
            masked.update(indices)
            exclusions.append({**entity, 'reason': reason})
        else:
            assignments.append((indices, kind, owner))
    for indices, kind, owner in assignments:
        for position, index in enumerate(indices):
            allowed[index] = [False] * len(LABELS)
            allowed[index][(1 if kind == 'SOFTWARE' else 3) + (position > 0)] = True
            owners[index] = owner
    for index in masked:
        allowed[index] = [True] * len(LABELS)
    results = []
    for wi, (start, end) in enumerate(windows):
        content_ids = ids[start:end]
        # BERT's two special positions have no source span or supervision.
        input_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
        if len(input_ids) != len(content_ids) + 2:
            raise ValueError('BERT tokenizer with two special tokens required')
        results.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids),
                        'offsets': [[0, 0], *[list(o) for o in offsets[start:end]], [0, 0]],
                        'token_indices': [-1, *range(start, end), -1],
                        'allowed_labels': [[True] * 5, *allowed[start:end], [True] * 5],
                        'active_mask': [False, *[owners[i] == wi and not all(allowed[i])
                                                for i in range(start, end)], False],
                        'content_start': start, 'content_end': end})
    return {'document_id': document['document_id'], 'windows': results,
            'exclusions': exclusions, 'token_count': len(ids), 'labels': LABELS,
            'word_ids': encoded.word_ids()}
