"""Source-aligned, masked features shared by attribute training and inference."""

from collections import Counter
from copy import deepcopy
from itertools import combinations
import re

from ..annotations.aliases import validate_alias_annotations
from ..annotations.validation import validate_task_occurrence
from ..contracts import INTENT_BITS
from ..data.jats import SEGMENTER_VERSION, sentence_regions
from ..data.manifest import digest, json_bytes
from .features import _span

MARKERS = ['[SW1]', '[/SW1]', '[SW2]', '[/SW2]']
SENTIMENT_LABELS = ['positive', 'negative', 'mixed', 'not_expressed']
STAGES = ('linker', 'intent', 'sentiment', 'alias')
CONTEXT_POLICY = {'schema_version': 'attribute-features-1', 'segmenter_version': SEGMENTER_VERSION,
                  'paragraph_policy': 'source-metadata-or-blank-lines-v1',
                  'pair_policy': 'same-or-adjacent-sentence-same-paragraph-v1',
                  'attribute_policy': 'previous-current-next-same-paragraph-v1',
                  'supervision_evidence_policy': 'wholly-visible-field-evidence-v1',
                  'max_length': 512, 'markers': MARKERS, 'distance_clip': 128}


def _prepare(document, tokenizer, config):
    if not tokenizer.is_fast:
        raise ValueError('fast tokenizer required for source offsets')
    if not isinstance(document.get('text'), str) or not document['text'].strip():
        raise ValueError('nonempty source text required')
    supplied = config.get('attribute_features', {})
    for key, value in supplied.items():
        if key not in ('marker_token_ids', 'tokenizer_class') and (key not in CONTEXT_POLICY or value != CONTEXT_POLICY[key]):
            raise ValueError('incompatible attribute feature policy: ' + key)
    tokenizer.add_special_tokens({'additional_special_tokens': MARKERS})
    marker_ids = tokenizer.convert_tokens_to_ids(MARKERS)
    if len(set(marker_ids)) != 4 or tokenizer.unk_token_id in marker_ids:
        raise ValueError('distinct endpoint markers required')
    policy = deepcopy(CONTEXT_POLICY)
    policy.update(marker_token_ids=marker_ids, tokenizer_class=type(tokenizer).__name__)
    if any(supplied[key] != policy[key] for key in ('marker_token_ids', 'tokenizer_class') if key in supplied):
        raise ValueError('incompatible frozen attribute tokenizer')
    base, text = document.get('offset_base', 0), document['text']
    blocks = document.get('paragraphs') or document.get('metadata', {}).get('paragraphs')
    if blocks is None:
        boundaries = [0, *[m.end() for m in re.finditer(r'\n\s*\n', text)], len(text)]
        blocks = [{'start': start + base, 'end': end + base, 'kind': 'p'}
                  for start, end in zip(boundaries, boundaries[1:]) if start < end]
    local = []
    for block in blocks:
        start, end = _span(block, base, len(text))
        local.append({**block, 'start': start - base, 'end': end - base,
                      'kind': block.get('kind', 'p')})
    if any(a['end'] > b['start'] for a, b in zip(sorted(local, key=lambda b: b['start']),
                                                 sorted(local, key=lambda b: b['start'])[1:])):
        raise ValueError('overlapping source paragraphs')
    sentences = sentence_regions({'text': text, 'paragraphs': local})
    for sentence in sentences:
        sentence['start'] += base
        sentence['end'] += base
        sentence['paragraph_span'] = {k: v + base for k, v in sentence['paragraph_span'].items()}
    return policy, sentences


def _endpoint(document, row, version=False):
    span = row['span' if version else 'name_span']
    base, text = document.get('offset_base', 0), document['text']
    start, end = _span(span, base, len(text))
    value = row['text' if version else 'name']
    if text[start-base:end-base] != value:
        raise ValueError('source span text mismatch')
    identity = row.get('version_id') if version else row.get('mention_id')
    identity = identity or ('version:' if version else 'mention:') + digest(json_bytes(
        [document['document_id'], document.get('text_revision'), start, end]))
    return {'id': identity, 'span': {'start': start, 'end': end}, 'text': value,
            'context_kind': row.get('context_kind', 'sentence')}


def _context(first, second, sentences):
    endpoints = [first] + ([second] if second else [])
    if any(endpoint['context_kind'] not in ('sentence', 'paragraph') for endpoint in endpoints):
        return None, 'unsupported_context'
    indices = [next((i for i, s in enumerate(sentences)
                     if s['start'] <= endpoint['span']['start'] < endpoint['span']['end'] <= s['end']), None)
               for endpoint in endpoints]
    if any(index is None for index in indices):
        return None, 'unsupported_context'
    paragraph = sentences[indices[0]]['paragraph_span']
    if second:
        if abs(indices[1] - indices[0]) > 1 or sentences[indices[1]]['paragraph_span'] != paragraph:
            return None, 'candidate_context'
        lo, hi = min(indices), max(indices)
    else:
        lo = indices[0] - int(indices[0] > 0 and sentences[indices[0]-1]['paragraph_span'] == paragraph)
        hi = indices[0] + int(indices[0]+1 < len(sentences) and sentences[indices[0]+1]['paragraph_span'] == paragraph)
    return {'start': sentences[lo]['start'], 'end': sentences[hi]['end']}, None


def _encode(document, first, second, context, tokenizer, policy):
    base = document.get('offset_base', 0)
    start, end = context['start'], context['end']
    text = document['text'][start-base:end-base]
    original = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                         truncation=False, verbose=False)
    source_offsets = [(a + start, b + start) for a, b in original['offset_mapping']]
    indices = []
    for endpoint in [first] + ([second] if second else []):
        a, b = endpoint['span']['start'], endpoint['span']['end']
        active = [i for i, (s, e) in enumerate(source_offsets) if s < b and a < e]
        if not active or source_offsets[active[0]][0] != a or source_offsets[active[-1]][1] != b:
            return None, 'token_boundary'
        indices.append(active[0])
    inserts = {}
    for endpoint, opening, closing in [(first, MARKERS[0], MARKERS[1]),
                                        *([(second, MARKERS[2], MARKERS[3])] if second else [])]:
        inserts.setdefault(endpoint['span']['start'] - start, []).append(opening)
        inserts.setdefault(endpoint['span']['end'] - start, []).insert(0, closing)
    pieces, source_map = [], []
    for position in range(len(text) + 1):
        if position in inserts:
            markers = ''.join(inserts[position])
            pieces.append(markers)
            source_map.extend([None] * len(markers))
        if position < len(text):
            pieces.append(text[position])
            source_map.append(start + position)
    encoded = tokenizer(''.join(pieces), return_offsets_mapping=True, truncation=False, verbose=False)
    if len(encoded['input_ids']) > policy['max_length']:
        return None, 'context_limit'
    offsets, content = [], []
    excluded_ids = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    for token_id, (a, b) in zip(encoded['input_ids'], encoded['offset_mapping']):
        mapped = source_map[a:b]
        active = bool(mapped) and token_id not in excluded_ids and all(p is not None for p in mapped)
        offsets.append([mapped[0], mapped[-1]+1] if active else [0, 0])
        content.append(active)
    masks = []
    for endpoint in [first] + ([second] if second else []):
        a, b = endpoint['span']['start'], endpoint['span']['end']
        mask = [active and a <= s < e <= b for (s, e), active in zip(offsets, content)]
        selected = [offset for offset, active in zip(offsets, mask) if active]
        if not selected or selected[0][0] != a or selected[-1][1] != b:
            return None, 'empty_endpoint_mask' if not selected else 'token_boundary'
        masks.append(mask)
    if not any(content):
        return None, 'empty_context_mask'
    distance = max(-128, min(128, indices[1] - indices[0])) + 128 if second else 0
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'],
            'first_mask': masks[0], 'second_mask': masks[1] if second else [False] * len(content),
            'context_mask': content, 'offsets': offsets, 'distance_bucket': distance,
            'order_flag': int(second['span']['start'] > first['span']['start']) if second else 0}, None


def _add(result, document, stage, first, second, sentences, tokenizer, targets=None, known=None, provenance=None,
         evidence_by_field=None):
    identity = {'document_id': document['document_id'], 'text_revision': document.get('text_revision'),
                'work_group_id': document.get('work_group_id', document['document_id']),
                'source': document.get('source', 'unknown'), 'stage': stage,
                'first_id': first['id'], 'second_id': second['id'] if second else None,
                'first_span': first['span'], 'second_span': second['span'] if second else None,
                'context_kind': first['context_kind'], 'provenance': deepcopy(provenance or {})}
    identity['feature_id'] = 'attribute:' + digest(json_bytes(
        [identity[k] for k in ('document_id', 'text_revision', 'stage', 'first_id', 'second_id')]))
    reason = None
    if second and first['span']['start'] < second['span']['end'] and second['span']['start'] < first['span']['end']:
        reason = 'overlapping_endpoint'
    elif stage == 'alias' and first['text'] == second['text']:
        reason = 'repeated_name'
    context, context_reason = _context(first, second, sentences)
    reason = reason or context_reason
    if known is not None:
        known = list(known)
    masked_fields = {}
    if reason is None:
        for field, (index, evidence) in (evidence_by_field or {}).items():
            outside = [span for span in evidence
                       if not context['start'] <= span['start'] < span['end'] <= context['end']]
            if known[index] and outside:
                known[index] = False
                masked_fields[field] = {'reason': 'evidence_outside_context', 'evidence_spans': deepcopy(outside)}
    if masked_fields:
        identity['provenance']['masked_fields'] = masked_fields
    if reason is None and known is not None and not any(known):
        reason = 'evidence_outside_context' if masked_fields else 'all_inactive'
    tensor, encoding_reason = _encode(document, first, second, context, tokenizer, result['config']) if reason is None else (None, None)
    reason = reason or encoding_reason
    if reason:
        result['excluded'].append({**identity, 'context_span': context, 'reason': reason})
        return
    row = {**identity, 'context_span': context, **tensor}
    if targets is not None:
        row.update(targets=targets, known=known)
    prior = next((old for old in result['features'][stage] if old['feature_id'] == row['feature_id']), None)
    if prior is not None:
        if any(prior.get(k) != row.get(k) for k in ('targets', 'known')):
            raise ValueError('conflicting duplicate attribute supervision')
        return
    result['features'][stage].append(row)


def _complete_relation(item, first, second):
    task, annotation = item['task'], item['annotation']
    lo = min(first['span']['start'], second['span']['start'])
    hi = max(first['span']['end'], second['span']['end'])
    owned = task['annotation_region']
    if not owned['start'] <= lo < hi <= owned['end']:
        return False
    if any(region['start'] < hi and lo < region['end'] for region in annotation.get('unresolved_regions', [])):
        return False
    return any(region.get('status') == 'complete' and region.get('fields', {}).get('software') is True
               and region.get('fields', {}).get('versions') is True
               and region['start'] <= lo < hi <= region['end'] for region in annotation['covered_regions'])


def build_attribute_features(document: dict, items: list[dict], tokenizer, config: dict) -> dict:
    policy, sentences = _prepare(document, tokenizer, config)
    result = {'features': {stage: [] for stage in STAGES}, 'excluded': [], 'config': policy}
    selected = [item for item in items if item['task']['document_id'] == document['document_id']]
    all_versions = {}
    validated = []
    for item in selected:
        task, annotation = item['task'], item['annotation']
        if task['text_revision'] != document.get('text_revision'):
            raise ValueError('attribute source revision mismatch')
        base = document.get('offset_base', 0)
        context = task['context_span']
        if document['text'][context['start']-base:context['end']-base] != task['text']:
            raise ValueError('attribute task source mismatch')
        occurrences = [validate_task_occurrence(task, occurrence) for occurrence in annotation['occurrences']]
        layer = validate_alias_annotations(task, occurrences, annotation.get('alias_annotations'))
        validated.append((item, occurrences, layer))
        for occurrence in occurrences:
            if occurrence['known']['versions']:
                for edge in occurrence['version_links']:
                    endpoint = _endpoint(document, edge, True)
                    all_versions[endpoint['id']] = endpoint
    for item, occurrences, layer in validated:
        task, annotation = item['task'], item['annotation']
        provenance = {key: deepcopy(annotation.get(key)) for key in
                      ('annotation_revision', 'review_status', 'annotator', 'evidence_check', 'human_passage_review')}
        provenance.update(task_id=task['task_id'], policy_hash=task.get('policy_hash'),
                          source_snapshot_sha256=item.get('source_snapshot_sha256'),
                          access_basis=document.get('access_basis'), text_license=document.get('text_license'),
                          source_ids=deepcopy(document.get('source_ids', {})))
        endpoints = {occurrence['mention_id']: _endpoint(document, occurrence) for occurrence in occurrences}
        for occurrence in occurrences:
            first = endpoints[occurrence['mention_id']]
            intents = occurrence['intents'] or []
            intent_evidence = occurrence['evidence']['intents']
            # Shared intent evidence supports positive bits; mentioned-only evidence is negative rationale.
            intent_fields = {bit: (index, intent_evidence) for index, bit in enumerate(INTENT_BITS)
                             if bit in intents or not any(label in intents for label in INTENT_BITS)}
            _add(result, document, 'intent', first, None, sentences, tokenizer,
                 [int(bit in intents) for bit in INTENT_BITS], [occurrence['known'][bit] for bit in INTENT_BITS],
                 provenance, intent_fields)
            sentiment = occurrence['sentiment']
            _add(result, document, 'sentiment', first, None, sentences, tokenizer,
                 [SENTIMENT_LABELS.index(sentiment) if sentiment in SENTIMENT_LABELS else 0],
                 [occurrence['known']['sentiment']], provenance,
                 {'sentiment': (0, occurrence['evidence']['sentiment'])})
            for second in all_versions.values():
                linked = any(edge['span'] == second['span'] for edge in occurrence['version_links'])
                known = occurrence['known']['versions'] and (linked or _complete_relation(item, first, second))
                _add(result, document, 'linker', first, second, sentences, tokenizer,
                     [int(linked)], [known], provenance)
        for relation in (layer or {}).get('relations', []):
            first, second = sorted([endpoints[mid] for mid in relation['member_mention_ids']],
                                   key=lambda endpoint: endpoint['span']['start'])
            _add(result, document, 'alias', first, second, sentences, tokenizer,
                 [int(relation['decision'] == 'alias')], [relation['decision'] in ('alias', 'not_alias')],
                 {**provenance, 'alias_relation': deepcopy(relation)}, {'alias': (0, relation['evidence_spans'])})
    return result


def build_inference_candidates(document: dict, occurrences: list[dict], versions: list[dict], tokenizer, config: dict) -> dict:
    policy, sentences = _prepare(document, tokenizer, config)
    result = {'features': {stage: [] for stage in STAGES}, 'excluded': [], 'config': policy}
    names = sorted([_endpoint(document, row) for row in occurrences], key=lambda endpoint: endpoint['span']['start'])
    version_endpoints = {_endpoint(document, row, True)['id']: _endpoint(document, row, True) for row in versions}
    for first in names:
        for stage in ('intent', 'sentiment'):
            _add(result, document, stage, first, None, sentences, tokenizer)
        for second in version_endpoints.values():
            _add(result, document, 'linker', first, second, sentences, tokenizer)
    for first, second in combinations(names, 2):
        _add(result, document, 'alias', first, second, sentences, tokenizer)
    return result


def summarize_support(features: dict) -> dict:
    stages = features.get('features', features)
    result = {}
    for stage in STAGES:
        rows = stages.get(stage, [])
        labels = list(INTENT_BITS) if stage == 'intent' else SENTIMENT_LABELS if stage == 'sentiment' else ['linked' if stage == 'linker' else 'alias']
        counts, capabilities, unavailable, warnings = {}, {}, [], []
        for index, label in enumerate(labels):
            observed = [(row['targets'][0 if stage == 'sentiment' else index], row.get('work_group_id', row.get('document_id')))
                        for row in rows if row.get('known', [False] * len(labels))[0 if stage == 'sentiment' else index]]
            if stage == 'sentiment':
                class_rows = [work for target, work in observed if target == index]
                counts[label] = {'count': len(class_rows), 'works': len(set(class_rows))}
                capabilities[label] = bool(class_rows)
                signs = [(label, len(class_rows), len(set(class_rows)))]
            else:
                signs = [(sign, sum(target == value for target, _ in observed),
                          len({work for target, work in observed if target == value}))
                         for sign, value in [('positive', 1), ('negative', 0)]]
                counts[label] = {key: value for sign, count, works in signs
                                 for key, value in [(sign, count), (sign + '_works', works)]}
                capabilities[label] = all(count > 0 for _, count, _ in signs)
            for sign, count, works in signs:
                if count == 0:
                    unavailable.append('missing_class:' + label if stage == 'sentiment' else 'missing_' + sign + ':' + label)
                elif count < 10 or works < 3:
                    warnings.append({'label': label, 'sign': sign, 'count': count, 'works': works,
                                     'reason': 'small_or_concentrated_support'})
        exclusions = Counter(row['reason'] for row in features.get('excluded', []) if row['stage'] == stage)
        result[stage] = {'labels': labels, 'counts': counts,
                         'examples': sum(any(row.get('known', [])) for row in rows),
                         'trainable': all(capabilities.values()), 'capabilities': capabilities,
                         'unavailable_reasons': unavailable, 'coverage_warnings': warnings,
                         'excluded_reasons': dict(sorted(exclusions.items()))}
        if stage == 'alias':
            subtypes = Counter(row.get('provenance', {}).get('alias_relation', {}).get('relation_type')
                               for row in rows if any(row.get('known', [])))
            result[stage]['subtype_counts'] = {key: value for key, value in sorted(subtypes.items()) if key is not None}
    return result
