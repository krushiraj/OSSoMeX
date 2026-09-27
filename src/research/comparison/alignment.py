"""Exact native alignment and conservative reduction of frozen windows."""

from copy import deepcopy
import math

from ..contracts import check_span, text_revision
from .contracts import (CAPABILITY_FIELDS, COMPLETED_STATUSES, LABEL_CAPABILITIES,
                        OFFSET_UNIT, RESULT_STATUSES, SCHEMA_VERSION, validate_input)


def native_span(text: str, span: dict, *, unit: str, offset_base: int = 0) -> dict:
    """Verify native boundaries locally before adding a code-point parent offset."""
    if not isinstance(text, str) or type(offset_base) is not int or offset_base < 0:
        raise ValueError('invalid text or code-point offset_base')
    start, end = check_span(span, {}, 'native_span')
    if unit == 'utf16':
        boundaries, units = {0: 0}, 0
        for index, character in enumerate(text):
            units += 2 if ord(character) > 0xFFFF else 1
            boundaries[units] = index + 1
        if start not in boundaries or end not in boundaries:
            raise ValueError('invalid UTF-16 boundary: out of bounds or splits a surrogate pair')
        start, end = boundaries[start], boundaries[end]
        method = 'native_utf16'
    elif unit in ('codepoint', OFFSET_UNIT):
        method = 'native_codepoint'
    else:
        raise ValueError(f'unsupported native offset unit: {unit}')
    if end > len(text) or span.get('text') != text[start:end]:
        raise ValueError('native span does not match exact source slice')
    return {**deepcopy(span), 'start': start + offset_base, 'end': end + offset_base,
            'alignment_method': method}


def _occurrences(text, value):
    first = text.find(value)
    if first < 0:
        return []
    second = text.find(value, first + 1)
    return [first] if second < 0 else [first, second]


def align_unique(text: str, value: str, context: str | None = None) -> dict:
    """Resolve exact text only when there is a single supported occurrence."""
    if not isinstance(text, str) or not isinstance(value, str):
        raise ValueError('text and value must be strings')
    unresolved = {'status': 'unresolved', 'text': value, 'context': context}
    if not value:
        return {**unresolved, 'reason': 'empty_value'}
    hits = _occurrences(text, value)
    if len(hits) == 1:
        return {'status': 'resolved', 'text': value, 'start': hits[0],
                'end': hits[0] + len(value), 'alignment_method': 'exact_unique'}
    if not hits:
        return {**unresolved, 'reason': 'text_not_found'}
    if context is None:
        return {**unresolved, 'reason': 'ambiguous_text'}
    if not isinstance(context, str) or not context:
        return {**unresolved, 'reason': 'invalid_context'}
    contexts, mentions = _occurrences(text, context), _occurrences(context, value)
    if len(contexts) != 1:
        return {**unresolved, 'reason': 'context_not_unique'}
    if len(mentions) != 1:
        return {**unresolved, 'reason': 'mention_not_unique_in_context'}
    start = contexts[0] + mentions[0]
    return {'status': 'resolved', 'text': value, 'start': start, 'end': start + len(value),
            'alignment_method': 'exact_unique_context'}


def _verify_window(document, result):
    window = result.get('window')
    if not isinstance(window, dict):
        raise ValueError('missing frozen window evidence')
    window_id = result.get('window_id')
    if not isinstance(window_id, str) or not window_id.strip() or window.get('window_id') != window_id:
        raise ValueError('window_id does not match frozen window')
    if any(window.get(key) != document[key] for key in ('document_id', 'text_revision')):
        raise ValueError('frozen window parent identity/revision mismatch')
    start, end = check_span(window, document, 'window', len(document['text']))
    text = document['text'][start:end]
    if window.get('text') != text or window.get('window_text_revision') != text_revision(text):
        raise ValueError('frozen window text/revision mismatch')
    if type(window.get('content_tokens')) is not int or not 0 <= window['content_tokens'] <= 480:
        raise ValueError('invalid frozen window content_tokens')
    return window


def _verified_span(window, span, capabilities):
    verified = native_span(window['text'], span, unit=OFFSET_UNIT, offset_base=window['start'])
    label = span.get('label')
    if label not in LABEL_CAPABILITIES or not capabilities[LABEL_CAPABILITIES[label]]:
        raise ValueError('span label is unsupported by arm capabilities')
    if 'score' not in span or 'score_kind' not in span:
        raise ValueError('span requires explicit score and score_kind')
    score, kind = span['score'], span['score_kind']
    if score is None:
        if kind is not None:
            raise ValueError('null extraction score requires null score_kind')
    elif (type(score) not in (int, float) or (type(score) is float and not math.isfinite(score))
          or not isinstance(kind, str) or not kind.strip()):
        raise ValueError('span score must be a finite native number with score_kind')
    method = span.get('alignment_method')
    if not isinstance(method, str) or not method.strip():
        raise ValueError('span requires alignment_method')
    verified['alignment_method'] = method
    return verified


def reduce_windows(document: dict, arm_id: str, results: list[dict], capabilities: dict) -> dict:
    """Assemble a document result; the runner supplies raw_artifact before validation."""
    document = validate_input(document)
    if not isinstance(arm_id, str) or not arm_id.strip():
        raise ValueError('arm_id must be nonblank')
    if (not isinstance(capabilities, dict) or set(capabilities) != set(CAPABILITY_FIELDS)
            or any(type(value) is not bool for value in capabilities.values())
            or any(capabilities[key] for key in CAPABILITY_FIELDS if key not in LABEL_CAPABILITIES.values())):
        raise ValueError('capabilities require the seven span-only boolean fields')
    if not isinstance(results, list):
        raise ValueError('window results must be a list')
    chunks, unresolved, verified_windows, seen = [], [], [], set()
    errors = [] if results else ['missing_window_results']
    for native in results:
        chunk = deepcopy(native) if isinstance(native, dict) else {'native_result': deepcopy(native)}
        chunks.append(chunk)
        try:
            window = _verify_window(document, chunk)
            if chunk['window_id'] in seen:
                raise ValueError('duplicate window result')
            seen.add(chunk['window_id'])
            if not all(key in chunk for key in ('status', 'spans', 'unresolved', 'raw', 'error')):
                raise ValueError('missing required window result field')
            if chunk['status'] not in RESULT_STATUSES:
                raise ValueError('invalid window status')
            for key in ('spans', 'unresolved'):
                if not isinstance(chunk[key], list) or any(not isinstance(item, dict) for item in chunk[key]):
                    raise ValueError(f'{key} must be an array of objects')
            unresolved.extend({**deepcopy(item), 'window_id': chunk['window_id']} for item in chunk['unresolved'])
            if chunk['status'] not in COMPLETED_STATUSES:
                raise ValueError('window_not_completed')
            if chunk['unresolved']:
                raise ValueError('unresolved_candidates')
            if chunk['error'] is not None:
                raise ValueError('completed_window_has_error')
            if (chunk['status'] == 'success') != bool(chunk['spans']):
                raise ValueError('window_status_span_mismatch')
            if not any(capabilities[key] for key in LABEL_CAPABILITIES.values()):
                raise ValueError('completed_window_requires_span_capability')
            spans = [_verified_span(window, span, capabilities) for span in chunk['spans']]
            verified_windows.append((window, spans))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
            chunk['native_status'] = chunk.get('status')
            chunk['status'] = 'failure'
            chunk['reduction_error'] = str(exc)
    unique = {}
    for window, spans in sorted(verified_windows, key=lambda item: item[0]['start']):
        for span in spans:
            key = (span['label'], span['start'], span['end'])
            if key not in unique:
                unique[key] = {**span, 'window_ids': [], 'native_scores': []}
            output = unique[key]
            if window['window_id'] not in output['window_ids']:
                output['window_ids'].append(window['window_id'])
            output['native_scores'].append({'window_id': window['window_id'], 'score': span['score'],
                                            'score_kind': span['score_kind']})
    result = {'schema_version': SCHEMA_VERSION, 'arm_id': arm_id,
              'document_id': document['document_id'], 'text_revision': document['text_revision'],
              'status': 'failure' if errors else ('success' if unique else 'no_mentions'),
              'offset_unit': OFFSET_UNIT, 'capabilities': deepcopy(capabilities),
              'scores_calibrated': False, 'spans': [] if errors else list(unique.values()),
              'unresolved': unresolved, 'chunks': chunks, 'raw_artifact': None,
              'provenance': {'reduction': 'exact_frozen_windows'}}
    if errors:
        result['reason'] = '; '.join(dict.fromkeys(errors))
    return result
