"""Deterministic passage ownership; selection signals are never annotation labels."""

import random
import re

from ..contracts import text_revision
from .jats import sentence_regions


def passage_candidates(parsed: dict) -> tuple[list[dict], list[dict]]:
    text = parsed['text']
    revision = text_revision(text)
    if parsed.get('text_revision', revision) != revision:
        raise ValueError('STALE_PASSAGE_TEXT')
    prior_end = 0
    for block in parsed['paragraphs']:
        start, end = block.get('start'), block.get('end')
        if (type(start) is not int or type(end) is not int
                or not prior_end <= start < end <= len(text)):
            raise ValueError('INVALID_PARAGRAPH_SPAN')
        prior_end = end
    eligible, issues = [], []
    for sentence in sorted(sentence_regions(parsed), key=lambda row: row['start']):
        context = sentence['paragraph_span']
        if context['end'] - context['start'] > 6000:
            issues.append({'code': 'CONTEXT_EXCEEDS_LIMIT', 'annotation_region': {
                'start': sentence['start'], 'end': sentence['end']}, 'context_span': context})
            continue
        eligible.append({
            'document_id': parsed.get('document_id'), 'text_revision': revision,
            'annotation_region': {'start': sentence['start'], 'end': sentence['end']},
            'context_span': dict(context), 'region_kind': 'sentence',
            'text': sentence['text'], 'context_text': text[context['start']:context['end']],
            'segmenter_version': sentence['segmenter_version'], 'whole_passage_audit': True,
        })
    return eligible, issues


def select_passages(parsed: dict, *, aliases: list[str], seed: int = 42,
                    random_count: int = 3, signal_count: int = 3) -> list[dict]:
    if (type(seed) is not int or seed != 42 or type(random_count) is not int
            or not 0 <= random_count <= 3 or type(signal_count) is not int
            or not 0 <= signal_count <= 3):
        raise ValueError('INVALID_PASSAGE_LIMITS')
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
        raise ValueError('INVALID_ALIASES')
    candidates, _ = passage_candidates(parsed)
    chosen = random.Random(seed).sample(candidates, min(random_count, len(candidates)))
    result = [{**row, 'selection_reason': 'random_whole_passage'} for row in chosen]
    selected = {row['annotation_region']['start'] for row in chosen}
    signals = []
    for row in candidates:
        if row['annotation_region']['start'] in selected:
            continue
        hits, version = [], False
        for alias in aliases:
            pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
            flags = 0 if len(alias) <= 3 else re.I
            matches = list(re.finditer(pattern, row['text'], flags))
            if matches:
                hits.append(alias)
            for match in matches:
                tail = row['text'][match.end():]
                version |= bool(re.match(
                    r'\s*[,(:-]?\s*(?:(?:version|ver\.?|release)\s*)?[vr]?\s*\d+(?:\.\d+)*(?:[a-z]\d*)?\b',
                    tail, re.I))
        if hits:
            signals.append((0 if version else 1, row['annotation_region']['start'], {
                **row, 'candidate_aliases': hits,
                'selection_reason': 'literal_name_version_candidate' if version else 'literal_name_candidate',
            }))
    result.extend(row for _, _, row in sorted(signals, key=lambda item: item[:2])[:signal_count])
    return result
