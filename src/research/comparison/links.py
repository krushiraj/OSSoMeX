"""Exact software-occurrence to version-span diagnostics, separate from detection."""

from copy import deepcopy
import json
import re

from ..contracts import check_span
from ..evaluation.metrics import prf
from .alignment import align_unique, native_span
from .contracts import COMPLETED_STATUSES, RESULT_STATUSES, validate_reference
from .references import REVIEW_KINDS, _documents, _permitted


def _span(document, value):
    start, end = check_span(value, document, 'endpoint', len(document['text']))
    if value.get('text') != document['text'][start:end]:
        raise ValueError('endpoint text does not match frozen source')
    return start, end


def _edges(document, row):
    spans = set()
    for span in row['spans']:
        if span.get('label') not in ('SOFTWARE', 'VERSION'):
            raise ValueError('unsupported endpoint label')
        spans.add((span['label'], *_span(document, span)))
    result = set()
    for edge in row['version_links']:
        name, version = _span(document, edge['software']), _span(document, edge['version'])
        if ('SOFTWARE', *name) not in spans or ('VERSION', *version) not in spans:
            raise ValueError('link requires both detected/reference endpoints')
        result.add((*name, *version))
    return result, spans


def _eligible(edge, regions, ignored, ignored_names):
    a, b, c, d = edge
    # Mask overlapping fragments as well as exact spans of an unresolved version.
    return (not any(lo < d and c < hi for lo, hi in ignored)
            and not any(lo < b and a < hi for lo, hi in ignored_names)
            and any(lo <= a < b <= hi and lo <= c < d <= hi for lo, hi in regions))


def _counts(gold, predicted):
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def _null():
    return dict.fromkeys(('tp', 'fp', 'fn', 'precision', 'recall', 'f1'))


def score_links(documents, references, predictions, *, review_kind):
    if review_kind not in REVIEW_KINDS:
        raise ValueError('invalid review_kind')
    docs = _documents(documents)
    if not docs:
        raise ValueError('nonempty diagnostic population required')
    refs = {}
    for doc in docs.values():
        _permitted(doc)
    for ref in references:
        ident = ref.get('document_id')
        if ident not in docs or ident in refs:
            raise ValueError('unknown or duplicate reference document')
        _permitted(ref)
        validate_reference(docs[ident], ref)
        gold, endpoints = _edges(docs[ident], ref)
        ignored = [_span(docs[ident], span) for span in ref.get('ignored_versions', [])]
        ignored_names = [_span(docs[ident], span) for span in ref.get('ignored_software', [])]
        regions = []
        for region in ref['link_coverage']:
            start, end = check_span(region, ref, 'link_coverage', len(docs[ident]['text']))
            if region.get('complete') is not True or region.get('review_kind') not in REVIEW_KINDS:
                raise ValueError('explicit complete link review required')
            if region['review_kind'] == review_kind:
                regions.append((start, end))
        refs[ident] = (gold, endpoints, regions, ignored, ignored_names)
    arms = {}
    for raw in predictions:
        ident, arm = raw.get('document_id'), raw.get('arm_id')
        if ident not in docs or not isinstance(arm, str) or not arm.strip():
            raise ValueError('known document and nonblank arm required')
        rows = arms.setdefault(arm, {})
        if ident in rows:
            raise ValueError('duplicate arm/document prediction')
        if type(raw.get('supported')) is not bool or raw.get('status') not in RESULT_STATUSES:
            raise ValueError('explicit support and valid status required')
        if rows and next(iter(rows.values()))['supported'] != raw['supported']:
            raise ValueError('arm support changed across documents')
        row = deepcopy(raw)
        row['invalid'] = False
        try:
            if row.get('text_revision') != docs[ident]['text_revision']:
                raise ValueError('prediction revision mismatch')
            if row['status'] not in COMPLETED_STATUSES or not row['supported']:
                if row['version_links']:
                    raise ValueError('inactive/unsupported output cannot carry links')
                # Detection can succeed independently of linking; retain its denominator.
                row['edges'], row['endpoints'] = _edges(docs[ident], row)
            else:
                row['edges'], row['endpoints'] = _edges(docs[ident], row)
        except (KeyError, TypeError, ValueError) as exc:
            row.update(status='failure', invalid=True, reason=str(exc), edges=set(), endpoints=set())
        rows[ident] = row
    report = {'mode': 'version_link_diagnostic', 'heldout_quality_evaluated': False,
              'review_kind': review_kind, 'population_documents': len(docs), 'arms': {}}
    for arm, rows in arms.items():
        supported = next(iter(rows.values()))['supported']
        active = supported and any(r['status'] not in ('unavailable', 'unsupported') for r in rows.values())
        totals, conditional, details = [0, 0, 0], [0, 0, 0], []
        conditional_gold = completed = invalid = 0
        for ident, doc in docs.items():
            gold, endpoints, regions, ignored, ignored_names = refs.get(ident, (set(), set(), [], [], []))
            row = rows.get(ident, {'status': 'failure' if active else 'unavailable',
                                  'edges': set(), 'endpoints': set(), 'reason': 'missing_result'})
            selected_gold = {edge for edge in gold if _eligible(edge, regions, ignored, ignored_names)}
            pred = {edge for edge in row['edges'] if _eligible(edge, regions, ignored, ignored_names)}
            counts = _counts(selected_gold, pred)
            matched = endpoints & row['endpoints']
            def both(edge):
                return ('SOFTWARE', *edge[:2]) in matched and ('VERSION', *edge[2:]) in matched
            conditional_g = {edge for edge in selected_gold if both(edge)}
            conditional_p = {edge for edge in pred if both(edge)}
            for i, count in enumerate(counts):
                totals[i] += count
            for i, count in enumerate(_counts(conditional_g, conditional_p)):
                conditional[i] += count
            conditional_gold += len(conditional_g)
            completed += row['status'] in COMPLETED_STATUSES
            invalid += row.get('invalid', False)
            details.append({'document_id': ident, 'status': row['status'], 'reason': row.get('reason'),
                            'eligible_gold_edges': len(selected_gold), 'excluded_gold_edges': len(gold - selected_gold),
                            'excluded_predictions': len(row['edges'] - pred),
                            'operational': prf(*counts) if active else _null(),
                            'false_positive_edges': sorted(pred - selected_gold) if active else [],
                            'missed_edges': sorted(selected_gold - pred) if active else []})
        report['arms'][arm] = {'supported': supported, 'inference_started': active,
                              'operational': prf(*totals) if active else _null(),
                              'conditional_on_detected_endpoints': prf(*conditional) if active else _null(),
                              'conditional_gold_edges': conditional_gold if active else None,
                              'completed_documents': completed, 'invalid_documents': invalid,
                              'per_document': details}
    return report


def softcite_links(chunks, offset_unit):
    """Keep the native name-version envelope; never infer ownership by proximity."""
    result = []
    for chunk in chunks:
        window = chunk['window']
        native = chunk['raw'].get('native')
        if native is None and chunk['raw'].get('status_code') == 204:
            continue
        if not isinstance(native, dict):
            raise ValueError('missing Softcite native evidence')
        mentions = native.get('software', native.get('mentions'))
        if not isinstance(mentions, list):
            raise ValueError('missing Softcite mention array')
        if offset_unit not in ('codepoint', 'unicode_codepoint_half_open', 'utf16'):
            if not window['text'].isascii():
                raise ValueError('unverified Softcite Unicode offset contract')
            unit = 'codepoint'
        else:
            unit = offset_unit
        for mention in mentions:
            if mention.get('version') is None:
                continue
            edge = {}
            for name, native_key in [('software', 'software-name'), ('version', 'version')]:
                value = mention[native_key]
                span = native_span(window['text'], {'text': value['rawForm'],
                    'start': value['offsetStart'], 'end': value['offsetEnd']},
                    unit=unit, offset_base=window['start'])
                edge[name] = {key: span[key] for key in ('start', 'end', 'text')}
            result.append(edge)
    return result


def ollama_links(chunks):
    """Project historical five-field records only when both endpoints align uniquely."""
    from ..schema import validate_public_records
    result = []
    for chunk in chunks:
        content = chunk['raw']['content'].strip()
        if content.startswith('```'):
            match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, flags=re.DOTALL)
            if not match:
                raise ValueError('incomplete JSON code fence')
            content = match.group(1)
        records = json.loads(content)
        if not isinstance(records, list):
            raise ValueError('complete record array required')
        records, invalid = validate_public_records(records)
        if invalid:
            raise ValueError('schema-invalid extraction records')
        window = chunk['window']
        for record in records:
            if record['version'] is None:
                continue
            edge = {}
            for field, key in [('software', 'name'), ('version', 'version')]:
                aligned = align_unique(window['text'], record[key], record['context_sentence'])
                if aligned['status'] != 'resolved':
                    raise ValueError('unresolved link endpoint')
                edge[field] = {'text': aligned['text'], 'start': aligned['start'] + window['start'],
                               'end': aligned['end'] + window['start']}
            result.append(edge)
    return result
