"""Experimental composition of verified local stages, with explicit incomplete fields."""

from copy import deepcopy
import math
from pathlib import Path
import re
import uuid

import torch

from ..contracts import (INTENT_BITS, check_span, export_public, occurrence_id,
                        validate_document, validate_occurrence)
from ..data.jats import sentence_regions
from ..data.manifest import digest
from .attribute_features import SENTIMENT_LABELS, STAGES, build_inference_candidates
from .full_label_train import AttributeCheckpoint, batch_attribute_features, load_pipeline_manifest
from .predict import Detector

STAGE_NAMES = ('detector', *STAGES)
STATUSES = {'success', 'partial', 'not_applicable', 'unavailable', 'failed', 'blocked'}
NONCANDIDATES = {'candidate_context', 'repeated_name', 'overlapping_endpoint'}
WARNINGS = ['experimental_small_data', 'uncalibrated_scores']


def _stage(status, candidates=0, completed=0, reasons=()):
    return {'status': status, 'candidate_count': candidates, 'completed_count': completed,
            'occurrence_count': 0, 'reasons': list(reasons)}


def _field(status, value=None, scores=None, reasons=(), context=None):
    return {'status': status, 'value': value, 'scores': scores, 'reasons': list(reasons),
            'context_span': context, 'evidence_method': 'model_input_context' if context else None}


def _probability(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('finite probability in [0,1] required')


def _alias_groups(pairs):
    positive = [pair for pair in pairs if pair['label'] == 'alias']
    adjacency = {}
    for pair in positive:
        a, b = pair['member_mention_ids']
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    groups, visited = [], set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        members, pending = set(), [root]
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(adjacency[current] - members)
        visited.update(members)
        edges = [pair['pair_id'] for pair in positive if set(pair['member_mention_ids']) <= members]
        chords = [pair['pair_id'] for pair in pairs if pair['label'] == 'not_alias'
                  and set(pair['member_mention_ids']) <= members]
        groups.append({'group_id': 'alias-group:' + digest('|'.join(sorted(members)).encode()),
            'member_mention_ids': sorted(members), 'contributing_pair_ids': edges,
            'negative_chord_pair_ids': chords, 'relation_type': None,
            'preferred_mention_id': None, 'preferred_name': None,
            'review_reasons': ['model_alias_proposal', *(['negative_chord_conflict'] if chords else [])]})
    return groups


class FullLabelPipeline:
    def __init__(self, checkpoint: Path, device: str = 'auto'):
        checkpoint = Path(checkpoint)
        self.manifest = load_pipeline_manifest(checkpoint)
        self.identity = digest((checkpoint / 'manifest.json').read_bytes())
        self.detector = Detector(checkpoint / self.manifest['stages']['detector']['path'], device)
        self.stages = {stage: AttributeCheckpoint(checkpoint / entry['path'], device)
                       for stage, entry in self.manifest['stages'].items()
                       if stage != 'detector' and entry['status'] == 'available'}

    def _envelope(self, document):
        hashes = {stage: 'sha256:' + entry['manifest_sha256']
                  for stage, entry in self.manifest['stages'].items() if entry['status'] == 'available'}
        return {'schema_version': 'full-label-prediction-1', 'document_id': document['document_id'],
            'text_revision': document['text_revision'], 'run_id': 'full-label:' + uuid.uuid4().hex,
            'offset_unit': 'unicode_codepoint_half_open', 'checkpoint_sha256': self.identity,
            'checkpoint_hashes': hashes, 'capabilities': deepcopy(self.manifest['capabilities']),
            'label_support': {stage: deepcopy(entry.get('support')) for stage, entry in self.manifest['stages'].items()},
            'scores_calibrated': False, 'quality_evaluated': False, 'review_required': True,
            'review_reasons': list(WARNINGS), 'evidence_method': 'model_input_context',
            'stage_status': {}, 'detector_diagnostics': None, 'field_predictions': [],
            'occurrences': [], 'pair_predictions': {'linker': []}, 'alias_predictions': {'pairs': [], 'groups': []},
            'public_rows': [], 'public_mappings': [], 'exclusions': [],
            'public_contract_complete': False, 'pipeline_complete': False, 'status': 'failure'}

    def failure_result(self, document, error):
        document = validate_document(document)
        result = self._envelope(document)
        result['stage_status'] = {stage: _stage('failed' if stage == 'detector' else 'blocked',
            reasons=[type(error).__name__ if stage == 'detector' else 'detector_failure']) for stage in STAGE_NAMES}
        result['review_reasons'].append('incomplete_document')
        return validate_full_label_prediction(result, document)

    def _evaluate(self, stage, rows, excluded, result, document):
        checkpoint = self.stages[stage]
        evaluations = []
        for row in rows:
            context = row['context_span']
            item = {'feature_id': row['feature_id'], 'first_id': row['first_id'], 'second_id': row['second_id'],
                'first_span': row['first_span'], 'second_span': row['second_span'], 'context_span': context,
                'context_text': document['text'][context['start']:context['end']],
                'evidence_method': 'model_input_context', 'checkpoint_sha256': checkpoint.identity,
                'status': 'success', 'scores': None, 'label': None, 'reasons': []}
            try:
                with torch.inference_mode():
                    batch = batch_attribute_features([row], checkpoint.tokenizer.pad_token_id, checkpoint.device)
                    logits = checkpoint.model(**batch).logits
                    width = 3 if stage == 'intent' else 4 if stage == 'sentiment' else 1
                    if tuple(logits.shape) != (1, width) or not torch.isfinite(logits).all():
                        raise ValueError('invalid or nonfinite stage logits')
                    probabilities = (logits.softmax(-1) if stage == 'sentiment' else logits.sigmoid())[0].cpu().tolist()
                labels = INTENT_BITS if stage == 'intent' else SENTIMENT_LABELS if stage == 'sentiment' else ('linked' if stage == 'linker' else 'alias',)
                item['scores'] = dict(zip(labels, probabilities))
                if stage == 'intent':
                    item['label'] = [label for label, probability in item['scores'].items() if probability >= .5] or ['mentioned']
                    confidence = min(max(p, 1 - p) for p in probabilities)
                elif stage == 'sentiment':
                    item['label'] = labels[max(range(width), key=lambda index: probabilities[index])]
                    confidence = max(probabilities)
                else:
                    item['label'] = labels[0] if probabilities[0] >= .5 else 'not_' + labels[0]
                    confidence = max(probabilities[0], 1 - probabilities[0])
                if confidence < .8:
                    item['reasons'].append('low_confidence:' + stage)
                    result['review_reasons'].append('low_confidence:' + stage)
            except Exception as exc:
                item.update(status='failed', reasons=[type(exc).__name__ + ': ' + str(exc)])
            evaluations.append(item)
        meaningful = [row for row in excluded if row['reason'] not in NONCANDIDATES]
        total = len(rows) + len(meaningful)
        completed = sum(row['status'] == 'success' for row in evaluations)
        status = 'not_applicable' if not total else 'success' if total == completed else 'partial' if completed else 'failed'
        reasons = [row['reason'] for row in meaningful] + [reason for row in evaluations if row['status'] != 'success' for reason in row['reasons']]
        result['stage_status'][stage] = _stage(status, total, completed, reasons)
        return evaluations

    def predict(self, document: dict) -> dict:
        document = validate_document(document)
        result = self._envelope(document)
        try:
            detected = self.detector.predict(document)
        except Exception as exc:
            return self.failure_result(document, exc)
        result['detector_diagnostics'] = deepcopy(detected)
        detector_ok = detected['status'] in ('success', 'no_mentions') and all(
            chunk.get('status') == 'success' for chunk in detected.get('chunks', []))
        result['stage_status']['detector'] = _stage('success' if detector_ok else 'failed',
            len(detected.get('chunks', [])), sum(c.get('status') == 'success' for c in detected.get('chunks', [])),
            [] if detector_ok else ['incomplete_detector_windows'])
        names, versions = [], []
        boundaries = [0, *[match.end() for match in re.finditer(r'\n\s*\n', document['text'])], len(document['text'])]
        document['metadata'] = document.get('metadata') or {}
        paragraphs = document.get('paragraphs') or document['metadata'].get('paragraphs') or [
            {'start': start, 'end': end, 'kind': 'p'} for start, end in zip(boundaries, boundaries[1:]) if start < end]
        sentences = sentence_regions({**document, 'paragraphs': paragraphs})
        for span in detected.get('spans', []):
            start, end = check_span(span, document, 'detector.span', len(document['text']))
            if document['text'][start:end] != span['text']:
                raise ValueError('detector source span mismatch')
            _probability(span['score'])
            if span['score'] < .8:
                result['review_reasons'].append('low_confidence:detector')
            source_span = {'start': start, 'end': end}
            if span['label'] == 'SOFTWARE':
                sentence = next((s for s in sentences if s['start'] <= start < end <= s['end']),
                                {'start': 0, 'end': len(document['text'])})
                context = {'start': sentence['start'], 'end': sentence['end']}
                names.append({'mention_id': occurrence_id(document['document_id'], document['text_revision'], start, end),
                    'name': span['text'], 'name_span': source_span, 'software_score': span['score'],
                    'context_span': context, 'context_sentence': document['text'][context['start']:context['end']],
                    'context_kind': 'sentence'})
            elif span['label'] == 'VERSION':
                versions.append({'version_id': 'version:' + occurrence_id(document['document_id'], document['text_revision'], start, end),
                                 'text': span['text'], 'span': source_span})
        names.sort(key=lambda name: (name['name_span']['start'], name['name_span']['end']))
        result['stage_status']['detector']['occurrence_count'] = len(names)
        if not detector_ok:
            result['stage_status'].update({stage: _stage('blocked', reasons=['detector_failure']) for stage in STAGES})
            result['review_reasons'].append('incomplete_document')
            return validate_full_label_prediction(result, document)
        evaluations, exclusions = {}, {}
        for stage in STAGES:
            applicable = bool(names) and (bool(versions) if stage == 'linker' else len(names) > 1 if stage == 'alias' else True)
            if not applicable:
                result['stage_status'][stage] = _stage('not_applicable')
                evaluations[stage], exclusions[stage] = [], []
                continue
            checkpoint = self.stages.get(stage)
            try:
                tokenizer = checkpoint.tokenizer if checkpoint else deepcopy(self.detector.tokenizer)
                candidates = build_inference_candidates(document, names, versions, tokenizer,
                    checkpoint.manifest['recipe'] if checkpoint else {})
            except Exception as exc:
                result['stage_status'][stage] = _stage('failed', reasons=['candidate_construction:' + type(exc).__name__ + ': ' + str(exc)])
                evaluations[stage], exclusions[stage] = [], []
                continue
            exclusions[stage] = [row for row in candidates['excluded'] if row['stage'] == stage]
            result['exclusions'].extend(exclusions[stage])
            if checkpoint is None:
                reasons = self.manifest['stages'][stage]['unavailable_reasons']
                evaluations[stage] = [{**{key: row[key] for key in ('feature_id', 'first_id', 'second_id', 'first_span', 'second_span', 'context_span')},
                    'status': 'unavailable', 'scores': None, 'label': None, 'reasons': list(reasons),
                    'evidence_method': None, 'checkpoint_sha256': None} for row in candidates['features'][stage]]
                result['stage_status'][stage] = _stage('unavailable', len(evaluations[stage]) + sum(
                    row['reason'] not in NONCANDIDATES for row in exclusions[stage]), reasons=reasons)
                if stage in ('linker', 'alias') and not result['stage_status'][stage]['candidate_count']:
                    result['stage_status'][stage] = _stage('not_applicable')
            else:
                evaluations[stage] = self._evaluate(stage, candidates['features'][stage], exclusions[stage], result, document)
        for stage in STAGES:
            rows = [*evaluations[stage], *[row for row in exclusions[stage] if row['reason'] not in NONCANDIDATES]]
            members = {row['first_id'] for row in rows}
            if stage == 'alias':
                members.update(row['second_id'] for row in rows)
            result['stage_status'][stage]['occurrence_count'] = len(members)
        result['pair_predictions']['linker'] = evaluations['linker']
        for row in evaluations['alias']:
            result['alias_predictions']['pairs'].append({**row, 'pair_id': row['feature_id'],
                'member_mention_ids': [row['first_id'], row['second_id']], 'relation_type': None,
                'preferred_mention_id': None, 'preferred_name': None})
        for row in exclusions['alias']:
            if row['reason'] not in NONCANDIDATES:
                result['alias_predictions']['pairs'].append({**row, 'pair_id': row['feature_id'],
                    'member_mention_ids': [row['first_id'], row['second_id']], 'status': 'partial',
                    'scores': None, 'label': None, 'reasons': [row['reason']], 'relation_type': None,
                    'preferred_mention_id': None, 'preferred_name': None, 'evidence_method': None})
        result['alias_predictions']['groups'] = _alias_groups(result['alias_predictions']['pairs'])
        if any(group['negative_chord_pair_ids'] for group in result['alias_predictions']['groups']):
            result['review_reasons'].append('negative_chord_conflict')
        by_version = {row['version_id']: row for row in versions}
        for name in names:
            identity = name['mention_id']
            fields = {key: deepcopy(value) for key, value in name.items() if key != 'software_score'}
            fields['software'] = _field('success', name['name'], {'software': name['software_score']})
            links = [row for row in evaluations['linker'] if row['first_id'] == identity]
            missed = [row for row in exclusions['linker'] if row['first_id'] == identity and row['reason'] not in NONCANDIDATES]
            incomplete = bool(missed) or any(row['status'] != 'success' for row in links)
            if result['stage_status']['linker']['status'] == 'unavailable':
                fields['versions'] = _field('unavailable', reasons=['missing_stage'])
            elif result['stage_status']['linker']['status'] == 'failed' and not links:
                fields['versions'] = _field('failed', reasons=result['stage_status']['linker']['reasons'])
            elif incomplete:
                fields['versions'] = _field('partial', reasons=['unevaluated_version_candidate'])
            else:
                edges = [{'text': by_version[row['second_id']]['text'], 'span': by_version[row['second_id']]['span'],
                          'status': 'explicit_local'} for row in links if row['label'] == 'linked']
                scores = [row['scores']['linked'] for row in links if row['label'] == 'linked']
                fields['versions'] = _field('success', edges, scores, [] if edges else ['predicted_absence'])
            for stage, field in [('intent', 'intents'), ('sentiment', 'sentiment')]:
                rows = [row for row in evaluations[stage] if row['first_id'] == identity]
                if rows and rows[0]['status'] == 'success':
                    row = rows[0]
                    fields[field] = _field('success', row['label'], row['scores'], row['reasons'], row['context_span'])
                    fields[field]['context_text'] = row['context_text']
                else:
                    status = 'unavailable' if stage not in self.stages else 'failed'
                    reasons = [reason for row in rows for reason in row['reasons']] or [row['reason'] for row in exclusions[stage] if row['first_id'] == identity] or result['stage_status'][stage]['reasons']
                    fields[field] = _field(status, reasons=reasons)
            result['field_predictions'].append(fields)
            if all(fields[field]['status'] == 'success' for field in ('software', 'versions', 'intents', 'sentiment')):
                occurrence = {**{key: deepcopy(value) for key, value in name.items() if key != 'software_score'},
                    'schema_version': '2.0', 'document_id': document['document_id'], 'text_revision': document['text_revision'],
                    'run_id': result['run_id'], 'checkpoint_hashes': result['checkpoint_hashes'],
                    'version_links': fields['versions']['value'],
                    'version_status': 'explicit' if fields['versions']['value'] else 'absent',
                    'intents': fields['intents']['value'], 'sentiment': fields['sentiment']['value'],
                    'scores': {'software': name['software_score'], 'version_links': fields['versions']['scores'],
                        'intents': fields['intents']['scores'], 'sentiment': fields['sentiment']['scores']},
                    'calibration': 'uncalibrated', 'needs_review': True, 'review_reasons': list(WARNINGS),
                    'evidence': {'intents': [fields['intents']['context_span']] if fields['intents']['value'] != ['mentioned'] else [],
                        'sentiment': [fields['sentiment']['context_span']] if fields['sentiment']['value'] != 'not_expressed' else []}}
                occurrence['review_reasons'].extend(reason for field in ('intents', 'sentiment') for reason in fields[field]['reasons'])
                occurrence['review_reasons'].extend(reason for row in links for reason in row['reasons'])
                if name['software_score'] < .8:
                    occurrence['review_reasons'].append('low_confidence:detector')
                result['occurrences'].append(validate_occurrence(occurrence, document['text']))
        result['public_rows'], result['public_mappings'] = export_public(result['occurrences'])
        result['public_contract_complete'] = len(result['occurrences']) == len(names)
        result['pipeline_complete'] = all(stage['status'] in ('success', 'not_applicable') for stage in result['stage_status'].values())
        result['status'] = ('no_mentions' if not names else 'success') if result['pipeline_complete'] else 'partial'
        if not result['pipeline_complete']:
            result['review_reasons'].append('incomplete_pipeline')
        result['review_reasons'] = list(dict.fromkeys(result['review_reasons']))
        return validate_full_label_prediction(result, document)


def validate_full_label_prediction(result: dict, document: dict) -> dict:
    document = validate_document(document)
    if (result.get('schema_version') != 'full-label-prediction-1' or result.get('document_id') != document['document_id']
            or result.get('text_revision') != document['text_revision'] or result.get('offset_unit') != 'unicode_codepoint_half_open'):
        raise ValueError('incompatible prediction identity')
    stages = result.get('stage_status', {})
    if set(stages) != set(STAGE_NAMES) or any(row.get('status') not in STATUSES for row in stages.values()):
        raise ValueError('invalid stage status')
    for stage in stages.values():
        if any(type(stage.get(key)) is not int or stage[key] < 0 for key in ('candidate_count', 'completed_count', 'occurrence_count')) or stage['completed_count'] > stage['candidate_count']:
            raise ValueError('invalid stage counts')
        if stage['status'] in ('partial', 'unavailable', 'failed', 'blocked') and not stage['reasons']:
            raise ValueError('incomplete stage requires explicit reasons')
    complete = all(row['status'] in ('success', 'not_applicable') for row in stages.values())
    if type(result.get('pipeline_complete')) is not bool or result['pipeline_complete'] != complete:
        raise ValueError('inconsistent pipeline completeness')
    fields = result['field_predictions']
    ids, complete_ids = set(), set()
    for row in fields:
        start, end = check_span(row['name_span'], document, 'name_span', len(document['text']))
        if document['text'][start:end] != row['name'] or row['mention_id'] != occurrence_id(document['document_id'], document['text_revision'], start, end):
            raise ValueError('field source span mismatch')
        if row['mention_id'] in ids:
            raise ValueError('duplicate field occurrence')
        ids.add(row['mention_id'])
        for field in ('software', 'versions', 'intents', 'sentiment'):
            prediction = row[field]
            if prediction['status'] not in STATUSES or (prediction['status'] == 'success') != (prediction['value'] is not None):
                raise ValueError('inconsistent field prediction')
            if prediction['status'] != 'success' and (prediction['scores'] is not None or not prediction['reasons']):
                raise ValueError('incomplete field requires null scores and reason')
            if prediction['status'] == 'success':
                values = prediction['scores'].values() if isinstance(prediction['scores'], dict) else prediction['scores']
                for value in values:
                    _probability(value)
                stage = {'software': 'detector', 'versions': 'linker', 'intents': 'intent', 'sentiment': 'sentiment'}[field]
                if stages[stage]['status'] not in ('success', 'partial', 'not_applicable'):
                    raise ValueError('successful field conflicts with stage failure')
                if field == 'intents':
                    if set(prediction['scores']) != set(INTENT_BITS):
                        raise ValueError('intent scores require every bit')
                    expected = [bit for bit in INTENT_BITS if prediction['scores'][bit] >= .5] or ['mentioned']
                    if prediction['value'] != expected:
                        raise ValueError('intent labels do not match independent thresholds')
                if field == 'sentiment':
                    if set(prediction['scores']) != set(SENTIMENT_LABELS):
                        raise ValueError('sentiment scores require every class')
                    expected = max(SENTIMENT_LABELS, key=lambda label: prediction['scores'][label])
                    if prediction['value'] != expected:
                        raise ValueError('sentiment label does not match class scores')
            if prediction.get('context_span'):
                cs, ce = check_span(prediction['context_span'], document, 'context_span', len(document['text']))
                if prediction.get('context_text') != document['text'][cs:ce] or prediction['evidence_method'] != 'model_input_context':
                    raise ValueError('model input context mismatch')
        if all(row[field]['status'] == 'success' for field in ('software', 'versions', 'intents', 'sentiment')):
            complete_ids.add(row['mention_id'])
    occurrences = [validate_occurrence(row, document['text']) for row in result['occurrences']]
    if any('known' in row for row in occurrences) or {row['mention_id'] for row in occurrences} != complete_ids:
        raise ValueError('canonical occurrences must match complete prediction fields')
    by_id = {row['mention_id']: row for row in fields}
    for occurrence in occurrences:
        row = by_id[occurrence['mention_id']]
        if (occurrence['name'] != row['name'] or occurrence['name_span'] != row['name_span']
                or occurrence['version_links'] != row['versions']['value']
                or occurrence['intents'] != row['intents']['value'] or occurrence['sentiment'] != row['sentiment']['value']
                or occurrence['scores'] != {'software': row['software']['scores']['software'],
                    'version_links': row['versions']['scores'], 'intents': row['intents']['scores'], 'sentiment': row['sentiment']['scores']}):
            raise ValueError('canonical fields do not match resolved predictions')
    public, mappings = export_public(occurrences)
    if public != result['public_rows'] or mappings != result['public_mappings']:
        raise ValueError('public mappings do not match canonical occurrences')
    detector_ok = stages['detector']['status'] == 'success'
    detected = result['detector_diagnostics']
    if detector_ok:
        if not isinstance(detected, dict) or detected['status'] not in ('success', 'no_mentions') or any(
                chunk['status'] != 'success' for chunk in detected['chunks']):
            raise ValueError('detector diagnostics do not prove successful coverage')
        detected_ids = {occurrence_id(document['document_id'], document['text_revision'], span['start'], span['end'])
                        for span in detected['spans'] if span['label'] == 'SOFTWARE'}
        if ids != detected_ids:
            raise ValueError('field coverage must retain every detected occurrence')
    expected_public_complete = detector_ok and len(complete_ids) == len(fields)
    if type(result.get('public_contract_complete')) is not bool or result['public_contract_complete'] != expected_public_complete:
        raise ValueError('inconsistent public completeness')
    if not detector_ok and (fields or occurrences or public or any(stages[s]['status'] != 'blocked' for s in STAGES)):
        raise ValueError('detector failure must block downstream execution')
    expected_status = 'failure' if not detector_ok else 'partial' if not complete else 'success' if fields else 'no_mentions'
    if result['status'] != expected_status:
        raise ValueError('inconsistent document status')
    for pair in [*result['pair_predictions']['linker'], *result['alias_predictions']['pairs']]:
        if pair['status'] == 'success':
            if pair['scores'] is None or pair['label'] is None:
                raise ValueError('successful pair requires scores and label')
            for value in pair['scores'].values():
                _probability(value)
        elif pair['scores'] is not None or pair['label'] is not None or not pair['reasons']:
            raise ValueError('incomplete pair cannot carry negative predictions')
        for key in ('first_span', 'second_span', 'context_span'):
            if pair.get(key):
                check_span(pair[key], document, key, len(document['text']))
    for pair in result['alias_predictions']['pairs']:
        if any(pair.get(key) is not None for key in ('relation_type', 'preferred_mention_id', 'preferred_name')):
            raise ValueError('binary alias model cannot predict subtype or preferred name')
        if not set(pair['member_mention_ids']) <= ids or len(set(pair['member_mention_ids'])) != 2:
            raise ValueError('alias endpoints require two detected occurrences')
    if result['alias_predictions']['groups'] != _alias_groups(result['alias_predictions']['pairs']):
        raise ValueError('alias groups do not match model edges')
    return deepcopy(result)
