"""Regenerate an offline dashboard from immutable, population-matched evidence."""

import json
from copy import deepcopy
from pathlib import Path

from ..data.artifacts import atomic_write_new
from ..evaluation.metrics import evaluate_v2
from .attributes import pipeline_occurrences, softcite_occurrences, score_alias_pairs
from .backends import REPO_ROOT, sha256
from .html_render import render_dashboard
from .link_report import full_label_rows
from .links import _eligible, score_links
from .metrics import score_reference
from .references import REVIEW_KINDS
from .report import _rows, _safe_path, _verified_run
from .runner import json_bytes, publish, verify_files
from .pipeline_timing import _summary as timing_summary


def _sidecar(root, schema, run_hash):
    manifest_bytes = _safe_path(root, 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get('schema_version') != schema or manifest.get('status') != 'reported'
            or manifest.get('run_manifest_sha256') != run_hash):
        raise ValueError('sidecar schema/status/run identity mismatch')
    evidence = {}
    for record in manifest['files']:
        name = record['path']
        if name in evidence or name == 'manifest.json':
            raise ValueError('duplicate artifact path')
        data = _safe_path(root, name).read_bytes()
        if sha256(data) != record['sha256']:
            raise ValueError(f'artifact hash mismatch: {name}')
        evidence[name] = data
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(evidence) | {'manifest.json'}:
        raise ValueError('sidecar artifact inventory mismatch')
    report = json.loads(evidence['report.json'])
    if report.get('run_manifest_sha256') != run_hash:
        raise ValueError('report run identity mismatch')
    return report, evidence, manifest_bytes


def _span_keys(spans):
    return sorted((s['label'], s['start'], s['end'], s['text']) for s in spans)


def _compact_identity(identity):
    keys = ('backend', 'checkpoint', 'checkpoint_sha256', 'verified', 'config_verified',
            'config_sha256', 'model', 'model_digest', 'offset_unit', 'offset_unit_verified',
            'scores_calibrated', 'policy_differences')
    result = {k: identity[k] for k in keys if k in identity}
    checkpoint = identity.get('checkpoint_manifest', {})
    result.update({k: checkpoint[k] for k in ('recipe', 'decoding', 'files') if k in checkpoint})
    return result


def _summary(models, field):
    eligible = [m for m in models if m['metrics'][field] and m['metrics'][field]['f1'] is not None]
    if not eligible:
        return ['No scored reference coverage for this field and review kind; N/A is not zero accuracy.']
    best = max(m['metrics'][field]['f1'] for m in eligible)
    winners = [m['arm_id'] for m in eligible if m['metrics'][field]['f1'] == best]
    lines = [f"Highest observed F1 on this diagnostic subset: {', '.join(winners)} ({best:.1%}). This is not a held-out winner claim."]
    for model in eligible:
        metric = model['metrics'][field]
        if model['kind'] in ('pipeline', 'softcite'):
            lines.append(f"{model['arm_id']}: {metric['tp']} exact matches, {metric['fp']} false positives, {metric['fn']} missed references.")
    lines.append({'software': 'Improve missed names and boundaries; downstream heads cannot reliably recover a name that the detector missed.',
                  'version': 'Detecting a release string does not establish which software owns it; see the separate ownership score.',
                  'links': 'Prioritize wrong-owner links and scoped negative training examples. Masked endpoints are excluded, not judged correct.'}[field])
    return lines


def _document_summary(arms, ref, span_scores, document_id):
    lines = []
    for arm in arms:
        details = arm['link_details']
        if details and details['operational']['f1'] is not None:
            m = details['operational']
            lines.append(f"{arm['arm_id']}: links {m['tp']} correct / {m['fp']} extra / {m['fn']} missed; {details['excluded_predictions']} excluded.")
        scored = span_scores.get(arm.get('span_source_arm') or arm['arm_id'])
        if scored:
            row = next((d for d in scored['per_document'] if d['document_id'] == document_id), None)
            if row:
                metric = row['labels']['SOFTWARE']['operational']
                if metric['tp'] is not None and any(r['label'] == 'SOFTWARE' for r in ref.get('coverage', [])):
                    lines.append(f"{arm['arm_id']}: names {metric['tp']} correct / {metric['fp']} extra / {metric['fn']} missed.")
    note = ref.get('provenance', {}).get('review_notes')
    if note:
        lines.append('Reference review note: ' + note)
    return lines


def _selected_reference(ref, kind):
    coverage = [r for r in ref.get('coverage', []) if r['review_kind'] == kind]
    link_coverage = [r for r in ref.get('link_coverage', []) if r['review_kind'] == kind]
    spans = [s for s in ref.get('spans', []) if any(r['label'] == s['label'] and
             r['start'] <= s['start'] < s['end'] <= r['end'] for r in coverage)]
    regions = [(r['start'], r['end']) for r in link_coverage]
    ignored = [(r['start'], r['end']) for r in ref.get('ignored_versions', [])]
    ignored_names = [(r['start'], r['end']) for r in ref.get('ignored_software', [])]
    edges = [e for e in ref.get('version_links', []) if _eligible(
        (e['software']['start'], e['software']['end'], e['version']['start'], e['version']['end']),
        regions, ignored, ignored_names)]
    return {**ref, 'spans': spans, 'version_links': edges, 'coverage': coverage, 'link_coverage': link_coverage}


def _verify_appendix(value, frozen):
    observed = [e for e in value.get('examples', []) if e.get('kind', '').startswith('observed_')]
    if not observed:
        return
    evidence = value.get('evidence', {})
    if not evidence.get('probe_report_path') or not evidence.get('probe_report_sha256'):
        raise ValueError('observed examples require a pinned probe receipt')
    payload = _safe_path(REPO_ROOT, evidence['probe_report_path']).read_bytes()
    if sha256(payload) != evidence['probe_report_sha256']:
        raise ValueError('probe receipt hash mismatch')
    receipt = json.loads(payload)
    if receipt.get('schema_version') != 'base-scibert-probe-1' or receipt.get('local_files_only') is not True:
        raise ValueError('offline base probe receipt required')
    for declared, key in (('checkpoint_model_id', 'model_id'), ('checkpoint_revision', 'revision'),
                          ('checkpoint_weights_sha256', 'weights_sha256'), ('checkpoint_config_sha256', 'config_sha256')):
        if evidence.get(declared) != receipt.get(key):
            raise ValueError('probe checkpoint identity mismatch')
    heads = receipt['head_evidence']
    for declared, key in (('tensor_key_count', 'tensor_key_count'), ('task_classifier_keys', 'task_classifier_keys'),
                          ('config_id2label', 'config_id2label'), ('non_encoder_key_families', 'non_encoder_prefixes')):
        if evidence.get(declared) != heads.get(key):
            raise ValueError('probe head evidence mismatch')
    for example in observed:
        probe = next((p for p in receipt['probes'] if p['input'] == example.get('input')), None)
        if probe is None:
            raise ValueError('observed example missing from probe')
        expected = {'top5_tokens': [{k: token[k] for k in ('token', 'probability')} for token in probe['masked_token_top5']],
                    'last_hidden_shape': probe['representation_shape']}
        if example.get('kind') != 'observed_base_masked_token' or example.get('output') != expected:
            raise ValueError('observed example differs from probe')
    frozen['probe-report.json'] = payload


def _timing_evidence(paths, documents, models, natives, configurations, frozen):
    """Attach only document-identical, hash-joined timing arms."""
    expected = {d['document_id']: d for d in documents}
    rows_out = []
    seen = set()
    for index, path in enumerate(paths):
        root = Path(path)
        manifest_bytes = _safe_path(root, 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get('schema_version') != 'pipeline-timing-1':
            raise ValueError('timing schema mismatch')
        files = {}
        for item in manifest['files']:
            name = item['path']
            if name in files or name == 'manifest.json':
                raise ValueError('duplicate timing artifact')
            payload = _safe_path(root, name).read_bytes()
            if sha256(payload) != item['sha256']:
                raise ValueError('timing artifact hash mismatch')
            files[name] = payload
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if actual != set(files) | {'manifest.json'}:
            raise ValueError('timing artifact inventory mismatch')
        if set(files) != {'inputs.jsonl', 'requests.jsonl', 'first_pass.jsonl', 'softcite-config.json'}:
            raise ValueError('timing artifact inventory incomplete')
        if (sha256(files['inputs.jsonl']) != manifest.get('input_file_sha256')
                or sha256(files['softcite-config.json']) != manifest.get('softcite_config_sha256')):
            raise ValueError('timing input/config hash mismatch')
        timed_documents = _rows(files['inputs.jsonl'])
        if (len(timed_documents) != len(expected) or
                any(expected.get(d.get('document_id')) != d for d in timed_documents)):
            raise ValueError('timing document population/text/revision mismatch')
        order = manifest['order']
        if sorted(order) != sorted(expected) or manifest['document_count'] != len(expected):
            raise ValueError('timing document order/count mismatch')
        requests, first_pass = _rows(files['requests.jsonl']), _rows(files['first_pass.jsonl'])
        arm_ids = [a['arm_id'] for a in manifest['arms']]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError('duplicate timing arm')
        if manifest['repeats'] < 1 or manifest['batch_size'] != 1:
            raise ValueError('invalid timing schedule')
        expected_requests = [(a, repeat, d) for a in arm_ids
                             for repeat in range(1, manifest['repeats'] + 1) for d in order]
        if [(r.get('arm_id'), r.get('repeat'), r.get('document_id')) for r in requests] != expected_requests:
            raise ValueError('timing request rows differ from schedule')
        for row in requests:
            doc = expected[row['document_id']]
            if row.get('text_revision') != doc['text_revision'] or row.get('input_sha256') != doc['text_revision']:
                raise ValueError('timing request revision mismatch')
            elapsed = row.get('elapsed_seconds')
            if elapsed is not None and (type(elapsed) not in (float, int) or elapsed < 0):
                raise ValueError('invalid timing request elapsed seconds')
        first_by_key = {(r['arm_id'], r['document_id']): r for r in first_pass}
        if len(first_by_key) != len(first_pass):
            raise ValueError('duplicate timing first pass')
        active_arms = {a['arm_id'] for a in manifest['arms'] if a.get('load_status') == 'ready' and
                       a.get('warmup_status') in (None, 'success', 'no_mentions')}
        if set(first_by_key) != {(arm_id, doc_id) for arm_id in active_arms for doc_id in expected}:
            raise ValueError('timing first pass population mismatch')
        for (arm_id, doc_id), first in first_by_key.items():
            if first.get('input_sha256') != expected[doc_id]['text_revision']:
                raise ValueError('timing first pass revision mismatch')
            native = first.get('native')
            if isinstance(native, dict):
                if native.get('document_id', native.get('window_id')) != doc_id:
                    raise ValueError('timing native document identity mismatch')
                if native.get('document_id') is not None and native.get('text_revision') != expected[doc_id]['text_revision']:
                    raise ValueError('timing native revision mismatch')
        for arm in manifest['arms']:
            own_rows = [r for r in requests if r['arm_id'] == arm['arm_id']]
            if json_bytes(timing_summary(own_rows)) != json_bytes(arm['summary']):
                raise ValueError('timing summary differs from request rows')
            if arm['arm_id'] == 'softcite':
                matches = [m for m in models if m['kind'] == 'softcite' and
                           configurations[m['arm_id']]['config_source']['sha256'] == manifest['softcite_config_sha256']]
            else:
                checkpoint = arm.get('checkpoint_manifest_sha256')
                matches = [m for m in models if m['kind'] == 'pipeline' and
                           m['identity']['checkpoint_sha256'] == checkpoint]
            if len(matches) != 1:
                raise ValueError('unmatched timing checkpoint/config identity')
            model = matches[0]
            if (index, arm['arm_id']) in seen:
                raise ValueError('duplicate timing join')
            seen.add((index, arm['arm_id']))
            if model['kind'] == 'pipeline':
                for doc_id in expected:
                    first = first_by_key.get((arm['arm_id'], doc_id))
                    if first and first.get('measurement_error') is None:
                        measured, scored_native = first.get('native'), natives[(model['arm_id'], doc_id)]
                        if (not isinstance(measured, dict) or measured.get('document_id') != doc_id
                                or measured.get('text_revision') != expected[doc_id]['text_revision']):
                            raise ValueError('timing native input identity mismatch')
                        if (measured.get('checkpoint_hashes') is not None and
                                measured['checkpoint_hashes'] != scored_native.get('checkpoint_hashes')):
                            raise ValueError('timing native stage identity mismatch')
            joined = {'timing_arm_id': arm['arm_id'], 'arm_id': model['arm_id'],
                      'device': manifest['hardware']['device'], 'hardware': manifest['hardware'],
                      'load_seconds': arm.get('load_seconds'), 'load_semantics': arm.get('load_semantics'),
                      'load_status': arm.get('load_status'), 'summary': arm['summary'],
                      'policy': manifest['timing_policy']}
            model['timing'] = [*(model.get('timing') or [])] + [joined]
            rows_out.append(joined)
        frozen[f'timing-{index}/manifest.json'] = manifest_bytes
        frozen.update({f'timing-{index}/{name}': payload for name, payload in files.items()})
    return rows_out


def _field_evidence(documents, refs, models, natives, by_result, review_kind):
    if not any('attribute_occurrences' in ref for ref in refs):
        return None
    if any(not {'attribute_occurrences', 'attribute_coverage', 'alias_pairs'} <= ref.keys() for ref in refs):
        raise ValueError('attribute reference population incomplete')
    gold = [row for ref in refs for row in ref['attribute_occurrences']
            if row.get('review', {}).get('status', row.get('review_kind')) == review_kind]
    coverage = [row for ref in refs for row in ref['attribute_coverage'] if row.get('review_kind') == review_kind]
    masked_documents = sorted(ref['document_id'] for ref in refs if ref.get('ignored_versions'))
    gold, coverage = deepcopy(gold), deepcopy(coverage)
    for row in gold:
        if row['document_id'] in masked_documents:
            row['known']['versions'] = False
            row['version_status'] = 'ambiguous'
    for row in coverage:
        if row['document_id'] in masked_documents:
            row['fields']['versions'] = False
    aliases = []
    for ref in refs:
        for pair in ref['alias_pairs']:
            if pair.get('review', {}).get('status', pair.get('review_kind')) != review_kind or pair.get('known') is not True:
                continue
            members = pair['members']
            if len(members) != 2:
                raise ValueError('reviewed alias pair requires two endpoints')
            aliases.append({'document_id': pair['document_id'],
                            'left': [members[0]['span']['start'], members[0]['span']['end']],
                            'right': [members[1]['span']['start'], members[1]['span']['end']],
                            'label': pair['decision']})
    scores = {}
    for model in models:
        if model['kind'] not in ('pipeline', 'softcite'):
            continue
        predictions, statuses, positive_aliases = [], [], []
        pipeline_capabilities = None
        for doc in documents:
            ident = doc['document_id']
            if model['kind'] == 'pipeline':
                native = natives[(model['arm_id'], ident)]
                if not isinstance(native.get('capabilities'), dict):
                    raise ValueError('missing pipeline capabilities')
                if pipeline_capabilities is not None and pipeline_capabilities != native['capabilities']:
                    raise ValueError('inconsistent pipeline capabilities')
                pipeline_capabilities = native['capabilities']
                predictions.extend(pipeline_occurrences(doc, native))
                statuses.append({'document_id': ident, 'status': native['status']})
                for pair in native.get('alias_predictions', {}).get('pairs', []):
                    if pair.get('status') == 'success' and pair.get('label') == 'alias':
                        positive_aliases.append({'document_id': ident,
                            'left': [pair['first_span']['start'], pair['first_span']['end']],
                            'right': [pair['second_span']['start'], pair['second_span']['end']],
                            'label': 'alias'})
            else:
                result = by_result[(model['arm_id'], ident)]
                if result['status'] in ('success', 'no_mentions'):
                    offset_unit = model['identity'].get('offset_unit')
                    if offset_unit not in ('codepoint', 'unicode_codepoint_half_open', 'utf16'):
                        if not doc['text'].isascii():
                            raise ValueError('Softcite field scoring requires verified Unicode offset unit')
                        offset_unit = 'codepoint'
                    predictions.extend(softcite_occurrences(doc, result['chunks'], offset_unit))
                statuses.append({'document_id': ident, 'status': result['status']})
        capabilities = {'software': True, 'versions': True, 'version_offsets': True,
                        'intents': True, 'sentiment': model['kind'] == 'pipeline'}
        if pipeline_capabilities is not None:
            capabilities = {'software': pipeline_capabilities.get('software_spans') is True,
                            'versions': pipeline_capabilities.get('version_linking') is True,
                            'version_offsets': pipeline_capabilities.get('version_linking') is True,
                            'intents': pipeline_capabilities.get('intent') is True,
                            'sentiment': pipeline_capabilities.get('sentiment') is True}
        scored = evaluate_v2(documents, gold, predictions, coverage, statuses, capabilities)
        scored['version_field_masked_documents'] = masked_documents
        scored['aliases'] = score_alias_pairs(aliases, positive_aliases,
            supported=pipeline_capabilities is not None and pipeline_capabilities.get('aliases') is True)
        scores[model['arm_id']] = scored
    return scores


def _reference_limit(refs):
    if refs and all(ref.get('provenance', {}).get('source_only') is True and
                    ref.get('provenance', {}).get('prediction_exposed') is False for ref in refs):
        return ('Source-only fresh diagnostic sample with agent-provisional references. '
                'This report does not establish held-out quality or general model superiority.')
    return 'Development diagnostic, not a held-out benchmark. No general superiority claim follows from these scores.'


def _attach_notes(notes, passages, review_kind):
    by_id = {p['document_id']: p for p in passages}
    selected = []
    for note in notes:
        if note.get('scope') == 'scored_population':
            if note.get('review_kind') not in REVIEW_KINDS:
                raise ValueError('scored notes require a valid review kind')
            ids = note.get('document_ids', [])
            if not isinstance(ids, list) or any(not isinstance(i, str) or i not in by_id for i in ids):
                raise ValueError('scored note refers to documents outside the report population')
            if note['review_kind'] != review_kind:
                continue
            for ident in dict.fromkeys(ids):
                by_id[ident]['summaries'].append(f"Analysis: {note['title']}. {note['body']}")
        selected.append(note)
    return selected


def build_html_report(run, spans, links, output, *, appendix=None, notes=None,
                      review_kind='agent_provisional', timing=()):
    run, spans, output = Path(run), Path(spans), Path(output)
    links = [Path(p) for p in links]
    if review_kind not in REVIEW_KINDS:
        raise ValueError('unknown review kind')
    if not links:
        raise ValueError('at least one link sidecar required')
    if output.exists():
        raise FileExistsError(output)
    timing = [Path(p) for p in timing]
    if any(output.resolve().is_relative_to(p.resolve()) for p in [run, spans, *links, *timing]):
        raise ValueError('output must be outside immutable inputs')
    manifest, run_bytes, documents, results = _verified_run(run)
    run_hash = sha256(run_bytes)
    span_report, span_files, span_manifest = _sidecar(spans, 'comparison-report-manifest-1', run_hash)
    if span_report.get('schema_version') != 'comparison-report-1' or 'references.jsonl' not in span_files:
        raise ValueError('a scored span reference sidecar is required')
    refs = _rows(span_files['references.jsonl'])
    reference_bytes = span_files.get('reference-source.jsonl', span_files['references.jsonl'])
    reference_hash = sha256(reference_bytes)
    span_scores = {kind: score_reference(documents, results, refs, review_kind=kind) for kind in REVIEW_KINDS}
    if span_scores != span_report.get('references'):
        raise ValueError('span score differs from recomputed evidence')
    frozen = {'run/manifest.json': run_bytes, 'spans/manifest.json': span_manifest}
    for record in manifest['files']:
        payload = _safe_path(run, record['path']).read_bytes()
        if sha256(payload) != record['sha256']:
            raise ValueError('run artifact changed during report assembly')
        frozen['run/' + record['path']] = payload
    frozen.update({'spans/' + name: payload for name, payload in span_files.items()})
    predictions, natives, pipeline_identities = {}, {}, {}
    for index, link_path in enumerate(links):
        report, evidence, side_manifest = _sidecar(link_path, 'version-link-report-manifest-1', run_hash)
        if (report.get('schema_version') != 'version-link-report-1'
                or report.get('reference_sha256') != reference_hash
                or evidence.get('reference-source.jsonl') != reference_bytes):
            raise ValueError('link reference identity mismatch')
        rows = _rows(evidence['predictions.jsonl'])
        recomputed = {kind: score_links(documents, refs, rows, review_kind=kind) for kind in REVIEW_KINDS}
        if json_bytes(recomputed) != json_bytes(report.get('references')):
            raise ValueError('link score differs from recomputed evidence')
        for row in rows:
            key = row['arm_id'], row['document_id']
            if key in predictions and predictions[key] != row:
                raise ValueError('conflicting duplicate link prediction')
            predictions[key] = row
        for i, source in enumerate(report['full_label_sources']):
            data = evidence[f'full-label-{i}.jsonl']
            if sha256(data) != source['sha256']:
                raise ValueError('full-label source hash mismatch')
            native_rows = _rows(data)
            projected, identity = full_label_rows(documents, native_rows, source['arm_id'])
            if identity != source['checkpoint_sha256']:
                raise ValueError('pipeline identity mismatch')
            for row, raw in zip(projected, native_rows):
                key = row['arm_id'], row['document_id']
                if predictions.get(key) != row or (key in natives and natives[key] != raw):
                    raise ValueError('native pipeline differs from scored predictions')
                natives[key] = raw
            stage_hashes = [raw['checkpoint_hashes'] for raw in native_rows]
            if any(value != stage_hashes[0] for value in stage_hashes):
                raise ValueError('inconsistent pipeline stage identities')
            pipeline_identities[source['arm_id']] = {'checkpoint_sha256': identity, 'stages': stage_hashes[0]}
        frozen[f'links-{index}/manifest.json'] = side_manifest
        for name, data in evidence.items():
            frozen[f'links-{index}/{name}'] = data
    link_scores = {kind: score_links(documents, refs, list(predictions.values()), review_kind=kind) for kind in REVIEW_KINDS}
    active_spans, active_links = span_scores[review_kind]['arms'], link_scores[review_kind]['arms']
    configurations = json.loads(frozen['run/arms.json'])
    kinds = {c['arm_id']: {'scibert': 'detector'}.get(c['backend'], c['backend']) for c in configurations}
    by_result = {(r['arm_id'], r['document_id']): r for r in results}
    models = []
    for arm in manifest['arms']:
        arm_id = arm['arm_id']
        scored = active_spans[arm_id]
        linked = active_links.get(arm_id, {})
        operations = scored['operations']
        status = ('partial' if operations['completed_documents'] else 'failure') if operations['failed_documents'] else arm['status']
        models.append({'arm_id': arm_id, 'kind': kinds[arm_id], 'status': status,
                       'identity': _compact_identity(arm['identity']), 'span_source_arm': None,
                       'metrics': {'software': scored['labels']['SOFTWARE']['operational'],
                                   'version': scored['labels']['VERSION']['operational'],
                                   'links': linked.get('operational')},
                       'excluded_predictions': sum(d.get('excluded_predictions', 0) for d in linked.get('per_document', [])),
                       'operations': operations, 'timing': arm.get('timing'),
                       'notes': (['No trained task head: extraction is unsupported, not a zero-F1 system.'] if kinds[arm_id] == 'base'
                                 else ['Detector-only timing, not the full labeling pipeline.'] if kinds[arm_id] == 'detector'
                                 else ['Native service identity is not fully verified.'] if arm['identity'].get('verified') is not True else []) +
                       [f"{operations['completed_documents']}/{len(documents)} documents completed; {operations['failed_documents']} failed."]})
    for arm_id, identity in pipeline_identities.items():
        detector_hash = identity['stages']['detector'].removeprefix('sha256:')
        candidates = [a['arm_id'] for a in manifest['arms']
                      if a['identity'].get('checkpoint_sha256') == detector_hash]
        source = next((candidate for candidate in candidates if all(
            _span_keys(by_result[candidate, d['document_id']]['spans']) ==
            _span_keys(predictions[arm_id, d['document_id']]['spans']) for d in documents)), None)
        if candidates and source is None:
            raise ValueError('matching detector identity emitted different spans; cannot reuse span scores')
        scored = active_spans.get(source)
        linked = active_links[arm_id]
        models.append({'arm_id': arm_id, 'kind': 'pipeline', 'identity': identity,
                       'status': 'complete' if all(natives[arm_id, d['document_id']]['pipeline_complete'] for d in documents) else 'partial',
                       'span_source_arm': source,
                       'metrics': {'software': scored['labels']['SOFTWARE']['operational'] if scored else None,
                                   'version': scored['labels']['VERSION']['operational'] if scored else None,
                                   'links': linked['operational']},
                       'excluded_predictions': sum(d.get('excluded_predictions', 0) for d in linked['per_document']),
                       'operations': {'linking': {k: linked[k] for k in ('completed_documents', 'invalid_documents', 'supported')},
                                      'pipeline_complete_documents': sum(natives[arm_id, d['document_id']]['pipeline_complete'] for d in documents)},
                       'timing': None, 'notes': ['Intent, sentiment and aliases are visible but unscored. Full-pipeline timing was not measured.'] +
                       ([] if source else ['No identity-matched detector arm: span scores unavailable.'])})
    timing_rows = _timing_evidence(timing, documents, models, natives,
                                   {c['arm_id']: c for c in configurations}, frozen) if timing else []
    field_scores = _field_evidence(documents, refs, models, natives, by_result, review_kind)
    ref_by_id = {r['document_id']: r for r in refs}
    presentation = []
    for document in documents:
        ident = document['document_id']
        ref = _selected_reference(ref_by_id.get(ident, {}), review_kind)
        arms = []
        for model in models:
            arm_id = model['arm_id']
            prediction = predictions.get((arm_id, ident), by_result.get((arm_id, ident), {}))
            raw = natives.get((arm_id, ident), {})
            details = next((d for d in active_links.get(arm_id, {}).get('per_document', []) if d['document_id'] == ident), None)
            arms.append({'arm_id': arm_id, 'status': raw.get('status', prediction.get('status', 'unavailable')),
                         'spans': prediction.get('spans', []), 'version_links': prediction.get('version_links', []),
                         'span_source_arm': model['span_source_arm'], 'link_details': details,
                         'fields': raw.get('field_predictions', []), 'alias_groups': raw.get('alias_predictions', {}).get('groups', []),
                         'notes': [prediction['reason']] if prediction.get('reason') else []})
        presentation.append({**document, 'reference_spans': ref.get('spans', []),
                             'reference_links': ref.get('version_links', []), 'coverage': ref.get('coverage', []),
                             'link_coverage': ref.get('link_coverage', []),
                             'ignored_versions': ref.get('ignored_versions', []), 'ignored_software': ref.get('ignored_software', []),
                             'arms': arms, 'summaries': _document_summary(arms, ref, active_spans, ident)})
    appendix_data, feedback = None, []
    for path, key in ((appendix, 'appendix.json'), (notes, 'notes.json')):
        if path:
            payload = Path(path).read_bytes()
            frozen[key] = payload
            value = json.loads(payload)
            if key == 'appendix.json':
                if not isinstance(value, dict):
                    raise ValueError('appendix object required')
                _verify_appendix(value, frozen)
                appendix_data = value
            else:
                if not isinstance(value, list) or any(not isinstance(v, dict) or not {'title', 'body', 'provenance'} <= v.keys() for v in value):
                    raise ValueError('notes must be a list of attributed feedback records')
                feedback = _attach_notes(value, presentation, review_kind)
    data = {'schema_version': 'comparison-dashboard-1', 'title': 'Software extraction · comparison workbench',
            'provenance': {'review_kind': review_kind, 'population_documents': len(documents),
                           'run_manifest_sha256': run_hash, 'reference_sha256': reference_hash,
                           'run_status': manifest['status'], 'heldout_quality_evaluated': False,
                           'coverage': span_scores[review_kind]['coverage'], 'timing_policy': manifest['timing_policy']},
            'limitations': [_reference_limit(refs),
                            'Human-reviewed and agent-provisional scores are separate; missing coverage is N/A, not accuracy zero.',
                            'Ownership masks exclude every edge touching the ignored endpoint, including wrong owners. Inspect them below.',
                            'Intent, sentiment, aliases and confidence calibration have not been scored here.',
                            'Timing is engineering context only: span detector and remote/local service paths are not equivalent to full-label pipelines.'],
            'models': models, 'documents': presentation,
            'section_summaries': {f: _summary(models, f) for f in ('software', 'version', 'links')},
            'feedback': feedback, 'appendix': appendix_data,
            'scored_references': {'spans': span_scores, 'links': link_scores}}
    if timing_rows:
        data['timing_rows'] = timing_rows
        data['limitations'][4] = ('Timing is engineering context on identical passages. CPU/MPS measurements use their stated devices; '
                                   'model outputs may differ between timing and scored runs. Softcite service health is not cold model load, '
                                   'and AMD64 service timing is a separate environment.')
        for model in models:
            if model['kind'] == 'pipeline' and any(row['summary']['attempted'] for row in model.get('timing') or []):
                model['notes'] = [note.replace(' Full-pipeline timing was not measured.', '') for note in model['notes']]
    if field_scores is not None:
        data['field_scores'] = field_scores
        data['limitations'][3] = ('Intent, sentiment, and complete-occurrence scores use only the selected review kind; '
                                   'missing classes leave full macro scores N/A. Alias scores are conditional on explicitly reviewed endpoint pairs.')
        masked = sorted({ident for score in field_scores.values() for ident in score['version_field_masked_documents']})
        if masked:
            data['limitations'].append(f'Attribute version and complete-occurrence scoring conservatively excludes {len(masked)} '
                'passage(s) with ignored version endpoints; dedicated version and ownership panels keep precise endpoint masks. '
                'Excluded document IDs: ' + ', '.join(masked))
        for model in models:
            if model['kind'] == 'pipeline':
                model['notes'] = [note.replace('Intent, sentiment and aliases are visible but unscored.',
                                               'Selected-review field scores are shown below.') for note in model['notes']]
    pipelines = [m for m in models if m['kind'] == 'pipeline']
    for previous, candidate in zip(pipelines, pipelines[1:]):
        changed = [k for k in previous['identity']['stages'] if previous['identity']['stages'][k] != candidate['identity']['stages'].get(k)]
        data['section_summaries']['links'].append(
            f"{previous['arm_id']} → {candidate['arm_id']}: changed stage identities: {', '.join(changed) or 'none'}. "
            'Shared detector identity does not constitute a new name-detection improvement.')
    html = render_dashboard(data)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name, payload in frozen.items():
        publish(output, files, 'sources/' + name, payload)
    publish(output, files, 'report.json', json_bytes(data))
    publish(output, files, 'report.html', html.encode('utf-8'))
    verify_files(output, files)
    atomic_write_new(output / 'manifest.json', json_bytes({'schema_version': 'comparison-dashboard-manifest-1',
                     'status': 'reported', 'run_manifest_sha256': run_hash, 'files': files}))
    return data
