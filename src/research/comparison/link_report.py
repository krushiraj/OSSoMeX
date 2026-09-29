"""Offline version-link sidecar for verified comparisons and full-label CLI JSONL."""

import json
from pathlib import Path

from ..data.artifacts import atomic_write_new
from .backends import sha256
from .contracts import COMPLETED_STATUSES
from .links import ollama_links, score_links, softcite_links
from .references import REVIEW_KINDS
from .report import _rows, _verified_run
from .runner import json_bytes, jsonl_bytes, publish, verify_files


def full_label_rows(documents, raw_rows, arm):
    from ..training.full_label import validate_full_label_prediction
    by_id = {doc['document_id']: doc for doc in documents}
    rows, seen, identities = [], set(), set()
    for raw in raw_rows:
        ident = raw.get('document_id')
        if ident not in by_id or ident in seen:
            raise ValueError('unknown or duplicate full-label document')
        seen.add(ident)
        native = validate_full_label_prediction(raw, by_id[ident])
        identities.add(native['checkpoint_sha256'])
        supported = native['capabilities'].get('version_linking') is True
        complete = (native['stage_status']['detector']['status'] == 'success'
                    and native['stage_status']['linker']['status'] in ('success', 'not_applicable')
                    and all(row['versions']['status'] == 'success' for row in native['field_predictions']))
        edges = []
        if complete and supported:
            for row in native['field_predictions']:
                edges.extend({'software': {**row['name_span'], 'text': row['name']},
                              'version': {**edge['span'], 'text': edge['text']}}
                             for edge in row['versions']['value'])
        rows.append({'arm_id': arm, 'document_id': ident, 'text_revision': native['text_revision'],
                     'supported': supported, 'status': 'unsupported' if not supported else
                     ('success' if complete else 'failure'),
                     'reason': None if complete else 'incomplete_detector_or_linker',
                     'spans': (native['detector_diagnostics'] or {}).get('spans', [])
                     if native['stage_status']['detector']['status'] == 'success' else [],
                     'version_links': edges})
    if len(identities) != 1:
        raise ValueError('one full-label checkpoint identity required per arm')
    if seen != set(by_id):
        raise ValueError('full-label output must cover the exact frozen population')
    return rows, identities.pop()


def build_link_report(run, references, full_labels, output):
    run, references, output = Path(run), Path(references), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError('report must be outside immutable run')
    manifest, manifest_bytes, documents, results = _verified_run(run)
    ref_bytes = references.read_bytes()
    refs = _rows(ref_bytes)
    predictions, sources = [], []
    arms = {row['arm_id']: row for row in manifest['arms']}
    # Existing span-only backends remain N/A here unless native ownership is available.
    config_arms = json.loads((run / 'arms.json').read_bytes())
    softcite_arms = {row['arm_id'] for row in config_arms if row.get('backend') == 'softcite'}
    ollama_arms = {row['arm_id'] for row in config_arms if row.get('backend') == 'ollama'}
    for result in results:
        arm = result['arm_id']
        row = {key: result[key] for key in ('arm_id', 'document_id', 'text_revision', 'status', 'spans')}
        row.update(supported=arm in softcite_arms | ollama_arms, version_links=[], reason=result.get('reason'))
        if row['supported'] and row['status'] in COMPLETED_STATUSES:
            try:
                row['version_links'] = (softcite_links(result['chunks'], arms[arm]['identity'].get('offset_unit'))
                                        if arm in softcite_arms else ollama_links(result['chunks']))
            except (KeyError, ValueError, TypeError) as exc:
                row.update(status='failure', spans=[], version_links=[], reason=str(exc))
        predictions.append(row)
    used = set(arms)
    frozen = []
    for entry in full_labels:
        arm, separator, path = entry.partition('=')
        if not separator or not arm.strip() or arm in used:
            raise ValueError('full-label requires a unique ARM=PATH')
        used.add(arm)
        data = Path(path).read_bytes()
        rows, checkpoint = full_label_rows(documents, _rows(data), arm)
        predictions.extend(rows)
        sources.append({'arm_id': arm, 'path': str(Path(path).resolve()), 'sha256': sha256(data),
                        'checkpoint_sha256': checkpoint})
        frozen.append(data)
    report = {'schema_version': 'version-link-report-1', 'heldout_quality_evaluated': False,
              'run_manifest_sha256': sha256(manifest_bytes), 'full_label_sources': sources,
              'reference_sha256': sha256(ref_bytes), 'source_arm_metadata': manifest['arms'],
              'notes': ['Agent-provisional and human review are separate.',
                        'Conditional scores use only endpoints detected exactly, not oracle reruns.',
                        'Full-label inputs are whole frozen documents; span backends use frozen windows.',
                        'No alias, intent, sentiment, calibration or speed claim is made by this link scorer.'],
              'references': {kind: score_links(documents, refs, predictions, review_kind=kind)
                             for kind in REVIEW_KINDS}}
    lines = ['# Version-link diagnostic', '', 'No held-out quality or winner claim. Unsupported/unavailable arms are N/A.', '']
    for kind, section in report['references'].items():
        lines += [f'## {kind}', '', '| Arm | TP / FP / FN | Precision / Recall / F1 |', '| --- | --- | --- |']
        for arm, row in section['arms'].items():
            metric = row['operational']
            counts = ' / '.join(str(metric[k]) if metric[k] is not None else 'N/A' for k in ('tp', 'fp', 'fn'))
            scores = ' / '.join(f'{metric[k]:.3f}' if metric[k] is not None else 'N/A' for k in ('precision', 'recall', 'f1'))
            lines.append(f'| {arm.replace(chr(124), chr(47))} | {counts} | {scores} |')
        lines.append('')
    lines.extend(report['notes'])
    output.mkdir(parents=True, exist_ok=False)
    files = []
    publish(output, files, 'reference-source.jsonl', ref_bytes)
    for i, data in enumerate(frozen):
        publish(output, files, f'full-label-{i}.jsonl', data)
    publish(output, files, 'predictions.jsonl', jsonl_bytes(predictions))
    publish(output, files, 'report.json', json_bytes(report))
    publish(output, files, 'report.md', ('\n'.join(lines) + '\n').encode())
    verify_files(output, files)
    atomic_write_new(output / 'manifest.json', json_bytes({'schema_version': 'version-link-report-manifest-1',
                     'status': 'reported', 'run_manifest_sha256': sha256(manifest_bytes), 'files': files}))
    return report
