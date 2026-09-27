"""Reviewable agreement or explicitly scoped references from verified run bytes."""

from copy import deepcopy
import html
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from ..data.artifacts import atomic_write_new
from .backends import sha256
from .contracts import validate_result
from .metrics import agreement, score_reference
from .references import references_from_snapshot
from .runner import json_bytes, jsonl_bytes, publish, run_is_partial, validate_population


def _safe_path(root, name):
    if not isinstance(name, str):
        raise ValueError('unsafe artifact path')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or not path.parts or name != path.as_posix()
            or any(char in name for char in ('\\', ':', '\0'))):
        raise ValueError('unsafe artifact path')
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError('unsafe artifact symlink')
    return target


def _rows(payload):
    values = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    if any(not isinstance(row, dict) for row in values):
        raise ValueError('JSONL objects required')
    return values


def _verified_run(run):
    manifest_bytes = (run / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 'comparison-run-1' or manifest.get('status') not in ('complete', 'partial'):
        raise ValueError('terminal comparison run manifest required')
    evidence = {}
    for record in manifest['files']:
        name = record['path']
        if name in evidence or name == 'manifest.json':
            raise ValueError('duplicate artifact path')
        data = _safe_path(run, name).read_bytes()
        if sha256(data) != record['sha256']:
            raise ValueError(f'artifact hash mismatch: {name}')
        evidence[name] = data
    required = {'inputs.jsonl', 'windows.jsonl', 'results.jsonl', 'arms.json', 'preflight.json', 'context.json'}
    if not required <= evidence.keys():
        raise ValueError('missing required run artifacts')
    actual = {str(path.relative_to(run)) for path in run.rglob('*') if path.is_file()}
    if actual != set(evidence) | {'manifest.json'}:
        raise ValueError('artifact inventory mismatch')
    documents, windows = validate_population(_rows(evidence['inputs.jsonl']), _rows(evidence['windows.jsonl']))
    results = _rows(evidence['results.jsonl'])
    arms, preflight = json.loads(evidence['arms.json']), json.loads(evidence['preflight.json'])
    arm_ids = [arm['arm_id'] for arm in arms]
    if (not arm_ids or len(set(arm_ids)) != len(arm_ids) or arm_ids != manifest['arm_order']
            or preflight != manifest['arms'] or [row['arm_id'] for row in preflight] != arm_ids
            or json.loads(evidence['context.json']) != manifest['provenance']):
        raise ValueError('inconsistent frozen arm metadata')
    expected = {(arm, document['document_id']) for arm in arm_ids for document in documents}
    if len(results) != len(expected) or {(r['arm_id'], r['document_id']) for r in results} != expected:
        raise ValueError('incomplete arm/document population')
    if any(manifest[key] != value for key, value in [('document_count', len(documents)), ('window_count', len(windows)), ('result_count', len(results))]):
        raise ValueError('manifest population count mismatch')
    partial = run_is_partial(results, preflight)
    if manifest['status'] != ('partial' if partial else 'complete') or manifest['exit_code'] != int(partial):
        raise ValueError('manifest status differs from operational outcomes')
    by_id, by_arm = {d['document_id']: d for d in documents}, {r['arm_id']: r for r in preflight}
    for result in results:
        validate_result(by_id[result['document_id']], result)
        if result['capabilities'] != by_arm[result['arm_id']]['capabilities']:
            raise ValueError('inconsistent arm capabilities')
        index_path = result['raw_artifact']
        if index_path not in evidence or not index_path.startswith('raw/'):
            raise ValueError('missing verified raw index')
        index = json.loads(evidence[index_path])
        expected_ids = [w['window_id'] for w in windows if w['document_id'] == result['document_id']]
        if (index['arm_id'] != result['arm_id'] or index['document_id'] != result['document_id']
                or index['expected_window_ids'] != expected_ids):
            raise ValueError('raw index identity/population mismatch')
        found = [item['window_id'] for item in index['windows']]
        if found != (expected_ids if by_arm[result['arm_id']]['status'] == 'ready' else []):
            raise ValueError('raw window population mismatch')
        for item in index['windows']:
            if item['path'] not in evidence or not item['path'].startswith('raw/'):
                raise ValueError('missing verified raw window')
    return manifest, manifest_bytes, documents, results


def _reference_rows(references, documents):
    original = None
    if references.is_dir():
        rows = references_from_snapshot(references, documents)
        source = {'kind': 'verified_snapshot', 'path': str(references.resolve()),
                  'manifest_sha256': sha256((references / 'manifest.json').read_bytes())}
    else:
        data = references.read_bytes()
        original = data
        rows = _rows(data)
        source = {'kind': 'reference_jsonl', 'path': str(references.resolve()), 'sha256': sha256(data)}
    return rows, source, original


def _presentation(documents, results, arms):
    indexed = {(row['arm_id'], row['document_id']): row for row in results}
    rows = []
    for document in documents:
        comparisons = []
        for arm in arms:
            result = indexed[arm['arm_id'], document['document_id']]
            flags = ['native_scores_uncalibrated']
            if arm['identity'].get('verified') is not True:
                flags.append('model_identity_unverified')
            if result['status'] not in ('success', 'no_mentions'):
                flags.append('incomplete_extraction')
            if result['unresolved']:
                flags.append('unresolved_candidates')
            comparisons.append({**deepcopy(result), 'uncertainty_flags': flags,
                                'capability_gaps': [key for key, supported in arm['capabilities'].items() if not supported],
                                'policy_differences': arm['identity'].get('policy_differences', [])})
        rows.append({'document_id': document['document_id'], 'text_revision': document['text_revision'],
                     'text': document['text'], 'metadata': {k: v for k, v in document.items() if k != 'text'}, 'arms': comparisons})
    return rows


def _cell(value):
    return html.escape(str(value)).replace('|', '&#124;').replace('\n', '<br>').replace('\r', '&#13;')


def _markdown(report, run, output):
    lines = ['# Span comparison diagnostic', '', f"Mode: {report['mode']}. Run status: {report['run_status']}.", '',
             'Native scores are uncalibrated and are not compared across arms. This diagnostic makes no held-out quality or winner claim.', '',
             '## Timing and availability', '', '| Arm | Status | Load (s) | Warmup (s) | Measured (s) | Windows/s |',
             '| --- | --- | --- | --- | --- | --- |']
    for arm in report['arms_metadata']:
        timing = arm['timing']
        lines.append('| ' + ' | '.join(_cell(value) for value in [arm['arm_id'], arm['status'], timing['load_seconds'],
                     timing['warmup_seconds'], timing['measured_seconds'], timing['windows_per_second']]) + ' |')
    lines.extend(['', report['timing_policy'], ''])
    for document in report['documents']:
        lines.extend([f"## Document {_cell(document['document_id'])}", '', f"<pre>{html.escape(document['text'])}</pre>", '',
                      '| Arm | Status | Spans (code points) | Raw evidence | Uncertainty / gaps / policy |',
                      '| --- | --- | --- | --- | --- |'])
        for arm in document['arms']:
            spans = '; '.join(f"{span['label']} {span['start']}:{span['end']} {span['text']}" for span in arm['spans']) or 'none'
            raw = quote(os.path.relpath(run / arm['raw_artifact'], output), safe='/')
            notes = {'uncertainty': arm['uncertainty_flags'], 'capability_gaps': arm['capability_gaps'],
                     'policy_differences': arm['policy_differences'], 'reason': arm.get('reason')}
            lines.append(f"| {_cell(arm['arm_id'])} | {_cell(arm['status'])} | {_cell(spans)} | [raw index]({raw}) | {_cell(json.dumps(notes, ensure_ascii=False))} |")
        lines.append('')
    if report.get('references'):
        lines.extend(['## Scoped reference diagnostics', '', 'Human-reviewed and agent-provisional evidence are reported separately. Full metrics require explicit complete coverage.', ''])
        for kind, section in report['references'].items():
            lines.extend([f'### {kind}', '', '| Arm | Label | Operational TP / FP / FN | Precision / recall / F1 | Completed-only documents |',
                          '| --- | --- | --- | --- | --- |'])
            for arm_id, arm in section['arms'].items():
                for label, scored in arm['labels'].items():
                    metric = scored['operational']
                    lines.append('| ' + ' | '.join(_cell(v) for v in [arm_id, label,
                                 ' / '.join(str(metric[k]) for k in ('tp', 'fp', 'fn')),
                                 ' / '.join(str(metric[k]) for k in ('precision', 'recall', 'f1')),
                                 scored['denominators']['completed_only_documents']]) + ' |')
            lines.append('')
    else:
        lines.extend(['## Agreement', '', '| Left arm | Right arm | Label | Intersection / union | Jaccard |', '| --- | --- | --- | --- | --- |'])
        for pair in report['agreement']['pairs']:
            for label, metric in pair['labels'].items():
                lines.append('| ' + ' | '.join(_cell(v) for v in [pair['left_arm_id'], pair['right_arm_id'], label,
                             f"{metric['intersection']} / {metric['union']}", metric['jaccard']]) + ' |')
    return '\n'.join(lines) + '\n'


def build_report(run: Path, output: Path, *, references: Path | None = None) -> dict:
    run, output = Path(run), Path(output)
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError('report output must be outside the immutable run')
    if output.exists():
        raise FileExistsError(output)
    manifest, manifest_bytes, documents, results = _verified_run(run)
    report = {'schema_version': 'comparison-report-1', 'status': 'reported',
              'mode': 'reference_diagnostic' if references is not None else 'unlabelled_agreement',
              'heldout_quality_evaluated': False,
              'run_status': manifest['status'], 'run_manifest_sha256': sha256(manifest_bytes),
              'agreement': agreement(documents, results), 'arms_metadata': manifest['arms'],
              'timing_policy': manifest['timing_policy'], 'documents': _presentation(documents, results, manifest['arms'])}
    reference_rows, reference_original = None, None
    if references is not None:
        reference_rows, source, reference_original = _reference_rows(Path(references), documents)
        report['reference_source'] = source
        report['references'] = {kind: score_reference(documents, results, reference_rows, review_kind=kind)
                                for kind in ('human_reviewed', 'agent_provisional')}
    markdown = _markdown(report, run, output)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    if reference_rows is not None:
        publish(output, files, 'references.jsonl', jsonl_bytes(reference_rows))
    if reference_original is not None:
        publish(output, files, 'reference-source.jsonl', reference_original)
    publish(output, files, 'report.json', json_bytes(report))
    publish(output, files, 'report.md', markdown.encode('utf-8'))
    atomic_write_new(output / 'manifest.json', json_bytes({'schema_version': 'comparison-report-manifest-1',
                     'status': 'reported', 'run_manifest_sha256': sha256(manifest_bytes), 'files': files}))
    return report
