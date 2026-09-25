"""Prepare a fresh review fork with checked, system-authored source issues."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from research.annotations.review_store import apply_decision, export_reference, get_item, open_store
from research.annotations.snapshots import restore_review_snapshot, validate_snapshot_store
from research.data.manifest import digest, json_bytes, write_once


def _rows(connection):
    connection.execute('BEGIN')
    try:
        return {table: [dict(row) for row in connection.execute(f'SELECT * FROM {table} ORDER BY {key}')]
                for table, key in [('items', 'task_id'), ('decisions', 'rowid'), ('metadata', 'rowid')]}
    finally:
        connection.rollback()


def _hashes(rows):
    return {table: digest(json.dumps(records, ensure_ascii=False, separators=(',', ':')).encode())
            for table, records in rows.items()}


def _readonly(path):
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    return connection


def _requests(connection, issues_bytes, report_hash):
    issues = []
    for number, line in enumerate(issues_bytes.decode('utf-8').splitlines(), 1):
        try:
            issue = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'Invalid issue JSON on line {number}') from exc
        required = {'issue_id', 'code', 'message', 'span', 'task_id', 'document_id',
                    'text_revision', 'source_report_sha256'}
        if not isinstance(issue, dict) or set(issue) != required:
            raise ValueError(f'Invalid issue fields on line {number}')
        if issue['source_report_sha256'] != report_hash:
            raise ValueError('SOURCE_REPORT_HASH_MISMATCH')
        issues.append(issue)
    if not issues:
        raise ValueError('ISSUES_REQUIRED')
    seen = set()
    requests = []
    for issue in issues:
        if not isinstance(issue['issue_id'], str) or issue['issue_id'] in seen:
            raise ValueError('DUPLICATE_OR_INVALID_ISSUE_ID')
        seen.add(issue['issue_id'])
        if not isinstance(issue['task_id'], str):
            raise ValueError('INVALID_TASK_ID')
        current = get_item(connection, issue['task_id'])
        task = current['task']
        if any(issue[key] != task[key] for key in ('document_id', 'text_revision')):
            raise ValueError('SOURCE_REVISION_MISMATCH')
        request = {'decision_id': 'source-issue:' + digest(json_bytes(issue)),
                   **{key: task[key] for key in ('task_id', 'document_id', 'text_revision')},
                   'base_annotation_revision': current['annotation_revision'],
                   'reviewer': 'selection-review-preparation', 'actor_kind': 'system',
                   'action': 'apply_review_batch', 'reason': 'Import checked source report issue.',
                   'value': {'schema_version': '1.0', 'completion': 'save', 'proposals_revealed': False,
                             'operations': [{'operation_id': 'source-issue:' + issue['issue_id'],
                                             'action': 'record_source_issue', 'reason_code': 'broken_passage',
                                             'value': issue}]}}
        if connection.execute('SELECT 1 FROM decisions WHERE decision_id=?', (request['decision_id'],)).fetchone():
            raise ValueError('SOURCE_ISSUE_ALREADY_IMPORTED')
        # Exercise the same identity, span, provenance and duplicate checks as the final import.
        apply_decision(connection, request)
        requests.append(request)
    return requests


def prepare_preview(source_store: Path, destination: Path, issues_file: Path, report_file: Path) -> dict:
    """Never mutate the source; leave failed destinations intact for diagnosis."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.resolve()
    source_store, issues_file, report_file = [Path(path).resolve() for path in (source_store, issues_file, report_file)]
    for path in (source_store, issues_file, report_file):
        if path == destination or path.is_relative_to(destination) or destination.is_relative_to(path):
            raise ValueError('SOURCE_DESTINATION_OVERLAP')
        if not path.is_file():
            raise FileNotFoundError(path)
    issues_bytes, report_bytes = issues_file.read_bytes(), report_file.read_bytes()
    report_hash = digest(report_bytes)
    with closing(_readonly(source_store)) as source, closing(sqlite3.connect(':memory:')) as captured:
        captured.row_factory = sqlite3.Row
        # export_reference owns BEGIN; pin a WAL-aware SQLite snapshot before its transaction.
        source.execute('BEGIN')
        source.execute('SELECT count(*) FROM items').fetchone()
        try:
            source.backup(captured)
        finally:
            source.rollback()
        baseline = _rows(captured)
        with closing(sqlite3.connect(':memory:', isolation_level=None)) as validation:
            validation.row_factory = sqlite3.Row
            captured.backup(validation)
            requests = _requests(validation, issues_bytes, report_hash)
        destination.mkdir(parents=True, exist_ok=False)
        export_reference(captured, destination / 'baseline')
        restore_review_snapshot(destination / 'baseline', destination / 'review.sqlite')
        with closing(open_store(destination / 'review.sqlite')) as prepared:
            if _rows(prepared) != baseline:
                raise ValueError('BASELINE_RESTORE_MISMATCH')
            for request in requests:
                apply_decision(prepared, request)
            final = _rows(prepared)
            if final['decisions'][:len(baseline['decisions'])] != baseline['decisions']:
                raise ValueError('DECISION_PREFIX_MISMATCH')
            export_reference(prepared, destination / 'prepared')
        validate_snapshot_store(destination / 'baseline', destination / 'review.sqlite')
        validate_snapshot_store(destination / 'prepared', destination / 'review.sqlite')
        write_once(destination / 'source-issues.jsonl', issues_bytes)
        write_once(destination / 'source-report.md', report_bytes)
        after = _rows(source)
        unchanged = after == baseline
        result = {'schema_version': '1.0', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                  'source_store': str(source_store), 'destination': str(destination),
                  'source_report': str(report_file), 'source_issues': str(issues_file),
                  'source_report_sha256': report_hash, 'issues_sha256': digest(issues_bytes),
                  'baseline_table_sha256': _hashes(baseline), 'source_after_table_sha256': _hashes(after),
                  'prepared_table_sha256': _hashes(final), 'source_unchanged': unchanged,
                  'ready_for_review': unchanged, 'decision_prefix_preserved': True,
                  'baseline_decision_count': len(baseline['decisions']),
                  'prepared_decision_count': len(final['decisions']),
                  'imported_system_issue_count': len(requests),
                  'warning': None if unchanged else 'Source changed during preparation. Prepare a new destination before review.'}
        write_once(destination / 'preservation.json', json_bytes(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-store', 'destination', 'issues', 'report'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_preview(args.source_store, args.destination, args.issues, args.report)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'Preparation failed: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0 if result['ready_for_review'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
