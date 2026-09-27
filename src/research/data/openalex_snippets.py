"""Pure, bounded parsing of diagnostic OpenAlex search excerpts.

``parse_response`` returns one row per selected snippet, including rejected blank
candidates. Count all rows against the acquisition budget; never refill rejected
slots. Accepted rows are frozen input documents. Rejected rows retain normalized
text/provenance but have no document ID or revision, and carry
``status='rejected', rejection_reason='blank_snippet'``.

``deduplicate_snippets`` returns accepted documents and identity-only reservations
for every candidate's parent, including rejected candidates. Reservations are
keyed by the exact (canonical OpenAlex ID, observed DOI) pair, so conflicting DOI
associations remain separate. Consumers must read all source associations and
identity rows rather than treating a document's first source_ids as exhaustive.
"""

from copy import deepcopy
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode

from ..comparison.contracts import validate_input
from ..contracts import text_revision
from .artifacts import atomic_write_new
from .manifest import digest, json_bytes
from .openalex_transport import ENDPOINT, QUERIES, fetch_snippet_response, validate_policy


NORMALIZATION_VERSION = "openalex-literal-em-html-unescape-1"
_WORK_ID = re.compile(r"(?:https://openalex\.org/)?(W[1-9][0-9]*)")
_MARKUP = re.compile(r"<(?:/?[A-Za-z][^>]*|![^>]*|\?[^>]*)>")
_POLICY = {
    "source": "openalex",
    "role": "diagnostic",
    "training_eligible": False,
    "future_untouched_test_eligible": False,
    "language": "unknown",
    "license_status": "unknown",
}


class OpenAlexSchemaError(ValueError):
    """Named response-schema failure; callers must not treat it as no results."""

    def __init__(self, code: str, field: str):
        self.code = code
        self.field = field
        super().__init__(f"{code}: {field}")


def normalize_snippet(raw: str) -> dict:
    """Remove literal highlight tags, decode once, and preserve all whitespace."""
    if not isinstance(raw, str):
        raise OpenAlexSchemaError("OPENALEX_INVALID_SNIPPET", "snippet")
    text = html.unescape(raw.replace("<em>", "").replace("</em>", ""))
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OpenAlexSchemaError("OPENALEX_INVALID_SNIPPET", "snippet.utf8") from exc
    return {
        "text": text,
        "normalization_version": NORMALIZATION_VERSION,
        "review_flags": ["unexpected_markup"] if _MARKUP.search(text) else [],
    }


def _validate_request(query: str, request: dict) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("OPENALEX_INVALID_QUERY")
    if not isinstance(request, dict):
        raise ValueError("OPENALEX_INVALID_REQUEST: requires an object")
    for key in ("url", "response_sha256", "retrieved_at"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"OPENALEX_INVALID_REQUEST: {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", request["response_sha256"]):
        raise ValueError("OPENALEX_INVALID_REQUEST: response_sha256")


def _source_ids(work: dict, field: str) -> dict:
    if not isinstance(work, dict):
        raise OpenAlexSchemaError("OPENALEX_INVALID_WORK", field)
    for key in ("id", "doi", "snippets", "relevance_score"):
        if key not in work:
            raise OpenAlexSchemaError("OPENALEX_MISSING_WORK_FIELD", f"{field}.{key}")
    match = _WORK_ID.fullmatch(work["id"]) if isinstance(work["id"], str) else None
    if match is None:
        raise OpenAlexSchemaError("OPENALEX_INVALID_WORK_ID", f"{field}.id")
    if work["doi"] is not None and (not isinstance(work["doi"], str) or not work["doi"].strip()):
        raise OpenAlexSchemaError("OPENALEX_INVALID_DOI", f"{field}.doi")
    if not isinstance(work["snippets"], list):
        raise OpenAlexSchemaError("OPENALEX_INVALID_SNIPPETS", f"{field}.snippets")
    score = work["relevance_score"]
    if type(score) not in (int, float) or (type(score) is float and not math.isfinite(score)):
        raise OpenAlexSchemaError("OPENALEX_INVALID_RELEVANCE_SCORE", f"{field}.relevance_score")
    return {"openalex": match[1], "doi": work["doi"]}


def parse_response(payload: dict, *, query: str, request: dict,
                   work_limit: int = 5, snippet_limit: int = 2) -> list[dict]:
    """Parse selected original ranks; request supplies raw response evidence.

    Required request fields are ``url``, bare lowercase 64-hex
    ``response_sha256``, and ``retrieved_at``. The caller hashes actual received
    bytes and supplies the retrieval timestamp; the parser cannot recover those
    from a decoded dictionary. Extra request fields are retained verbatim.
    """
    _validate_request(query, request)
    for key, value, maximum in (("work_limit", work_limit, 5), ("snippet_limit", snippet_limit, 2)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"OPENALEX_INVALID_LIMIT: {key}")
    if not isinstance(payload, dict):
        raise OpenAlexSchemaError("OPENALEX_INVALID_RESPONSE", "response")
    if "results" not in payload:
        raise OpenAlexSchemaError("OPENALEX_MISSING_RESULTS", "results")
    if not isinstance(payload["results"], list):
        raise OpenAlexSchemaError("OPENALEX_INVALID_RESULTS", "results")
    rows = []
    for work_rank, work in enumerate(payload["results"][:work_limit]):
        source_ids = _source_ids(work, f"results[{work_rank}]")
        for snippet_rank, raw in enumerate(work["snippets"][:snippet_limit]):
            if not isinstance(raw, str):
                raise OpenAlexSchemaError("OPENALEX_INVALID_SNIPPET", f"results[{work_rank}].snippets[{snippet_rank}]")
            normalized = normalize_snippet(raw)
            association = {
                "source_ids": deepcopy(source_ids),
                "work_id": work["id"],
                "query": query,
                "request": deepcopy(request),
                "request_url": request["url"],
                "response_sha256": request["response_sha256"],
                "retrieved_at": request["retrieved_at"],
                "work_rank": work_rank,
                "snippet_rank": snippet_rank,
                "raw_snippet": raw,
                "relevance_score": work["relevance_score"],
                "normalization_version": normalized["normalization_version"],
                "review_flags": deepcopy(normalized["review_flags"]),
            }
            row = {**_POLICY, **normalized, "source_ids": deepcopy(source_ids),
                   "source_associations": [association]}
            if not row["text"].strip():
                row.update(status="rejected", rejection_reason="blank_snippet")
            else:
                revision = text_revision(row["text"])
                row.update(status="accepted", text_revision=revision,
                           document_id="openalex-snippet:" + revision.removeprefix("sha256:"))
                row = validate_input(row)
            rows.append(row)
    return rows


def deduplicate_snippets(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplicate frozen exact text while retaining parent/query associations."""
    documents = {}
    identities = {}
    for candidate in rows:
        row = deepcopy(candidate)
        if row.get("status") == "accepted":
            row = validate_input(row)
            revision = row["text_revision"]
            if row["document_id"] != "openalex-snippet:" + revision.removeprefix("sha256:"):
                raise ValueError("OPENALEX_INVALID_CANDIDATE: document_id")
            if revision not in documents:
                documents[revision] = deepcopy(row)
            else:
                document = documents[revision]
                document["source_associations"].extend(deepcopy(row["source_associations"]))
                for flag in row["review_flags"]:
                    if flag not in document["review_flags"]:
                        document["review_flags"].append(flag)
        elif row.get("status") != "rejected" or row.get("rejection_reason") != "blank_snippet" or row.get("text", "").strip():
            raise ValueError("OPENALEX_INVALID_CANDIDATE: status/rejection_reason/text")
        for association in row["source_associations"]:
            source_ids = association["source_ids"]
            key = (source_ids["openalex"], source_ids["doi"])
            if key not in identities:
                identities[key] = {**_POLICY, "source_ids": deepcopy(source_ids), "source_associations": []}
            identities[key]["source_associations"].append(deepcopy(association))
    return list(documents.values()), list(identities.values())


def _verify_config_source(config: dict) -> None:
    source = config.get('config_source')
    if source is None:
        return
    if (not isinstance(source, dict) or set(source) != {'path', 'sha256', 'bytes_utf8'}
            or not isinstance(source['path'], str) or not Path(source['path']).is_absolute()
            or not isinstance(source['bytes_utf8'], str)):
        raise ValueError('invalid config_source')
    original = source['bytes_utf8'].encode('utf-8')
    if digest(original) != source['sha256'] or Path(source['path']).read_bytes() != original:
        raise ValueError('changed config_source bytes/hash')
    parsed = json.loads(original)
    if parsed != {key: value for key, value in config.items() if key != 'config_source'}:
        raise ValueError('config_source differs from effective acquisition config')


def _collection_config(config: dict) -> dict:
    bounds = {'page': (1, 1), 'per_page': (1, 5), 'snippets_per_work': (1, 2),
              'max_candidates': (1, 100), 'max_concurrency': (1, 2)}
    keys = set(bounds) | {'endpoint', 'queries', 'timeout_seconds', 'max_retries', 'max_bytes'}
    if not isinstance(config, dict) or not keys <= set(config) or set(config) - keys - {'config_source'}:
        raise ValueError('invalid collection config keys')
    config = deepcopy(config)
    if config['endpoint'] != ENDPOINT or config['queries'] != list(QUERIES):
        raise ValueError('invalid collection endpoint/queries')
    validate_policy({key: config[key] for key in ('timeout_seconds', 'max_retries', 'max_bytes')})
    for key, (minimum, maximum) in bounds.items():
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f'invalid collection bound: {key}')
    _verify_config_source(config)
    return config


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf-8')


def _empty_identities(payload: dict, query: str, request: dict, work_limit: int) -> list[dict]:
    identities = []
    for rank, work in enumerate(payload['results'][:work_limit]):
        source_ids = _source_ids(work, f'results[{rank}]')
        if work['snippets']:
            continue
        association = {'source_ids': source_ids, 'work_id': work['id'], 'query': query,
                       'request': deepcopy(request), 'request_url': request['url'],
                       'response_sha256': request['response_sha256'],
                       'retrieved_at': request['retrieved_at'], 'work_rank': rank,
                       'reason': 'no_snippets', 'relevance_score': work['relevance_score']}
        identities.append({**_POLICY, 'source_ids': source_ids, 'reason': 'no_snippets',
                           'source_associations': [association]})
    return identities


def collect_snippets(config: dict, output: Path) -> dict:
    """Publish a new immutable, diagnostic-only bundle with the manifest last."""
    config = _collection_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def publish(relative: str, payload: bytes) -> None:
        atomic_write_new(output / relative, payload)
        files.append({'path': relative, 'sha256': digest(payload)})

    plan = [{'query': query, 'query_rank': rank,
             'url': ENDPOINT + '?' + urlencode({'search': query, 'page': 1, 'per_page': config['per_page']})}
            for rank, query in enumerate(config['queries'])]
    publish('config.json', json_bytes(config))
    publish('request-plan.jsonl', _jsonl_bytes(plan))
    (output / 'raw').mkdir()
    policy = {key: config[key] for key in ('timeout_seconds', 'max_retries', 'max_bytes')}
    requests_log, candidates, identities, issues, stop_reasons = [], [], [], [], []
    completed = failed = 0
    stop = None
    for planned in plan:
        if stop is not None or len(candidates) >= config['max_candidates']:
            reason = stop or 'candidate_budget'
            if reason not in stop_reasons:
                stop_reasons.append(reason)
            requests_log.append({**planned, 'status': 'skipped', 'reason': reason, 'attempts': []})
            issues.append({**planned, 'reason': reason, 'kind': 'skipped_query'})
            continue
        raw_dir = Path('raw') / f'query-{planned["query_rank"]:02d}'
        record = fetch_snippet_response(planned['url'], output / raw_dir, policy=policy)
        for attempt in record['attempts']:
            if attempt['raw_path'] is not None:
                relative = (raw_dir / attempt['raw_path']).as_posix()
                if digest((output / relative).read_bytes()) != attempt['response_sha256']:
                    raise ValueError(f'raw response hash mismatch: {relative}')
                files.append({'path': relative, 'sha256': attempt['response_sha256']})
        transport_record = raw_dir / 'record.json'
        files.append({'path': transport_record.as_posix(),
                      'sha256': digest((output / transport_record).read_bytes())})
        entry = {**planned, **deepcopy(record), 'raw_directory': raw_dir.as_posix()}
        requests_log.append(entry)
        if record['stop_collection']:
            stop = record['reason']
            stop_reasons.append(stop)
        if record['status'] != 'completed':
            failed += 1
            issues.append({**planned, 'reason': record['reason'], 'kind': 'request_failed'})
            continue
        raw_path = output / raw_dir / record['raw_path']
        raw = raw_path.read_bytes()
        if digest(raw) != record['response_sha256']:
            raise ValueError('raw response hash mismatch')
        request = {key: record[key] for key in ('url', 'response_sha256', 'retrieved_at')}
        request.update(query_rank=planned['query_rank'], raw_path=raw_path.relative_to(output).as_posix())
        try:
            payload = json.loads(raw)
            rows = parse_response(payload, query=planned['query'], request=request,
                                  work_limit=config['per_page'], snippet_limit=config['snippets_per_work'])
            empty = _empty_identities(payload, planned['query'], request, config['per_page'])
        except (OpenAlexSchemaError, ValueError, UnicodeError) as exc:
            failed += 1
            entry.update(status='failed', reason='schema_error', schema_error=str(exc))
            issues.append({**planned, 'reason': 'schema_error', 'error': str(exc), 'kind': 'request_failed'})
            continue
        remaining = config['max_candidates'] - len(candidates)
        candidates.extend(rows[:remaining])
        identities.extend(empty)
        _, skipped_identities = deduplicate_snippets(rows[remaining:])
        for identity in skipped_identities:
            identity['reason'] = 'candidate_budget'
            for association in identity['source_associations']:
                association['reason'] = 'candidate_budget'
        identities.extend(skipped_identities)
        for row in rows[remaining:]:
            issues.append({'kind': 'skipped_candidate', 'reason': 'candidate_budget',
                           'source_associations': row['source_associations']})
        if rows[remaining:]:
            stop_reasons.append('candidate_budget')
        for work_rank, work in enumerate(payload['results']):
            if work_rank >= config['per_page']:
                issues.append({**planned, 'kind': 'skipped_work', 'reason': 'work_limit', 'work_rank': work_rank})
            elif len(work['snippets']) > config['snippets_per_work']:
                issues.append({**planned, 'kind': 'skipped_snippets', 'reason': 'snippet_limit',
                               'work_rank': work_rank, 'count': len(work['snippets']) - config['snippets_per_work']})
        completed += 1
    documents, candidate_identities = deduplicate_snippets(candidates)
    identities = candidate_identities + identities
    for row in candidates:
        if row['status'] == 'rejected':
            issues.append({'kind': 'rejected_candidate', 'reason': row['rejection_reason'], 'candidate': row})
    publish('requests.jsonl', _jsonl_bytes(requests_log))
    publish('inputs.jsonl', _jsonl_bytes(documents))
    publish('identities.jsonl', _jsonl_bytes(identities))
    publish('issues.jsonl', _jsonl_bytes(issues))
    _verify_config_source(config)
    for file in files:
        if digest((output / file['path']).read_bytes()) != file['sha256']:
            raise ValueError(f'artifact hash mismatch: {file["path"]}')
    status = 'failed' if not completed else 'partial' if failed or stop_reasons else 'completed'
    manifest = {**_POLICY, 'schema_version': 'openalex-diagnostic-1', 'status': status,
                'normalization_version': NORMALIZATION_VERSION, 'files': files,
                'stop_reasons': list(dict.fromkeys(stop_reasons)),
                'counts': {'planned_queries': len(plan), 'completed_queries': completed,
                           'failed_queries': failed, 'skipped_queries': len(plan) - completed - failed,
                           'candidates': len(candidates), 'rejected': sum(row['status'] == 'rejected' for row in candidates),
                           'documents': len(documents), 'identities': len(identities)},
                'diagnostic_limitations': ['query_selected_highlights', 'not_gold_labels',
                                           'paper_language_and_license_not_verified']}
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    return manifest
