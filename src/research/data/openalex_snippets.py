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
import math
import re

from ..comparison.contracts import validate_input
from ..contracts import text_revision


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
