from copy import deepcopy
import hashlib
import importlib

import pytest

from research.comparison.contracts import validate_input


@pytest.fixture
def api():
    return importlib.import_module("research.data.openalex_snippets")


@pytest.fixture
def request_metadata():
    return {
        "url": "https://api.openalex.org/funder-search?search=NumPy&page=1&per_page=5",
        "response_sha256": "a" * 64,
        "retrieved_at": "2026-09-27T11:00:00Z",
        "attempt": 1,
    }


def work(work_id="W123", snippets=None, doi=None):
    return {
        "id": "https://openalex.org/" + work_id,
        "doi": doi,
        "snippets": ["<em>NumPy</em>"] if snippets is None else snippets,
        "relevance_score": 28.5,
    }


def test_normalization_preserves_offsets(api):
    normalized = api.normalize_snippet("<em>NumPy</em>&#x20;\r\n")
    assert normalized["text"] == "NumPy \r\n"
    assert normalized["normalization_version"]
    assert normalized["review_flags"] == []
    assert api.normalize_snippet("&amp;#x20;")["text"] == "&#x20;"
    assert api.normalize_snippet(" A\t😀\r\n B ")["text"] == " A\t😀\r\n B "


@pytest.mark.parametrize("raw,expected", [
    ("<b>NumPy</b>", "<b>NumPy</b>"),
    ('<em class="hit">NumPy</em>', '<em class="hit">NumPy'),
    ("<EM>NumPy</EM>", "<EM>NumPy</EM>"),
    ("&lt;em&gt;NumPy&lt;/em&gt;", "<em>NumPy</em>"),
    ("<!-- comment -->NumPy", "<!-- comment -->NumPy"),
])
def test_unexpected_markup_is_flagged_and_retained(api, raw, expected):
    normalized = api.normalize_snippet(raw)
    assert normalized["text"] == expected
    assert "unexpected_markup" in normalized["review_flags"]


def test_provenance_and_diagnostic_policy(api, request_metadata):
    payload = {"results": [work(snippets=["<em>NumPy</em>\r\n"])]}
    original = deepcopy(payload)
    rows = api.parse_response(payload, query="NumPy", request=request_metadata)
    documents, identities = api.deduplicate_snippets(rows)
    assert payload == original
    assert len(rows) == len(documents) == len(identities) == 1
    document = documents[0]
    digest = hashlib.sha256(b"NumPy\r\n").hexdigest()
    assert document["document_id"] == "openalex-snippet:" + digest
    assert document["text_revision"] == "sha256:" + digest
    assert validate_input(document) == document
    for record in [rows[0], document, identities[0]]:
        assert record["role"] == "diagnostic"
        assert record["training_eligible"] is False
        assert record["future_untouched_test_eligible"] is False
        assert record["license_status"] == "unknown"
        assert record["language"] == "unknown"
    association = document["source_associations"][0]
    assert association["raw_snippet"] == "<em>NumPy</em>\r\n"
    assert association["query"] == "NumPy"
    assert association["request"] == request_metadata
    assert association["request_url"] == request_metadata["url"]
    assert association["response_sha256"] == request_metadata["response_sha256"]
    assert association["retrieved_at"] == request_metadata["retrieved_at"]
    assert association["work_rank"] == association["snippet_rank"] == 0
    assert association["relevance_score"] == 28.5
    assert association["source_ids"] == {"openalex": "W123", "doi": None}
    assert document["source_ids"] == association["source_ids"]
    assert "score" not in document
    assert identities[0]["source_ids"] == document["source_ids"]
    request_metadata["attempt"] = 99
    rows[0]["source_associations"][0]["raw_snippet"] = "mutated"
    assert association["request"]["attempt"] == 1
    assert association["raw_snippet"] == "<em>NumPy</em>\r\n"


def test_every_query_and_parent_survives_exact_text_deduplication(api, request_metadata):
    payload = {"results": [work("W123"), work("W456", ["NumPy"], "https://doi.org/10.1/b")]}
    rows = []
    for query in ("NumPy", "Python"):
        rows += api.parse_response(payload, query=query, request=request_metadata)
    documents, identities = api.deduplicate_snippets(rows)
    assert len(documents) == 1
    associations = documents[0]["source_associations"]
    assert [(row["query"], row["source_ids"]["openalex"]) for row in associations] == [
        ("NumPy", "W123"), ("NumPy", "W456"), ("Python", "W123"), ("Python", "W456"),
    ]
    assert [row["source_ids"]["openalex"] for row in identities] == ["W123", "W456"]
    assert [len(row["source_associations"]) for row in identities] == [2, 2]
    assert documents[0]["source_ids"] == {"openalex": "W123", "doi": None}


def test_conflicting_dois_are_retained_as_distinct_reservations(api, request_metadata):
    payload = {"results": [work(doi="https://doi.org/10.1/a"), work(doi="https://doi.org/10.1/b")]}
    documents, identities = api.deduplicate_snippets(api.parse_response(payload, query="R", request=request_metadata))
    assert len(documents) == 1
    assert [row["source_ids"]["doi"] for row in identities] == ["https://doi.org/10.1/a", "https://doi.org/10.1/b"]


def test_blank_candidates_count_without_replacements_and_reserve_parents(api, request_metadata):
    payload = {"results": [
        work("W1", ["<em> </em>", "\r\n", "replacement"]),
        work("W2", []),
        work("W3", ["third", "third two", "ignored"]),
        work("W4", ["fourth"]),
        work("W5", ["fifth"]),
        work("W6", ["sixth"]),
    ]}
    rows = api.parse_response(payload, query="R", request=request_metadata)
    assert len(rows) == 6
    assert [row["status"] for row in rows] == ["rejected", "rejected", "accepted", "accepted", "accepted", "accepted"]
    assert [row["rejection_reason"] for row in rows[:2]] == ["blank_snippet", "blank_snippet"]
    assert all("document_id" not in row for row in rows[:2])
    assert [row["text"] for row in rows] == [" ", "\r\n", "third", "third two", "fourth", "fifth"]
    documents, identities = api.deduplicate_snippets(rows)
    assert [row["text"] for row in documents] == ["third", "third two", "fourth", "fifth"]
    assert [row["source_ids"]["openalex"] for row in identities] == ["W1", "W3", "W4", "W5"]
    assert [row["snippet_rank"] for row in identities[0]["source_associations"]] == [0, 1]


@pytest.mark.parametrize("payload,code", [
    (None, "OPENALEX_INVALID_RESPONSE"),
    ({}, "OPENALEX_MISSING_RESULTS"),
    ({"results": None}, "OPENALEX_INVALID_RESULTS"),
    ({"results": {}}, "OPENALEX_INVALID_RESULTS"),
    ({"results": [None]}, "OPENALEX_INVALID_WORK"),
])
def test_bad_response_schema_has_named_error(api, request_metadata, payload, code):
    with pytest.raises(api.OpenAlexSchemaError, match=code) as error:
        api.parse_response(payload, query="R", request=request_metadata)
    assert error.value.code == code


@pytest.mark.parametrize("field,value,code", [
    ("id", None, "OPENALEX_INVALID_WORK_ID"),
    ("id", "https://openalex.org/A123", "OPENALEX_INVALID_WORK_ID"),
    ("id", "https://evil.org/W123", "OPENALEX_INVALID_WORK_ID"),
    ("id", "https://openalex.org/W123/", "OPENALEX_INVALID_WORK_ID"),
    ("doi", 12, "OPENALEX_INVALID_DOI"),
    ("snippets", None, "OPENALEX_INVALID_SNIPPETS"),
    ("snippets", "NumPy", "OPENALEX_INVALID_SNIPPETS"),
    ("snippets", [None], "OPENALEX_INVALID_SNIPPET"),
    ("snippets", [{"text": "NumPy"}], "OPENALEX_INVALID_SNIPPET"),
    ("relevance_score", None, "OPENALEX_INVALID_RELEVANCE_SCORE"),
    ("relevance_score", True, "OPENALEX_INVALID_RELEVANCE_SCORE"),
    ("relevance_score", float("nan"), "OPENALEX_INVALID_RELEVANCE_SCORE"),
    ("relevance_score", float("inf"), "OPENALEX_INVALID_RELEVANCE_SCORE"),
])
def test_bad_work_schema_has_named_error(api, request_metadata, field, value, code):
    result = work()
    result[field] = value
    with pytest.raises(api.OpenAlexSchemaError, match=code):
        api.parse_response({"results": [result]}, query="R", request=request_metadata)


@pytest.mark.parametrize("field", ["id", "doi", "snippets", "relevance_score"])
def test_missing_observed_work_field_is_schema_drift(api, request_metadata, field):
    result = work()
    del result[field]
    with pytest.raises(api.OpenAlexSchemaError, match="OPENALEX_MISSING_WORK_FIELD"):
        api.parse_response({"results": [result]}, query="R", request=request_metadata)


def test_limits_select_original_ranks_and_do_not_validate_unselected_candidates(api, request_metadata):
    payload = {"results": [work("W1", ["one", None]), None]}
    rows = api.parse_response(payload, query="R", request=request_metadata, work_limit=1, snippet_limit=1)
    assert [row["text"] for row in rows] == ["one"]


@pytest.mark.parametrize("limits", [
    {"work_limit": 6}, {"work_limit": True}, {"work_limit": 0},
    {"snippet_limit": 3}, {"snippet_limit": 1.5}, {"snippet_limit": -1},
])
def test_invalid_limits_cannot_expand_candidate_budget(api, request_metadata, limits):
    with pytest.raises(ValueError, match="OPENALEX_INVALID_LIMIT"):
        api.parse_response({"results": []}, query="R", request=request_metadata, **limits)


@pytest.mark.parametrize("field", ["url", "response_sha256", "retrieved_at"])
def test_missing_request_evidence_fails(api, request_metadata, field):
    del request_metadata[field]
    with pytest.raises(ValueError, match="OPENALEX_INVALID_REQUEST"):
        api.parse_response({"results": [work()]}, query="R", request=request_metadata)


def test_no_inferred_license_or_snippet_language(api, request_metadata):
    result = {**work(), "language": "en", "license": "cc-by"}
    rows = api.parse_response({"results": [result]}, query="R", request=request_metadata)
    assert rows[0]["language"] == "unknown"
    assert rows[0]["license_status"] == "unknown"


def test_deduplication_revalidates_explicit_text_revision(api, request_metadata):
    rows = api.parse_response({"results": [work()]}, query="R", request=request_metadata)
    rows[0]["text"] = "tampered"
    with pytest.raises(ValueError, match="text_revision"):
        api.deduplicate_snippets(rows)


def test_empty_response_is_valid_and_preserves_empty_outputs(api, request_metadata):
    rows = api.parse_response({"results": []}, query="R", request=request_metadata)
    assert rows == []
    assert api.deduplicate_snippets(rows) == ([], [])


def test_dedup_does_not_decode_again_or_collapse_whitespace(api, request_metadata):
    payload = {"results": [work("W1", ["&amp;#x20;", "&#x26;#x20;"]), work("W2", [" NumPy", "NumPy"])]}
    rows = api.parse_response(payload, query="R", request=request_metadata)
    documents, _ = api.deduplicate_snippets(rows)
    assert [row["text"] for row in documents] == ["&#x20;", " NumPy", "NumPy"]
    assert len(documents[0]["source_associations"]) == 2


@pytest.mark.parametrize("raw", [None, 1, "\ud800"])
def test_invalid_snippet_text_has_named_failure(api, raw):
    with pytest.raises(api.OpenAlexSchemaError, match="OPENALEX_INVALID_SNIPPET"):
        api.normalize_snippet(raw)


@pytest.mark.parametrize("value", [None, "bad", "sha256:" + "a" * 64, "A" * 64])
def test_response_hash_requires_raw_byte_digest_shape(api, request_metadata, value):
    request_metadata["response_sha256"] = value
    with pytest.raises(ValueError, match="OPENALEX_INVALID_REQUEST"):
        api.parse_response({"results": []}, query="R", request=request_metadata)


def test_compact_work_ids_and_full_urls_reserve_the_same_parent(api, request_metadata):
    payload = {"results": [work(), {**work(), "id": "W123"}]}
    documents, identities = api.deduplicate_snippets(api.parse_response(payload, query="R", request=request_metadata))
    assert len(documents) == len(identities) == 1
    assert [row["work_id"] for row in identities[0]["source_associations"]] == ["https://openalex.org/W123", "W123"]
