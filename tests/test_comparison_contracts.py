from copy import deepcopy
import hashlib

import pytest


def contracts():
    from research.comparison import contracts as module
    return module


@pytest.fixture
def document():
    text = "🧪 NumPy 1.26 and NumPy.\r\n"
    return {"document_id": "paper-1", "text": text,
            "text_revision": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "metadata": {"role": "diagnostic", "training_eligible": False}}


@pytest.fixture
def result(document):
    return {"schema_version": "span-comparison-1", "arm_id": "detector",
            "document_id": document["document_id"], "text_revision": document["text_revision"],
            "status": "success", "offset_unit": "unicode_codepoint_half_open",
            "capabilities": {"software_spans": True, "version_spans": True,
                             "version_linking": False, "aliases": False, "intent": False,
                             "sentiment": False, "full_contract": False},
            "scores_calibrated": False,
            "spans": [{"label": "SOFTWARE", "text": "NumPy", "start": 2, "end": 7,
                       "score": 0.8, "score_kind": "geometric_mean_token_softmax",
                       "alignment_method": "native_offsets"}],
            "unresolved": [], "chunks": [{"window_id": "w1", "status": "success"}],
            "raw_artifact": "raw/document-1/index.json", "provenance": {"backend": "fixture"}}


@pytest.fixture
def reference(document):
    return {"document_id": document["document_id"], "text_revision": document["text_revision"],
            "spans": [{"label": "SOFTWARE", "text": "NumPy", "start": 2, "end": 7}],
            "coverage": [{"start": 0, "end": 25, "label": "SOFTWARE", "complete": True,
                          "review_kind": "human_reviewed"}],
            "provenance": {"source": "fixture"}}


def test_reject_missing_input_revision():
    with pytest.raises(ValueError):
        contracts().validate_input({"document_id": "x", "text": "NumPy"})


def test_validate_input_preserves_frozen_text_and_metadata(document):
    validated = contracts().validate_input(document)
    assert validated == document
    assert validated["text"] == "🧪 NumPy 1.26 and NumPy.\r\n"
    validated["metadata"]["role"] = "changed"
    assert document["metadata"]["role"] == "diagnostic"


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(text_revision="sha256:" + "0" * 64),
    lambda d: d.update(text_revision=None),
    lambda d: d.update(text="\ud800"),
    lambda d: d.update(document_id=" "),
    lambda d: d.update(metadata=[]),
])
def test_reject_invalid_input(document, mutation):
    mutation(document)
    with pytest.raises(ValueError):
        contracts().validate_input(document)


def test_result_preserves_native_scores_and_independent_versions(document, result):
    result["spans"].append({"label": "VERSION", "text": "1.26", "start": 8, "end": 12,
                            "score": None, "score_kind": None, "alignment_method": "exact_unique"})
    validated = contracts().validate_result(document, result)
    assert validated == result
    assert validated["spans"][1]["score"] is None
    assert "version_links" not in validated
    validated["spans"][0]["score"] = 0.5
    assert result["spans"][0]["score"] == 0.8


@pytest.mark.parametrize("boundary", [
    {"start": True}, {"end": False}, {"start": -1}, {"end": 26},
    {"start": 7}, {"start": 8}, {"start": 2.0},
], ids=["boolean_start", "boolean_end", "negative", "out_of_range", "empty", "reversed", "float"])
def test_reject_invalid_span_offsets(document, result, boundary):
    result["spans"][0].update(boundary)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_reject_mismatched_source_slice(document, result):
    result["spans"][0]["text"] = "numpy"
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_reject_duplicate_result_spans(document, result):
    duplicate = deepcopy(result["spans"][0])
    duplicate["score"] = 0.99
    result["spans"].append(duplicate)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(schema_version="detector-prediction-1"),
    lambda r: r.update(document_id="other"),
    lambda r: r.update(text_revision="sha256:" + "0" * 64),
    lambda r: r.update(offset_unit="utf16"),
    lambda r: r.update(status="ready"),
    lambda r: r.update(arm_id=" "),
    lambda r: r.update(scores_calibrated=1),
    lambda r: r.update(provenance=None),
    lambda r: r.update(chunks=["w1"]),
    lambda r: r.update(unresolved=["NumPy"]),
    lambda r: r["capabilities"].update(software_spans=1),
    lambda r: r["capabilities"].pop("full_contract"),
    lambda r: r["capabilities"].update(software_spans=False),
    lambda r: r["spans"][0].update(label="INTENT"),
    lambda r: r["spans"][0].update(alignment_method=" "),
    lambda r: r["spans"][0].pop("score_kind"),
    lambda r: r["spans"][0].update(score=None),
    lambda r: r["spans"][0].update(score_kind=None),
])
def test_reject_invalid_result_fields(document, result, mutation):
    mutation(result)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


@pytest.mark.parametrize("score", [True, "0.8", float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_or_nonnumeric_scores(document, result, score):
    result["spans"][0]["score"] = score
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


@pytest.mark.parametrize("field", ["schema_version", "arm_id", "document_id", "text_revision", "status",
                                   "offset_unit", "capabilities", "scores_calibrated", "spans", "unresolved",
                                   "chunks", "raw_artifact", "provenance"])
def test_reject_missing_result_fields(document, result, field):
    result.pop(field)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


@pytest.mark.parametrize("path", [None, "", "/tmp/raw.json", "../raw.json", "raw/../../raw.json",
                                  "raw\\file.json", "https://example.test/raw.json"])
def test_reject_unsafe_or_missing_success_raw_artifact(document, result, path):
    result["raw_artifact"] = path
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_empty_extraction_is_no_mentions(document, result):
    result["spans"] = []
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)
    result["status"] = "no_mentions"
    assert contracts().validate_result(document, result)["status"] == "no_mentions"


@pytest.mark.parametrize("status", ["no_mentions", "failure", "unavailable", "unsupported"])
def test_non_success_cannot_carry_scored_spans(document, result, status):
    result.update(status=status, reason="fixture_reason")
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_unresolved_prediction_is_failure(document, result):
    result["unresolved"] = [{"label": "SOFTWARE", "text": "NumPy", "reason": "ambiguous"}]
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)
    result.update(status="failure", spans=[], reason="ambiguous_prediction")
    assert contracts().validate_result(document, result)["unresolved"] == result["unresolved"]


def test_failed_chunk_cannot_be_success(document, result):
    result["chunks"][0]["status"] = "failure"
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


@pytest.mark.parametrize("status", ["failure", "unavailable", "unsupported"])
def test_non_success_requires_reason(document, result, status):
    result.update(status=status, spans=[], chunks=[], raw_artifact=None)
    result["capabilities"] = dict.fromkeys(result["capabilities"], False)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)
    result["reason"] = "not_ready"
    assert contracts().validate_result(document, result)["reason"] == "not_ready"


def test_unsupported_base_has_no_extraction_capability(document, result):
    result.update(arm_id="scibert-base", status="unsupported", reason="no_task_head",
                  spans=[], chunks=[], raw_artifact=None)
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)
    result["capabilities"] = dict.fromkeys(result["capabilities"], False)
    assert contracts().validate_result(document, result)["status"] == "unsupported"
    result["reason"] = "missing_model"
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_base_cannot_claim_success(document, result):
    result["arm_id"] = "scibert-base"
    with pytest.raises(ValueError):
        contracts().validate_result(document, result)


def test_reference_deduplicates_gold_without_inventing_coverage(document, reference):
    reference["spans"].append(deepcopy(reference["spans"][0]))
    reference["coverage"] = []
    validated = contracts().validate_reference(document, reference)
    assert validated["spans"] == [{"label": "SOFTWARE", "text": "NumPy", "start": 2, "end": 7}]
    assert validated["coverage"] == []
    assert len(reference["spans"]) == 2


def test_reference_preserves_review_kind_and_field_coverage(document, reference):
    reference["coverage"].append({"start": 0, "end": 12, "label": "VERSION", "complete": True,
                                  "review_kind": "agent_provisional"})
    validated = contracts().validate_reference(document, reference)
    assert validated == reference
    validated["coverage"][0]["label"] = "VERSION"
    assert reference["coverage"][0]["label"] == "SOFTWARE"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(document_id="other"),
    lambda r: r.update(text_revision="sha256:" + "0" * 64),
    lambda r: r.pop("provenance"),
    lambda r: r.update(coverage=None),
    lambda r: r["spans"][0].update(start=True),
    lambda r: r["spans"][0].update(end=99),
    lambda r: r["spans"][0].update(text="Wrong"),
    lambda r: r["spans"][0].update(label="version"),
    lambda r: r["coverage"][0].update(start=False),
    lambda r: r["coverage"][0].update(end=26),
    lambda r: r["coverage"][0].update(start=25),
    lambda r: r["coverage"][0].update(complete=False),
    lambda r: r["coverage"][0].update(complete=1),
    lambda r: r["coverage"][0].update(label="software"),
    lambda r: r["coverage"][0].update(review_kind="accepted_candidate"),
])
def test_reject_invalid_reference_or_coverage(document, reference, mutation):
    mutation(reference)
    with pytest.raises(ValueError):
        contracts().validate_reference(document, reference)
