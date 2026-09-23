from copy import deepcopy
import json

import pytest

from research.contracts import ContractError, FIELDS, validate_document


def metrics():
    from research.evaluation import metrics as module
    return module


def evaluate(case, predictions=None, *, gold=None, fields=None, caps=None, statuses=None):
    doc = validate_document(case["input"])
    coverage = [{"document_id": doc["document_id"], "text_revision": doc["text_revision"],
                 "start": 0, "end": len(doc["text"]), "fields": fields or dict.fromkeys(FIELDS, True),
                 "status": "complete", "provenance": {"kind": "synthetic"}}]
    return metrics().evaluate_v2(
        [doc], case["expected_occurrences"] if gold is None else gold,
        case["expected_occurrences"] if predictions is None else predictions,
        coverage, [case["expected_status"]] if statuses is None else statuses,
        caps or {"software": True, "versions": True, "version_offsets": True, "intents": True, "sentiment": True},
    )


def test_wrong_version_is_fp_and_fn(fixture_case):
    gold = fixture_case("fixture-multiversion")["expected_occurrences"]
    gold[0]["version_links"] = gold[0]["version_links"][:1]
    pred = deepcopy(gold)
    pred[0]["version_links"][0]["text"] = "1.26"
    assert metrics().count_value_edges(gold, pred) == {"tp": 0, "fp": 1, "fn": 1}


@pytest.mark.parametrize("counts, expected", [
    ((0, 0, 1), {"precision": None, "recall": 0.0, "f1": 0.0}),
    ((0, 1, 0), {"precision": 0.0, "recall": None, "f1": 0.0}),
    ((0, 0, 0), {"precision": None, "recall": None, "f1": None}),
    ((1, 1, 1), {"precision": 0.5, "recall": 0.5, "f1": 0.5}),
])
def test_prf_undefined_is_not_zero_failure(counts, expected):
    result = metrics().prf(*counts)
    for key, value in expected.items():
        assert result[key] == value


def test_perfect_multiversion_counts_one_occurrence(fixture_case):
    result = evaluate(fixture_case("fixture-multiversion"))
    assert result["mention_detection"]["tp"] == 1
    assert result["version"]["value_edges"]["tp"] == 2
    assert result["version"]["strict_edges"]["tp"] == 2
    assert result["intents"]["per_label"]["used"]["tp"] == 1
    assert result["sentiment"]["per_class"]["not_expressed"]["tp"] == 1
    assert result["complete_occurrence"]["f1"] == 1.0
    assert result["evaluator_version"] == "scibert-v2.0"
    assert result["input_hashes"]["predictions"]


def test_missed_mentions_contribute_attribute_false_negatives(fixture_case):
    result = evaluate(fixture_case("fixture-multiversion"), [])
    assert result["mention_detection"]["fn"] == 1
    assert result["mention_detection"]["f1"] == 0.0
    assert result["version"]["value_edges"]["fn"] == 2
    assert result["intents"]["per_label"]["used"]["fn"] == 1
    assert result["sentiment"]["per_class"]["not_expressed"]["fn"] == 1
    assert result["complete_occurrence"]["fn"] == 1
    assert result["intents"]["exact_set_accuracy"] == 0.0


def test_spurious_mentions_contribute_attribute_false_positives(fixture_case):
    case = fixture_case("fixture-multiversion")
    result = evaluate(case, gold=[])
    assert result["mention_detection"]["fp"] == 1
    assert result["version"]["value_edges"]["fp"] == 2
    assert result["intents"]["per_label"]["used"]["fp"] == 1
    assert result["sentiment"]["per_class"]["not_expressed"]["fp"] == 1
    assert result["zero_mention_documents"]["false_positive_rate"] == 1.0


def test_extra_versions_on_absent_gold_are_errors(fixture_case):
    case = fixture_case("fixture-multiversion")
    gold = deepcopy(case["expected_occurrences"])
    gold[0].update(version_links=[], version_status="absent")
    result = evaluate(case, gold=gold)
    assert result["version"]["value_edges"]["fp"] == 2
    assert result["version"]["value_edges"]["f1"] == 0.0
    assert result["complete_occurrence"]["tp"] == 0
    assert result["complete_occurrence"]["fp"] == 1
    assert result["complete_occurrence"]["fn"] == 1


def test_mentioned_is_exclusive_and_scored(fixture_case):
    case = fixture_case("fixture-negated-planned")
    result = evaluate(case)
    assert result["intents"]["per_label"]["mentioned"]["tp"] == 2
    predicted = deepcopy(case["expected_occurrences"])
    predicted[0]["intents"] = ["used"]
    result = evaluate(case, predicted)
    assert result["intents"]["per_label"]["mentioned"]["fn"] == 1
    assert result["intents"]["per_label"]["used"]["fp"] == 1
    assert result["intents"]["exact_set_accuracy"] == 0.5


def test_partial_intent_masks_only_unknown_bits(fixture_case):
    case = fixture_case("fixture-created-shared")
    gold = deepcopy(case["expected_occurrences"])
    gold[0]["intents"] = ["created"]
    gold[0]["known"].update(used=False, shared=False, sentiment=False)
    gold[0]["sentiment"] = None
    result = evaluate(case, gold=gold)
    assert result["intents"]["per_label"]["created"]["tp"] == 1
    assert result["intents"]["per_label"]["shared"]["fp"] == 0
    assert result["intents"]["per_label"]["shared"]["f1"] is None
    assert result["sentiment"]["per_class"]["not_expressed"]["fp"] == 0
    assert result["complete_occurrence"]["f1"] is None


def test_wrong_sentiment_and_false_opinion_use_fixed_gold_denominator(fixture_case):
    case = fixture_case("fixture-repeated-mentions")
    pred = deepcopy(case["expected_occurrences"][:1])
    pred[0]["sentiment"] = "positive"
    result = evaluate(case, pred)
    assert result["sentiment"]["per_class"]["positive"]["fp"] == 1
    assert result["sentiment"]["per_class"]["not_expressed"]["fn"] == 2
    assert result["sentiment"]["false_opinion_rate"] == 0.5
    assert result["mention_detection"]["recall"] == 0.5


def test_sparse_macro_does_not_silently_drop_missing_classes(fixture_case):
    result = evaluate(fixture_case("fixture-multiversion"))
    assert result["intents"]["macro"]["f1"] is None
    assert result["intents"]["observed_macro_f1"] == 1.0
    assert result["sentiment"]["macro_f1"] is None
    assert result["sentiment"]["missing_classes"] == ["positive", "negative", "mixed"]


def test_unknown_coverage_does_not_create_false_negatives_or_positives(fixture_case):
    case = fixture_case("fixture-multiversion")
    result = evaluate(case, gold=[], fields=dict.fromkeys(FIELDS, False))
    assert result["mention_detection"]["fp"] == 0
    assert result["version"]["value_edges"]["fp"] == 0
    assert result["exclusions"]["prediction_outside_software_coverage"] == 1


def test_timeout_keeps_eligible_document_and_all_misses(fixture_case):
    case = fixture_case("fixture-multiversion")
    status = {"document_id": case["input"]["document_id"], "status": "timeout", "chunks_expected": 3, "chunks_succeeded": 2, "chunks_failed": 1}
    result = evaluate(case, [], statuses=[status])
    assert result["mention_detection"]["fn"] == 1
    assert result["failures"]["failed_documents"] == 1
    assert result["failures"]["failed_chunks"] == 1
    assert len(result["per_document"]) == 1


def test_no_mentions_is_undefined_f1_but_valid_negative_document(fixture_case):
    result = evaluate(fixture_case("fixture-negative"))
    assert result["mention_detection"]["f1"] is None
    assert result["zero_mention_documents"]["false_positive_rate"] == 0.0


def test_unsupported_fields_cannot_pass_from_placeholder_labels(fixture_case):
    result = evaluate(fixture_case("fixture-multiversion"), caps={"software": True, "versions": False, "intents": False, "sentiment": False})
    assert result["version"]["value_edges"]["f1"] is None
    assert result["sentiment"]["supported"] is False
    assert result["sentiment"]["macro_f1"] is None
    assert result["complete_occurrence"]["supported"] is False


def test_duplicate_predictions_and_conflicting_revisions(fixture_case):
    case = fixture_case("fixture-multiversion")
    pred = deepcopy(case["expected_occurrences"] * 2)
    assert evaluate(case, pred)["mention_detection"]["tp"] == 1
    pred[1]["text_revision"] = "sha256:" + "0" * 64
    with pytest.raises(ContractError, match="revision"):
        evaluate(case, pred)


def test_converter_collapses_version_rows_and_preserves_unaligned(fixture_case):
    from research.evaluation.legacy import convert_legacy
    case = fixture_case("fixture-multiversion")
    doc = validate_document(case["input"])
    rows = [{**row, "document_id": doc["document_id"], "name_span": {"start": 8, "end": 13}} for row in case["expected_public_records"]]
    rows.append({**rows[0], "name": "Ghost", "name_span": None})
    result = convert_legacy(rows, [doc], {"software": True, "versions": True, "intents": True, "sentiment": True})
    assert len(result["occurrences"]) == 1
    assert [e["text"] for e in result["occurrences"][0]["version_links"]] == ["1.24", "1.26"]
    assert all(e["span"] is None for e in result["occurrences"][0]["version_links"])
    assert len(result["unresolved_predictions"]) == 1
    scored = evaluate(case, result["occurrences"] + result["unresolved_predictions"])
    assert scored["mention_detection"]["fp"] == 1
    assert scored["version"]["value_edges"]["tp"] == 2
    assert scored["version"]["strict_edges"]["f1"] is None


def test_legacy_empty_labels_are_unknown_not_mentioned(fixture_case):
    from research.evaluation.legacy import convert_legacy
    case = fixture_case("fixture-multiversion")
    row = {"document_id": case["input"]["document_id"], "name": "NumPy", "name_span": {"start": 8, "end": 13}, "version": None, "intents": []}
    result = convert_legacy([row], [case["input"]], {"software": True, "versions": True, "intents": True, "sentiment": True})
    occ = result["occurrences"][0]
    assert occ["known"]["used"] is False
    assert occ["known"]["sentiment"] is False
    assert occ["intents"] is None


def test_explicit_version_dispatch_keeps_legacy_available(fixture_case):
    from research.evaluate import Evaluator
    case = fixture_case("fixture-multiversion")
    doc = validate_document(case["input"])
    result = Evaluator(metric_version="v2").evaluate(
        case["expected_occurrences"], case["expected_occurrences"], documents=[doc],
        coverage=[], statuses=[], capabilities={"software": True})
    assert result["evaluator_version"] == "scibert-v2.0"
    with pytest.raises(ValueError):
        Evaluator(metric_version="unknown")


def test_all_fixture_metrics_are_bounded_and_json_serializable(contract_pack):
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, float):
                    assert 0 <= item <= 1, (key, item)
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    for case in contract_pack["extraction_cases"]:
        for pred in ([], case["expected_occurrences"]):
            result = evaluate(case, pred)
            check(result)
            json.dumps(result, allow_nan=False)


def test_legacy_source_case_restored_only_at_verified_offsets(fixture_case):
    from research.evaluation.legacy import convert_legacy
    case = fixture_case("fixture-multiversion")
    row = {"document_id": case["input"]["document_id"], "name": "numpy", "name_span": {"start": 8, "end": 13}, "intents": ["used"], "version": None}
    result = convert_legacy([row], [case["input"]], {"software": True, "intents": True})
    assert result["occurrences"][0]["name"] == "NumPy"
    assert result["issues"][0]["code"] == "SOURCE_CASE_RESTORED"


def test_conflicting_legacy_attributes_are_invalid_not_cherry_picked(fixture_case):
    from research.evaluation.legacy import convert_legacy
    case = fixture_case("fixture-multiversion")
    row = {**case["expected_public_records"][0], "document_id": case["input"]["document_id"], "name_span": {"start": 8, "end": 13}}
    caps = {"software": True, "versions": True, "intents": True, "sentiment": True}
    result = convert_legacy([row, {**row, "intents": ["created"]}], [case["input"]], caps)
    assert result["occurrences"][0]["intents"] is None
    assert result["issues"][0]["code"] == "CONFLICTING_LEGACY_LABELS"
    score = evaluate(case, result["occurrences"])
    assert score["intents"]["per_label"]["used"]["fn"] == 1
    assert score["exclusions"]["invalid_prediction_intents"] == 1
    with pytest.raises(ContractError):
        convert_legacy([row, {**row, "intents": ["created"]}], [case["input"]], caps, reference=True)


def test_matched_only_diagnostic_is_separate_from_end_to_end(fixture_case):
    case = fixture_case("fixture-repeated-mentions")
    result = evaluate(case, case["expected_occurrences"][:1])
    assert result["intents"]["exact_set_accuracy"] == 0.5
    assert result["diagnostics"]["matched_only"]["intent_exact_accuracy"] == 1.0


def test_null_metadata_is_valid_and_unknown_software_gold_is_rejected(fixture_case):
    case = fixture_case("fixture-multiversion")
    case["input"]["metadata"] = None
    assert evaluate(case)["mention_detection"]["tp"] == 1
    case["expected_occurrences"][0]["known"]["software"] = False
    with pytest.raises(ContractError):
        evaluate(case)


def test_negative_failure_counts_and_bad_version_offsets_are_rejected(fixture_case):
    case = fixture_case("fixture-multiversion")
    status = {"document_id": case["input"]["document_id"], "status": "success", "chunks_failed": -1}
    with pytest.raises(ContractError):
        evaluate(case, statuses=[status])
    case["expected_occurrences"][0]["version_links"][0]["span"] = {"start": 0, "end": 4}
    with pytest.raises(ContractError):
        evaluate(case)
