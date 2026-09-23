from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from research import aggregate


def contracts():
    from research import contracts as module
    return module


def test_two_versions_count_once(fixture_case):
    case = fixture_case("fixture-multiversion")
    rows, links = contracts().export_public(case["expected_occurrences"])
    assert rows == case["expected_public_records"]
    assert [link["export_ordinal"] for link in links] == [0, 1]
    assert [link["version_edge_ordinal"] for link in links] == [0, 1]
    assert len({link["mention_id"] for link in links}) == 1
    result = aggregate.aggregate_occurrences(case["expected_occurrences"])
    assert result["mention_count"] == 1
    assert result["export_row_count"] == 2
    assert result["intent_counts"] == {"used": 1}
    assert result["sentiment_counts"] == {"not_expressed": 1}
    assert result["version_buckets"] == {"1.24": 1, "1.26": 1}


def test_all_examples_validate_export_and_count(contract_pack):
    for case in contract_pack["extraction_cases"]:
        original = deepcopy(case)
        doc = contracts().validate_document(case["input"])
        assert doc["text_revision"] == "sha256:" + hashlib.sha256(doc["text"].encode()).hexdigest()
        for occ in case["expected_occurrences"]:
            assert contracts().validate_occurrence(occ, doc["text"]) == occ
        rows, mappings = contracts().export_public(case["expected_occurrences"])
        assert rows == case["expected_public_records"]
        result = aggregate.aggregate_occurrences(case["expected_occurrences"])
        for key, expected in case["expected_counts"].items():
            assert result[key] == expected
        assert len(mappings) == len(rows)
        assert case == original


def test_partial_annotation_valid_but_cannot_be_exported(contract_pack):
    case = deepcopy(contract_pack["partial_annotation_case"])
    module = contracts()
    module.validate_occurrence(case["annotation"], case["input"]["text"])
    with pytest.raises(module.ContractError, match="partial"):
        module.export_public([case["annotation"]])


@pytest.mark.parametrize("mutation", [
    lambda o: o.update(name="Wrong"),
    lambda o: o["name_span"].update(start=True),
    lambda o: o["context_span"].update(end=999),
    lambda o: o.update(text_revision="sha256:" + "0" * 64),
    lambda o: o.update(mention_id="arbitrary"),
    lambda o: o["version_links"].append(deepcopy(o["version_links"][0])),
    lambda o: o["version_links"][0].update(text="NA"),
    lambda o: o.update(version_status="absent"),
    lambda o: o.update(intents=["mentioned", "used"]),
    lambda o: o.update(intents=["used", "used"]),
    lambda o: o.update(sentiment=None),
    lambda o: o["known"].update(used=False),
    lambda o: o["known"].update(software=False),
    lambda o: o["evidence"].update(intents=[]),
    lambda o: o.update(sentiment="positive"),
])
def test_invalid_occurrences_rejected_with_structured_errors(fixture_case, mutation):
    case = fixture_case("fixture-multiversion")
    occ = case["expected_occurrences"][0]
    mutation(occ)
    module = contracts()
    with pytest.raises(module.ContractError) as error:
        module.validate_occurrence(occ, case["input"]["text"])
    assert error.value.issues[0]["document_id"] == case["input"]["document_id"]
    assert error.value.issues[0]["field"]
    assert error.value.issues[0]["code"]


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(document_id=" "),
    lambda d: d.update(text=""),
    lambda d: d.update(text="\ud800"),
    lambda d: d.update(text_revision="sha256:bad"),
    lambda d: d.update(sections=[{"start": 0, "end": 1000}]),
    lambda d: d.update(page_spans=[{"start": -1, "end": 3}]),
])
def test_invalid_documents_rejected(fixture_case, mutation):
    doc = fixture_case("fixture-multiversion")["input"]
    mutation(doc)
    with pytest.raises(contracts().ContractError):
        contracts().validate_document(doc)


def test_unicode_offsets_are_code_points(fixture_case):
    case = fixture_case("fixture-multiversion")
    doc = case["input"]
    prefix = "🧪 e\u0301 "
    doc["text"] = prefix + doc["text"]
    occ = case["expected_occurrences"][0]
    for span in [occ["name_span"], occ["context_span"], *occ["evidence"]["intents"],
                 *(v["span"] for v in occ["version_links"])]:
        span["start"] += len(prefix)
        span["end"] += len(prefix)
    occ["text_revision"] = "sha256:" + hashlib.sha256(doc["text"].encode()).hexdigest()
    occ["mention_id"] = contracts().occurrence_id(doc["document_id"], occ["text_revision"], 13, 18)
    assert contracts().validate_occurrence(occ, doc["text"])["name_span"] == {"start": 13, "end": 18}


def test_duplicate_occurrences_are_not_double_counted(fixture_case):
    occ = fixture_case("fixture-multiversion")["expected_occurrences"][0]
    result = aggregate.aggregate_occurrences([occ, deepcopy(occ)])
    assert result["mention_count"] == 1
    assert result["intent_counts"] == {"used": 1}
    conflict = deepcopy(occ)
    conflict["intents"] = ["created"]
    with pytest.raises(contracts().ContractError, match="conflict"):
        aggregate.aggregate_occurrences([occ, conflict])


def test_json_schemas_cover_fixtures_and_require_run_provenance(contract_pack):
    root = Path(__file__).resolve().parents[1] / "schemas/scibert-v2"
    schemas = {name: json.loads((root / f"{name}.schema.json").read_text())
               for name in ("document", "occurrence", "annotation", "run")}
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    for case in contract_pack["extraction_cases"]:
        jsonschema.validate(case["input"], schemas["document"])
        for occ in case["expected_occurrences"]:
            jsonschema.validate(occ, schemas["occurrence"])
    jsonschema.validate(contract_pack["partial_annotation_case"]["annotation"], schemas["occurrence"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"run_id": "no-provenance"}, schemas["run"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"task_id": "no-coverage"}, schemas["annotation"])
