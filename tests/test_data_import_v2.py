from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.contracts import ContractError, validate_occurrence


def tei(xml):
    from research.data.tei import read_tei
    return read_tei(xml)


def brat(text, ann):
    from research.data.brat import read_brat
    return read_brat(text, ann)


def mapped(parsed, fields=None, end=None):
    from research.data.coverage import map_annotations
    policy = {
        "policy_version": "test-native-v1",
        "software_types": {"software": {}, "Application_Usage": {"intent_bits": {"used": True}},
                           "Application_Deposition": {}},
        "version_types": ["version", "Version"],
        "version_relations": ["version_of", "Version_of"],
        "ignored_types": ["bibr"],
        "coverage": [{"start": 0, "end": end or len(parsed["text"]),
                      "fields": fields or {"software": True, "versions": True},
                      "provenance": {"kind": "test"}}],
    }
    return map_annotations(parsed, {"document_id": "d", "text": parsed["text"]}, policy)


def test_xml_ids_versions_and_tail_once():
    result = tei(b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><p>A <hi>B</hi> C <rs xml:id="s" type="software">X</rs> <rs xml:id="v" type="version" corresp="#s">1</rs>.</p></text></TEI>')
    assert result["text"] == "A B C X 1."
    assert result["spans"][0]["source_id"] == "s"
    assert result["relations"] == [{"kind": "version_of", "source_id": "v", "target_id": "s"}]


def test_block_separator_offsets_and_multiple_versions():
    parsed = tei(b'<TEI><text><p>First</p><p><rs type="software" xml:id="s">X</rs> <rs type="version" xml:id="v1" corresp="s">1</rs> and <rs type="version" xml:id="v2" corresp="#s">2</rs>.</p></text></TEI>')
    assert parsed["text"] == "First\nX 1 and 2."
    assert parsed["insertions"] == [{"offset": 5, "text": "\n", "reason": "block_boundary"}]
    result = mapped(parsed)
    assert len(result["occurrences"]) == 1
    occ = result["occurrences"][0]
    assert occ["name_span"] == {"start": 6, "end": 7}
    assert [e["text"] for e in occ["version_links"]] == ["1", "2"]
    validate_occurrence(occ, result["document"]["text"])


def test_existing_block_whitespace_not_duplicated_and_corresp_keeps_all_ids():
    parsed = tei(b'<TEI><text><p><rs type="software" xml:id="a">X</rs></p>\n<p><rs type="software" xml:id="b">Y</rs> <rs type="version" xml:id="v" corresp="#a b">2</rs></p></text></TEI>')
    assert parsed["text"] == "X\nY 2"
    assert parsed["insertions"] == []
    assert [r["target_id"] for r in parsed["relations"]] == ["a", "b"]


def test_brat_keeps_multiple_relations_attributes_and_discontinuous_segments():
    parsed = brat("North and South X 1 2", "T1\tsoftware 0 5;16 17\tNorth X\nT2\tVersion 18 19\t1\nT3\tVersion 20 21\t2\nR1\tVersion_of Arg1:T2 Arg2:T1\nR2\tVersion_of Arg2:T1 Arg1:T3\nA1\tNegation T1\n")
    assert parsed["spans"][0]["segments"] == [{"start": 0, "end": 5}, {"start": 16, "end": 17}]
    assert [r["source_id"] for r in parsed["relations"]] == ["T2", "T3"]
    assert parsed["native_records"][-1]["raw"] == "A1\tNegation T1"
    result = mapped(parsed)
    assert result["occurrences"] == []
    assert any(e["code"] == "DISCONTINUOUS_SPAN" for e in result["exclusions"])
    assert all(not (r["start"] <= 0 < r["end"] and r["fields"]["software"]) for r in result["coverage"])


def test_repeated_name_projects_its_own_offset_after_whitespace_normalization():
    parsed = brat("X\t\t X  1", "T1\tsoftware 4 5\tX\nT2\tVersion 7 8\t1\nR1\tVersion_of Arg1:T2 Arg2:T1")
    result = mapped(parsed)
    occ = result["occurrences"][0]
    assert result["document"]["text"] == "X X 1"
    assert occ["name_span"] == {"start": 2, "end": 3}
    assert occ["version_links"][0]["span"] == {"start": 4, "end": 5}
    assert result["normalization"]["source_spans"][1] == {"start": 1, "end": 4}


def test_bad_alignment_is_quarantined_not_relocated_to_equal_name():
    parsed = brat("X Y X", "T1\tsoftware 2 3\tX")
    result = mapped(parsed)
    assert result["occurrences"] == []
    assert any(e["code"] == "ALIGNMENT_UNRESOLVED" for e in result["exclusions"])


def test_partial_and_unknown_native_labels_do_not_manufacture_negatives():
    parsed = brat("X Y Z", "T1\tApplication_Usage 0 1\tX\nT2\tApplication_Deposition 2 3\tY\nT3\tUnmappedThing 4 5\tZ")
    result = mapped(parsed, fields={"software": True})
    used, deposition = result["occurrences"]
    assert used["intents"] == ["used"]
    assert used["known"]["used"] is True
    assert used["known"]["created"] is False
    assert deposition["intents"] is None
    assert deposition["sentiment"] is None
    assert deposition["version_status"] == "unannotated"
    assert result["native_annotations"]["spans"] == parsed["spans"]
    assert any(e["code"] == "UNMAPPED_TYPE" for e in result["exclusions"])
    for occ in result["occurrences"]:
        validate_occurrence(occ, result["document"]["text"])


def test_nested_mentions_are_masked_and_methods_coverage_does_not_expand():
    parsed = brat("Long X. Other X", "T1\tsoftware 0 6\tLong X\nT2\tsoftware 5 6\tX\nT3\tsoftware 14 15\tX")
    result = mapped(parsed, end=7)
    assert result["occurrences"] == []
    assert {e["code"] for e in result["exclusions"]} == {"OVERLAPPING_ENTITY", "OUTSIDE_COVERAGE"}
    assert max(r["end"] for r in result["coverage"]) == 7
    assert not any(r["fields"]["software"] and r["start"] < 6 for r in result["coverage"])


def test_unlinked_and_dangling_versions_are_reported_without_guessing():
    parsed = tei(b'<TEI><text><rs xml:id="s" type="software">X</rs> <rs xml:id="v" type="version">1</rs> <rs xml:id="v2" type="version" corresp="#missing">2</rs></text></TEI>')
    assert any(i["code"] == "DANGLING_RELATION" for i in parsed["issues"])
    result = mapped(parsed)
    assert result["occurrences"][0]["version_links"] == []
    assert result["occurrences"][0]["known"]["versions"] is False
    assert sum(e["code"] == "UNLINKED_VERSION" for e in result["exclusions"]) == 2


@pytest.mark.parametrize("xml", [b'<TEI>', b'<TEI><text><rs xml:id="x">a</rs><rs xml:id="x">b</rs></text></TEI>', b'<!DOCTYPE a [<!ENTITY x "a">]><TEI><text>&x;</text></TEI>', b'<TEI/>'])
def test_malformed_or_unsafe_xml_is_an_explicit_error(xml):
    with pytest.raises(ContractError):
        tei(xml)


@pytest.mark.parametrize("ann", ["T1\tsoftware bad\tX", "T1\tsoftware 0 1\tX\nT1\tsoftware 0 1\tX", "R1\tVersion_of Arg1:T1", "T1\tsoftware 0 50\tX"])
def test_malformed_brat_is_an_explicit_error(ann):
    with pytest.raises(ContractError):
        brat("X", ann)


def test_versioned_import_cli_writes_new_bundle_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "paper.xml").write_text('<TEI><text><rs xml:id="s" type="software">X</rs></text></TEI>')
    output = tmp_path / "bundle"
    command = [sys.executable, "scripts/ingest_sofair.py", "--schema-version", "2", "--sofair-dir", str(source), "--pick-all", "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    docs = [json.loads(line) for line in (output / "documents.jsonl").read_text().splitlines()]
    assert docs[0]["text"] == "X"
    occ = json.loads((output / "occurrences.jsonl").read_text().splitlines()[0])
    assert occ["known"]["versions"] is False
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "exist" in second.stderr.lower()


def test_brat_import_cli_preserves_native_labels_without_assuming_intent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "paper.txt").write_text("X 1")
    (source / "paper.ann").write_text("T1\tApplication_Usage 0 1\tX\nT2\tVersion 2 3\t1\nR1\tVersion_of Arg1:T2 Arg2:T1")
    output = tmp_path / "bundle"
    command = [sys.executable, "scripts/ingest_somesci.py", "--schema-version", "2", "--label-dir", str(source), "--subset", ".", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    occ = json.loads((output / "occurrences.jsonl").read_text().splitlines()[0])
    assert occ["version_links"][0]["text"] == "1"
    assert occ["intents"] is None
    assert occ["known"]["used"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sources"][0]["annotation_sha256"]
    assert manifest["document_count"] == 1


def test_crlf_combining_marks_and_newline_runs_project_without_substring_guessing():
    from research.data.coverage import normalize_with_offsets, project_span
    normalized = normalize_with_offsets("🧪\r\ne\u0301\n\n\nX")
    assert normalized["text"] == "🧪\ne\u0301\n\nX"
    assert project_span({"start": 3, "end": 5}, normalized) == {"start": 2, "end": 4}
    assert project_span({"start": 8, "end": 9}, normalized) == {"start": 6, "end": 7}
    assert project_span({"start": 2, "end": 3}, normalized) is None


def test_custom_brat_relation_roles_preserved_without_guessing_direction():
    parsed = brat("X Y", "T1\tsoftware 0 1\tX\nT2\tsoftware 2 3\tY\nR1\tCoref Anaphor:T1 Antecedent:T2")
    assert parsed["relations"][0]["arguments"] == {"Anaphor": "T1", "Antecedent": "T2"}
    assert parsed["relations"][0]["source_id"] is None
