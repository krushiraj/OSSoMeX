import hashlib
import json
import subprocess
import sys


def _fixture(tmp_path):
    text = "ImageJ®\r\nNumPy π"
    revision = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    document_id = "snippet:one"
    reference = {
        "document_id": document_id,
        "text_revision": revision,
        "spans": [{"label": "SOFTWARE", "start": 0, "end": 6, "text": "ImageJ"}],
        "attribute_occurrences": [{"name": "ImageJ", "name_span": {"start": 0, "end": 6},
                                   "intents": ["mentioned"], "sentiment": "not_expressed",
                                   "known": {"software": True, "versions": False}}],
        "coverage": [{"label": "SOFTWARE", "start": 0, "end": len(text),
                      "complete": True, "review_kind": "agent_provisional"}],
        "attribute_coverage": [], "ignored_software": [], "ignored_versions": [],
        "version_links": [], "link_coverage": [],
        "alias_pairs": [{"known": False, "members": []}],
        "provenance": {"review_kind": "agent_provisional"},
    }
    references = tmp_path / "references.jsonl"
    references.write_text(json.dumps(reference, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "provenance": {"reference_sha256": hashlib.sha256(references.read_bytes()).hexdigest()},
        "documents": [{"document_id": document_id, "text_revision": revision, "text": text,
                       "source": "openalex", "source_ids": {"openalex": "W1"},
                       "source_associations": [{"query": "ImageJ"}],
                       "license_status": "unknown", "role": "diagnostic",
                       "training_eligible": False, "future_untouched_test_eligible": False,
                       "arms": [{"arm_id": "full-label-006", "status": "success",
                                 "spans": [{"label": "SOFTWARE", "start": 0, "end": 7,
                                 "text": "ImageJ®"}], "fields": [{"name": "ImageJ®",
                                                                  "name_span": {"start": 0, "end": 7},
                                                                  "versions": {"status": "success", "value": []},
                                                                  "intents": {"status": "success", "value": ["used"]},
                                                                  "sentiment": {"status": "success", "value": "not_expressed"}}],
                                 "version_links": [], "alias_groups": []},
                                {"arm_id": "softcite-0.8.1", "status": "success",
                                 "spans": [{"label": "SOFTWARE", "start": 0, "end": 6,
                                            "text": "ImageJ"}], "fields": [],
                                 "version_links": [], "alias_groups": []}]}],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return report_path, references, reference, text


def _run(report, references, output, *extra):
    return subprocess.run(
        [sys.executable, "-m", "research.evaluation.review_export", "--report", str(report),
         "--references", str(references), "--output", str(output), *extra],
        capture_output=True, text=True,
    )


def test_cli_preserves_full_unicode_text_reference_masks_and_separate_predictions(tmp_path):
    report, references, reference, text = _fixture(tmp_path)
    output = tmp_path / "export"

    result = _run(report, references, output, "--arms", "full-label-006", "softcite-0.8.1")

    assert result.returncode == 0, result.stderr
    row = json.loads((output / "review.jsonl").read_text(encoding="utf-8").strip())
    assert row["text"] == text
    assert row["reference"] == reference
    assert row["source_ids"] == {"openalex": "W1"}
    assert row["source_associations"] == [{"query": "ImageJ"}]
    assert row["local_review_only"] is True
    assert row["human_review_pending"] is True
    assert row["training_eligible"] is False
    assert row["future_untouched_test_eligible"] is False
    assert [arm["arm_id"] for arm in row["predictions"]] == ["full-label-006", "softcite-0.8.1"]
    assert row["predictions"][0]["spans"][0]["text"] == "ImageJ®"
    markdown = (output / "review.md").read_bytes().decode("utf-8")
    assert "ImageJ®\r\nNumPy π" in markdown
    assert "| Name | Offsets | Versions | Intent | Sentiment | Alias |" in markdown
    assert "| ImageJ® | 0:7 | none predicted | used | not_expressed | none predicted |" in markdown
    assert "<details><summary>Show saved predictions</summary>" in markdown
    assert str(report) not in markdown
    assert "Human review: pending" in markdown


def test_fully_masked_passage_is_unknown_and_eligibility_is_forced_false(tmp_path):
    report, references, reference, _ = _fixture(tmp_path)
    reference["spans"] = []
    reference["attribute_occurrences"] = []
    reference["coverage"] = []
    reference["ignored_software"] = [{"start": 0, "end": 16, "text": "ImageJ®\r\nNumPy π"}]
    reference["ignored_versions"] = reference["ignored_software"]
    references.write_text(json.dumps(reference, ensure_ascii=False) + "\n")
    report_data = json.loads(report.read_text())
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report_data["documents"][0]["training_eligible"] = True
    report_data["documents"][0]["future_untouched_test_eligible"] = True
    report.write_text(json.dumps(report_data))
    output = tmp_path / "export"

    result = _run(report, references, output)

    assert result.returncode == 0, result.stderr
    row = json.loads((output / "review.jsonl").read_text())
    assert row["report_metadata"]["training_eligible"] is False
    assert row["report_metadata"]["future_untouched_test_eligible"] is False
    markdown = (output / "review.md").read_text()
    assert "Fully masked" in markdown
    assert "unknown" in markdown.lower()
    assert "Reviewed absence of names across the full passage." not in markdown


def test_reviewed_empty_passage_marks_checked_absence(tmp_path):
    report, references, reference, _ = _fixture(tmp_path)
    reference["spans"] = []
    reference["attribute_occurrences"] = []
    references.write_text(json.dumps(reference) + "\n")
    report_data = json.loads(report.read_text())
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))
    output = tmp_path / "export"

    result = _run(report, references, output)

    assert result.returncode == 0, result.stderr
    assert "Reviewed absence of names" in (output / "review.md").read_text()


def test_versions_aliases_and_version_candidates_keep_their_roles(tmp_path):
    report, references, reference, _ = _fixture(tmp_path)
    occurrence = reference["attribute_occurrences"][0]
    occurrence["mention_id"] = "a"
    occurrence["version_links"] = [{"text": "π", "span": {"start": 15, "end": 16},
                                    "status": "explicit_local"}]
    occurrence["version_status"] = "explicit"
    occurrence["known"]["versions"] = True
    reference["spans"].append({"label": "VERSION", "start": 15, "end": 16, "text": "π"})
    reference["alias_pairs"] = [{"known": True, "decision": "alias",
                                 "member_mention_ids": ["a", "b"],
                                 "members": [{"mention_id": "a", "name": "ImageJ",
                                              "span": {"start": 0, "end": 6}},
                                             {"mention_id": "b", "name": "NumPy",
                                              "span": {"start": 9, "end": 14}}]}]
    references.write_text(json.dumps(reference, ensure_ascii=False) + "\n")
    report_data = json.loads(report.read_text())
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    scibert, softcite = report_data["documents"][0]["arms"]
    scibert["spans"].append({"label": "VERSION", "start": 9, "end": 14, "text": "NumPy"})
    scibert["version_links"] = [{"software": {"start": 0, "end": 7, "text": "ImageJ®"},
                                  "version": {"start": 9, "end": 14, "text": "NumPy"}}]
    softcite["version_links"] = [{"software": {"start": 0, "end": 6, "text": "ImageJ"},
                                   "version": {"start": 9, "end": 14, "text": "NumPy"}}]
    report.write_text(json.dumps(report_data, ensure_ascii=False))
    output = tmp_path / "export"

    result = _run(report, references, output)

    assert result.returncode == 0, result.stderr
    markdown = (output / "review.md").read_text()
    assert "π [15:16]" in markdown
    assert "ImageJ [0:6] ↔ NumPy [9:14]" in markdown
    assert "Version candidates: NumPy [9:14]" in markdown
    assert "ImageJ [0:6] → NumPy [9:14]" in markdown
    assert "| NumPy | 9:14 |" not in markdown


def test_rejects_reference_hash_mismatch_before_creating_output(tmp_path):
    report, references, _, _ = _fixture(tmp_path)
    with references.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "export"

    result = _run(report, references, output)

    assert result.returncode != 0
    assert "reference SHA-256" in result.stderr
    assert not output.exists()


def test_rejects_duplicate_or_missing_reference_documents(tmp_path):
    report, references, reference, _ = _fixture(tmp_path)
    output = tmp_path / "export"
    references.write_text(json.dumps(reference) + "\n" + json.dumps(reference) + "\n")
    report_data = json.loads(report.read_text())
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))

    duplicate = _run(report, references, output)
    assert duplicate.returncode != 0
    assert "duplicate" in duplicate.stderr.lower()
    assert not output.exists()

    references.write_text("")
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))
    missing = _run(report, references, output)
    assert missing.returncode != 0
    assert "missing" in missing.stderr.lower()
    assert not output.exists()


def test_rejects_existing_output_and_text_revision_or_span_mismatch(tmp_path):
    report, references, reference, _ = _fixture(tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    existing = _run(report, references, output)
    assert existing.returncode != 0
    assert (output / "sentinel").read_text() == "keep"

    output = tmp_path / "new"
    reference["spans"][0]["text"] = "wrong"
    references.write_text(json.dumps(reference) + "\n")
    report_data = json.loads(report.read_text())
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))
    bad_span = _run(report, references, output)
    assert bad_span.returncode != 0
    assert "span" in bad_span.stderr.lower()
    assert not output.exists()

    reference["spans"][0]["text"] = "ImageJ"
    reference["text_revision"] = "sha256:wrong"
    references.write_text(json.dumps(reference) + "\n")
    report_data["provenance"]["reference_sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))
    bad_revision = _run(report, references, output)
    assert bad_revision.returncode != 0
    assert "text_revision" in bad_revision.stderr
    assert not output.exists()
