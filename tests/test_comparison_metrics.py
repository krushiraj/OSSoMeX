from copy import deepcopy

import pytest

from research.comparison.contracts import CAPABILITY_FIELDS
from research.contracts import text_revision


def metrics():
    from research.comparison import metrics as module
    return module


def document(ident="d", text="Alpha 1.0 and Beta 2.0."):
    return {"document_id": ident, "text": text, "text_revision": text_revision(text)}


def span(doc, label, start, end, **extra):
    return {"label": label, "start": start, "end": end, "text": doc["text"][start:end], **extra}


def result(doc, spans=(), *, arm="a", status=None, **extra):
    capabilities = dict.fromkeys(CAPABILITY_FIELDS, False)
    capabilities.update(software_spans=True, version_spans=True)
    status = status or ("success" if spans else "no_mentions")
    if status == "unsupported":
        capabilities.update(software_spans=False, version_spans=False)
    return {"schema_version": "span-comparison-1", "arm_id": arm,
            "document_id": doc["document_id"], "text_revision": doc["text_revision"],
            "status": status, "offset_unit": "unicode_codepoint_half_open",
            "capabilities": capabilities, "scores_calibrated": False,
            "spans": [{**s, "score": None, "score_kind": None, "alignment_method": "native_codepoint"}
                      for s in spans], "unresolved": [], "chunks": [],
            "raw_artifact": "raw/fixture.json", "provenance": {"kind": "synthetic"}, **extra}


def coverage(doc, label, *, start=0, end=None, kind="human_reviewed"):
    return {"start": start, "end": len(doc["text"]) if end is None else end,
            "label": label, "complete": True, "review_kind": kind}


def reference(doc, spans=(), regions=(), **extra):
    return {"document_id": doc["document_id"], "text_revision": doc["text_revision"],
            "spans": list(spans), "coverage": list(regions), "provenance": {}, **extra}


def score(docs, rows, refs, kind="human_reviewed"):
    return metrics().score_reference(docs, rows, refs, review_kind=kind)


@pytest.mark.parametrize("label", ["SOFTWARE", "VERSION"])
@pytest.mark.parametrize("gold_offsets,pred_offsets,expected", [
    ([(0, 5), (14, 18)], [(0, 5), (14, 17)],
     {"tp": 1, "fp": 1, "fn": 1, "precision": .5, "recall": .5, "f1": .5}),
    ([], [(0, 5)], {"tp": 0, "fp": 1, "fn": 0, "precision": 0., "recall": None, "f1": 0.}),
    ([], [], {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}),
])
def test_exact_span_metrics_and_undefined_ratios(label, gold_offsets, pred_offsets, expected):
    doc = document()
    gold = [span(doc, label, *offset) for offset in gold_offsets]
    pred = [span(doc, label, *offset) for offset in pred_offsets]
    report = score([doc], [result(doc, pred)], [reference(doc, gold, [coverage(doc, label)])])
    assert report["mode"] == "reference_diagnostic"
    assert report["arms"]["a"]["labels"][label]["operational"] == expected


def test_software_complete_version_unknown_never_creates_version_negatives():
    doc = document()
    report = score([doc], [result(doc, [span(doc, "VERSION", 6, 9)])],
                   [reference(doc, [], [coverage(doc, "SOFTWARE")])])
    label = report["arms"]["a"]["labels"]["VERSION"]
    assert label["operational"] == {"tp": 0, "fp": 0, "fn": 0,
                                    "precision": None, "recall": None, "f1": None}
    assert label["coverage"]["eligible_documents"] == 0
    assert label["exclusions"]["prediction_uncovered"] == 1


def test_boundary_crossing_and_uncovered_spans_are_counted_and_excluded():
    doc = document()
    gold = [span(doc, "SOFTWARE", 0, 5), span(doc, "SOFTWARE", 14, 18)]
    report = score([doc], [result(doc, gold)],
                   [reference(doc, gold, [coverage(doc, "SOFTWARE", end=4)])])
    label = report["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["operational"]["fp"] == label["operational"]["fn"] == 0
    assert label["exclusions"] == {"gold_boundary_crossing": 1, "gold_uncovered": 1,
                                    "prediction_boundary_crossing": 1, "prediction_uncovered": 1}


def test_duplicate_gold_and_coverage_do_not_inflate_counts():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    ref = reference(doc, [gold, gold], [coverage(doc, "SOFTWARE")] * 2)
    report = score([doc], [result(doc, [gold])], [ref, deepcopy(ref)])
    label = report["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["operational"]["tp"] == 1
    assert label["coverage"]["eligible_gold_spans"] == 1
    assert label["coverage"]["covered_codepoints"] == len(doc["text"])


def test_candidate_only_review_is_positive_recovery_without_full_scores():
    doc = document()
    gold = [span(doc, "SOFTWARE", 0, 5, review_kind="human_reviewed"),
            span(doc, "SOFTWARE", 14, 18, review_kind="human_reviewed")]
    report = score([doc], [result(doc, [gold[0], span(doc, "SOFTWARE", 6, 9)])],
                   [reference(doc, gold)])
    label = report["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["positive_recovery"] == {"eligible_positives": 2, "recovered": 1, "missed": 1,
                                           "recovery_rate": .5, "eligible_documents": 1}
    assert label["operational"]["precision"] is None
    assert label["operational"]["recall"] is None
    assert label["operational"]["f1"] is None
    assert not {"precision", "recall", "f1"} & label["positive_recovery"].keys()


def test_untagged_candidates_are_unclassified_and_explicit_provenance_tags_are_respected():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    refs = [reference(doc, [gold])]
    assert score([doc], [result(doc, [gold])], refs)["arms"]["a"]["labels"]["SOFTWARE"][
        "positive_recovery"]["eligible_positives"] == 0
    refs[0]["provenance"]["review_kind"] = "agent_provisional"
    assert score([doc], [result(doc, [gold])], refs)["arms"]["a"]["labels"]["SOFTWARE"][
        "positive_recovery"]["eligible_positives"] == 0
    assert score([doc], [result(doc, [gold])], refs, "agent_provisional")["arms"]["a"]["labels"][
        "SOFTWARE"]["positive_recovery"]["recovered"] == 1


def test_human_and_provisional_coverage_are_separate():
    doc = document()
    gold = [span(doc, "SOFTWARE", 0, 5), span(doc, "SOFTWARE", 14, 18)]
    ref = reference(doc, gold, [coverage(doc, "SOFTWARE", end=10),
                    coverage(doc, "SOFTWARE", start=10, kind="agent_provisional")])
    for kind, expected in [("human_reviewed", (1, 0)), ("agent_provisional", (0, 1))]:
        report = score([doc], [result(doc, gold[:1])], [ref], kind)
        label = report["arms"]["a"]["labels"]["SOFTWARE"]
        assert (label["operational"]["tp"], label["operational"]["fn"]) == expected
        assert report["review_kind"] == kind
        assert report["coverage"][kind]["SOFTWARE"]["documents"] == 1


@pytest.mark.parametrize("status,arm,reason", [("unavailable", "a", "missing_checkpoint"),
                                             ("unsupported", "scibert-base", "no_task_head")])
def test_preflight_inactive_arms_have_null_metrics(status, arm, reason):
    doc = document()
    report = score([doc], [result(doc, arm=arm, status=status, reason=reason)],
                   [reference(doc, [span(doc, "SOFTWARE", 0, 5)], [coverage(doc, "SOFTWARE")])])
    label = report["arms"][arm]["labels"]["SOFTWARE"]
    assert set(label["operational"].values()) == {None}
    assert report["arms"][arm]["inference_started"] is False
    assert label["coverage"]["eligible_gold_spans"] == 1
    assert label["denominators"]["operational_documents"] == 0


def test_failures_and_missing_documents_keep_gold_in_primary_denominator():
    docs = [document(str(i)) for i in range(3)]
    gold = [span(docs[0], "SOFTWARE", 0, 5)]
    refs = [reference(d, gold, [coverage(d, "SOFTWARE")]) for d in docs]
    rows = [result(docs[0], gold), result(docs[1], status="failure", reason="timeout")]
    report = score(docs, rows, refs)
    arm = report["arms"]["a"]
    label = arm["labels"]["SOFTWARE"]
    assert label["operational"]["tp"] == 1
    assert label["operational"]["fn"] == 2
    assert label["secondary_completed_only"]["fn"] == 0
    assert label["secondary_completed_only"]["f1"] == 1.
    assert label["denominators"] == {"operational_documents": 3, "completed_only_documents": 1}
    assert arm["operations"]["failed_documents"] == 2
    assert arm["operations"]["missing_documents"] == 1


def test_mixed_valid_and_unresolved_output_discards_all_scored_predictions():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    row = result(doc, [gold], unresolved=[{"text": "Beta", "reason": "ambiguous_text"}])
    original = deepcopy(row)
    report = score([doc], [row], [reference(doc, [gold], [coverage(doc, "SOFTWARE")])])
    arm = report["arms"]["a"]
    assert arm["labels"]["SOFTWARE"]["operational"]["tp"] == 0
    assert arm["labels"]["SOFTWARE"]["operational"]["fn"] == 1
    assert arm["operations"]["invalid_documents"] == 1
    assert arm["per_document"][0]["discarded_predictions"] == 1
    assert row == original


def test_missing_label_capability_has_null_metrics_but_retains_coverage():
    doc = document()
    row = result(doc)
    row["capabilities"]["version_spans"] = False
    report = score([doc], [row], [reference(doc, [span(doc, "VERSION", 6, 9)],
                                                       [coverage(doc, "VERSION")])])
    label = report["arms"]["a"]["labels"]["VERSION"]
    assert label["supported"] is False
    assert label["operational"]["fn"] is None
    assert label["coverage"]["eligible_gold_spans"] == 1


def test_reference_provenance_and_exposure_flags_are_retained_without_mutation():
    doc = document()
    doc.update(exposed=True, training_overlap=True, selection_bias="query_sample")
    ref = reference(doc, provenance={"reviewer": "fixture", "selection": "candidate_only"})
    before = deepcopy(ref)
    report = score([doc], [result(doc)], [ref])
    assert report["population"][0]["exposed"] is True
    assert report["population"][0]["training_overlap"] is True
    assert report["reference_provenance"][0]["provenance"] == ref["provenance"]
    assert ref == before


@pytest.mark.parametrize("mutation", ["revision", "unknown_document", "incomplete", "private", "test"])
def test_invalid_references_are_rejected(mutation):
    doc = document()
    ref = reference(doc, [], [coverage(doc, "SOFTWARE")])
    if mutation == "revision": ref["text_revision"] = "sha256:" + "0" * 64
    elif mutation == "unknown_document": ref["document_id"] = "other"
    elif mutation == "incomplete": ref["coverage"][0]["complete"] = False
    elif mutation == "private": ref["provenance"]["private"] = True
    else: ref["provenance"]["role"] = "test"
    with pytest.raises(ValueError):
        score([doc], [result(doc)], [ref])


def assert_no_accuracy_keys(value):
    if isinstance(value, dict):
        assert not {"precision", "recall", "f1"} & value.keys()
        for child in value.values(): assert_no_accuracy_keys(child)
    elif isinstance(value, list):
        for child in value: assert_no_accuracy_keys(child)


def test_agreement_only_jointly_valid_documents_with_population_visible():
    docs = [document(str(i)) for i in range(3)]
    rows = [result(docs[0], [span(docs[0], "SOFTWARE", 0, 5)], arm="a"),
            result(docs[0], [span(docs[0], "SOFTWARE", 0, 5), span(docs[0], "SOFTWARE", 14, 18)], arm="b"),
            result(docs[1], arm="a"), result(docs[1], arm="b", status="failure", reason="timeout"),
            result(docs[2], arm="a")]
    report = metrics().agreement(docs, rows)
    assert report["mode"] == "unlabelled_agreement"
    assert report["population_documents"] == 3
    pair = report["pairs"][0]
    assert pair["jointly_valid_documents"] == 1
    assert pair["population_documents"] == 3
    assert pair["labels"]["SOFTWARE"] == {"intersection": 1, "union": 2, "left_only": 0,
                                             "right_only": 1, "jaccard": .5}
    assert pair["labels"]["VERSION"]["jaccard"] is None
    assert report["arms"]["a"]["candidate_counts"]["SOFTWARE"] == 1
    assert_no_accuracy_keys(report)


def test_agreement_malformed_outputs_are_not_jointly_valid():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    rows = [result(doc, [gold]), result(doc, [gold], arm="b", unresolved=[{"text": "missing"}])]
    report = metrics().agreement([doc], rows)
    assert report["pairs"][0]["jointly_valid_documents"] == 0
    assert report["pairs"][0]["labels"]["SOFTWARE"]["union"] == 0
    assert report["pairs"][0]["labels"]["SOFTWARE"]["jaccard"] is None
    assert report["arms"]["b"]["operations"]["invalid_documents"] == 1
    assert_no_accuracy_keys(report)


@pytest.mark.parametrize("bad", ["duplicate_input", "duplicate_result", "unknown_result", "kind"])
def test_ambiguous_populations_and_review_kind_are_rejected(bad):
    doc = document()
    docs, rows, kind = [doc], [result(doc)], "human_reviewed"
    if bad == "duplicate_input": docs.append(doc)
    elif bad == "duplicate_result": rows.append(rows[0])
    elif bad == "unknown_result": rows[0]["document_id"] = "other"
    else: kind = "accepted_candidate"
    with pytest.raises(ValueError): score(docs, rows, [], kind)


@pytest.mark.parametrize("bad", ["capabilities_missing", "capabilities_change", "base_completed"])
def test_invalid_arm_metadata_cannot_silently_remove_failure_gold(bad):
    docs = [document("a"), document("b")]
    rows = [result(d) for d in docs]
    if bad == "capabilities_missing": rows[0].pop("capabilities")
    elif bad == "capabilities_change": rows[0]["capabilities"]["version_spans"] = False
    else:
        for row in rows: row["arm_id"] = "scibert-base"
    with pytest.raises(ValueError): score(docs, rows, [])


@pytest.mark.parametrize("bad", ["stale_revision", "wrong_slice", "bad_span", "failed_chunk"])
def test_invalid_document_output_is_an_operational_miss(bad):
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    row = result(doc, [gold])
    if bad == "stale_revision": row["text_revision"] = "sha256:" + "0" * 64
    elif bad == "wrong_slice": row["spans"][0]["text"] = "wrong"
    elif bad == "bad_span": row["spans"][0]["start"] = True
    else: row["chunks"] = [{"status": "failure"}]
    label = score([doc], [row], [reference(doc, [gold], [coverage(doc, "SOFTWARE")])])["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["operational"] == {"tp": 0, "fp": 0, "fn": 1, "precision": None, "recall": 0., "f1": 0.}


def test_adjacent_complete_regions_do_not_silently_admit_crossing_span():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    ref = reference(doc, [gold], [coverage(doc, "SOFTWARE", end=3),
                                coverage(doc, "SOFTWARE", start=3, end=7)])
    label = score([doc], [result(doc, [gold])], [ref])["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["operational"]["tp"] == 0
    assert label["exclusions"]["gold_boundary_crossing"] == 1


def test_agreement_label_denominator_requires_both_arm_capabilities():
    doc = document()
    rows = [result(doc), result(doc, arm="b")]
    rows[1]["capabilities"]["version_spans"] = False
    pair = metrics().agreement([doc], rows)["pairs"][0]
    assert pair["jointly_valid_documents"] == 1
    assert pair["label_jointly_valid_documents"] == {"SOFTWARE": 1, "VERSION": 0}
    assert pair["labels"]["VERSION"]["jaccard"] is None


def test_explicit_span_kind_takes_precedence_over_provenance_kind():
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5, review_kind="agent_provisional")
    ref = reference(doc, [gold], provenance={"review_kind": "human_reviewed"})
    label = score([doc], [result(doc, [gold])], [ref])["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["positive_recovery"]["eligible_positives"] == 0


@pytest.mark.parametrize("kind", ["human_reviewed", "agent_provisional"])
@pytest.mark.parametrize("order", [("agent_provisional", "human_reviewed"),
                                   ("human_reviewed", "agent_provisional")])
@pytest.mark.parametrize("grouped", [True, False])
def test_duplicate_positive_review_kinds_survive_order_and_row_grouping(kind, order, grouped):
    doc = document()
    gold = span(doc, "SOFTWARE", 0, 5)
    evidence = [{**gold, "review_kind": review_kind} for review_kind in order] * 2
    refs = [reference(doc, evidence)] if grouped else [reference(doc, [s]) for s in evidence]
    report = score([doc], [result(doc, [gold])], refs, kind)
    label = report["arms"]["a"]["labels"]["SOFTWARE"]
    assert label["positive_recovery"] == {"eligible_positives": 1, "recovered": 1, "missed": 0,
                                           "recovery_rate": 1., "eligible_documents": 1}
    assert label["operational"] == {"tp": 0, "fp": 0, "fn": 0,
                                    "precision": None, "recall": None, "f1": None}
