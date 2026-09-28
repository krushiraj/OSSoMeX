from copy import deepcopy
import json

import pytest

from research.annotations import review_store as store
from research.annotations.tasks import make_tasks
from research.annotations.validation import validate_task_occurrence
from research.contracts import FIELDS, text_revision
from research.data.manifest import json_bytes


def references(snapshot, documents):
    from research.comparison.references import references_from_snapshot
    return references_from_snapshot(snapshot, documents)


def fixture_item(*, role="train", fields=("software", "versions"), shared_version=False):
    text = "😀 Alpha 1.0 and Beta 2.0."
    doc = {"document_id": "fixture", "text": text, "text_revision": text_revision(text),
           "source": "synthetic-contract-fixture", "split": role, "public": True,
           "access_basis": "synthetic fixture", "text_license": "CC-BY-4.0",
           "exposed": True, "training_overlap": True}
    task = make_tasks(doc, {"policy_version": "scibert-poc-2.0", "policy_hash": "a" * 64})[0]
    task.update(exposed=True, training_overlap=True)
    occurrences = []
    for name, version in [("Alpha", "1.0"), ("Beta", "1.0" if shared_version else "2.0")]:
        start, vstart = text.index(name), text.index(version)
        occurrences.append(validate_task_occurrence(task, {
            "schema_version": "2.0", "document_id": doc["document_id"], "text_revision": doc["text_revision"],
            "name": name, "name_span": {"start": start, "end": start + len(name)},
            "context_sentence": text, "context_span": task["context_span"], "context_kind": "paragraph",
            "version_links": [{"text": version, "span": {"start": vstart, "end": vstart + len(version)},
                               "status": "explicit_local"}], "version_status": "explicit", "intents": ["used"],
            "sentiment": "not_expressed", "known": dict.fromkeys(FIELDS, True),
            "evidence": {"intents": [task["context_span"]], "sentiment": []},
            "review": {"status": "agent_provisional", "reasons": []}}))
    masks = {field: field in fields for field in FIELDS}
    annotation = {"occurrences": occurrences, "covered_regions": [
        {**task["annotation_region"], "fields": masks, "status": "partial",
         "provenance": {"kind": "agent_provisional"}}], "unresolved_regions": [],
        "status": "partial", "review_status": "agent_provisional", "annotation_revision": 1}
    return doc, {"task": task, "annotation": annotation}


def snapshot(tmp_path, *, decisions=(), **options):
    doc, item = fixture_item(**options)
    c = store.open_store(tmp_path / "new-fixture.sqlite")
    try:
        store.import_items(c, [item], item["task"]["split"])
        for index, action in enumerate(decisions, 1):
            task = item["task"]
            request = {"decision_id": f"decision-{index}", "task_id": task["task_id"],
                       "document_id": task["document_id"], "text_revision": task["text_revision"],
                       "base_annotation_revision": index, "reviewer": "Synthetic reviewer",
                       "action": action, "reason": "Synthetic bounded review", "value": None}
            if action == "accept_occurrence":
                request["target_name_span"] = item["annotation"]["occurrences"][0]["name_span"]
            store.apply_decision(c, request)
        path = tmp_path / "new-snapshot"
        store.export_reference(c, path)
    finally:
        c.close()
    return doc, item, path


@pytest.mark.parametrize("role", ["train", "dev"])
def test_snapshot_projection_preserves_field_masks_and_provisional_status(tmp_path, role):
    doc, item, path = snapshot(tmp_path, role=role, fields=("software",))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    rows = references(path, [doc])
    assert len(rows) == 1
    ref = rows[0]
    assert [(r["label"], r["review_kind"]) for r in ref["coverage"]] == [("SOFTWARE", "agent_provisional")]
    assert all(s["review_kind"] == "agent_provisional" for s in ref["spans"])
    assert ref["provenance"]["covered_regions"] == item["annotation"]["covered_regions"]
    assert ref["provenance"]["exposed"] is True
    assert ref["provenance"]["training_overlap"] is True
    assert ref["provenance"]["snapshot_manifest_sha256"]
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_candidate_acceptance_cannot_become_human_passage_coverage(tmp_path):
    doc, _, path = snapshot(tmp_path, decisions=["accept_occurrence"])
    ref = references(path, [doc])[0]
    assert all(r["review_kind"] == "agent_provisional" for r in ref["coverage"])
    assert [(s["label"], s["text"]) for s in ref["spans"] if s["review_kind"] == "human_reviewed"] == [
        ("SOFTWARE", "Alpha"), ("VERSION", "1.0")]


def test_passage_name_audit_does_not_promote_version_masks(tmp_path):
    doc, _, path = snapshot(tmp_path, decisions=["accept_passage"])
    ref = references(path, [doc])[0]
    assert {(r["label"], r["review_kind"]) for r in ref["coverage"]} == {
        ("SOFTWARE", "human_reviewed"), ("VERSION", "agent_provisional")}
    assert all(s["review_kind"] == "agent_provisional" for s in ref["spans"] if s["label"] == "VERSION")


def test_confirmed_shared_version_edges_become_one_distinct_span(tmp_path):
    doc, _, path = snapshot(tmp_path, shared_version=True)
    ref = references(path, [doc])[0]
    assert [(s["start"], s["end"], s["text"]) for s in ref["spans"] if s["label"] == "VERSION"] == [
        (8, 11, "1.0")]
    assert not any("version_links" in s or "software_span" in s for s in ref["spans"])


def test_loader_rejects_stale_revision_and_changed_context(tmp_path):
    doc, _, path = snapshot(tmp_path)
    changed = {**doc, "text": doc["text"] + " changed"}
    changed["text_revision"] = text_revision(changed["text"])
    with pytest.raises(ValueError, match="revision"):
        references(path, [changed])
    with pytest.raises(ValueError, match="duplicate"):
        references(path, [doc, doc])


@pytest.mark.parametrize("change", [{"role": "test"}, {"private": True}, {"public": False}])
def test_private_or_test_snapshot_rejected_from_manifest_before_sidecars(tmp_path, change):
    path = tmp_path / "metadata-only"
    path.mkdir()
    (path / "manifest.json").write_bytes(json_bytes({"role": "train", **change}))
    doc, _ = fixture_item()
    with pytest.raises(ValueError, match="not permitted"):
        references(path, [doc])


def test_snapshot_hash_tampering_is_rejected_by_validated_reader(tmp_path):
    doc, _, path = snapshot(tmp_path)
    (path / "occurrences.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed artifact"):
        references(path, [doc])


def test_private_document_cannot_be_used_as_reference(tmp_path):
    doc, _, path = snapshot(tmp_path)
    doc["public"] = False
    with pytest.raises(ValueError, match="not permitted"):
        references(path, [doc])


def test_snapshot_does_not_invent_coverage_from_occurrence_known_masks(tmp_path):
    doc, item = fixture_item(fields=())
    c = store.open_store(tmp_path / "fresh.sqlite")
    try:
        store.import_items(c, [item], "train")
        store.export_reference(c, tmp_path / "snapshot")
    finally:
        c.close()
    ref = references(tmp_path / "snapshot", [doc])[0]
    assert ref["coverage"] == []
    assert len(ref["spans"]) == 4


def test_stale_human_name_audit_is_not_promoted_by_leftover_region_marker(tmp_path):
    doc, item = fixture_item()
    c = store.open_store(tmp_path / "fresh.sqlite")
    try:
        store.import_items(c, [item], "train")
        task = item["task"]
        base = {"task_id": task["task_id"], "document_id": task["document_id"],
                "text_revision": task["text_revision"], "reviewer": "Synthetic reviewer", "reason": "fixture"}
        store.apply_decision(c, {**base, "decision_id": "audit", "base_annotation_revision": 1,
                                "action": "accept_passage", "value": None})
        occurrence = deepcopy(item["annotation"]["occurrences"][0])
        store.apply_decision(c, {**base, "decision_id": "edit", "base_annotation_revision": 2,
                                "action": "upsert_occurrence", "value": occurrence,
                                "target_name_span": occurrence["name_span"]})
        store.export_reference(c, tmp_path / "snapshot")
    finally:
        c.close()
    assert all(r["review_kind"] == "agent_provisional"
               for r in references(tmp_path / "snapshot", [doc])[0]["coverage"])


def test_verified_workflow_approval_promotes_only_explicit_complete_fields(tmp_path):
    doc, item = fixture_item(fields=("software", "versions"))
    c = store.open_store(tmp_path / "fresh.sqlite")
    try:
        store.import_items(c, [item], "train")
        task = item["task"]
        store.apply_decision(c, {"decision_id": "approve", "task_id": task["task_id"],
            "document_id": task["document_id"], "text_revision": task["text_revision"],
            "reviewer": "Synthetic reviewer", "reason": "Full fixture review", "actor_kind": "human",
            "base_annotation_revision": 1, "action": "apply_review_batch",
            "value": {"schema_version": "1.0", "completion": "approve", "proposals_revealed": True,
                      "operations": []}})
        store.export_reference(c, tmp_path / "snapshot")
    finally:
        c.close()
    ref = references(tmp_path / "snapshot", [doc])[0]
    assert {(r["label"], r["review_kind"]) for r in ref["coverage"]} == {
        ("SOFTWARE", "human_reviewed"), ("VERSION", "human_reviewed")}
    assert all(s["review_kind"] == "human_reviewed" for s in ref["spans"])


@pytest.mark.parametrize("bad", ["boolean_mask", "outside_owned_region", "unknown_versions"])
def test_snapshot_field_coverage_is_independently_checked(tmp_path, bad):
    doc, item = fixture_item()
    region = item["annotation"]["covered_regions"][0]
    if bad == "boolean_mask": region["fields"]["software"] = 1
    elif bad == "outside_owned_region": region["end"] += 1
    else:
        occurrence = item["annotation"]["occurrences"][0]
        occurrence["known"]["versions"] = False
        occurrence["version_status"] = "ambiguous"
    c = store.open_store(tmp_path / "fresh.sqlite")
    try:
        store.import_items(c, [item], "train")
        store.export_reference(c, tmp_path / "snapshot")
    finally:
        c.close()
    if bad == "unknown_versions":
        ref = references(tmp_path / "snapshot", [doc])[0]
        assert all(r["label"] != "VERSION" for r in ref["coverage"])
        assert [(s["text"], s["start"], s["end"]) for s in ref["spans"] if s["label"] == "VERSION"] == [
            ("2.0", 21, 24)]
    else:
        with pytest.raises(ValueError, match="coverage"):
            references(tmp_path / "snapshot", [doc])


@pytest.mark.parametrize('masked', [True, False])
def test_version_positive_recovery_respects_known_mask_without_full_coverage(tmp_path, masked):
    from test_comparison_metrics import result, score, span

    doc, item = fixture_item(fields=("software", "versions") if masked else ())
    if masked:
        for occurrence in item['annotation']['occurrences']:
            occurrence['known']['versions'] = False
            occurrence['version_status'] = 'ambiguous'
    connection = store.open_store(tmp_path / 'review.sqlite')
    try:
        store.import_items(connection, [item], 'train')
        store.export_reference(connection, tmp_path / 'snapshot')
    finally:
        connection.close()
    before = {path.name: path.read_bytes() for path in (tmp_path / 'snapshot').iterdir()}
    refs = references(tmp_path / 'snapshot', [doc])
    predictions = [result(doc, [span(doc, 'VERSION', 8, 11), span(doc, 'VERSION', 21, 24)])]
    label = score([doc], predictions, refs, kind='agent_provisional')['arms']['a']['labels']['VERSION']
    assert label['positive_recovery'] == {
        'eligible_positives': 0 if masked else 2, 'recovered': 0 if masked else 2, 'missed': 0,
        'recovery_rate': None if masked else 1.0, 'eligible_documents': 0 if masked else 1}
    assert label['operational'] == {'tp': 0, 'fp': 0, 'fn': 0,
                                    'precision': None, 'recall': None, 'f1': None}
    assert not any(region['label'] == 'VERSION' for region in refs[0]['coverage'])
    if masked:
        assert refs[0]['provenance']['unscored_version_candidates'] == [
            {'mention_id': occurrence['mention_id'], 'known': False, 'version_status': 'ambiguous',
             'version_links': occurrence['version_links']} for occurrence in item['annotation']['occurrences']]
    assert {path.name: path.read_bytes() for path in (tmp_path / 'snapshot').iterdir()} == before
