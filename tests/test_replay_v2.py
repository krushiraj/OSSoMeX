from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


def test_replay_writes_new_hashed_report_without_changing_inputs(tmp_path, fixture_case):
    from research.evaluation.replay import replay_legacy
    case = fixture_case("fixture-multiversion")
    document_id = case["input"]["document_id"]
    docs = tmp_path / "docs.jsonl"
    gold = tmp_path / "gold.jsonl"
    run = tmp_path / "old-run"
    run.mkdir()
    docs.write_text(json.dumps(case["input"]) + "\n")
    row = {"document_id": document_id, "name": "NumPy", "name_span": {"start": 8, "end": 13},
           "versions": ["1.24", "1.26"], "intents": ["used"]}
    gold.write_text(json.dumps(row) + "\n")
    (run / "predictions.jsonl").write_text("".join(json.dumps({**p, "document_id": document_id, "name_span": row["name_span"]}) + "\n" for p in case["expected_public_records"]))
    (run / "run.json").write_text(json.dumps({"feature_support": {"name": True, "version": True, "intents": True, "sentiment": True}}))
    (run / "metrics.json").write_text('{"legacy": true}')
    paths = [docs, gold, *run.iterdir()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = tmp_path / "new-replay"
    result = replay_legacy(docs, gold, run, output, text_policy="legacy-normalize")
    assert result["mention_detection"]["tp"] == 1
    assert result["version"]["value_edges"]["tp"] == 2
    assert result["sentiment"]["macro_f1"] is None
    assert result["replay"]["file_sha256"][str(gold)] == before[gold]
    assert json.loads((output / "metrics.json").read_text())["evaluator_version"] == "scibert-v2.0"
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before
    with pytest.raises(FileExistsError):
        replay_legacy(docs, gold, run, output, text_policy="legacy-normalize")
