"""Non-destructive v2 replay of frozen historical documents and predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from ..contracts import ContractError, FIELDS, validate_document
from ..text import normalize
from .legacy import convert_legacy
from .metrics import evaluate_v2


def _rows(path):
    result = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected object")
                result.append(row)
            except ValueError as exc:
                raise ContractError({}, f"{path}:{number}", str(exc), "INVALID_JSONL") from exc
    return result


def replay_legacy(documents_path: Path, gold_path: Path, run_dir: Path, output: Path, *, text_policy: str) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if text_policy not in ("legacy-normalize", "frozen"):
        raise ValueError("select frozen or legacy-normalize text policy explicitly")
    docs = _rows(documents_path)
    if text_policy == "legacy-normalize":
        docs = [{**doc, "text": normalize(doc["text"]).text} for doc in docs]
    docs = [validate_document(doc) for doc in docs]
    gold_rows = _rows(gold_path)
    prediction_path = run_dir / "predictions.jsonl"
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    capabilities = run.get("feature_support") or {}
    gold = convert_legacy(gold_rows, docs, {"software": True, "versions": True, "intents": True, "sentiment": True}, reference=True)
    if gold["unresolved_predictions"]:
        raise ContractError({}, "reference", "unaligned historical reference; adjudication required")
    predictions = convert_legacy(_rows(prediction_path), docs, capabilities)
    coverage = []
    for doc in docs:
        doc_gold = [row for row in gold["occurrences"] if row["document_id"] == doc["document_id"]]
        fields = {field: bool(doc_gold) and all(row["known"][field] for row in doc_gold) for field in FIELDS}
        fields["software"] = True
        coverage.append({"document_id": doc["document_id"], "text_revision": doc["text_revision"],
                         "start": 0, "end": len(doc["text"]), "fields": fields, "status": "complete",
                         "provenance": {"kind": "legacy_replay_assumption", "human_gold": False}})
    status_path = run_dir / "document_status.jsonl"
    statuses = _rows(status_path) if status_path.exists() else []
    result = evaluate_v2(docs, gold["occurrences"], predictions["occurrences"] + predictions["unresolved_predictions"], coverage, statuses, capabilities)
    paths = [documents_path, gold_path, prediction_path, run_path]
    if status_path.exists():
        paths.append(status_path)
    result["replay"] = {
        "kind": "historical_same_text_repaired_metrics", "text_policy": text_policy,
        "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        (Path(__file__), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("legacy.py"), Path(__file__).parents[1] / "text.py")},
        "conversion_issues": gold["issues"] + predictions["issues"],
        "limitations": ["Original text and reference retained, including known importer defects.",
                        "Complete software coverage reproduces the historical reference assumption; it is not a fresh coverage audit.",
                        "Historical intent labels are retained without asserting policy compatibility.",
                        "Conflicting predicted attributes are invalid outputs, not a best-of-row selection.",
                        "Legacy status records may be missing or incomplete; do not infer a successful empty run.",
                        "No statistical superiority claim or corrected-source rerun is made by this replay."],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-policy", choices=("legacy-normalize", "frozen"), required=True)
    args = parser.parse_args()
    result = replay_legacy(args.documents, args.gold, args.run, args.output, text_policy=args.text_policy)
    print(json.dumps({"output": str(args.output), "evaluator_version": result["evaluator_version"],
                      "mention_detection": result["mention_detection"]}, indent=2))


if __name__ == "__main__":
    main()
