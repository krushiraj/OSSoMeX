"""research CLI (protocol section 09 command contract)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import aggregate as aggregate_mod
from . import chunking as chunking_mod
from . import evaluate as evaluate_mod
from . import io as io_mod
from . import schema as schema_mod
from . import text as text_mod
from .adapters import get_adapter
from .adapters.base import PredictionInput

ROOT = Path(__file__).resolve().parents[2]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cmd_inventory(args: argparse.Namespace) -> int:
    import hashlib

    inputs = Path(args.input)
    out = Path(args.output)
    rows: list[dict] = []
    seen: set[str] = set()
    for path in sorted(inputs.iterdir() if inputs.is_dir() else [inputs]):
        if path.suffix == ".jsonl":
            for doc in io_mod.read_jsonl(path):
                doc_id = doc.get("document_id")
                if not doc_id:
                    continue
                text = doc.get("text", "")
                sha = io_mod.sha256_text(text)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                rows.append(
                    {
                        "document_id": doc_id,
                        "content_sha256": sha,
                        "text_revision": doc.get("text_revision") or f"sha256:{sha}",
                        "byte_count": len(text.encode("utf-8")),
                        "char_count": len(text),
                        "metadata": doc.get("metadata"),
                        "page_spans": doc.get("page_spans"),
                        "sections": doc.get("sections"),
                        "cohort": "D0",
                        "source_file": path.name,
                        "eligible": True,
                        "exclusion_reason": None,
                    }
                )
    io_mod.write_jsonl(out, rows)
    print(f"inventory: {len(rows)} documents -> {out}")
    return 0


def cmd_prepare_text(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    source_dir = Path(args.source)
    out_dir = ROOT / "manifests" / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible = {
        doc["document_id"]: doc
        for doc in io_mod.read_jsonl(manifest)
        if doc.get("eligible", True)
    }
    existing_path = ROOT / "manifests" / "prepared.jsonl"
    rows = list(io_mod.read_jsonl(existing_path)) if existing_path.exists() else []
    seen_ids: set[str] = {doc["document_id"] for doc in rows}
    for path in sorted(source_dir.glob("*.jsonl")):
        for doc in io_mod.read_jsonl(path):
            doc_id = doc.get("document_id")
            if doc_id not in eligible or doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            normalized = text_mod.normalize(doc.get("text", ""))
            sections = doc.get("sections") or text_mod.detect_sections(normalized.text)
            record = {
                "document_id": doc_id,
                "text_revision": doc.get("text_revision"),
                "normalized_text": normalized.text,
                "sections": sections,
                "index_map": normalized.index_map,
                "normalization_changes": normalized.changes,
                "text_revision_normalized": f"sha256:{io_mod.sha256_text(normalized.text)}",
            }
            (out_dir / f"{doc_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            rows.append(
                {
                    "document_id": doc_id,
                    "text_revision": doc.get("text_revision"),
                    "text_revision_normalized": record["text_revision_normalized"],
                    "chars_original": len(doc.get("text", "")),
                    "chars_normalized": len(normalized.text),
                    "sections": sections,
                    "sections_known": bool(doc.get("sections")),
                }
            )
    io_mod.write_jsonl(ROOT / "manifests" / "prepared.jsonl", rows)
    print(f"prepared: {len(rows)} documents -> manifests/prepared.jsonl")
    return 0


def _load_prepared(doc_id: str) -> dict:
    path = ROOT / "manifests" / "prepared" / f"{doc_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_validate_gold(args: argparse.Namespace) -> int:
    gold = list(io_mod.read_jsonl(Path(args.gold)))
    problems = evaluate_mod.validate_gold_rows(gold)
    if problems:
        print(json.dumps(problems, indent=2))
        return 1
    print(f"gold ok: {len(gold)} rows")
    return 0


def cmd_make_chunks(args: argparse.Namespace) -> int:
    import hashlib

    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    target = int(config.get("target_content_tokens", 480))
    overlap = int(config.get("overlap_tokens", 64))
    policy = config.get("policy_version", "1.0")
    token_sample = config.get("tokenizer_counter", False)
    counter = chunking_mod.estimate_tokens
    tokenizer = None
    if token_sample:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(token_sample)
            counter = chunking_mod.make_tokenizer_counter(tokenizer)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: tokenizer {token_sample} unavailable, using fallback: {exc}")

    out_rows = []
    summaries = []
    for doc in io_mod.read_jsonl(ROOT / "manifests" / "prepared.jsonl"):
        prepared = _load_prepared(doc["document_id"])
        normalized = text_mod.NormalizedText(
            text=prepared["normalized_text"],
            original="",
            index_map=prepared["index_map"],
            changes=prepared["normalization_changes"],
        )
        chunks = chunking_mod.make_chunks(
            doc["document_id"],
            doc["text_revision"] or doc["text_revision_normalized"],
            normalized,
            prepared["sections"],
            target_content_tokens=target,
            overlap_tokens=overlap,
            policy_version=policy,
            counter=counter,
        )
        for chunk in chunks:
            out_rows.append(chunk.as_dict())
        coverage = chunking_mod.coverage(chunks, len(normalized.text))
        summaries.append(
            {
                "document_id": doc["document_id"],
                "chars_normalized": len(normalized.text),
                "chunks": len(chunks),
                "coverage_fraction": round(coverage, 4),
                "tokenizer": token_sample or "chars_per_4",
            }
        )
    io_mod.write_jsonl(ROOT / "manifests" / "chunks.jsonl", out_rows)
    print(json.dumps(summaries, indent=2))
    print(f"chunks written: {len(out_rows)}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    arm = args.arm
    run_id = args.run_id
    run_dir = ROOT / "runs" / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    adapter = get_adapter(arm)
    load_events = adapter.load(config)
    io_mod.write_jsonl(
        run_dir / "load_events.jsonl",
        [{"event": "load", "result": load_events, "time_utc": _utc()}],
    )

    run_manifest = {
        "run_id": run_id,
        "status": "running",
        "protocol_version": "1.0",
        "schema_version": "1.0",
        "policy_version": "1.0",
        "track": args.track,
        "arm": arm,
        "model_id": load_events.get("model_id") or config.get("model_id"),
        "feature_support": adapter.feature_support,
        "split": args.split,
        "started_at_utc": _utc(),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    split_docs = _load_chunk_lines(args.split)
    predictions: list[dict] = []
    evidence: list[dict] = []
    doc_status: list[dict] = []
    events: list[dict] = []
    events_file = run_dir / "events.jsonl"
    events_buf = open(events_file, "a", encoding="utf-8")
    done_count = 0
    total_chunks = sum(len(chunks) for chunks in split_docs.values())

    for doc_id, chunks in split_docs.items():
        prepared = _load_prepared(doc_id)
        normalized = prepared["normalized_text"]
        doc_start = time.monotonic()
        chunk_statuses = []
        for chunk in chunks:
            chunk_text = normalized[chunk["start"]:chunk["end"]]
            item = PredictionInput(
                document_id=doc_id,
                text_revision=chunk["text_revision"],
                chunk_id=chunk["chunk_id"],
                character_offsets=(chunk["start"], chunk["end"]),
                section_type=chunk["section_type"],
                text=chunk_text,
                page_map=None,
            )
            t0 = time.monotonic()
            result = adapter.predict_one(item, {"run_id": run_id, "variant": args.variant, "split": args.split})
            duration = time.monotonic() - t0
            chunk_statuses.append(result.status)
            events_buf.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "document_id": doc_id,
                        "chunk_id": chunk["chunk_id"],
                        "status": result.status,
                        "duration_seconds": round(duration, 4),
                        "input_chars": len(chunk_text),
                        "usage": result.usage,
                        "warnings": result.warnings,
                        "error": result.error,
                        "time_utc": _utc(),
                    }
                )
                + "\n"
            )
            events_buf.flush()
            done_count += 1
            for rec, ev in zip(result.public_records, result.evidence_records):
                merged = {
                    "run_id": run_id,
                    "document_id": doc_id,
                    "chunk_id": chunk["chunk_id"],
                    **ev,
                    **rec,
                }
                merged["run_id"] = run_id
                predictions.append(merged)
            for ev in result.evidence_records:
                ev.setdefault("run_id", run_id)
                evidence.append(ev)
        wall = time.monotonic() - doc_start
        doc_status.append(
            {
                "run_id": run_id,
                "document_id": doc_id,
                "cohort": "D0",
                "text_chars": len(normalized),
                "chunks_expected": len(chunks),
                "chunks_completed": len(chunk_statuses),
                "failure_reason": None,
                "final_status": "completed",
                "wall_seconds": round(wall, 4),
            }
        )
        print(f"progress: {done_count}/{total_chunks} chunks, doc {doc_id} wall {round(wall,1)}s", flush=True)

    events_buf.close()
    io_mod.write_jsonl(run_dir / "predictions.jsonl", predictions)
    io_mod.write_jsonl(run_dir / "evidence.jsonl", evidence)
    io_mod.write_jsonl(run_dir / "document_status.jsonl", doc_status)

    run_manifest.update(
        status="completed" if doc_status else "failed",
        finished_at_utc=_utc(),
        predictions=len(predictions),
        evidence=len(evidence),
        documents=len(doc_status),
    )
    (run_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"predict {arm} -> {run_id}: {len(predictions)} predictions, {len(evidence)} evidence, {len(doc_status)} docs")
    return 0


def _load_chunk_lines(split: str) -> dict[str, list[dict]]:
    splits = json.loads((ROOT / "manifests" / "splits.json").read_text(encoding="utf-8"))
    doc_ids = set(splits.get(split, []))
    groups: dict[str, list[dict]] = {}
    for chunk in io_mod.read_jsonl(ROOT / "manifests" / "chunks.jsonl"):
        if chunk["document_id"] in doc_ids:
            groups.setdefault(chunk["document_id"], []).append(chunk)
    return groups


def cmd_evaluate(args: argparse.Namespace) -> int:
    run_id = args.run_id
    run_dir = ROOT / "runs" / run_id
    predictions = list(io_mod.read_jsonl(run_dir / "predictions.jsonl"))
    gold = list(io_mod.read_jsonl(Path(args.gold)))

    for pred in predictions:
        pred.setdefault("version", None)
        pred.setdefault("intents", [])
        pred.setdefault("sentiment", None)
        pred.setdefault("name_span", None)

    feature_support = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("feature_support")
    score_sentiment = bool(feature_support.get("sentiment", True))
    evaluator = evaluate_mod.Evaluator(score_sentiment=score_sentiment)
    result = evaluator.evaluate(gold, predictions)

    summary = {k: v for k, v in result.items() if k != "details"}
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    io_mod.write_jsonl(run_dir / "matched_details.jsonl", result["details"])
    print(json.dumps(summary, indent=2))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    run_id = args.run_id
    run_dir = ROOT / "runs" / run_id
    predictions = list(io_mod.read_jsonl(run_dir / "predictions.jsonl"))
    result = aggregate_mod.aggregate_documents(predictions)
    io_mod.write_jsonl(run_dir / "aggregated.jsonl", [result[doc] for doc in sorted(result)])
    total_mentions = sum(v["distinct_mention_count"] for v in result.values())
    total_rows = sum(v["export_row_count"] for v in result.values())
    print(f"aggregate {run_id}: {len(result)} docs, {total_mentions} distinct mentions, {total_rows} export rows")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    run_dir = ROOT / "runs" / args.run_id
    events = list(io_mod.read_jsonl(run_dir / "events.jsonl"))
    durations = sorted(e["duration_seconds"] for e in events)
    n = len(durations)
    if not n:
        print("no events to benchmark")
        return 0
    pct = lambda p: durations[min(n - 1, int(p / 100 * n))]
    total = sum(durations)
    stats = {
        "run_id": args.run_id,
        "n_requests": n,
        "total_seconds": round(total, 4),
        "mean_seconds": round(total / n, 4),
        "p50_seconds": pct(50),
        "p95_seconds": pct(95),
        "p99_seconds": pct(99),
        "error_events": sum(1 for e in events if e.get("status") not in ("success", "no_mentions")),
    }
    io_mod.write_jsonl(run_dir / "benchmark.jsonl", [stats])
    print(json.dumps(stats, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report  # noqa: PLC0415

    out = build_report(ROOT / args.runs, ROOT / "reports")
    print(f"report written: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research")
    sub = parser.add_subparsers(dest="command", required=True)
    from .data.cli import register as register_data
    register_data(sub)

    p = sub.add_parser("inventory")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("prepare-text")
    p.add_argument("--manifest", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_prepare_text)

    p = sub.add_parser("validate-gold")
    p.add_argument("--gold", required=True)
    p.add_argument("--splits", default=None)
    p.set_defaults(func=cmd_validate_gold)

    p = sub.add_parser("make-chunks")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_make_chunks)

    p = sub.add_parser("predict")
    p.add_argument("--arm", required=True)
    p.add_argument("--track", default="T1")
    p.add_argument("--split", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-id", required=True)
    p.add_argument("--gold", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("aggregate")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("report")
    p.add_argument("--runs", default="runs")
    p.add_argument("--output", default="reports")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
