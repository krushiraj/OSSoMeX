"""Compile run metrics into the comparison report (protocol section 18)."""

from __future__ import annotations

from pathlib import Path


def _round(value, digits: int = 3):
    if value is None:
        return "n/a"
    return round(float(value), digits)


def build_report(runs_dir: Path, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted(runs_dir.glob("[!.]*/")):
        metrics_path = run_dir / "metrics.json"
        run_path = run_dir / "run.json"
        if not metrics_path.exists() or not run_path.exists():
            continue
        run = __import__("json").loads(run_path.read_text(encoding="utf-8"))
        metrics = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": run.get("run_id"),
                "arm": run.get("arm"),
                "model_id": run.get("model_id"),
                "status": run.get("status"),
                "mention_f1": _round(metrics.get("mention_detection", {}).get("f1")),
                "mention_precision": _round(metrics.get("mention_detection", {}).get("precision")),
                "mention_recall": _round(metrics.get("mention_detection", {}).get("recall")),
                "version_edge_f1": _round(metrics.get("version", {}).get("explicit_edge_f1")),
                "exact_intent_set": _round(metrics.get("intents", {}).get("exact_set_accuracy")),
                "sentiment_macro_f1": _round(metrics.get("sentiment", {}).get("macro_f1")),
                "core_complete": _round(metrics.get("complete_occurrence", {}).get("core_accuracy")),
            }
        )

    lines = [
        "# Software mention extraction - pilot comparison",
        "",
        "Protocol v1.0. Status: pilot, silver gold, local-only. See README.md and reports/BLOCKERS.md.",
        "",
        "## Quality leaderboard (silver gold, provisional)",
        "",
        "| run | arm | model | status | mention F1 | mention P | mention R | version F1 | intent exact | sentiment macro | core complete |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {arm} | {model_id} | {status} | {mention_f1} | {mention_precision} | "
            "{mention_recall} | {version_edge_f1} | {exact_intent_set} | {sentiment_macro_f1} | {core_complete} |".format(
                **row
            )
        )
    if not rows:
        lines.append("| _(no evaluated runs yet)_ |")

    blocker_path = reports_dir / "BLOCKERS.md"
    blockers = blocker_path.read_text(encoding="utf-8") if blocker_path.exists() else ""
    lines += ["", blockers]
    out = reports_dir / "comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out