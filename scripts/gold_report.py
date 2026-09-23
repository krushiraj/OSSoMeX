"""Gold-evaluation aggregation for gold trials (track G1).

Scores every runs/GOLD-{split}-* arm against the given gold file using the same
Evaluator as the pipeline, then writes:
  - reports/gold_{tag}.html  (standalone summary report)
  - reports/gold_{tag}.md
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from research import evaluate as evaluate_mod
from research import io as io_mod

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

PCT = lambda v: "—" if v is None else f"{v * 100:.1f}"

_EXTRA_LABELS = {"fastdict": "dictionary longest-match", "softcite": "Softcite 0.8.1"}


def _fold(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def _relaxed_metrics(gold, pred: list[dict]) -> dict:
    from collections import defaultdict as dd

    gold_by_doc = dd(list)
    pred_by_doc = dd(list)
    for g in gold:
        gold_by_doc[g["document_id"]].append(_fold(g["name"]))
    for p in pred:
        name = p.get("name")
        if name:
            pred_by_doc[p.get("document_id")].append(_fold(name))
    tp = fp = fn = 0
    for doc, gold_keys in gold_by_doc.items():
        pred_keys = pred_by_doc.get(doc, [])
        used = [False] * len(pred_keys)
        for gk in gold_keys:
            for i, pk in enumerate(pred_keys):
                if used[i]:
                    continue
                if gk == pk or (len(pk) >= 4 and gk.startswith(pk)) or (len(gk) >= 4 and pk.startswith(gk)):
                    used[i] = True
                    break
            else:
                fn += 1
        tp += sum(used)
        fp += sum(not u for u in used)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r and p + r else None
    return {"P": p, "R": r, "F1": f1, "tp": tp, "fp": fp, "fn": fn}


def arm_label(run_id: str, split: str) -> str:
    m = run_id.replace(f"GOLD-{split}-", "").replace("-001", "")
    return _EXTRA_LABELS.get(m, {"gemma2": "gemma2:2b", "gemma3": "gemma3:4b",
                                   "qwen25": "qwen2.5:3b", "llama32": "llama3.2:3b"}.get(m, m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="somesci")
    ap.add_argument("--gold", default=None, help="path to gold jsonl (default inputs/<split>/<split>_gold.jsonl)")
    ap.add_argument("--tag", default=None, help="file tag / label prefix (default = split)")
    args = ap.parse_args()
    split = args.split
    tag = args.tag or split
    gold_path = Path(args.gold) if args.gold else ROOT / "inputs" / split / f"{split}_gold.jsonl"

    gold = list(io_mod.read_jsonl(gold_path))
    runs = sorted(RUNS.glob(f"GOLD-{split}-*/"))
    print(f"[{tag}] gold rows: {len(gold)} | runs: {[r.name for r in runs]}")
    gold_has_intents = any(bool(g.get("intents")) for g in gold)

    results: dict[str, dict] = {}
    relaxed: dict[str, dict] = {}
    for run_dir in runs:
        run_id = run_dir.name
        pred = list(io_mod.read_jsonl(run_dir / "predictions.jsonl"))
        fs = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("feature_support", {})
        ev = evaluate_mod.Evaluator(score_sentiment=bool(fs.get("sentiment", True)))
        res = ev.evaluate(gold, pred)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({k: v for k, v in res.items() if k != "details"}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        io_mod.write_jsonl(run_dir / "matched_details.jsonl", res["details"])
        results[run_id] = res
        relaxed[run_id] = _relaxed_metrics(gold, pred)
        md = res["mention_detection"]
        print(
            f"{arm_label(run_id, split):28s} P={PCT(md['precision'])} R={PCT(md['recall'])} "
            f"F1={PCT(md['f1'])} tp={md['tp']} fp={md['fp']} fn={md['fn']}"
        )

    _write_markdown(results, gold, relaxed, split, tag, gold_has_intents)
    _write_html(results, gold, relaxed, split, tag, gold_has_intents)


def _write_markdown(results, gold, relaxed, split, tag, gold_has_intents) -> None:
    lines = [f"# Gold Evaluation: {tag} (track G1)", ""]
    n_docs = len({r['document_id'] for r in gold})
    lines.append(
        f"Gold: {len(gold)} mention-level rows across {n_docs} docs (split `{split}`)."
    )
    lines.append("Matching: exact normalized-text name span (one-to-one); relaxed = same folded name in same doc.")
    lines.append("")
    lines.append("| arm | P | R | F1 | tp | fp | fn | pred | gold |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run_id, res in sorted(results.items()):
        md = res["mention_detection"]
        lines.append(
            f"| {arm_label(run_id, split)} | {PCT(md['precision'])} | {PCT(md['recall'])} | "
            f"{PCT(md['f1'])} | {md['tp']} | {md['fp']} | {md['fn']} | "
            f"{md['tp'] + md['fp']} | {res['support']['gold_mentions']} |"
        )
    lines.append("")
    lines.append("Relaxed (folded name match, any position):")
    lines.append("")
    lines.append("| arm | P | R | F1 | tp | fp | fn |")
    lines.append("|---|---|---|---|---|---|---|")
    for run_id, r in sorted(relaxed.items()):
        lines.append(
            f"| {arm_label(run_id, split)} | {PCT(r['P'])} | {PCT(r['R'])} | {PCT(r['F1'])} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} |"
        )
    lines += ["", "## Matching quality on matched mentions", ""]
    for run_id, res in sorted(results.items()):
        quality = f"name-correct = {PCT(res['name_correct_on_matched'])}"
        if gold_has_intents:
            quality += (
                f"; intent macro-F1 = {PCT(res['intents']['macro']['f1'])} "
                f"(exact-set acc {PCT(res['intents']['exact_set_accuracy'])})"
            )
        quality += (
            f"; version edge-F1 = {PCT(res['version']['explicit_edge_f1'])} "
            f"({res['support']['gold_explicit_versions']} gold versions); "
            f"complete-occurrence core acc = {PCT(res['complete_occurrence']['core_accuracy'])}."
        )
        lines.append(f"- **{arm_label(run_id, split)}**: {quality}")
    out = ROOT / "reports" / f"gold_{tag}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _write_html(results, gold, relaxed, split, tag, gold_has_intents) -> None:
    from html import escape

    rows_html = ""
    for run_id, res in sorted(results.items()):
        md = res["mention_detection"]
        rows_html += (
            f"<tr><td><b>{escape(arm_label(run_id, split))}</b></td>"
            f"<td>{PCT(md['precision'])}</td><td>{PCT(md['recall'])}</td><td><b>{PCT(md['f1'])}</b></td>"
            f"<td class=num>{md['tp']}</td><td class=num>{md['fp']}</td><td class=num>{md['fn']}</td>"
            f"<td class=num>{md['tp'] + md['fp']}</td></tr>"
        )
    detail_rows = ""
    for run_id, res in sorted(results.items()):
        label = arm_label(run_id, split)
        top_fp = Counter()
        top_fn = Counter()
        for d in res["details"]:
            if not d["matched"]:
                if d.get("false_positive"):
                    top_fp[(d["document_id"], d["name"])] += 1
                else:
                    top_fn[(d["document_id"], d["name"])] += 1
        fp_str = ", ".join(f"{name} ×{c}" for (_, name), c in top_fp.most_common(6))
        fn_str = ", ".join(f"{name} ×{c}" for (_, name), c in top_fn.most_common(6))
        intent_cell = f"{PCT(res['intents']['macro']['f1'])}" if gold_has_intents else "—"
        detail_rows += (
            f"<tr><td><b>{escape(label)}</b></td>"
            f"<td>{PCT(res['name_correct_on_matched'])}</td>"
            f"<td>{intent_cell}</td>"
            f"<td>{PCT(res['version']['explicit_edge_f1'])}</td>"
            f"<td>{PCT(res['complete_occurrence']['core_accuracy'])}</td>"
            f"<td class=sm>{escape(fp_str)}</td><td class=sm>{escape(fn_str)}</td></tr>"
        )
    relaxed_rows = ""
    for run_id, r in sorted(relaxed.items()):
        relaxed_rows += (
            f"<tr><td><b>{escape(arm_label(run_id, split))}</b></td>"
            f"<td>{PCT(r['P'])}</td><td>{PCT(r['R'])}</td><td><b>{PCT(r['F1'])}</b></td>"
            f"<td class=num>{r['tp']}</td><td class=num>{r['fp']}</td><td class=num>{r['fn']}</td></tr>"
        )
    html_doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Gold Evaluation: {escape(tag)}</title><style>
body{{font-family:-apple-system,sans-serif;margin:2rem;color:#222}}
h1,h2{{color:#0b3d61}} table{{border-collapse:collapse;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#eef3f8}} td.num{{text-align:right}} .sm{{font-size:12px;max-width:340px}}
.note{{color:#555;max-width:900px}}
</style></head><body>
<h1>Gold Evaluation: {escape(tag)}</h1>
<p class=note>Metrics are against gold; matching = exact normalized-text name span, one-to-one; false positives = predictions with no corresponding gold span. {"" if gold_has_intents else "<b>Intents not present in this gold set — intent column withheld.</b>"}</p>
<h2>Mention detection (P / R / F1)</h2>
<table><tr><th>arm</th><th>P</th><th>R</th><th>F1</th><th>tp</th><th>fp</th><th>fn</th><th>pred</th></tr>{rows_html}</table>
<h2>Mention detection — relaxed (folded name match, any position)</h2>
<table><tr><th>arm</th><th>P</th><th>R</th><th>F1</th><th>tp</th><th>fp</th><th>fn</th></tr>{relaxed_rows}</table>
<h2>Matched-mentioned quality</h2>
<table><tr><th>arm</th><th>name-correct</th><th>intent macro-F1</th><th>version edge-F1</th><th>complete-occ core</th><th>top FP names</th><th>top fn names</th></tr>{detail_rows}</table>
<hr><p class=note>Generated {__import__('datetime').date.today().isoformat()} | track G1 | scripts/gold_report.py</p>
</body></html>"""
    out = ROOT / "reports" / f"gold_{tag}.html"
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()