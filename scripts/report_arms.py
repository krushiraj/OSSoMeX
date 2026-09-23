"""Build reports/comparison.md from completed run directories.

Reusable: re-run as runs complete; it discovers runs on the fly.

python scripts/report_arms.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

ROOT = "/Users/krushi/work/openalex-sw-mentions"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slug_of(doc_id: str) -> str:
    pre = doc_id.split("oa-", 1)[1].rsplit("-W", 1)[0]
    return norm(pre)


def load_runs() -> list[dict]:
    import os
    runs = []
    base = f"{ROOT}/runs"
    for name in sorted(os.listdir(base)):
        if name.startswith("smoke"):
            continue
        rd = f"{base}/{name}"
        preds_f = f"{rd}/predictions.jsonl"
        if not os.path.exists(preds_f):
            continue
        run = json.load(open(f"{rd}/run.json"))
        preds = [json.loads(l) for l in open(preds_f)]
        events = [json.loads(l) for l in open(f"{rd}/events.jsonl")] if os.path.exists(f"{rd}/events.jsonl") else []
        runs.append({"name": name, "manifest": run, "preds": preds, "events": events})
    return runs


def qrecall(run: dict, dev: list[dict]) -> tuple[int, int]:
    per_doc: dict[str, set[str]] = defaultdict(set)
    for p in run["preds"]:
        per_doc[p["document_id"]].add(p["name"])
    qselig = hit = 0
    for d in dev:
        q = slug_of(d["document_id"])
        text = norm(d["text"])
        if len(q) >= 3 and q not in text:
            continue
        qselig += 1
        names = per_doc.get(d["document_id"], set())
        if any(
            q == norm(nm) or (len(q) >= 3 and q in norm(nm)) or (len(norm(nm)) >= 3 and norm(nm) in q)
            for nm in names
        ):
            hit += 1
    return qselig, hit


def main() -> None:
    sm = json.load(open(f"{ROOT}/scripts/software_mentions.json"))
    dev_rows = [json.loads(l) for l in open(f"{ROOT}/inputs/dev/openalex_snippets_dev.jsonl")]
    dev = list({d["document_id"]: d for d in dev_rows}.values())
    runs = load_runs()

    lines: list[str] = []
    add = lines.append
    add("# Arm comparison — dev split (D0, 530 chunks / 190 docs)")
    add(f"*generated {datetime.now(timezone.utc).isoformat()}*")
    add("")
    add("## Run summary")
    add("")
    add("| run | arm | model | model_id | finished | preds | docs w/ preds | q-recall (docs) | note |")
    add("|-----|-----|-------|----------|----------|-------|---------------|-----------------|------|")
    for r in runs:
        m = r["manifest"]
        qselig, hit = qrecall(r, dev)
        n_docs = len({p["document_id"] for p in r["preds"]})
        add(
            f"| {r['name']} | {m.get('arm')} | {m.get('model_id','?')} | {m.get('model_id','?')} | "
            f"{m.get('finished_at_utc','?')[:16]} | {len(r['preds'])} | {n_docs} | {hit}/{qselig} | "
            f"{m.get('status','?')} |"
        )
    add("")
    add("## Per-run chunk status")
    add("")
    for r in runs:
        st = defaultdict(int)
        for e in r["events"]:
            st[e["status"]] += 1
        add(f"- **{r['name']}**: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    add("")

    add("## Per-software query recall (OpenAlex-snippet dev docs)")
    add("")
    add("`hit` = model produced a name equivalent to the query seed for at least one of that query's docs.")
    add("")
    add(f"| query | docs | " + " | ".join(f"{r['name']}" for r in runs) + f" |")
    add(f"|-------|------|" + "|".join("---" for _ in runs) + "|")
    for e in sm:
        q = norm(e["name"])
        dids = [d["document_id"] for d in dev if slug_of(d["document_id"]) == q]
        cells = []
        for r in runs:
            names = set()
            for d in dids:
                names |= {p["name"] for p in r["preds"] if p["document_id"] == d}
            hit = any(
                q == norm(nm) or (len(q) >= 3 and q in norm(nm)) or (len(norm(nm)) >= 3 and norm(nm) in q)
                for nm in names
            )
            cells.append("Y" if hit else ("-" if not names else "partial"))
        add(f"| {e['name']} | {len(dids)} | " + " | ".join(cells) + " |")
    add("")

    add("## Pickups, artifacts, notable names (all runs, top occurrences)")
    add("")
    for r in runs:
        add(f"### {r['name']}")
        add("")
        cnt: dict[str, int] = defaultdict(int)
        for p in r["preds"]:
            cnt[p["name"]] += 1
        top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        add(", ".join(f"{n} ({c})" for n, c in top))
        add("")

    out = "\n".join(lines)
    with open(f"{ROOT}/reports/comparison.md", "w") as fh:
        fh.write(out)
    print(f"wrote reports/comparison.md ({len(lines)} lines, {len(runs)} runs)")


if __name__ == "__main__":
    main()