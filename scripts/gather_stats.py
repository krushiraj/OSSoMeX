"""Gather structured stats for all pilot runs into data/stats.json.

Read-only; used to build the detailed results report.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

ROOT = "/Users/krushi/work/openalex-sw-mentions"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slug_of(doc_id: str) -> str:
    pre = doc_id.split("oa-", 1)[1].rsplit("-W", 1)[0]
    return norm(pre)


def ev_sum(events, key):
    s = 0
    n = 0
    for e in events:
        u = e.get("usage") or {}
        v = u.get(key)
        if isinstance(v, int):
            s += v
            n += 1
    return s, n


def parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def main() -> None:
    dev = [json.loads(l) for l in open(f"{ROOT}/inputs/dev/openalex_snippets_dev.jsonl")]
    dev = list({d["document_id"]: d for d in dev}.values())
    sm = json.load(open(f"{ROOT}/scripts/software_mentions.json"))

    by_query = defaultdict(list)
    for d in dev:
        by_query[slug_of(d["document_id"])].append(d["document_id"])

    out: dict = {"runs": {}, "queries": {}}
    for q, names in sorted(by_query.items()):
        out["queries"][q] = names

    for name in sorted(os.listdir(f"{ROOT}/runs")):
        if name.startswith("smoke"):
            continue
        rd = f"{ROOT}/runs/{name}"
        if not os.path.exists(f"{rd}/predictions.jsonl"):
            continue
        manifest = json.load(open(f"{rd}/run.json"))
        preds = [json.loads(l) for l in open(f"{rd}/predictions.jsonl")]
        events = [json.loads(l) for l in open(f"{rd}/events.jsonl")]
        evid = [json.loads(l) for l in open(f"{rd}/evidence.jsonl")]
        doc_status = [json.loads(l) for l in open(f"{rd}/document_status.jsonl")]

        st = Counter(e["status"] for e in events)
        dur = [e.get("duration_seconds") or 0 for e in events]
        total_time = sum(dur)
        in_tok, n_in = ev_sum(events, "input_tokens")
        out_tok, n_out = ev_sum(events, "output_tokens")
        align = Counter(x.get("alignment_status") for x in evid)
        intents = Counter(tuple(sorted(p.get("intents", []))) for p in preds)
        names = Counter(p["name"] for p in preds)
        versions = Counter((p.get("version") or "").strip() for p in preds if p.get("version"))

        # per-doc pred counts -> extraction-volume spread
        per_doc = Counter(p["document_id"] for p in preds)
        docs_with = len(per_doc)

        # per-query hit/miss
        qhit: dict[str, str] = {}
        for q in by_query:
            names_q = set()
            for did in by_query[q]:
                names_q |= {p["name"] for p in preds if p["document_id"] == did}
            hit = any(
                q == norm(nm) or (len(q) >= 3 and q in norm(nm)) or (len(norm(nm)) >= 3 and norm(nm) in q)
                for nm in names_q
            )
            qhit[q] = "Y" if hit else "partial" if names_q else "-"

        # error samples
        err_sample = []
        for e in events:
            if e.get("error"):
                err_sample.append({"chunk": e["chunk_id"], "status": e["status"], "error": e["error"][:140]})
                if len(err_sample) >= 6:
                    break

        start = parse_ts(manifest.get("started_at_utc"))
        end = parse_ts(manifest.get("finished_at_utc"))
        wall = (end - start).total_seconds() if (start and end) else None

        out["runs"][name] = {
            "manifest": {
                "arm": manifest.get("arm"),
                "model_id": manifest.get("model_id"),
                "variant": manifest.get("variant"),
                "track": manifest.get("track"),
                "status": manifest.get("status"),
                "started_at_utc": manifest.get("started_at_utc"),
                "finished_at_utc": manifest.get("finished_at_utc"),
            },
            "totals": {
                "preds": len(preds),
                "evidence": len(evid),
                "docs_with_preds": docs_with,
                "docs_total": len(doc_status),
                "chunks": len(events),
                "wall_seconds": round(wall, 1) if wall else None,
                "sum_chunk_seconds": round(total_time, 1),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
            },
            "chunk_status": dict(st),
            "alignment": dict(align),
            "intent_counts": {"|".join(k): c for k, c in intents.most_common(8)},
            "top_names": names.most_common(25),
            "version_top": versions.most_common(10),
            "docs": docs_with,
            "qhit": qhit,
            "error_sample": err_sample,
        }

    with open(f"{ROOT}/data/stats.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("wrote data/stats.json with", len(out["runs"]), "runs")


if __name__ == "__main__":
    main()