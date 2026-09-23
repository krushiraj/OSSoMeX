"""Per-query recall for completed runs on the dev OpenAlex-snippet subset.

For every dev document the OpenAlex query seed is derivable from its
document_id (`oa-<query>-W...`).  This is a legitimate *pseudo-gold* for a
coarse recall check: the snippet was returned for that software query, so a
model that extracts software names should surface that software in that
document.  Nothing here is fabricated; it only measures model behavior against
the known sample key.

Equivalence rules (applied to case-folded, alnum-only strings):
  - exact equality, or
  - containment either direction when the longer side is >= 3 chars
    (covers multi-name groups like napari-imagej vs napari).

Usage: python scripts/compare_runs.py [runs/...]
"""

from __future__ import annotations

import json
import re
import sys

ROOT = "/Users/krushi/work/openalex-sw-mentions"

ALIASES: dict[str, set[str]] = {}
for entry in json.load(open(f"{ROOT}/scripts/software_mentions.json")):
    name = entry["name"]
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    ALIASES.setdefault(slug, set()).add(name)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def query_slug(doc_id: str) -> str:
    pre = doc_id.split("oa-", 1)[1].rsplit("-W", 1)[0]
    return norm(pre)


def matches(qnorm: str, predicted: str) -> bool:
    pnorm = norm(predicted)
    if not pnorm:
        return False
    if qnorm == pnorm:
        return True
    if len(qnorm) >= 3 and qnorm in pnorm:
        return True
    if len(pnorm) >= 3 and pnorm in qnorm:
        return True
    return False


def main(paths: list[str]) -> None:
    docs = [json.loads(l) for l in open(f"{ROOT}/inputs/dev/openalex_snippets_dev.jsonl")]
    by_doc = {d["document_id"]: d for d in docs}
    total = len(docs)
    print(f"{'run':<32} {'docs':>5} {'qselig':>7} {'hit':>5} {'recall':>7}  note")
    for path in paths:
        preds = [json.loads(l) for l in open(path)]
        per_doc: dict[str, set[str]] = {}
        for p in preds:
            per_doc.setdefault(p["document_id"], set()).add(p["name"])
        qselig = hit = 0
        misses: list[tuple[str, str, str]] = []
        for doc_id, d in by_doc.items():
            qslugs = {query_slug(doc_id)}
            text = norm(d["text"])
            if not any(q in text for q in qslugs if len(q) >= 3):
                continue
            qselig += 1
            pred_names = per_doc.get(doc_id, set())
            if any(matches(q, pred) for q in qslugs for pred in pred_names):
                hit += 1
            elif qselig - hit <= 15:
                misses.append((doc_id, sorted(qslugs)[0], "; ".join(sorted(pred_names))[:80]))
        recall = hit / qselig if qselig else 0.0
        run = path.rsplit("/", 2)[-2]
        print(f"{run:<32} {total:>5} {qselig:>7} {hit:>5} {recall:>6.2%}  {len(preds)} preds")
        if misses:
            print(f"  sample misses (first {len(misses)} of {qselig - hit}):")
            for doc_id, q, p in misses:
                print(f"    {doc_id}  q={q:<16} preds=({p})")


if __name__ == "__main__":
    main(sys.argv[1:] or [f"{ROOT}/runs/L1-dev-001/predictions.jsonl", f"{ROOT}/runs/S0-dev-001/predictions.jsonl"])