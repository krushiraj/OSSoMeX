"""Ingest SoMeSci (Zenodo 4968738) BRAT annotations into the workspace.

Reads a Label subset (default PLoS_methods), selects docs by size/mention
count, writes:
  - inputs/somesci/somesci_docs.jsonl (document_id, text)   -> pipeline corpus
  - inputs/somesci/somesci_gold.jsonl (document_id, name, name_span
    [normalized coords], version, intents)                  -> evaluator gold

Span coordinates are mapped from BRAT raw offsets into the workspace
normalized-text space using text_mod.normalize + index_map, with a
substring-verification fallback.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from research import io as io_mod
from research import text as text_mod

SOFTWARE_TYPES = {
    "Application_Creation",
    "Application_Deposition",
    "Application_Usage",
    "Application_Mention",
    "PlugIn_Creation",
    "PlugIn_Deposition",
    "PlugIn_Usage",
    "PlugIn_Mention",
    "ProgrammingEnvironment_Usage",
    "ProgrammingEnvironment_Mention",
    "OperatingSystem_Usage",
    "OperatingSystem_Mention",
    "SoftwareCoreference_Deposition",
}

INTENT_BY_CLASS = {
    "_Usage": "used",
    "_Creation": "created",
    "_Deposition": "mentioned",
    "_Mention": "mentioned",
}

ATTR_TYPES = {
    "Version", "Developer", "URL", "License", "Citation",
    "Abbreviation", "AlternativeName", "Release",
}


def parse_brat(ann_text: str) -> tuple[list[dict], list[dict]]:
    entities = []
    relations = []
    for line in ann_text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split("\t")
        if not parts[0].startswith("T") and not parts[0].startswith("R"):
            continue
        if parts[0].startswith("R"):
            body = parts[1]
            m = re.match(r"(\w+)\s+Arg1:(T\d+)\s+Arg2:(T\d+)", body)
            if m:
                relations.append({"type": m.group(1), "arg1": m.group(2), "arg2": m.group(3)})
            continue
        m = re.match(r"(.+?)\s+(\d+)\s+(\d+)", parts[1])
        if not m:
            continue
        entities.append(
            {
                "id": parts[0],
                "type": m.group(1),
                "start": int(m.group(2)),
                "end": int(m.group(3)),
                "text": parts[2] if len(parts) > 2 else "",
            }
        )
    return entities, relations


def build_gold(entities, relations, raw_text, normalized) -> list[dict]:
    ent_by_id = {e["id"]: e for e in entities}
    vers_of: dict[str, str] = {}
    for r in relations:
        if r["type"] == "Version_of":
            vers_of[r["arg2"]] = r["arg1"]

    orig_to_norm = _orig_to_norm_mapping(normalized)
    rows = []
    for ent in entities:
        if ent["type"] not in SOFTWARE_TYPES:
            continue
        if not ent["text"].strip():
            continue
        span = _map_span(ent, raw_text, normalized, orig_to_norm)
        if span is None:
            continue
        name = normalized.text[span[0]:span[1]]
        cls = next((suf for suf in ("_Usage", "_Creation", "_Deposition", "_Mention") if ent["type"].endswith(suf)), "_Mention")
        intents = [INTENT_BY_CLASS[cls]]
        version = None
        if ent["id"] in vers_of:
            v = ent_by_id.get(vers_of[ent["id"]])
            if v and v["text"].strip():
                version = v["text"].strip()
        rows.append(
            {
                "document_id": None,
                "name": name,
                "name_span": {"start": span[0], "end": span[1]},
                "version": version,
                "intents": intents,
                "mention_id": ent["id"],
                "gold_type": ent["type"],
            }
        )
    return rows


def _orig_to_norm_mapping(normalized) -> dict[int, int]:
    """Index-map walk: built for lookups at char boundaries."""
    index_map = normalized.index_map
    n = len(normalized.text)
    out: dict[int, int] = {}
    prev_src = -1
    for i in range(n + 1):
        src = index_map[i] if i < len(index_map) else index_map[-1] + 1
        if i == 0:
            continue
        pass
    # output position i covers input range [index_map[i], index_map[i+1])
    for i in range(n):
        lo = index_map[i]
        hi = index_map[i + 1] if i + 1 < len(index_map) else lo + 1
        out.setdefault(lo, i)
    return out


def _map_span(ent, raw_text, normalized, orig_to_norm) -> tuple[int, int] | None:
    s, e = ent["start"], ent["end"]
    lo = orig_to_norm.get(s)
    hi = orig_to_norm.get(e - 1) if e > s else None
    if lo is not None and hi is not None and lo <= hi and hi + 1 <= len(normalized.text):
        cand = (lo, hi + 1)
        if normalized.text[cand[0]:cand[1]] == ent["text"]:
            return cand
    # fallback: search normalized substring
    key = re.sub(r"\s+", " ", ent["text"]).strip()
    idx = normalized.text.find(key)
    if idx >= 0:
        return (idx, idx + len(key))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--subset", default="PLoS_methods")
    ap.add_argument("--max-docs", type=int, default=10)
    ap.add_argument("--min-chars", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=35000)
    ap.add_argument("--min-mentions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    subset_dir = Path(args.label_dir) / args.subset

    candidates = []
    dropped = 0
    misc = Counter()
    for txt in sorted(subset_dir.glob("*.txt")):
        ann = txt.with_suffix(".ann")
        if not ann.exists():
            misc["no ann"] += 1
            continue
        raw = txt.read_text(encoding="utf-8", errors="replace")
        n = len(raw)
        if not (args.min_chars <= n <= args.max_chars):
            dropped += 1
            continue
        entities, relations = parse_brat(ann.read_text(encoding="utf-8", errors="replace"))
        n_soft = sum(1 for e in entities if e["type"] in SOFTWARE_TYPES)
        if n_soft < args.min_mentions:
            dropped += 1
            continue
        candidates.append((txt.stem, n, n_soft, entities, relations, raw))
        misc[f">= {args.min_mentions}"] += 1

    candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
    picked = candidates[: args.max_docs]

    docs_rows, gold_rows = [], []
    for pmid, size, n_soft, entities, relations, raw in picked:
        doc_id = f"somesci_{pmid}"
        normalized = text_mod.normalize(raw)
        g = build_gold(entities, relations, raw, normalized)
        for row in g:
            row["document_id"] = doc_id
        docs_rows.append({"document_id": doc_id, "text": raw, "metadata": {"source": f"SoMeSci:{args.subset}:{pmid}"}})
        gold_rows.extend(g)
        print(f"{doc_id}: {size} chars, {n_soft} brat mentions -> {len(g)} gold rows")

    out = root / "inputs" / "somesci"
    out.mkdir(parents=True, exist_ok=True)
    io_mod.write_jsonl(out / "somesci_docs.jsonl", docs_rows)
    io_mod.write_jsonl(out / "somesci_gold.jsonl", gold_rows)

    n_intent = Counter()
    for r in gold_rows:
        n_intent[tuple(r["intents"])] += 1
    print(f"wrote {len(docs_rows)} docs, {len(gold_rows)} gold rows -> {out}")
    print("intent histogram:", dict(n_intent))
    print("candidate pool:", len(candidates), "| skipped:", dropped, "| reasons:", dict(misc))


if __name__ == "__main__":
    main()