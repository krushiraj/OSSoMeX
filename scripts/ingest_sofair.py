"""Ingest SoFAIR dataset (ORDO 30374830, file 59502416) TEI annotations.

Extracts plain text + <rs type="software"> spans (+ <rs type="version">
corresp-linked) into the workspace shape used by ingest_somesci:
  - inputs/sofair/sofair_docs.jsonl
  - inputs/sofair/sofair_gold.jsonl  (name_span in normalized coords)
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from research import io as io_mod
from research import text as text_mod

TEI_NS = "{http://www.tei-c.org/ns/1.0}"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inputs" / "sofair"

PICK = [
    ("SoFAIR_AD_Environmental_Science_papers", "479167285"),
    ("SoFAIR_AD_Digital_Humanties_papers", "DH47"),
    ("SoFAIR_AD_Mathematics_papers", "566501142"),
    ("SoFAIR_AD_Physics_papers", "589952941"),
    ("SoFAIR_AD_Comp_Sci_papers", "523405194"),
    ("SoFAIR_Cultural_Studies_papers", "481727773"),
]


def extract_tei(xml_bytes: bytes):
    tree = ET.fromstring(xml_bytes)
    text_el = tree.find(f"{TEI_NS}text")
    if text_el is None:
        return None
    parts: list[str] = []
    spans: list[dict] = []  # {id,type,start,end}
    offset = 0

    def push(s: str) -> None:
        nonlocal offset
        parts.append(s)
        offset += len(s)

    def walk(el) -> None:
        nonlocal offset
        if el.text:
            push(el.text)
        tail_after = None
        for child in el:
            tag = child.tag
            if isinstance(tag, str):
                local = tag.split("}")[-1]
            else:
                local = str(child)  # comment/pi
            if local == "rs":
                attrs = child.attrib
                rtype = attrs.get(TEI_NS + "type") or attrs.get("type")
                rid = attrs.get(TEI_NS + "id") or attrs.get("id")
                if rtype in ("software", "version"):
                    start = offset
                    inner = []
                    inner_off = [offset]

                    def push_inner(s: str) -> None:
                        inner.append(s)
                        inner_off[0] += len(s)

                    def walk_inner(e) -> None:
                        if e.text:
                            push_inner(e.text)
                        for g in e:
                            walk_inner(g)
                            if g.tail:
                                push_inner(g.tail)

                    walk_inner(child)
                    s = "".join(inner)
                    push(s)
                    spans.append({"id": rid, "type": rtype, "start": start, "end": offset})
                else:
                    walk(child)
                    if child.tail:
                        push(child.tail)
                if child.tail:
                    push(child.tail)
            else:
                walk(child)
                if child.tail:
                    push(child.tail)
        if el.tail and el != text_el:
            push(el.tail)

    walk(text_el)
    return "".join(parts), spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sofair-dir", required=True)
    ap.add_argument("--pick-all", action="store_true", help="ingest every doc (overrides PICK)")
    args = ap.parse_args()
    base = Path(args.sofair_dir)

    files = []
    if args.pick_all:
        files = sorted(base.rglob("*.xml"))
    else:
        for disc, ident in PICK:
            cand = list((base / "tei-annotated-deduplicated" / disc).glob(f"{ident}*tei.xml"))
            if not cand:
                cand = list((base / disc).glob(f"{ident}*tei.xml"))
            if not cand:
                raise SystemExit(f"missing {disc}/{ident}")
            files.append(cand[0])

    docs_rows, gold_rows = [], []
    stats = Counter()
    for fn in files:
        disc = fn.parent.name
        ident = fn.name.split(".")[0]
        try:
            raw, spans = extract_tei(fn.read_bytes())
        except ET.ParseError as exc:
            print("parse err", fn.name, exc)
            continue
        if raw is None or not raw.strip():
            continue
        doc_id = f"sofair_{ident}"
        normalized = text_mod.normalize(raw)
        n_sw = sum(1 for s in spans if s["type"] == "software")

        # version map: id -> version text span
        vers_by_id = {}
        for s in spans:
            if s["type"] == "version" and s.get("id"):
                vers_by_id[s["id"]] = raw[s["start"]:s["end"]]
        # corresp linking: version rs has corresp=software id
        corresp = {}
        for s in spans:
            if s["type"] == "version":
                corresp.setdefault((s["start"], s["end"]), None)
        # gather software→version via corresp attr
        vers_for = {}
        # re-walk rs nodes for corresp
        tree = ET.fromstring(fn.read_bytes())
        for el in tree.iter():
            if el.tag == TEI_NS + "rs":
                attrs = el.attrib
                at = attrs.get("type")
                if at == "version":
                    c = attrs.get("corresp") or attrs.get(TEI_NS + "corresp")
                    v = (el.text or "").strip()
                    if c and v:
                        vers_for.setdefault(c, v)

        for s in spans:
            if s["type"] != "software":
                continue
            name = raw[s["start"]:s["end"]].strip()
            if not name:
                continue
            span = _map_span(s["start"], s["end"], raw, normalized)
            if span is None:
                continue
            gname = normalized.text[span[0]:span[1]]
            version = vers_for.get(s["id"]) if s.get("id") else None
            gold_rows.append({
                "document_id": doc_id,
                "name": gname,
                "name_span": {"start": span[0], "end": span[1]},
                "version": version,
                "intents": [],
                "mention_id": s["id"],
                "gold_type": "software",
            })
        docs_rows.append({"document_id": doc_id, "text": raw,
                          "metadata": {"source": f"SoFAIR:{disc}:{ident}"}})
        stats[disc] += 1
        print(f"{doc_id:24s} {len(raw):>7d} chars  {n_sw} soft tags -> {sum(1 for g in gold_rows if g['document_id']==doc_id)} gold")

    OUT.mkdir(parents=True, exist_ok=True)
    io_mod.write_jsonl(OUT / "sofair_docs.jsonl", docs_rows)
    io_mod.write_jsonl(OUT / "sofair_gold.jsonl", gold_rows)
    print(f"wrote {len(docs_rows)} docs, {len(gold_rows)} gold rows -> {OUT}")
    print("docs per discipline:", dict(stats))


def _map_span(s: int, e: int, raw: str, normalized) -> tuple[int, int] | None:
    # reconstruct orig->norm mapping like ingest_somesci
    index_map = normalized.index_map
    orig_to_norm = {}
    for i in range(len(normalized.text)):
        lo = index_map[i]
        orig_to_norm.setdefault(lo, i)
    lo = orig_to_norm.get(s)
    hi = orig_to_norm.get(e - 1) if e > s else None
    if lo is not None and hi is not None and lo <= hi and hi + 1 <= len(normalized.text):
        cand = (lo, hi + 1)
        if normalized.text[cand[0]:cand[1]] == raw[s:e]:
            return cand
    key = re.sub(r"\s+", " ", raw[s:e]).strip()
    idx = normalized.text.find(key)
    if idx >= 0:
        return (idx, idx + len(key))
    return None


if __name__ == "__main__":
    main()