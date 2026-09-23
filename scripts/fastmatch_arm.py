"""Fast dictionary / text-match arm (simulates WP4 'fast text matching').

Builds a mention dictionary from external sources (JSONL of names + optional
curated list + CZI frequency file), then applies longest-match substring
scanning over a given split's docs. Emits predictions.jsonl in the pipeline run
format so scripts/gold_report.py can score it like any other arm.

Dictionary sources are recorded in run.json so the trial stays reproducible —
never derive the dictionary from the gold you are scoring it against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "manifests" / "chunks.jsonl"
PREPARED = ROOT / "manifests" / "prepared.jsonl"


def norm_key(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def build_dictionary(name_files: list[Path], curated: Path | None, czi_freq: Path | None) -> dict[str, str]:
    dict_: dict[str, str] = {}
    for nf in name_files:
        for line in nf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                name = json.loads(line).get("name") or line
            except json.JSONDecodeError:
                name = line
            nk = norm_key(name)
            if 2 <= len(nk) <= 48:
                dict_[nk] = name
    if curated:
        for line in curated.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            nk = norm_key(line)
            if 2 <= len(nk) <= 48:
                dict_[nk] = line
    if czi_freq:
        for line in czi_freq.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            try:
                name = line.split("\t", 1)[1]
            except ValueError:
                continue
            nk = norm_key(name)
            if 2 <= len(nk) <= 48:
                dict_[nk] = name.strip()
    return dict_


def longest_match_spans(folded: str, dict_: dict[str, str]) -> list[tuple[int, int, str]]:
    positions: list[tuple[int, int, str]] = []
    i = 0
    n = len(folded)
    while i < n:
        best = None
        for end in range(min(n, i + 48), i, -1):
            word = folded[i:end]
            nk = word.replace(" ", "")
            if nk in dict_:
                best = (i, end, dict_[nk])
                break
        if best:
            start, end, name = best
            s0 = start
            while s0 < end and not folded[s0].isalnum():
                s0 += 1
            e0 = end
            while e0 > s0 and not folded[e0 - 1].isalnum():
                e0 -= 1
            if e0 - s0 >= 2:
                positions.append((s0, e0, name))
            i = end
        else:
            i += 1
    return positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dict", action="append", default=[], help="jsonl/text file of dictionary names (>=1)")
    ap.add_argument("--curated", default=None)
    ap.add_argument("--czi-freq", default=None)
    args = ap.parse_args()

    src = [Path(p) for p in args.dict]
    dict_ = build_dictionary(src, Path(args.curated) if args.curated else None,
                             Path(args.czi_freq) if args.czi_freq else None)
    print(f"dictionary size: {len(dict_)} from {[p.name for p in src]}{' + curated' if args.curated else ''}")

    splits = json.loads((ROOT / "manifests" / "splits.json").read_text(encoding="utf-8"))
    doc_ids = set(splits.get(args.split, []))
    prep_dir = ROOT / "manifests" / "prepared"
    present = [d for d in doc_ids if (prep_dir / f"{d}.json").exists()]
    print(f"split {args.split}: {len(doc_ids)} docs, prepared present: {len(present)}")

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for doc_id in sorted(present):
        norm = json.loads((prep_dir / f"{doc_id}.json").read_text(encoding="utf-8"))["normalized_text"]
        folded = norm.lower()
        for start, end, name in longest_match_spans(folded, dict_):
            if end - start < 2:
                continue
            predictions.append({
                "document_id": doc_id,
                "name": name,
                "name_span": {"start": start, "end": end},
                "version": None,
                "intents": ["used"],
                "sentiment": None,
                "mention_id": f"fd-{doc_id}-{start}-{end}",
            })

    run_dir.joinpath("run.json").write_text(json.dumps({
        "run_id": args.run_id, "status": "completed", "arm": "fastdict",
        "model_id": "dict-longest-match",
        "feature_support": {"name": True, "version": False, "intents": True, "sentiment": False},
        "split": args.split, "predictions": len(predictions),
        "dictionary_files": [p.name for p in src],
        "curated_file": args.curated,
        "dictionary_size": len(dict_),
    }, indent=2), encoding="utf-8")

    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(run_dir / "evidence.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps({"document_id": p["document_id"], "name_span": p["name_span"], "name": p["name"]}, ensure_ascii=False) + "\n")
    print(f"{args.run_id}: {len(predictions)} predictions across {len(present)} docs")


if __name__ == "__main__":
    main()