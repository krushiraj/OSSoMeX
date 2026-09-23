"""Create a bounded development subset of the OpenAlex snippet corpus.

Running every arm over all snippets is expensive; the pilot uses the first
N passages per software-mention query. The full file is retained for later runs.

Run: .venv/bin/python scripts/subsample_openalex.py --per-query 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "inputs" / "openalex_snippets.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--output", default="inputs/openalex_snippets_dev.jsonl")
    args = parser.parse_args()

    counts: Counter = Counter()
    kept = []
    with SRC.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            query = row["metadata"]["query"]
            if counts[query] < args.per_query:
                counts[query] += 1
                kept.append(row)
    out = ROOT / args.output
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(kept)} passages across {len(counts)} queries -> {out}")


if __name__ == "__main__":
    main()
