"""Collect software-mention text snippets from the public OpenAlex search API.

The owner pointed at the pattern:
    https://api.openalex.org/funder-search?search=CodeT5

That endpoint returns full-text search hits, each carrying a ``snippets`` array
of matching passages. We use those passages as an extra, out-of-PDF comparison
corpus for the extraction arms.

Run: .venv/bin/python scripts/collect_openalex.py
Writes data/openalex_snippets/raw/<slug>.json and inputs/openalex_snippets.jsonl.

No document text is uploaded; these are public read-only metadata queries.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "openalex_snippets" / "raw"
OUT = ROOT / "inputs" / "openalex_snippets.jsonl"
MENTIONS = ROOT / "scripts" / "software_mentions.json"

BASE = "https://api.openalex.org/funder-search"
PER_PAGE = 25
MAX_PASSAGES_PER_QUERY = 40
SLEEP_SECONDS = 0.25

_EM = re.compile(r"</?em>")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def clean(text: str) -> str:
    text = _EM.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    mentions = json.loads(MENTIONS.read_text(encoding="utf-8"))
    total = 0
    summary = []
    with OUT.open("w", encoding="utf-8") as out:
        for entry in mentions:
            name = entry["name"]
            params = {"search": name, "per_page": PER_PAGE}
            resp = requests.get(BASE, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            slug = slugify(name)
            (RAW / f"{slug}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            passages = 0
            for rank, result in enumerate(payload.get("results", [])):
                wid = result.get("id", "").rsplit("/", 1)[-1]
                for i, snippet in enumerate(result.get("snippets", []) or []):
                    text = clean(snippet)
                    if not text:
                        continue
                    doc_id = f"oa-{slug}-{wid}-{i}"
                    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    out.write(
                        json.dumps(
                            {
                                "document_id": doc_id,
                                "text": text,
                                "text_revision": f"sha256:{text_hash}",
                                "metadata": {
                                    "title": None,
                                    "doi": result.get("doi"),
                                    "language": "en",
                                    "domain": "openalex-snippet",
                                    "source": "openalex:funder-search",
                                    "extraction_method": "openalex-api",
                                    "query": name,
                                    "category": entry["category"],
                                    "work_id": result.get("id"),
                                    "rank": rank,
                                },
                                "page_spans": None,
                                "sections": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    passages += 1
                    total += 1
                    if passages >= MAX_PASSAGES_PER_QUERY:
                        break
                if passages >= MAX_PASSAGES_PER_QUERY:
                    break
            summary.append(
                {
                    "query": name,
                    "count": payload.get("meta", {}).get("count"),
                    "passages_saved": passages,
                }
            )
            time.sleep(SLEEP_SECONDS)
    (ROOT / "data" / "openalex_snippets" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"queries={len(mentions)} passages={total}")
    print(json.dumps(summary[:8], indent=2))


if __name__ == "__main__":
    main()
