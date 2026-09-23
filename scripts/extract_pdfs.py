"""Extract full text from the reference PDFs into the D0 development inputs.

Run: .venv/bin/python scripts/extract_pdfs.py
Writes inputs/raw/<document_id>.txt and inputs/reference_docs.jsonl.

This is an intake step only. It performs no semantic interpretation and applies
no software-keyword filtering; every page's text is retained verbatim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
RAW = INPUTS / "raw"

# document_id -> (path, sha256 from protocol section 03, download source)
SOURCES = {
    "ref-codet5plus": (
        "/Users/krushi/Downloads/2305.07922v2.pdf",
        "f4ac864b2905cd9068e508796d55fb2b433f68116bebbefa603f3a8b813d3804",
    ),
    "ref-napari-imagej": (
        "/Users/krushi/Downloads/s41592-023-01990-0.pdf",
        "67d66308358ef8bc8f046dc19e9a7e5bd440c4f9cb19556dd2dde2258d6cb87b",
    ),
    "ref-cell-segmentation": (
        "/Users/krushi/Downloads/s41598-025-01763-z.pdf",
        "a9bca88bd0190b8de0abc4e867ce945a74fa98b2b9b2425d2763d3313663fafb",
    ),
    "ref-mobile": (
        "/Users/krushi/Downloads/s41467-023-39729-2.pdf",
        "325ad95cdf67718bc4d4a2ec22ef018222ddfdfb2fd8d1cfc17848a40c8d54a9",
    ),
    "ref-softcite-two": (
        "/Users/krushi/work/softcite_pdfs/softcite_corpus/latex/softcite_two.pdf",
        "74bfbe0d18e01b0688969144aae9be1dc78bafce7e61128e92786b6d157118d3",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(document_id: str, pdf_path: Path, expected_hash: str) -> dict:
    actual = sha256_file(pdf_path)
    if actual != expected_hash:
        raise SystemExit(
            f"hash mismatch for {document_id}: expected {expected_hash}, got {actual}"
        )
    doc = pymupdf.open(pdf_path)
    pieces: list[str] = []
    page_spans: list[dict] = []
    title = None
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text")
        if title is None:
            first = text.strip().splitlines()
            title = first[0].strip() if first else None
        start = sum(len(p) for p in pieces)
        pieces.append(text)
        end = start + len(text)
        page_spans.append({"page": page_index + 1, "start": start, "end": end})
        pieces.append("\n\n")
    doc.close()
    # drop the trailing separator contributed after the last page
    full = "".join(pieces)
    if page_spans:
        page_spans[-1]["end"] = min(page_spans[-1]["end"], len(full))
    text_hash = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return {
        "document_id": document_id,
        "text_revision": f"sha256:{text_hash}",
        "text": full,
        "page_spans": page_spans,
        "sections": None,
        "metadata": {
            "title": title,
            "doi": None,
            "language": "en",
            "domain": None,
            "source": f"reference-pdf:{pdf_path.name}",
            "extraction_method": f"pymupdf-{pymupdf.__version__}",
            "pdf_sha256": actual,
            "page_count": len(page_spans),
        },
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = INPUTS / "reference_docs.jsonl"
    summary = []
    with out_path.open("w", encoding="utf-8") as out:
        for document_id, (path_str, expected_hash) in SOURCES.items():
            path = Path(path_str)
            if not path.exists():
                raise SystemExit(f"missing reference PDF: {path}")
            record = extract(document_id, path, expected_hash)
            (RAW / f"{document_id}.txt").write_text(record["text"], encoding="utf-8")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary.append(
                {
                    "document_id": document_id,
                    "pages": record["metadata"]["page_count"],
                    "chars": len(record["text"]),
                    "text_revision": record["text_revision"],
                }
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
