"""Text intake: reversible normalization, sentence segmentation, sections.

Protocol sections 02 and 08. Normalization is deliberately conservative: it
standardizes line endings and horizontal whitespace only, and keeps a
per-character map back to the original code points so every span can be
reported against the source text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SECTION_PATTERNS = [
    ("abstract", r"^abstract\b"),
    ("introduction", r"^(1\.?\s*)?introduction\b"),
    ("methods", r"^(materials and methods|methods|methodology|method|2\.?\s*methods)\b"),
    ("results", r"^(results|results and discussion)\b"),
    ("discussion", r"^(discussion|conclusions? and discussion)\b"),
    ("conclusion", r"^(conclusions?)\b"),
    ("references", r"^(references|bibliography|literature cited)\b"),
    ("acknowledgements", r"^(acknowledg(e)?ments?)\b"),
    ("availability", r"^(data|code|software) (availability|and .*availability)\b"),
    ("supplementary", r"^(supplementary|supporting information)\b"),
    ("funding", r"^(funding|author contributions)\b"),
]

_ABBREVIATIONS = {
    "e.g", "i.e", "et al", "vs", "cf", "fig", "figs", "eq", "eqs", "sec",
    "ref", "refs", "no", "approx", "ca", "dr", "prof", "mr", "ms", "mrs",
    "inc", "ltd", "co", "st", "etc", "v", "ver", "vol", "ch", "pp",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_URL_OR_DOTTED = re.compile(r"(https?://\S+|\b\d+(?:\.\d+)+\b|\b[A-Za-z]\.(?=[A-Za-z]))")


@dataclass
class NormalizedText:
    text: str
    original: str
    index_map: list[int]
    changes: list[dict] = field(default_factory=list)

    def to_source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open normalized span back to original code points."""
        if start >= len(self.index_map):
            start = len(self.index_map) - 1
        if end - 1 >= len(self.index_map):
            end = len(self.index_map)
        if start < 0 or end <= start:
            return (0, 0)
        return (self.index_map[start], self.index_map[end - 1] + 1)


def normalize(text: str) -> NormalizedText:
    out: list[str] = []
    index_map: list[int] = []
    changes: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\r":
            start = i
            if i + 1 < n and text[i + 1] == "\n":
                i += 2
            else:
                i += 1
            out.append("\n")
            index_map.append(start)
            changes.append({"kind": "line_ending", "src": [start, i]})
            continue
        if ch in " \t":
            start = i
            while i < n and text[i] in " \t":
                i += 1
            out.append(" ")
            index_map.append(start)
            if i - start > 1:
                changes.append({"kind": "whitespace", "src": [start, i]})
            continue
        if ch == "\u00a0":
            out.append(" ")
            index_map.append(i)
            changes.append({"kind": "nbsp", "src": [i, i + 1]})
            i += 1
            continue
        if ch == "\n":
            start = i
            run = 0
            while i < n and text[i] == "\n":
                run += 1
                i += 1
            kept = "\n" if run < 3 else "\n\n"
            for _ in kept:
                out.append("\n")
                index_map.append(start)
            if run >= 3:
                changes.append({"kind": "newline_run", "src": [start, i]})
            continue
        out.append(ch)
        index_map.append(i)
        i += 1
    return NormalizedText(text="".join(out), original=text, index_map=index_map, changes=changes)


def segment_sentences(text: str) -> list[tuple[int, int]]:
    """Return half-open (start, end) spans of sentences in normalized text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    protected = [(m.start(), m.end()) for m in _URL_OR_DOTTED.finditer(text)]

    def inside_protected(pos: int) -> bool:
        return any(a <= pos < b for a, b in protected)

    for match in _SENTENCE_END.finditer(text):
        end = match.start() + 1
        boundary = match.start()
        if inside_protected(boundary) or inside_protected(end - 1):
            continue
        window = text[max(0, boundary - 12):boundary].strip().lower()
        if window.rsplit(" ", 1)[-1].rstrip(".") in _ABBREVIATIONS:
            continue
        if re.search(r"\b[A-Z]$", text[max(0, boundary - 2):boundary]):
            continue
        spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    if not spans:
        spans = [(0, len(text)) if text else (0, 0)]
    return spans


def detect_sections(text: str) -> list[dict]:
    """Deterministic heading-based section map; provenance is recorded."""
    sections: list[dict] = []
    offset = 0
    current = {"type": "unknown", "start": 0}
    for line in text.split("\n"):
        stripped = line.strip().lower()
        heading = stripped[:80]
        for section_type, pattern in SECTION_PATTERNS:
            if re.match(pattern, heading):
                if offset > current["start"]:
                    sections.append({**current, "end": offset})
                current = {"type": section_type, "start": offset}
                break
        offset += len(line) + 1
    sections.append({**current, "end": len(text)})
    return [
        s for s in sections if s["end"] > s["start"] and s["type"] != "unknown"
    ] or [{"type": "unknown", "start": 0, "end": len(text)}]


def section_at(sections: list[dict], offset: int) -> str:
    for section in sections:
        if section["start"] <= offset < section["end"]:
            return section["type"]
    return "unknown"
