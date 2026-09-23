"""Sentence-aware chunking with a token budget and whole-sentence overlap.

Protocol section 08 steps 7-9. Windows are built from whole sentences wherever
possible so every arm processes identical character spans. A long single
sentence that cannot fit is split at character boundaries and flagged; it is
never silently truncated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

from . import text as text_mod


def estimate_tokens(content: str) -> int:
    """Cheap, documented fallback token estimate (chars / 4, min 1)."""
    if not content:
        return 0
    return max(1, (len(content) + 3) // 4)


def make_tokenizer_counter(tokenizer) -> Callable[[str], int]:
    """Token counter backed by a HuggingFace fast tokenizer."""

    def counter(content: str) -> int:
        return len(tokenizer(content, add_special_tokens=False)["input_ids"])

    return counter


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text_revision: str
    start: int
    end: int
    section_type: str
    token_estimate: int
    sentence_spans: list[tuple[int, int]] = field(default_factory=list)
    split_sentence: bool = False

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text_revision": self.text_revision,
            "start": self.start,
            "end": self.end,
            "section_type": self.section_type,
            "token_estimate": self.token_estimate,
            "sentence_spans": [list(s) for s in self.sentence_spans],
            "split_sentence": self.split_sentence,
        }


def _chunk_id(document_id: str, text_revision: str, start: int, end: int, policy: str) -> str:
    raw = f"{document_id}|{text_revision}|{start}|{end}|{policy}".encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:24]


def _split_oversized(
    content: str,
    start: int,
    counter: Callable[[str], int],
    target_tokens: int,
) -> list[tuple[int, int]]:
    """Character-boundary fallback for a sentence that exceeds the budget."""
    spans: list[tuple[int, int]] = []
    approx_chars = max(1, target_tokens * 4)
    pos = 0
    while pos < len(content):
        piece = content[pos:pos + approx_chars]
        if counter(piece) <= target_tokens or pos + approx_chars >= len(content):
            spans.append((start + pos, start + pos + len(piece)))
            pos += len(piece)
        else:
            cut = len(piece)
            while cut > 1 and counter(piece[:cut]) > target_tokens:
                cut -= max(1, cut // 10)
            cut = max(1, cut)
            spans.append((start + pos, start + pos + cut))
            pos += cut
    return [s for s in spans if s[1] > s[0]]


def _make_chunk(
    document_id: str,
    text_revision: str,
    normalized: text_mod.NormalizedText,
    sections: list[dict],
    start: int,
    end: int,
    counter: Callable[[str], int],
    policy_version: str,
    sentence_spans: list[tuple[int, int]],
    split_sentence: bool,
) -> Chunk:
    content = normalized.text[start:end]
    return Chunk(
        chunk_id=_chunk_id(document_id, text_revision, start, end, policy_version),
        document_id=document_id,
        text_revision=text_revision,
        start=start,
        end=end,
        section_type=text_mod.section_at(sections, start),
        token_estimate=counter(content),
        sentence_spans=sentence_spans,
        split_sentence=split_sentence,
    )


def make_chunks(
    document_id: str,
    text_revision: str,
    normalized: text_mod.NormalizedText,
    sections: list[dict],
    target_content_tokens: int = 480,
    overlap_tokens: int = 64,
    policy_version: str = "1.0",
    counter: Callable[[str], int] = estimate_tokens,
) -> list[Chunk]:
    content = normalized.text
    sentences = text_mod.segment_sentences(content)
    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        start = sentences[i][0]
        end = start
        used: list[tuple[int, int]] = []
        j = i
        split = False
        while j < len(sentences):
            s_start, s_end = sentences[j]
            if counter(content[start:s_end]) > target_content_tokens:
                if used:
                    break
                for piece in _split_oversized(content[s_start:s_end], s_start, counter, target_content_tokens):
                    chunks.append(
                        _make_chunk(document_id, text_revision, normalized, sections,
                                    piece[0], piece[1], counter, policy_version, [piece], True)
                    )
                j += 1
                split = True
                break
            used.append((s_start, s_end))
            end = s_end
            j += 1
        if used:
            chunks.append(
                _make_chunk(document_id, text_revision, normalized, sections,
                            start, end, counter, policy_version, used, split)
            )
        if j > i:
            i = _overlap_start(sentences, i, j, overlap_tokens, counter, content)
        else:
            i += 1

    if not chunks:
        chunks.append(
            _make_chunk(document_id, text_revision, normalized, sections, 0, len(content),
                        counter, policy_version, [(0, len(content))] if content else [], False)
        )
    return chunks


def _overlap_start(
    sentences: list[tuple[int, int]],
    first_index: int,
    next_index: int,
    overlap_tokens: int,
    counter: Callable[[str], int],
    content: str,
) -> int:
    """Rewind the next window start to include ~overlap_tokens of trailing text."""
    if overlap_tokens <= 0 or next_index <= first_index:
        return next_index
    start = next_index
    for k in range(next_index - 1, first_index, -1):
        if counter(content[sentences[k][0]:sentences[next_index - 1][1]]) > overlap_tokens:
            break
        start = k
    if start <= first_index or start >= next_index:
        return next_index
    return start


def coverage(chunks: list[Chunk], length: int) -> float:
    if length <= 0:
        return 1.0
    covered = [False] * length
    for chunk in chunks:
        for pos in range(max(0, chunk.start), min(length, chunk.end)):
            covered[pos] = True
    return sum(covered) / length
