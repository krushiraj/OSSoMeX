from research import chunking as c
from research import text as t


def _chunks(text):
    n = t.normalize(text)
    sections = t.detect_sections(n.text)
    return c.make_chunks("doc", "rev", n, sections, target_content_tokens=40, overlap_tokens=8)


def test_all_chars_covered():
    text = "Sentence one is here. Sentence two is here. " * 20
    chunks = _chunks(text)
    total = len(text)
    covered = [False] * total
    for chunk in chunks:
        for pos in range(chunk.start, chunk.end):
            covered[pos] = True
    assert all(covered), "some source characters were not covered"


def test_never_silently_truncated():
    text = "This is a very long sentence that cannot possibly fit in the target budget at all. " * 10
    chunks = _chunks(text)
    lengths = [chunk.end - chunk.start for chunk in chunks]
    assert sum(lengths) >= len(text)


def test_chunk_ids_deterministic():
    a = _chunks("The quick brown fox. Jumps over the dog.")
    b = _chunks("The quick brown fox. Jumps over the dog.")
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]