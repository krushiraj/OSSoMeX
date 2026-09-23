from research import text as t


def test_normalize_maps_back():
    original = "line1\r\nline2\t tab\u00a0x\n\n\n\ny"
    n = t.normalize(original)
    # "\r\n" collapses to "\n" so normalized "line2" sits at chars 6..11
    o_start, o_end = n.to_source_span(6, 11)
    assert original[o_start:o_end] == "line2"
    assert n.text == "line1\nline2 tab x\n\ny"


def test_normalize_maps_whitespace_run():
    original = "a\t\t b"
    n = t.normalize(original)
    assert n.text == "a b"
    o_start, o_end = n.to_source_span(1, 2)
    assert original[o_start] in " \t"


def test_dotted_version_not_split():
    text = "We used Python 3.10 and NumPy 1.24 and R 4.3"
    spans = t.segment_sentences(text)
    assert len(spans) == 1


def test_abbreviation_not_split():
    text = "We used R and e.g. SAS. Next sentence."
    spans = t.segment_sentences(text)
    assert len(spans) == 2


def test_sections_detected():
    text = "Abstract\nWe present X.\nReferences\n[1] A."
    sections = t.detect_sections(text)
    types = [s["type"] for s in sections]
    assert "abstract" in types and "references" in types