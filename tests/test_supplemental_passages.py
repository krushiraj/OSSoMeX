import importlib

import pytest

from research.contracts import text_revision


def parsed(paragraphs):
    text = '\n\n'.join(paragraphs)
    spans, offset = [], 0
    for paragraph in paragraphs:
        spans.append({'start': offset, 'end': offset + len(paragraph), 'kind': 'p'})
        offset += len(paragraph) + 2
    return {'text': text, 'paragraphs': spans, 'document_id': 'fixture', 'text_revision': text_revision(text)}


def select(doc, **kwargs):
    return importlib.import_module('research.data.supplemental_passages').select_passages(doc, **kwargs)


def test_zero_name_hits_still_supplies_random_whole_passages():
    doc = parsed(['Alpha prose.', 'Beta prose.', 'Gamma prose.', 'Delta prose.'])
    rows = select(doc, aliases=['ImageJ'])
    assert len(rows) == 3
    assert all(r['selection_reason'] == 'random_whole_passage' and r['whole_passage_audit'] is True for r in rows)
    assert [r['annotation_region']['start'] for r in rows] == [0, 41, 14]


def test_random_draw_ignores_alias_matches_then_signals_rank_versions_first():
    doc = parsed(['No target.', 'ImageJ used.', 'ImageJ version 1.2 used.', 'More text.',
                  'ImageJ 2.0 used.', 'Other prose.', 'ImageJ described.', 'Last prose.'])
    rows = select(doc, aliases=['ImageJ'])
    assert [r['text'] for r in rows[:3]] == ['ImageJ used.', 'No target.', 'Other prose.']
    assert [r['text'] for r in rows[3:]] == ['ImageJ version 1.2 used.', 'ImageJ 2.0 used.', 'ImageJ described.']
    assert [r['selection_reason'] for r in rows[3:]] == ['literal_name_version_candidate', 'literal_name_version_candidate', 'literal_name_candidate']
    assert select(doc, aliases=['absent'])[:3] == rows[:3]


def test_source_slices_and_disjoint_ownership_are_exact():
    doc = parsed(['Unicode α sentence. ImageJ 1.2 used. A final sentence.', 'ImageJ 2.3 comparison.'])
    rows = select(doc, aliases=['ImageJ'])
    owned = []
    for row in rows:
        start, end = row['annotation_region'].values()
        cs, ce = row['context_span'].values()
        assert row['document_id'] == 'fixture' and row['text_revision'] == doc['text_revision']
        assert row['region_kind'] == 'sentence' and row['text'] == doc['text'][start:end]
        assert row['context_text'] == doc['text'][cs:ce] and cs <= start < end <= ce
        assert all(end <= os or start >= oe for os, oe in owned)
        owned.append((start, end))


def test_short_passage_supply_and_oversized_context_have_no_fabricated_rows():
    doc = parsed(['Only eligible.', 'x' * 6001 + '.'])
    rows = select(doc, aliases=['ImageJ'])
    assert len(rows) == 1 and rows[0]['text'] == 'Only eligible.'
    module = importlib.import_module('research.data.supplemental_passages')
    _, issues = module.passage_candidates(doc)
    assert issues[0]['code'] == 'CONTEXT_EXCEEDS_LIMIT'


@pytest.mark.parametrize('kwargs', [{'random_count': 4}, {'signal_count': 4}, {'seed': 7}, {'random_count': True}])
def test_passage_bounds_cannot_expand_or_change_frozen_seed(kwargs):
    with pytest.raises(ValueError):
        select(parsed(['Text.']), aliases=['R'], **kwargs)


def test_integer_and_release_versions_are_candidate_signals_not_labels():
    doc = parsed(['SPSS discussion.', 'SPSS 28 analysis.', 'MATLAB R2024b analysis.'])
    rows = select(doc, aliases=['SPSS', 'MATLAB'], random_count=0)
    assert [row['text'] for row in rows] == ['SPSS 28 analysis.', 'MATLAB R2024b analysis.', 'SPSS discussion.']
    assert all('version_links' not in row and 'known' not in row for row in rows)


def test_corrupt_revision_or_overlapping_paragraphs_rejected():
    doc = parsed(['First prose.', 'Other prose.'])
    doc['text_revision'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError, match='STALE_PASSAGE_TEXT'):
        select(doc, aliases=[])
    doc.pop('text_revision')
    doc['paragraphs'][1]['start'] = 0
    with pytest.raises(ValueError, match='INVALID_PARAGRAPH_SPAN'):
        select(doc, aliases=[])
