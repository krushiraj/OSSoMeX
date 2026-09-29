import json
from pathlib import Path

import pytest


def test_fixture_freezes_exact_offsets_links_masks_and_no_human_gold(tmp_path):
    from research.comparison.diagnostic_fixture import freeze_fixture
    source = Path(__file__).parents[1] / 'examples/scibert-cli-diagnostic-001.json'
    result = freeze_fixture(source, tmp_path / 'data')
    assert result['documents'] == 7
    docs = [json.loads(line) for line in (tmp_path / 'data/inputs.jsonl').read_text().splitlines()]
    refs = [json.loads(line) for line in (tmp_path / 'data/references.jsonl').read_text().splitlines()]
    env = next(ref for ref in refs if ref['document_id'].endswith(':environment'))
    assert len(env['version_links']) == 6
    assert [v['text'] for v in env['ignored_versions']] == ['3.9.7', '3.11.4']
    assert len(next(r for r in refs if r['document_id'].endswith(':imagej'))['spans']) == 11
    for doc, ref in zip(docs, refs):
        assert doc['role'] == 'diagnostic'
        assert ref['provenance']['review_kind'] == 'agent_provisional'
        assert ref['provenance']['human_reviewed'] is False
        for span in ref['spans']:
            assert doc['text'][span['start']:span['end']] == span['text']
    assert 'human_reviewed' not in {r['review_kind'] for ref in refs for r in ref['coverage']}
    with pytest.raises(FileExistsError): freeze_fixture(source, tmp_path / 'data')


def test_rejects_unknown_or_duplicate_annotation_endpoints(tmp_path):
    from research.comparison.diagnostic_fixture import parse_annotated
    with pytest.raises(ValueError): parse_annotated('[[s1|Python]] [[s1|R]]')
    with pytest.raises(ValueError): parse_annotated('[[x1|Python]]')
    assert parse_annotated('[[s1|Python]]3')[0] == 'Python3'
