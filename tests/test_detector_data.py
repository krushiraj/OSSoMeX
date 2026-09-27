from copy import deepcopy

import pytest


def example():
    parent = {'document_id': 'd', 'text': 'We used NumPy.', 'text_revision': 'sha256:a',
              'split': 'train', 'source': 'ecosystems', 'public': True,
              'text_license': 'CC-BY-4.0', 'access_basis': {'article_url': 'https://example.org'}}
    task = {**parent, 'task_id': 't', 'offset_base': 0,
            'context_span': {'start': 0, 'end': 14}, 'annotation_region': {'start': 0, 'end': 14}}
    annotation = {'occurrences': [{'name': 'NumPy', 'name_span': {'start': 8, 'end': 13},
                                 'known': {'software': True, 'versions': True}, 'version_links': []}],
                  'covered_regions': [{'start': 0, 'end': 14, 'fields': {'software': True, 'versions': False}}]}
    return parent, {'task': task, 'annotation': annotation, 'status': 'reviewed'}


def test_select_preserves_partial_coverage_and_excludes_unknown_only():
    from research.training.data import select_supervision
    parent, item = example()
    unknown = deepcopy(item)
    unknown['task']['task_id'] = 'unknown'
    unknown['annotation'] = {'occurrences': [], 'covered_regions': []}
    result = select_supervision([parent], [item, unknown])
    assert result['items'] == [item]
    assert result['excluded'] == [{'task_id': 'unknown', 'reason': 'no_detector_supervision'}]
    assert result['summary']['software_spans'] == 1
    assert result['summary']['version_spans'] == 0
    assert result['items'][0]['annotation']['covered_regions'][0]['fields']['versions'] is False


@pytest.mark.parametrize('change', ['text', 'revision', 'split', 'license', 'ownership', 'duplicate'])
def test_select_rejects_invalid_or_duplicate_supervision(change):
    from research.training.data import select_supervision
    parent, item = example()
    if change == 'text': item['task']['text'] = 'wrong'
    if change == 'revision': item['task']['text_revision'] = 'wrong'
    if change == 'split': parent['split'] = 'test'
    if change == 'license': parent['text_license'] = 'unknown'
    if change == 'ownership': item['task']['annotation_region']['start'] = 10
    with pytest.raises(ValueError):
        select_supervision([parent], [item, deepcopy(item)] if change == 'duplicate' else [item])
