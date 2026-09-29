from copy import deepcopy
from html.parser import HTMLParser

import pytest


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.links = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == 'a':
            self.links.append(dict(attrs).get('href'))

    def handle_data(self, data):
        self.text.append(data)


def sample_data():
    source = '🧪 <tool> 3.9.7'
    software = {'label': 'SOFTWARE', 'start': 2, 'end': 8, 'text': '<tool>'}
    version = {'label': 'VERSION', 'start': 9, 'end': 14, 'text': '3.9.7'}
    edge = {'software': {key: software[key] for key in ('start', 'end', 'text')},
            'version': {key: version[key] for key in ('start', 'end', 'text')}}
    metric = {'tp': 0, 'fp': 1, 'fn': 1, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return {
        'title': 'Comparison <script>alert(1)</script>',
        'provenance': {'run': 'sha256:abc', 'review': 'Agent provisional'},
        'limitations': ['No human gold; exposed diagnostic.'],
        'models': [
            {'arm_id': 'scibert-base', 'kind': 'base', 'status': 'unsupported',
             'identity': {'reason': 'no_task_head'}, 'span_source_arm': None,
             'metrics': {'software': None, 'version': None, 'links': None},
             'excluded_predictions': None, 'operations': {}, 'timing': None,
             'notes': ['No extraction head']},
            {'arm_id': 'scibert-full-label-003', 'kind': 'pipeline', 'status': 'ready',
             'identity': {'checkpoint_sha256': 'sha256:def'},
             'span_source_arm': 'scibert-detector-003',
             'metrics': {'software': metric, 'version': metric, 'links': metric},
             'excluded_predictions': 2, 'operations': {'completed_documents': 1},
             'timing': {'measured_seconds': 4.5},
             'notes': ['Link score uses masked references']},
        ],
        'documents': [{
            'document_id': 'passage-1', 'text': source, 'text_revision': 'sha256:xyz',
            'reference_spans': [software, version], 'reference_links': [edge],
            'coverage': [{'label': label, 'start': 0, 'end': len(source),
                          'review_kind': 'agent_provisional', 'complete': True}
                         for label in ('SOFTWARE', 'VERSION')],
            'link_coverage': [{'start': 0, 'end': len(source),
                               'review_kind': 'agent_provisional', 'complete': True}],
            'ignored_versions': [version], 'ignored_software': [],
            'arms': [
                {'arm_id': 'scibert-base', 'status': 'unsupported', 'spans': [],
                 'version_links': [], 'link_details': None, 'fields': [],
                 'alias_groups': [], 'notes': ['no_task_head']},
                {'arm_id': 'scibert-full-label-003', 'status': 'success',
                 'spans': [software, version], 'version_links': [edge],
                 'link_details': {'false_positive_edges': [[2, 8, 9, 14]],
                                  'missed_edges': [], 'excluded_predictions': 1,
                                  'excluded_gold_edges': 0,
                                  'operational': metric},
                 'fields': [
                     {'mention_id': 'm1', 'name': '<tool>',
                      'intents': {'status': 'success', 'value': ['used']},
                      'versions': {'status': 'success', 'value': [{'text': '3.9.7'}]},
                      'sentiment': {'status': 'success', 'value': 'not_expressed'}},
                     {'mention_id': 'm2', 'name': '🧪',
                      'intents': {'status': 'success', 'value': ['mentioned']},
                      'versions': {'status': 'success', 'value': []},
                      'sentiment': {'status': 'failure', 'value': None}},
                 ],
                 'alias_groups': [{'member_mention_ids': ['m1', 'm2']}],
                 'notes': ['Ownership uncertain']},
            ],
            'summaries': ['One masked version needs review.'],
        }],
        'section_summaries': {'software': ['Exact names are sparse.'],
                              'version': ['Version spans align.'],
                              'links': ['Masked predictions are excluded.']},
        'feedback': [{'title': 'Later CLI feedback', 'body': 'macOS → 3.9.7',
                      'provenance': 'Unscored user observation'}],
        'appendix': {'title': 'Base model probe', 'summary': 'Instructional examples only',
                     'sources': [{'title': 'Paper', 'url': 'https://example.com/paper'}],
                     'examples': [{'title': 'Probe', 'kind': 'instructional',
                                   'input': '<script>bad()</script>',
                                   'output': {'answer': '<unsafe>'},
                                   'note': 'Not an extraction score',
                                   'source_url': 'javascript:alert(1)'}],
                     'evidence': {'base': 'no task head'}},
    }


def render(data):
    from research.comparison.html_render import render_dashboard
    return render_dashboard(data)


def test_escapes_source_and_user_text_but_marks_unicode_offsets():
    page = render(sample_data())
    parsed = PageParser()
    parsed.feed(page)
    assert '<script>' not in page
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in page
    assert '🧪 <mark class="span software">&lt;tool&gt;</mark>' in page
    assert '<mark class="span version">3.9.7</mark>' in page
    assert ('script', {}) not in parsed.tags
    assert all(not href.startswith('javascript:') for href in parsed.links if href)


def test_null_metric_is_na_and_zero_metric_is_zero_with_counts():
    page = render(sample_data())
    assert 'N/A' in page
    assert '0.0%' in page
    assert '0 / 1 / 1' in page
    assert 'Exact names are sparse.' in page
    assert 'Masked predictions are excluded.' in page
    assert 'scibert-detector-003' in page


def test_passage_shows_reference_masks_model_error_details_and_unscored_feedback():
    page = render(sample_data())
    parsed = PageParser()
    parsed.feed(page)
    visible = ' '.join(parsed.text)
    assert 'passage-1' in visible
    assert 'Ignored versions' in visible
    assert '1 excluded prediction' in visible
    assert 'False positive links' in visible
    assert 'no_task_head' in visible
    assert 'Other field outputs (unscored)' in visible
    assert 'Unscored user observation' in visible
    assert 'Instructional examples only' in visible


def test_model_inventory_exposes_identity_and_operational_context():
    page = render(sample_data())
    parsed = PageParser()
    parsed.feed(page)
    visible = ' '.join(parsed.text)
    assert 'Model identities' in visible
    assert 'sha256:def' in visible
    assert 'completed_documents' in visible
    assert 'measured_seconds' in visible


def test_navigation_targets_exist_when_optional_sections_are_absent():
    data = sample_data()
    data['feedback'] = []
    data['appendix'] = None
    parsed = PageParser()
    parsed.feed(render(data))
    ids = {attrs['id'] for _, attrs in parsed.tags if 'id' in attrs}
    assert all(link[1:] in ids for link in parsed.links if link and link.startswith('#'))


def test_overlapping_model_spans_remain_visible_on_one_source_track():
    data = sample_data()
    data['documents'][0]['arms'][1]['spans'] = [
        data['documents'][0]['reference_spans'][0],
        {'label': 'VERSION', 'start': 6, 'end': 10, 'text': 'l> 3'},
    ]
    page = render(data)
    assert '<mark class="span software version"' in page


def test_native_field_outputs_use_compact_unscored_table_with_alias_members():
    page = render(sample_data())
    parsed = PageParser()
    parsed.feed(page)
    visible = ' '.join(parsed.text)
    assert 'Software' in visible and 'Intent' in visible and 'Sentiment' in visible
    assert 'Alias group members' in visible
    assert '<tool>, 🧪' in visible
    assert '3.9.7' in visible and 'not_expressed' in visible
    assert 'unknown (failure)' in visible
    assert 'Raw field JSON' in visible


def test_span_lists_distinguish_matched_extra_and_missed_within_coverage():
    data = sample_data()
    data['documents'][0]['arms'][1]['spans'] = [
        data['documents'][0]['reference_spans'][0],
        {'label': 'SOFTWARE', 'start': 0, 'end': 1, 'text': '🧪'},
    ]
    page = render(data)
    parsed = PageParser()
    parsed.feed(page)
    visible = ' '.join(parsed.text)
    assert 'Matched spans' in visible
    assert 'Extra spans' in visible
    assert 'Missed spans' in visible
    assert '🧪' in visible and '3.9.7' in visible


def test_human_reference_label_and_incomplete_coverage_do_not_imply_miss():
    data = sample_data()
    data['provenance']['review_kind'] = 'human_reviewed'
    data['documents'][0]['coverage'] = [
        {'label': 'SOFTWARE', 'start': 0, 'end': 8,
         'review_kind': 'human_reviewed', 'complete': True}]
    data['documents'][0]['link_coverage'] = []
    page = render(data)
    parsed = PageParser()
    parsed.feed(page)
    visible = ' '.join(parsed.text)
    assert 'Human reviewed reference' in visible
    assert 'No complete VERSION coverage' in visible
    assert 'No complete ownership coverage' in visible
    assert 'Agent provisional reference' not in visible


def test_passage_index_jumps_to_each_document_and_back():
    data = sample_data()
    second = deepcopy(data['documents'][0])
    second['document_id'] = 'passage-2'
    data['documents'].append(second)
    parsed = PageParser()
    parsed.feed(render(data))
    ids = {attrs['id'] for _, attrs in parsed.tags if 'id' in attrs}
    assert {'passage-1', 'passage-2'} <= ids
    assert '#passage-1' in parsed.links and '#passage-2' in parsed.links
    assert parsed.links.count('#passages') >= 2


def test_evidence_stamp_and_metric_guide_reflect_selected_review():
    data = sample_data()
    data['provenance']['review_kind'] = 'human_reviewed'
    data['provenance']['population_documents'] = 7
    parsed = PageParser()
    parsed.feed(render(data))
    visible = ' '.join(parsed.text)
    assert 'Human reviewed' in visible
    assert '7 frozen passages' in visible
    assert 'TP' in visible and 'FP' in visible and 'FN' in visible
    assert 'Precision' in visible and 'Recall' in visible and 'F1' in visible
    assert 'N/A means' in visible


@pytest.mark.parametrize('damage', [
    {'start': -1}, {'end': 40}, {'text': 'wrong'}, {'start': 4, 'end': 8},
])
def test_invalid_highlight_rejected(damage):
    data = sample_data()
    data['documents'][0]['reference_spans'][0] = {
        **deepcopy(data['documents'][0]['reference_spans'][0]), **damage}
    with pytest.raises(ValueError, match='span'):
        render(data)
