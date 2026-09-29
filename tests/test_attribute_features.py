from copy import deepcopy

import pytest

transformers = pytest.importorskip('transformers')

from research.annotations.aliases import alias_relation_id
from research.annotations.tasks import make_tasks
from research.contracts import FIELDS, occurrence_id, text_revision


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                               'We', 'used', 'NumPy', 'Tool', '##X', '1', '.', '24',
                               'and', 'later', 'Before', 'After', 'good']) + '\n')
    return transformers.BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)


def bundle(text='We used NumPy 1.24 and ToolX.'):
    document = {'document_id': 'd', 'text': text, 'text_revision': text_revision(text),
                'source': 'ecosystems', 'work_group_id': 'w', 'split': 'train'}
    task = make_tasks(document, {'policy_version': 'scibert-poc-2.1',
                                 'policy_hash': 'a' * 64, 'alias_schema_version': '1.0'})[0]
    occurrences = []
    for name in ('NumPy', 'ToolX'):
        if name not in text:
            continue
        start = text.index(name)
        occurrences.append({'schema_version': '2.0', 'document_id': 'd',
                            'text_revision': document['text_revision'],
                            'mention_id': occurrence_id('d', document['text_revision'], start, start + len(name)),
                            'name': name, 'name_span': {'start': start, 'end': start + len(name)},
                            'context_sentence': task['text'], 'context_span': task['context_span'],
                            'context_kind': 'sentence', 'known': dict.fromkeys(FIELDS, True),
                            'version_links': [], 'version_status': 'absent', 'intents': ['used'],
                            'sentiment': 'not_expressed', 'evidence': {'intents': [task['context_span']], 'sentiment': []}})
    if '1.24' in text:
        start = text.index('1.24')
        occurrences[0]['version_links'] = [{'text': '1.24', 'span': {'start': start, 'end': start + 4},
                                          'status': 'explicit_local'}]
        occurrences[0]['version_status'] = 'explicit'
    annotation = {'occurrences': occurrences, 'annotation_revision': 1,
                  'review_status': 'agent_provisional', 'covered_regions': [{**task['annotation_region'],
                    'status': 'complete', 'fields': dict.fromkeys(FIELDS, True)}], 'unresolved_regions': [],
                  'alias_annotations': {'schema_version': '1.0', 'relations': []}}
    return document, {'task': task, 'annotation': annotation}


def test_pair_coverage_not_absence_supplies_negatives(tokenizer):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle()
    item['annotation']['covered_regions'][0]['fields']['versions'] = False
    result = build_attribute_features(document, [item], tokenizer, {})
    assert [row['targets'] for row in result['features']['linker']] == [[1]]
    item['annotation']['covered_regions'][0]['fields']['versions'] = True
    result = build_attribute_features(document, [item], tokenizer, {})
    assert sorted(row['targets'] for row in result['features']['linker']) == [[0], [1]]
    item['annotation']['occurrences'][1]['known']['versions'] = False
    item['annotation']['occurrences'][1]['version_status'] = 'ambiguous'
    result = build_attribute_features(document, [item], tokenizer, {})
    assert [row['targets'] for row in result['features']['linker']] == [[1]]
    document, item = bundle('We used NumPy and ToolX.')
    assert build_attribute_features(document, [item], tokenizer, {})['features']['linker'] == []


def test_empty_alias_layer_has_no_targets(tokenizer):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle()
    assert build_attribute_features(document, [item], tokenizer, {})['features']['alias'] == []
    members = sorted(o['mention_id'] for o in item['annotation']['occurrences'])
    relation = {'relation_id': alias_relation_id('d', document['text_revision'], members),
                'document_id': 'd', 'text_revision': document['text_revision'],
                'member_mention_ids': members, 'relation_type': 'explicit_alternative_name',
                'decision': 'unresolved', 'preferred_mention_id': None,
                'evidence_spans': [item['task']['context_span']],
                'review': {'status': 'agent_provisional', 'reasons': ['uncertain']}}
    item['annotation']['alias_annotations']['relations'] = [relation]
    assert build_attribute_features(document, [item], tokenizer, {})['features']['alias'] == []
    relation['decision'] = 'not_alias'
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['alias'][0]['targets'] == [0]
    from research.training.attribute_features import summarize_support
    assert summarize_support(result)['alias']['subtype_counts'] == {'explicit_alternative_name': 1}


def test_checked_version_negatives_survive_unannotated_other_fields(tokenizer):
    from research.annotations.validation import validate_reply
    from research.training.attribute_features import build_attribute_features
    document, item = bundle()
    annotation = item['annotation']
    fields = {field: field in ('software', 'versions') for field in FIELDS}
    for occurrence in annotation['occurrences']:
        occurrence.update(known=dict(fields), intents=None, sentiment=None,
                          evidence={'intents': [], 'sentiment': []})
    annotation['covered_regions'][0].update(status='partial', fields=fields)
    reply = {**annotation, **{key: item['task'][key] for key in
                             ('task_id', 'document_id', 'text_revision', 'policy_version')},
             'attempt_id': 'names-versions-only', 'status': 'partial',
             'annotator': {'runtime': 'current_codex_session', 'model_identifier': None,
                           'prompt_hash': 'b' * 64, 'run_identifier': 'test'}}
    item['annotation'] = validate_reply(item['task'], reply)
    result = build_attribute_features(document, [item], tokenizer, {})
    assert [(row['first_span'], row['targets'], row['known'])
            for row in result['features']['linker']] == [
                ({'start': 8, 'end': 13}, [1], [True]),
                ({'start': 23, 'end': 28}, [0], [True])]
    assert result['features']['intent'] == result['features']['sentiment'] == []

    # Losing either checked field must still mask the negative, not invent absence.
    for field in ('software', 'versions'):
        masked = deepcopy(item)
        masked['annotation']['covered_regions'][0]['fields'][field] = False
        rows = build_attribute_features(document, [masked], tokenizer, {})['features']['linker']
        assert [row['targets'] for row in rows] == [[1]]

    # Unresolved intervening text and endpoints outside owned coverage are not negatives.
    for mutation in ('unresolved', 'outside_coverage'):
        masked = deepcopy(item)
        if mutation == 'unresolved':
            masked['annotation']['unresolved_regions'] = [{'start': 19, 'end': 22}]
            masked['annotation']['covered_regions'] = [
                {**deepcopy(annotation['covered_regions'][0]), 'end': 19},
                {**deepcopy(annotation['covered_regions'][0]), 'start': 22}]
        else:
            masked['annotation']['covered_regions'][0]['end'] = 19
            masked['annotation']['unresolved_regions'] = [{'start': 19, 'end': 29}]
        masked['annotation'] = validate_reply(masked['task'], masked['annotation'])
        rows = build_attribute_features(document, [masked], tokenizer, {})['features']['linker']
        assert [row['targets'] for row in rows] == [[1]]


def test_unknown_intent_bits_and_sentiment_are_masked(tokenizer):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle('We used NumPy.')
    occurrence = item['annotation']['occurrences'][0]
    occurrence['known'].update(created=False, shared=False, sentiment=False)
    occurrence['sentiment'] = None
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['intent'][0]['targets'] == [0, 1, 0]
    assert result['features']['intent'][0]['known'] == [False, True, False]
    assert result['features']['sentiment'] == []
    assert any(row['stage'] == 'sentiment' and row['reason'] == 'all_inactive' for row in result['excluded'])


def test_context_and_markers_preserve_unicode_offsets(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    text = '🧪 Before. We used NumPy 1.24. After.\n\nToolX.'
    document, item = bundle(text)
    occurrence = item['annotation']['occurrences'][0]
    version = occurrence['version_links'][0]
    result = build_inference_candidates(document, [occurrence], [version], tokenizer, {})
    pair = result['features']['linker'][0]
    assert pair['first_span'] == {'start': 18, 'end': 23}
    assert pair['second_span'] == {'start': 24, 'end': 28}
    assert pair['context_span'] == {'start': 10, 'end': 29}
    assert pair['distance_bucket'] == 129 and pair['order_flag'] == 1
    attribute = result['features']['intent'][0]
    assert attribute['context_span'] == {'start': 0, 'end': 36}
    assert not any(attribute['second_mask'])
    tokens = tokenizer.convert_ids_to_tokens(pair['input_ids'])
    assert [token for token, active in zip(tokens, pair['first_mask']) if active] == ['NumPy']
    assert [token for token, active in zip(tokens, pair['second_mask']) if active] == ['1', '.', '24']
    assert all(not active for token, active in zip(tokens, pair['context_mask']) if token in ('[CLS]', '[SEP]', '[SW1]', '[/SW1]', '[SW2]', '[/SW2]'))
    assert 'targets' not in pair and 'known' not in pair
    assert document['text'] == text


def test_long_context_is_excluded_not_truncated(tokenizer):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle('We used NumPy ' + 'good ' * 510 + '.')
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['intent'] == [] and result['features']['sentiment'] == []
    assert {row['reason'] for row in result['excluded']} == {'context_limit'}


def test_support_counts_after_exclusion(tokenizer):
    from research.training.attribute_features import build_attribute_features, summarize_support
    document, item = bundle()
    result = build_attribute_features(document, [item], tokenizer, {})
    row = deepcopy(result['features']['intent'][0])
    row['targets'] = [1, 0, 1]
    row['known'] = [False, False, False]
    result['features']['intent'].append(row)
    support = summarize_support(result)
    assert support['linker']['counts']['linked'] == {'positive': 1, 'negative': 1, 'positive_works': 1, 'negative_works': 1}
    assert support['linker']['trainable'] is True
    assert support['intent']['counts']['created']['positive'] == 0
    assert support['intent']['counts']['used']['negative'] == 0
    assert support['intent']['trainable'] is False
    assert support['sentiment']['counts']['not_expressed'] == {'count': 2, 'works': 1}
    assert support['sentiment']['trainable'] is False
    assert support['alias']['trainable'] is False
    assert support['linker']['coverage_warnings']


def test_unaligned_and_cross_paragraph_endpoints_are_excluded(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    document, item = bundle('We used NumPy.\n\nToolX 1.24.')
    first, second = item['annotation']['occurrences']
    result = build_inference_candidates(document, [first, second], first['version_links'], tokenizer, {})
    assert len(result['features']['linker']) == 1
    assert any(row['reason'] == 'candidate_context' for row in result['excluded'])
    first['name'] = 'Num'
    first['name_span']['end'] -= 2
    result = build_inference_candidates(document, [first], [], tokenizer, {})
    assert result['features']['intent'] == []
    assert any(row['reason'] == 'token_boundary' for row in result['excluded'])


def test_unknown_source_tokens_still_have_context_means(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    document, item = bundle('🧪 We used NumPy.')
    result = build_inference_candidates(document, item['annotation']['occurrences'], [], tokenizer, {})
    row = result['features']['intent'][0]
    assert row['context_mask'][row['offsets'].index([0, 1])] is True
    document, item = bundle('[MASK] We used NumPy.')
    row = build_inference_candidates(document, item['annotation']['occurrences'], [], tokenizer, {})['features']['intent'][0]
    assert row['context_mask'][row['input_ids'].index(tokenizer.mask_token_id)] is False


def test_abutting_pair_markers_and_frozen_policy_roundtrip(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    document = {'document_id': 'd', 'text': 'ToolX1.24.'}
    names = [{'name': 'ToolX', 'name_span': {'start': 0, 'end': 5}, 'mention_id': 'tool'}]
    versions = [{'text': '1.24', 'span': {'start': 5, 'end': 9}}]
    # Both endpoint boundaries must already exist in the unmarked tokenization.
    names[0]['name'] = 'ToolX1'
    names[0]['name_span']['end'] = 6
    versions[0] = {'text': '.24', 'span': {'start': 6, 'end': 9}}
    result = build_inference_candidates(document, names, versions, tokenizer, {})
    row = result['features']['linker'][0]
    tokens = tokenizer.convert_ids_to_tokens(row['input_ids'])
    assert all(marker in tokens for marker in ['[SW1]', '[/SW1]', '[SW2]', '[/SW2]'])
    assert any(row['first_mask']) and any(row['second_mask'])
    replay = build_inference_candidates(document, names, versions, tokenizer,
                                         {'attribute_features': result['config']})
    assert replay == result


def test_source_paragraphs_and_repeated_aliases_control_candidates(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    document = {'document_id': 'd', 'text': 'NumPy. ToolX.',
                'paragraphs': [{'start': 0, 'end': 6, 'kind': 'p'}, {'start': 7, 'end': 13, 'kind': 'p'}]}
    names = [{'name': 'NumPy', 'name_span': {'start': 0, 'end': 5}},
             {'name': 'ToolX', 'name_span': {'start': 7, 'end': 12}}]
    result = build_inference_candidates(document, names, [], tokenizer, {})
    assert result['features']['alias'] == []
    assert result['features']['intent'][0]['context_span'] == {'start': 0, 'end': 6}
    document = {'document_id': 'd', 'text': 'NumPy NumPy.'}
    names[1] = {'name': 'NumPy', 'name_span': {'start': 6, 'end': 11}}
    result = build_inference_candidates(document, names, [], tokenizer, {})
    assert result['features']['alias'] == []
    assert any(row['reason'] == 'repeated_name' for row in result['excluded'])


def test_distance_uses_original_token_positions_and_clips(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    text = '1.24 ' + 'good ' * 140 + 'NumPy.'
    document = {'document_id': 'd', 'text': text}
    names = [{'name': 'NumPy', 'name_span': {'start': 705, 'end': 710}}]
    result = build_inference_candidates(document, names, [{'text': '1.24', 'span': {'start': 0, 'end': 4}}], tokenizer, {})
    row = result['features']['linker'][0]
    assert row['distance_bucket'] == 0 and row['order_flag'] == 0


def test_absolute_source_offsets_and_unsupported_context(tokenizer):
    from research.training.attribute_features import build_inference_candidates
    document = {'document_id': 'd', 'text': 'We used NumPy.', 'offset_base': 100}
    names = [{'name': 'NumPy', 'name_span': {'start': 108, 'end': 113}}]
    result = build_inference_candidates(document, names, [], tokenizer, {})
    assert result['features']['intent'][0]['context_span'] == {'start': 100, 'end': 114}
    assert [108, 113] in result['features']['intent'][0]['offsets']
    names[0]['context_kind'] = 'table'
    result = build_inference_candidates(document, names, [], tokenizer, {})
    assert result['features']['intent'] == []
    assert {row['reason'] for row in result['excluded']} == {'unsupported_context'}


def test_remote_combined_intent_evidence_masks_positive_bits_only(tokenizer):
    from research.training.attribute_features import build_attribute_features, summarize_support
    document, item = bundle('We used NumPy. Before. After. We later shared it.')
    occurrence = item['annotation']['occurrences'][0]
    occurrence['intents'] = ['used', 'shared']
    remote = {'start': document['text'].index('We later'), 'end': len(document['text'])}
    occurrence['evidence']['intents'] = [occurrence['name_span'], remote]
    original = deepcopy(item)
    result = build_attribute_features(document, [item], tokenizer, {})
    row = result['features']['intent'][0]
    assert row['known'] == [True, False, False]
    assert row['targets'] == [0, 1, 1]
    assert set(row['provenance']['masked_fields']) == {'used', 'shared'}
    assert row['provenance']['masked_fields']['used']['reason'] == 'evidence_outside_context'
    assert row['provenance']['masked_fields']['used']['evidence_spans'] == [remote]
    assert summarize_support(result)['intent']['counts']['used']['positive'] == 0
    assert item == original
    occurrence['known']['created'] = False
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['intent'] == []
    assert any(row['stage'] == 'intent' and row['reason'] == 'evidence_outside_context' for row in result['excluded'])


def test_remote_sentiment_and_negative_intent_rationale_are_excluded(tokenizer):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle('We used NumPy. Before. After. It was good.')
    occurrence = item['annotation']['occurrences'][0]
    remote = {'start': document['text'].index('It was'), 'end': len(document['text'])}
    occurrence['sentiment'] = 'positive'
    occurrence['evidence']['sentiment'] = [remote]
    occurrence['intents'] = ['mentioned']
    occurrence['evidence']['intents'] = [remote]
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['sentiment'] == [] and result['features']['intent'] == []
    excluded = {row['stage']: row for row in result['excluded']}
    assert excluded['sentiment']['reason'] == 'evidence_outside_context'
    assert set(excluded['intent']['provenance']['masked_fields']) == {'created', 'used', 'shared'}
    occurrence['evidence']['intents'] = []
    assert build_attribute_features(document, [item], tokenizer, {})['features']['intent'][0]['known'] == [True] * 3
    occurrence['evidence']['sentiment'] = [occurrence['name_span']]
    assert build_attribute_features(document, [item], tokenizer, {})['features']['sentiment'][0]['known'] == [True]


@pytest.mark.parametrize('decision', ['alias', 'not_alias'])
def test_alias_evidence_outside_pair_context_is_excluded(tokenizer, decision):
    from research.training.attribute_features import build_attribute_features
    document, item = bundle('We used NumPy and ToolX. Before. After.')
    members = sorted(o['mention_id'] for o in item['annotation']['occurrences'])
    item['annotation']['alias_annotations']['relations'] = [{
        'relation_id': alias_relation_id('d', document['text_revision'], members),
        'document_id': 'd', 'text_revision': document['text_revision'],
        'member_mention_ids': members, 'relation_type': 'explicit_alternative_name',
        'decision': decision, 'preferred_mention_id': members[0] if decision == 'alias' else None,
        'evidence_spans': [item['task']['context_span']],
        'review': {'status': 'agent_provisional', 'reasons': ['checked']}}]
    original = deepcopy(item)
    result = build_attribute_features(document, [item], tokenizer, {})
    assert result['features']['alias'] == []
    excluded = next(row for row in result['excluded'] if row['stage'] == 'alias')
    assert excluded['reason'] == 'evidence_outside_context'
    assert excluded['provenance']['masked_fields']['alias']['reason'] == 'evidence_outside_context'
    assert item == original
    item['annotation']['alias_annotations']['relations'][0]['evidence_spans'] = [{'start': 0, 'end': 23}]
    assert build_attribute_features(document, [item], tokenizer, {})['features']['alias'][0]['known'] == [True]
