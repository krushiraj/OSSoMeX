from copy import deepcopy

import pytest

from alias_fixtures import alias_item
from research.annotations.aliases import alias_relation_id, validate_alias_annotations, build_alias_groups
from research.annotations.tasks import make_tasks
from research.contracts import export_public, occurrence_id, text_revision


def test_two_mentions_link_to_one_group_without_changing_rows():
    item = alias_item()
    ann = item['annotation']
    original = deepcopy(ann['occurrences'])

    layer = validate_alias_annotations(item['task'], original, ann['alias_annotations'])
    groups = build_alias_groups(original, layer)

    assert len(groups) == 1 and groups[0]['preferred_name'] == 'ICEKAT'
    assert groups[0]['member_mention_ids'] == sorted(o['mention_id'] for o in original)
    assert len(export_public(original)[0]) == 2
    assert original == ann['occurrences']


@pytest.mark.parametrize('case_id', [
    'design-unknown-repeated-name', 'design-distinct-tools', 'design-negative'])
def test_unresolved_negative_and_empty_do_not_create_groups(case_id):
    item = alias_item(case_id)
    ann = item['annotation']
    layer = validate_alias_annotations(item['task'], ann['occurrences'], ann['alias_annotations'])

    assert build_alias_groups(ann['occurrences'], layer) == []
    assert build_alias_groups(ann['occurrences'], None) == []


@pytest.mark.parametrize('patch,code', [
    ({'schema_version': '99'}, 'ALIAS_SCHEMA_UNSUPPORTED'),
    ({'relations': None}, 'INVALID_ALIAS_RECORD'),
])
def test_partial_layer_is_not_silently_ignored(patch, code):
    item = alias_item()
    ann = item['annotation']
    with pytest.raises(ValueError, match=code):
        validate_alias_annotations(item['task'], ann['occurrences'],
                                   {**ann['alias_annotations'], **patch})


def test_design_fixture_identity_and_output_are_exact():
    item = alias_item()
    ann = item['annotation']
    relation = ann['alias_annotations']['relations'][0]
    assert alias_relation_id(item['task']['document_id'], item['task']['text_revision'],
                             relation['member_mention_ids']) == relation['relation_id']
    before = deepcopy(ann['alias_annotations'])
    layer = validate_alias_annotations(item['task'], ann['occurrences'], before)
    assert layer == before and layer is not before
    assert build_alias_groups(ann['occurrences'], layer) == [{
        'local_entity_id': 'local-software:21ce981a5d3b2bf07bbc1b07a3e3eb8b5d40519af5bdd981069a89d70675c71d',
        'document_id': item['task']['document_id'],
        'text_revision': item['task']['text_revision'],
        'member_mention_ids': relation['member_mention_ids'],
        'preferred_name': 'ICEKAT',
        'names': ['Interactive Continuous Enzyme Kinetics Analysis Tool', 'ICEKAT'],
        'quality': 'synthetic_fixture',
    }]
    assert export_public(ann['occurrences'])[0] == [
        {'name': 'Interactive Continuous Enzyme Kinetics Analysis Tool',
         'context_sentence': item['task']['text'], 'intents': ['created'],
         'sentiment': 'not_expressed', 'version': None},
        {'name': 'ICEKAT', 'context_sentence': item['task']['text'],
         'intents': ['created'], 'sentiment': 'not_expressed', 'version': None},
    ]


def _valid_pair():
    item = alias_item()
    task = item['task']
    occurrences = item['annotation']['occurrences']
    layer = item['annotation']['alias_annotations']
    relation = layer['relations'][0]
    return task, occurrences, layer, relation


@pytest.mark.parametrize('change,code', [
    (lambda r, o: r.update(member_mention_ids=[o[0]['mention_id'], 'missing']),
     'ALIAS_ENDPOINT_INVALID'),
    (lambda r, o: r.update(document_id='other-paper'), 'ALIAS_IDENTITY_MISMATCH'),
    (lambda r, o: r.update(text_revision='sha256:' + '0' * 64), 'ALIAS_IDENTITY_MISMATCH'),
    (lambda r, o: r.update(member_mention_ids=[o[0]['mention_id']] * 2),
     'INVALID_ALIAS_RECORD'),
    (lambda r, o: o[1]['name_span'].update(start=o[0]['name_span']['start']),
     'ALIAS_ENDPOINT_INVALID'),
    (lambda r, o: o[0]['name_span'].update(start=True), 'ALIAS_ENDPOINT_INVALID'),
    (lambda r, o: o[0]['name_span'].update(start=o[0]['name_span']['end']),
     'ALIAS_ENDPOINT_INVALID'),
    (lambda r, o: o[0].update(name='Different'), 'ALIAS_ENDPOINT_INVALID'),
    (lambda r, o: r.update(member_mention_ids=list(reversed(r['member_mention_ids']))),
     'INVALID_ALIAS_RECORD'),
    (lambda r, o: r.update(relation_id='alias:wrong'), 'ALIAS_ID_MISMATCH'),
    (lambda r, o: r.update(preferred_mention_id='missing'), 'ALIAS_PREFERENCE_INVALID'),
    (lambda r, o: r.update(evidence_spans=[]), 'ALIAS_EVIDENCE_INVALID'),
    (lambda r, o: r.update(evidence_spans=[{'start': 0, 'end': 66}]),
     'ALIAS_EVIDENCE_INVALID'),
    (lambda r, o: r.update(evidence_spans=[{'start': 0, 'end': 75},
                                           {'start': 0, 'end': 76}]),
     'ALIAS_EVIDENCE_INVALID'),
    (lambda r, o: r.update(review={'status': ' ', 'reasons': []}),
     'ALIAS_REVIEW_INVALID'),
    (lambda r, o: r.update(review={'status': 'human_reviewed', 'reasons': [],
                                  'reviewer': 'Alice'}), 'ALIAS_REVIEW_INVALID'),
])
def test_invalid_pair_is_rejected(change, code):
    task, occurrences, layer, relation = _valid_pair()
    change(relation, occurrences)
    with pytest.raises(ValueError, match=code):
        validate_alias_annotations(task, occurrences, layer)


def test_duplicate_pair_is_rejected_even_when_decisions_differ():
    task, occurrences, layer, relation = _valid_pair()
    duplicate = deepcopy(relation)
    duplicate.update(decision='not_alias', preferred_mention_id=None)
    layer['relations'].append(duplicate)
    with pytest.raises(ValueError, match='DUPLICATE_ALIAS_PAIR'):
        validate_alias_annotations(task, occurrences, layer)


def test_nonpositive_relation_requires_explicit_null_preference():
    task, occurrences, layer, relation = _valid_pair()
    relation['decision'] = 'not_alias'
    del relation['preferred_mention_id']
    with pytest.raises(ValueError, match='INVALID_ALIAS_RECORD'):
        validate_alias_annotations(task, occurrences, layer)


@pytest.mark.parametrize('patch,code', [
    ({'relation_type': 'cooccurrence'}, 'INVALID_ALIAS_RECORD'),
    ({'decision': 'maybe'}, 'INVALID_ALIAS_RECORD'),
    ({'preferred_mention_id': None}, 'ALIAS_PREFERENCE_INVALID'),
    ({'evidence_spans': [{'start': False, 'end': 75}]}, 'ALIAS_EVIDENCE_INVALID'),
])
def test_relation_fields_have_strict_types_and_enums(patch, code):
    task, occurrences, layer, relation = _valid_pair()
    relation.update(patch)
    with pytest.raises(ValueError, match=code):
        validate_alias_annotations(task, occurrences, layer)


def _three_names():
    text = 'We used Alpha, also called Beta and Gamma.'
    doc_id = 'synthetic-three-names'
    policy = {'policy_version': 'scibert-poc-2.1', 'policy_hash': 'a' * 64}
    task = make_tasks({'document_id': doc_id, 'text': text}, policy)[0]
    base = alias_item()['annotation']['occurrences'][0]
    occurrences = []
    for name in ('Alpha', 'Beta', 'Gamma'):
        start = text.index(name)
        end = start + len(name)
        occurrence = deepcopy(base)
        occurrence.update(document_id=doc_id, text_revision=task['text_revision'],
                          mention_id=occurrence_id(doc_id, task['text_revision'], start, end),
                          name=name, name_span={'start': start, 'end': end},
                          context_sentence=text, context_span={'start': 0, 'end': len(text)},
                          evidence={'intents': [{'start': 0, 'end': len(text)}], 'sentiment': []})
        occurrences.append(occurrence)

    def pair(first, second, decision='alias', preferred=None):
        members = sorted([occurrences[first]['mention_id'], occurrences[second]['mention_id']])
        return {'relation_id': alias_relation_id(doc_id, task['text_revision'], members),
                'document_id': doc_id, 'text_revision': task['text_revision'],
                'member_mention_ids': members, 'relation_type': 'explicit_alternative_name',
                'decision': decision,
                'preferred_mention_id': occurrences[preferred]['mention_id'] if preferred is not None else None,
                'evidence_spans': [{'start': 0, 'end': len(text)}],
                'review': {'status': 'synthetic_fixture', 'reasons': ['alias_link_review']}}

    return task, occurrences, pair


def test_transitive_negative_contradiction_is_rejected_during_validation():
    task, occurrences, pair = _three_names()
    layer = {'schema_version': '1.0', 'relations': [
        pair(0, 1, preferred=0), pair(1, 2, preferred=1), pair(0, 2, 'not_alias')]}
    with pytest.raises(ValueError, match='ALIAS_CONTRADICTION'):
        validate_alias_annotations(task, occurrences, layer)


def test_transitive_group_flags_conflicting_preferred_names():
    task, occurrences, pair = _three_names()
    layer = {'schema_version': '1.0', 'relations': [
        pair(0, 1, preferred=0), pair(1, 2, preferred=1)]}
    accepted = validate_alias_annotations(task, occurrences, layer)
    groups = build_alias_groups(occurrences, accepted)
    assert len(groups) == 1
    assert groups[0]['member_mention_ids'] == sorted(o['mention_id'] for o in occurrences)
    assert groups[0]['names'] == ['Alpha', 'Beta', 'Gamma']
    assert groups[0]['preferred_name'] is None
    assert groups[0]['review_reasons'] == ['alias_preference_conflict']


def test_same_acronym_in_different_documents_has_independent_ids():
    first = alias_item()
    second = deepcopy(first)
    second['task']['document_id'] = 'other-paper'
    for occurrence in second['annotation']['occurrences']:
        occurrence['document_id'] = 'other-paper'
        span = occurrence['name_span']
        occurrence['mention_id'] = occurrence_id('other-paper', second['task']['text_revision'],
                                                  span['start'], span['end'])
    relation = second['annotation']['alias_annotations']['relations'][0]
    relation['document_id'] = 'other-paper'
    relation['member_mention_ids'] = sorted(o['mention_id'] for o in second['annotation']['occurrences'])
    relation['preferred_mention_id'] = second['annotation']['occurrences'][1]['mention_id']
    relation['relation_id'] = alias_relation_id('other-paper', second['task']['text_revision'],
                                                relation['member_mention_ids'])
    groups = []
    for item in (first, second):
        layer = validate_alias_annotations(item['task'], item['annotation']['occurrences'],
                                           item['annotation']['alias_annotations'])
        groups.extend(build_alias_groups(item['annotation']['occurrences'], layer))
    assert len({group['local_entity_id'] for group in groups}) == 2
    assert [group['preferred_name'] for group in groups] == ['ICEKAT', 'ICEKAT']


def test_emoji_prefix_offsets_are_code_points():
    text = '😀 Alpha (A).'
    task = make_tasks({'document_id': 'emoji-fixture', 'text': text},
                      {'policy_version': 'scibert-poc-2.1', 'policy_hash': 'a' * 64})[0]
    source = alias_item()['annotation']['occurrences'][0]
    occurrences = []
    for name, start, end in [('Alpha', 2, 7), ('A', 9, 10)]:
        occurrence = deepcopy(source)
        occurrence.update(document_id='emoji-fixture', text_revision=text_revision(text),
                          mention_id=occurrence_id('emoji-fixture', text_revision(text), start, end),
                          name=name, name_span={'start': start, 'end': end},
                          context_sentence=text, context_span={'start': 0, 'end': len(text)})
        occurrences.append(occurrence)
    members = sorted(o['mention_id'] for o in occurrences)
    layer = {'schema_version': '1.0', 'relations': [{
        'relation_id': alias_relation_id('emoji-fixture', text_revision(text), members),
        'document_id': 'emoji-fixture', 'text_revision': text_revision(text),
        'member_mention_ids': members, 'relation_type': 'abbreviation', 'decision': 'alias',
        'preferred_mention_id': occurrences[1]['mention_id'],
        'evidence_spans': [{'start': 0, 'end': len(text)}],
        'review': {'status': 'synthetic_fixture', 'reasons': []}}]}
    assert [text[o['name_span']['start']:o['name_span']['end']] for o in occurrences] == ['Alpha', 'A']
    accepted = validate_alias_annotations(task, occurrences, layer)
    assert build_alias_groups(occurrences, accepted)[0]['names'] == ['Alpha', 'A']
