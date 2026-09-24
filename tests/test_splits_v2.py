from copy import deepcopy

import pytest

from research.contracts import text_revision
from research.data import splits, bundles
from research.data.manifest import read_jsonl
from research.data.manifest import digest, write_jsonl


def doc(ident, source='sofair', native=None, text=None, **extra):
    text = text or f'Unique research document {ident} with distinct contents.'
    return {'document_id': ident, 'text': text, 'text_revision': text_revision(text),
            'metadata': {'source': source}, 'source': source, 'source_ids': {},
            'native_split': native, 'fulltext_eligible': True, 'historical': False, **extra}


def test_alternative_id_cannot_hide_test_work():
    train = {'documents': [{'document_id': 'alias', 'work_group_id': 'w1', 'text_revision': 'sha256:a'}]}
    assert splits.check_training_manifest(train, {'work_group_ids': ['w1'], 'text_revisions': ['sha256:b']})[0]['code'] == 'HELDOUT_WORK_GROUP'


def test_transitive_cross_source_identifiers_and_exact_text_grouped():
    docs = [doc('a', source_ids={'doi':'https://doi.org/10.1/X'}),
            doc('b', 'somesci', source_ids={'doi':'10.1/x', 'pmcid':'PMC12'}),
            doc('c', 'openalex', source_ids={'pmcid':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12/'}),
            doc('d', text='Unique research document a with distinct contents.')]
    result = splits.group_works(docs)
    assert len(result['groups']) == 1
    assert result['groups'][0]['source'] == 'sofair'
    assert len(result['groups'][0]['documents']) == 4


def test_near_duplicate_and_excerpt_overlap_block_freeze():
    text = ' '.join('word'+str(i) for i in range(100))
    result = splits.group_works([doc('a', text=text), doc('b', text=text+' ending'),
                                doc('c', text=' '.join('word'+str(i) for i in range(10, 30)))])
    assert len(result['unresolved_overlaps']) == 3
    with pytest.raises(ValueError, match='UNRESOLVED_OVERLAPS'):
        splits.assign_splits(result['groups'], {'sofair': {'train':1, 'dev':0, 'test':1}}, 42)


def test_native_and_historical_exclusions_precede_quota_and_are_deterministic():
    docs = [doc('h', historical=True), doc('t', native='test'), doc('d', native='dev'),
            doc('a', native='train'), doc('b', native='train'), doc('c', native='train')]
    groups = splits.group_works(docs)['groups']
    quota = {'sofair': {'train':2, 'dev':1, 'test':1}}
    a = splits.assign_splits(groups, quota, 42)
    b = splits.assign_splits(list(reversed(groups)), quota, 42)
    assert a == b
    chosen = {r['document_id']: r['split'] for r in a['documents']}
    assert chosen['t'] == 'test' and chosen['d'] == 'dev'
    assert 'h' not in chosen
    assert all(chosen.get(x) != 'train' for x in ['h', 't', 'd'])


def test_quota_shortage_is_explicit_without_partial_split():
    groups = splits.group_works([doc('a')])['groups']
    with pytest.raises(splits.SplitError) as exc:
        splits.assign_splits(groups, {'sofair': {'train':2, 'dev':0, 'test':0}}, 42)
    assert exc.value.report['shortages'] == [{'source':'sofair', 'split':'train', 'required':2, 'available':1, 'shortage':1}]


def test_synthetic_and_native_heldout_are_rejected_by_training_guard():
    records = [doc('f', source='synthetic-contract-fixture'), doc('n', native='test')]
    issues = splits.check_training_manifest({'documents':records}, {})
    assert {x['code'] for x in issues} == {'SYNTHETIC_NOT_RESEARCH', 'NATIVE_HELDOUT'}


def test_materialized_training_bundle_contains_no_test_text_or_labels(tmp_path):
    a, b = doc('a'), doc('b', text='SECRET TEST TEXT')
    a.update(work_group_id='train-group', split='train')
    b.update(work_group_id='test-group', split='test')
    manifest = {'documents':[a,b], 'role':'train', 'heldout': {'work_group_ids':['test-group'], 'text_revisions':[b['text_revision']]}}
    bundles.materialize_bundle(manifest, tmp_path / 'train')
    assert [d['document_id'] for d in read_jsonl(tmp_path/'train/documents.jsonl')] == ['a']
    assert 'SECRET TEST TEXT' not in ''.join(p.read_text() for p in (tmp_path/'train').iterdir())
    with pytest.raises(FileExistsError):
        bundles.materialize_bundle(manifest, tmp_path / 'train')


def test_bundle_rejects_changed_text(tmp_path):
    a = doc('a'); a.update(split='train', work_group_id='g'); a['text'] = 'changed'
    with pytest.raises(ValueError):
        bundles.materialize_bundle({'documents':[a], 'role':'train', 'heldout':{}}, tmp_path/'train')
    assert not (tmp_path/'train').exists()


def test_corpus_build_preserves_native_scope_and_requires_access_decision(tmp_path):
    import io, zipfile
    from research.data.corpus import build_corpus
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('Label/PLoS_methods/PMC1.txt', 'We used X.')
        z.writestr('Label/PLoS_methods/PMC1.ann', 'T1\tApplication_Usage 8 9\tX\n')
        z.writestr('Label/PLoS_methods/PMC2.txt', '')
        z.writestr('Label/PLoS_methods/PMC2.ann', '')
    raw = tmp_path/'raw'; raw.mkdir()
    (raw/'archive').write_bytes(buf.getvalue())
    (raw/'split').write_text('{"train": ["PMC1"], "test": [], "devel": []}')
    write_jsonl(raw/'acquisitions.jsonl', [
        {'source':'somesci', 'role':'native_archive', 'path':'archive', 'sha256':digest(buf.getvalue()), 'public':True},
        {'source':'somesci-split', 'role':'source_metadata', 'path':'split', 'sha256':digest((raw/'split').read_bytes())}])
    report = build_corpus(raw/'acquisitions.jsonl', tmp_path/'corpus')
    records = read_jsonl(tmp_path/'corpus/documents.jsonl')
    assert report['document_count'] == 1
    assert report['input_failed'] == 1
    assert records[0]['supplied_text_scope'] == 'methods'
    assert records[0]['native_split'] == 'train'
    assert records[0]['fulltext_eligible'] is False
    assert len(read_jsonl(tmp_path/'corpus/native.jsonl')[0]['spans']) == 1


def test_missing_native_split_cannot_be_used_for_training(tmp_path):
    from research.data.corpus import native_partition
    with pytest.raises(ValueError, match='NATIVE_SPLIT_MISSING'):
        native_partition('PMC1', {})


def test_access_decision_cannot_promote_an_excerpt_or_different_revision():
    from research.data.corpus import apply_access_decision
    d = doc('a'); d.update(supplied_text_scope='methods', fulltext_eligible=False)
    decision = {'document_id':'a', 'text_revision':d['text_revision'], 'public':True,
                'language':'en', 'text_license':'CC-BY-4.0', 'access_basis':'https://example.org/license',
                'supplied_text_scope':'fulltext'}
    with pytest.raises(ValueError, match='EXCERPT_REQUIRES_NEW_TEXT'):
        apply_access_decision(d, decision)
    decision['text_revision'] = 'sha256:wrong'
    with pytest.raises(ValueError, match='REVISION_MISMATCH'):
        apply_access_decision(d, decision)


def test_native_test_identity_survives_different_acquired_text(tmp_path):
    import json
    from research.data.corpus import build_corpus
    raw=tmp_path/'raw';raw.mkdir()
    rows=[]
    for name,ident,text in [('somesci','PMC1','Original full paper.'),('openalex','W1','A different extraction of the same paper.')]:
        p=raw/name;p.write_text(text)
        rows.append({'source':name,'source_record_id':ident,'source_ids':{'pmcid':'PMC1','doi':'10.1/same'},
                     'role':'paper_text','format':'text','path':name,'sha256':digest(p.read_bytes()),
                     'public':True,'language':'en','supplied_text_scope':'fulltext',
                     'text_license':'CC-BY-4.0','access_basis':'https://example.org/license'})
    (raw/'split.json').write_text(json.dumps({'train':[],'test':['PMC1']}))
    rows.append({'source':'somesci-split','role':'source_metadata','path':'split.json','sha256':digest((raw/'split.json').read_bytes())})
    write_jsonl(raw/'acquisitions.jsonl',rows)
    build_corpus(raw/'acquisitions.jsonl',tmp_path/'corpus')
    docs=read_jsonl(tmp_path/'corpus/documents.jsonl')
    assert all(d['source_ids']['pmcid']=='PMC1' for d in docs)
    groups=splits.group_works(docs)['groups']
    assert len(groups)==1
    with pytest.raises(splits.SplitError,match='QUOTA_SHORTAGE'):
        splits.assign_splits(groups,{'somesci':{'train':1,'dev':0,'test':0}},42)
    rows[0]['source_ids']['pmcid']='PMC2'
    write_jsonl(raw/'conflicting.jsonl',rows)
    with pytest.raises(ValueError,match='SOURCE_ID_CONFLICT'):
        build_corpus(raw/'conflicting.jsonl',tmp_path/'conflicting-corpus')
