from copy import deepcopy
import json

import pytest

from research.contracts import FIELDS
from research.annotations.tasks import make_tasks
from research.annotations import review_store as store
from research.data.manifest import read_jsonl
from alias_fixtures import alias_item


def item(text='😀 We used X.'):
    task=make_tasks({'document_id':'d','text':text,'source':'synthetic-contract-fixture','split':'demo'},
                    {'policy_version':'scibert-poc-2.0','policy_hash':'a'*64})[0]
    task['whole_passage_audit']=True
    return {'task':task,'annotation':{'occurrences':[],'covered_regions':[], 'unresolved_regions':[task['annotation_region']],
                                     'status':'partial','review_status':'agent_provisional','annotation_revision':1}}


def decision(task, ident='decision-1', revision=1, **extra):
    return {'decision_id':ident,'task_id':task['task_id'],'document_id':task['document_id'],
            'text_revision':task['text_revision'],'base_annotation_revision':revision,'reviewer':'Krushi',
            'action':'accept_passage','reason':'Checked the full passage','value':None,**extra}


def policy_required_item(signal='version',role='train',source='ecosystems'):
    i=alias_item(); original=i['task']
    policy={'policy_version':'scibert-poc-2.1' if signal=='version' else 'scibert-poc-2.0',
            'policy_hash':'a'*64,'alias_schema_version':'1.0' if signal=='requested' else None}
    i['task']=make_tasks({'document_id':original['document_id'],'text':original['text'],
                          'split':role,'source':source},policy)[0]
    i['annotation'].update({key:i['task'][key] for key in ('task_id','policy_version')})
    if signal=='layer': i['annotation']['alias_annotations']['relations']=[]
    else: i['annotation'].pop('alias_annotations')
    return i


@pytest.mark.parametrize('signal',['version','requested','layer'])
@pytest.mark.parametrize('role,source',[('train','ecosystems'),('dev','ecosystems'),
    ('train','synthetic-contract-fixture'),('demo','ecosystems')])
def test_store_requires_frozen_policy_for_nonexempt_alias_tasks(tmp_path,signal,role,source):
    i=policy_required_item(signal,role,source); c=store.open_store(tmp_path/'r.sqlite')
    with pytest.raises(ValueError,match='POLICY_SNAPSHOT_REQUIRED'):
        store.import_items(c,[i],role)
    for table in ('items','decisions','metadata'):
        assert c.execute(f'SELECT count(*) FROM {table}').fetchone()[0]==0
    c.close()


@pytest.mark.parametrize('role',['train','dev'])
def test_store_accepts_legacy_research_policy_without_aliases(tmp_path,role):
    i=item(); i['task'].update(split=role,source='ecosystems')
    c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],role)
    assert c.execute('SELECT count(*) FROM items').fetchone()[0]==1
    c.close()


def test_stale_revision_idempotence_restart_and_export(tmp_path):
    path=tmp_path/'review.sqlite'; connection=store.open_store(path)
    i=item('No software.'); store.import_items(connection,[i],role='demo')
    first=decision(i['task'])
    saved=store.apply_decision(connection,first)
    assert saved['annotation_revision']==2
    assert store.apply_decision(connection,first)==saved
    with pytest.raises(store.ReviewError,match='STALE_REVISION'):
        store.apply_decision(connection,decision(i['task'],'decision-2'))
    with pytest.raises(store.ReviewError,match='DECISION_ID_CONFLICT'):
        store.apply_decision(connection,{**first,'reason':'different'})
    connection.close(); connection=store.open_store(path)
    assert store.get_item(connection,i['task']['task_id'])['annotation_revision']==2
    export=store.export_reference(connection,tmp_path/'export')
    assert export['role']=='demo' and export['quality']=='provisional'
    assert len(read_jsonl(tmp_path/'export/decisions.jsonl'))==1
    regions=read_jsonl(tmp_path/'export/coverage.jsonl')
    assert regions[0]['human_reviewed_fields']==['software']
    assert regions[0]['fields']['sentiment'] is False


def test_add_edit_link_unresolve_remove_and_exact_unicode_offsets(tmp_path):
    connection=store.open_store(tmp_path/'r.sqlite'); i=item(); task=i['task']; store.import_items(connection,[i],role='demo')
    occurrence={'schema_version':'2.0','document_id':'d','text_revision':task['text_revision'],
                'name':'X','name_span':{'start':10,'end':11}, 'context_sentence':task['text'],
                'context_span':task['context_span'],'context_kind':'paragraph','version_links':[],
                'version_status':'absent','intents':['used'],'sentiment':'not_expressed',
                'known':dict.fromkeys(FIELDS,True),'evidence':{'intents':[task['context_span']],'sentiment':[]},
                'review':{'status':'needs_review','reasons':[]}}
    saved=store.apply_decision(connection,decision(task,action='upsert_occurrence',value=occurrence,reason='Added a missed name'))
    assert saved['annotation_revision']==2
    current=store.get_item(connection,task['task_id']); o=current['annotation']['occurrences'][0]
    assert task['text'][o['name_span']['start']:o['name_span']['end']]=='X'
    assert current['status']!='reviewed'
    store.apply_decision(connection,decision(task,'u',2,action='mark_field_unresolved',field='sentiment',target_name_span=o['name_span'],reason='Uncertain opinion'))
    o=store.get_item(connection,task['task_id'])['annotation']['occurrences'][0]
    assert o['known']['sentiment'] is False and o['sentiment'] is None
    store.apply_decision(connection,decision(task,'r',3,action='remove_occurrence',target_name_span=o['name_span'],reason='Not software'))
    assert store.get_item(connection,task['task_id'])['annotation']['occurrences']==[]
    assert connection.execute('select count(*) from decisions').fetchone()[0]==3


def test_bad_span_reason_identity_and_nonloopback_inputs_rejected(tmp_path):
    c=store.open_store(tmp_path/'r.sqlite'); i=item();store.import_items(c,[i],role='demo')
    for patch in [{'reason':''},{'text_revision':'changed'},{'document_id':'other'},{'action':'unknown'}]:
        with pytest.raises(store.ReviewError):store.apply_decision(c,{**decision(i['task']),**patch})
    assert store.get_item(c,i['task']['task_id'])['annotation_revision']==1


def test_store_cannot_mix_splits_or_replace_original_items(tmp_path):
    c=store.open_store(tmp_path/'r.sqlite'); i=item();store.import_items(c,[i],role='demo')
    with pytest.raises(store.ReviewError):store.import_items(c,[i],role='test')
    bad=deepcopy(i);bad['task']['text']='Changed'
    with pytest.raises(store.ReviewError):store.import_items(c,[bad],role='demo')


def test_decision_log_is_append_only(tmp_path):
    import sqlite3
    c=store.open_store(tmp_path/'r.sqlite');i=item();store.import_items(c,[i],role='demo')
    store.apply_decision(c,decision(i['task']))
    with pytest.raises(sqlite3.DatabaseError):c.execute('delete from decisions')


def test_demo_cli_is_marked_synthetic_and_never_a_research_split(tmp_path):
    from research.cli import main
    assert main(['annotate','demo','--output',str(tmp_path/'demo')])==0
    manifest=json.loads((tmp_path/'demo/manifest.json').read_bytes())
    assert manifest['role']=='demo'
    items=read_jsonl(tmp_path/'demo/items.jsonl')
    assert len(items)>=8
    assert all(i['task']['source']=='synthetic-contract-fixture' for i in items)
    c=store.open_store(tmp_path/'demo.sqlite');store.import_items(c,items,role='demo')
    assert store.queue(c)['progress']['total']==len(items)


def test_name_audit_preserves_agent_coverage_and_recomputes_negative_regions(tmp_path):
    c=store.open_store(tmp_path/'r.sqlite');i=item('No tools.')
    region={**i['task']['annotation_region'],'status':'complete','fields':dict.fromkeys(FIELDS,True),
            'provenance':{'kind':'agent_provisional'}}
    i['annotation'].update(status='complete',covered_regions=[region],unresolved_regions=[],complete_negative_regions=[region])
    store.import_items(c,[i],role='demo');store.apply_decision(c,decision(i['task']))
    current=store.get_item(c,i['task']['task_id'])['annotation']
    assert all(current['covered_regions'][0]['fields'].values())
    assert current['covered_regions'][0]['provenance']=={'kind':'agent_provisional'}
    assert current['covered_regions'][0]['human_reviewed_fields']==['software']
    assert current['status']=='complete'
    assert current['complete_negative_regions'][0]['fields']['sentiment'] is True
    task=i['task']
    occurrence={'schema_version':'2.0','document_id':task['document_id'],'text_revision':task['text_revision'],
                'name':'tools','name_span':{'start':3,'end':8},'context_sentence':task['text'],
                'context_span':task['context_span'],'context_kind':'paragraph','version_links':[],
                'version_status':'absent','intents':['mentioned'],'sentiment':'not_expressed',
                'known':dict.fromkeys(FIELDS,True),'evidence':{'intents':[],'sentiment':[]},
                'review':{'status':'unreviewed','reasons':[]}}
    store.apply_decision(c,decision(task,'add',2,action='upsert_occurrence',value=occurrence))
    assert store.get_item(c,task['task_id'])['annotation']['complete_negative_regions']==[]
    store.apply_decision(c,decision(task,'remove',3,action='remove_occurrence',target_name_span=occurrence['name_span']))
    assert len(store.get_item(c,task['task_id'])['annotation']['complete_negative_regions'])==1


def test_alias_acceptance_independent_and_preserves_completed_name_audit(tmp_path):
    i=alias_item(); task=i['task']; c=store.open_store(tmp_path/'r.sqlite')
    store.import_items(c,[i],'demo')
    before=store.get_item(c,task['task_id'])['annotation']['occurrences']
    rid=i['annotation']['alias_annotations']['relations'][0]['relation_id']
    payload=decision(task,action='accept_alias',target_relation_id=rid)
    saved=store.apply_decision(c,payload)
    assert store.apply_decision(c,payload)==saved
    current=store.get_item(c,task['task_id'])
    assert current['annotation']['occurrences']==before
    assert 'human_passage_review' not in current['annotation']
    assert current['status']!='reviewed'
    assert current['annotation']['alias_annotations']['relations'][0]['review']['status']=='human_reviewed'
    store.apply_decision(c,decision(task,'passage',2))
    passage=store.get_item(c,task['task_id'])['annotation']['human_passage_review']
    store.apply_decision(c,decision(task,'reject',3,action='reject_alias',target_relation_id=rid))
    current=store.get_item(c,task['task_id'])
    assert current['status']=='reviewed'
    assert current['annotation']['human_passage_review']==passage
    assert store.queue(c)['progress']['reviewed']==1


@pytest.mark.parametrize('relation_decision',['alias','not_alias','unresolved'])
def test_alias_members_protected_and_attribute_edits_survive(tmp_path,relation_decision):
    i=alias_item(); relation=i['annotation']['alias_annotations']['relations'][0]
    relation['decision']=relation_decision
    if relation_decision!='alias': relation['preferred_mention_id']=None
    task=i['task']; c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo')
    before=store.get_item(c,task['task_id']); occurrence=before['annotation']['occurrences'][0]
    changed=deepcopy(occurrence)
    changed['name_span']['end']-=1; changed['name']=changed['name'][:-1]
    changed.pop('mention_id')
    for patch in [{'action':'remove_occurrence'}, {'action':'upsert_occurrence','value':changed}]:
        with pytest.raises(store.ReviewError,match='ALIAS_MEMBER_IN_USE'):
            store.apply_decision(c,decision(task,target_name_span=occurrence['name_span'],**patch))
        assert store.get_item(c,task['task_id'])==before
        assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==0
    store.apply_decision(c,decision(task,action='upsert_occurrence',target_name_span=occurrence['name_span'],value=occurrence))
    assert store.get_item(c,task['task_id'])['annotation']['alias_annotations']==i['annotation']['alias_annotations']


def test_alias_actions_atomic_and_server_owned_provenance(tmp_path):
    i=alias_item(); task=i['task']; c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo')
    relation=i['annotation']['alias_annotations']['relations'][0]; rid=relation['relation_id']
    before=store.get_item(c,task['task_id'])
    for patch,code in [({'action':'remove_alias','target_relation_id':'unknown'},'UNKNOWN_ALIAS_RELATION'),
                       ({'action':'upsert_alias','value':relation},'DUPLICATE_ALIAS_PAIR'),
                       ({'action':'accept_alias','target_relation_id':rid,'text_revision':'wrong'},'SOURCE_REVISION_MISMATCH'),
                       ({'action':'accept_alias','target_relation_id':rid,'base_annotation_revision':9},'STALE_REVISION')]:
        with pytest.raises(store.ReviewError,match=code): store.apply_decision(c,decision(task,**patch))
        assert store.get_item(c,task['task_id'])==before
        assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==0
    value={**relation,'review':{'status':'human_reviewed','reviewer':'forged'}}
    store.apply_decision(c,decision(task,action='upsert_alias',target_relation_id=rid,value=value))
    review=store.get_item(c,task['task_id'])['annotation']['alias_annotations']['relations'][0]['review']
    assert review['reviewer']=='Krushi'
    store.apply_decision(c,decision(task,'unresolve',2,action='unresolve_alias',target_relation_id=rid))
    with pytest.raises(store.ReviewError,match='ALIAS_NOT_POSITIVE'):
        store.apply_decision(c,decision(task,'accept',3,action='accept_alias',target_relation_id=rid))
    assert 'unresolved_alias' in store.queue(c)['items'][0]['reasons']
    store.apply_decision(c,decision(task,'remove',3,action='remove_alias',target_relation_id=rid))
    assert store.get_item(c,task['task_id'])['annotation']['alias_annotations']['relations']==[]


def test_store_rejects_null_alias_layer_and_preserves_legacy_absence(tmp_path):
    c=store.open_store(tmp_path/'r.sqlite'); i=item(); i['annotation']['alias_annotations']=None
    with pytest.raises(store.ReviewError,match='ALIAS_SCHEMA_UNSUPPORTED'): store.import_items(c,[i],'demo')
    assert c.execute('SELECT count(*) FROM items').fetchone()[0]==0
    del i['annotation']['alias_annotations']; store.import_items(c,[i],'demo')
    store.apply_decision(c,decision(i['task']))
    assert 'alias_annotations' not in store.get_item(c,i['task']['task_id'])['annotation']


def test_alias_edit_preserves_every_other_annotation_field(tmp_path):
    i=alias_item(); i['annotation']['occurrences'].reverse()
    c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo')
    before=store.get_item(c,i['task']['task_id'])['annotation']
    rid=before['alias_annotations']['relations'][0]['relation_id']
    store.apply_decision(c,decision(i['task'],action='accept_alias',target_relation_id=rid))
    after=store.get_item(c,i['task']['task_id'])['annotation']
    for annotation in (before,after):
        annotation.pop('alias_annotations'); annotation.pop('annotation_revision')
    assert after==before


@pytest.mark.parametrize('capability',['old_policy','not_requested'])
def test_alias_mutations_require_task_policy_capability(tmp_path,capability):
    i=alias_item()
    if capability=='old_policy': i['task']['policy_version']='scibert-poc-2.0'
    else: i['task']['requested_fields'].remove('aliases')
    c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo')
    before=store.get_item(c,i['task']['task_id'])
    with pytest.raises(store.ReviewError,match='ALIAS_TASK_UNSUPPORTED'):
        store.apply_decision(c,decision(i['task'],action='upsert_alias',value=i['annotation']['alias_annotations']['relations'][0]))
    assert store.get_item(c,i['task']['task_id'])==before
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==0


def test_alias_graph_contradiction_and_targeted_replacement_are_atomic(tmp_path):
    from test_aliases import _three_names
    task,occurrences,pair=_three_names(); task.update(split='demo',source='synthetic-contract-fixture',requested_fields=['aliases'])
    i={'task':task,'annotation':{'occurrences':occurrences,'annotation_revision':1,
        'alias_annotations':{'schema_version':'1.0','relations':[pair(0,1,preferred=0),pair(0,2,'not_alias')]}}}
    c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo'); before=store.get_item(c,task['task_id'])
    with pytest.raises(store.ReviewError,match='ALIAS_CONTRADICTION'):
        store.apply_decision(c,decision(task,action='upsert_alias',value=pair(1,2,preferred=1)))
    assert store.get_item(c,task['task_id'])==before
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==0
    rid=pair(0,2,'not_alias')['relation_id']
    store.apply_decision(c,decision(task,action='upsert_alias',target_relation_id=rid,value=pair(1,2,preferred=1)))
    relations=store.get_item(c,task['task_id'])['annotation']['alias_annotations']['relations']
    assert len(relations)==2 and all(r['relation_id']!=rid for r in relations)
    assert 'alias_preference_conflict' in store.queue(c)['items'][0]['reasons']


def test_failed_alias_log_insert_rolls_back_item_and_revision(tmp_path):
    import sqlite3
    i=alias_item(); c=store.open_store(tmp_path/'r.sqlite'); store.import_items(c,[i],'demo')
    before=store.get_item(c,i['task']['task_id'])
    c.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON decisions BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    with pytest.raises(sqlite3.DatabaseError,match='disk failure'):
        store.apply_decision(c,decision(i['task'],action='reject_alias',
                                      target_relation_id=i['annotation']['alias_annotations']['relations'][0]['relation_id']))
    assert store.get_item(c,i['task']['task_id'])==before
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==0


def test_alias_helper_is_copy_on_write_for_success_and_failure():
    from research.annotations.alias_decisions import apply_alias_decision
    i=alias_item(); before=deepcopy(i); rid=i['annotation']['alias_annotations']['relations'][0]['relation_id']
    provenance={'status':'human_reviewed','reviewer':'tester','decision_id':'pure','recorded_at_utc':'now','reasons':[]}
    result=apply_alias_decision(i['task'],i['annotation'],{'action':'accept_alias','target_relation_id':rid},provenance)
    result['alias_annotations']['relations'][0]['review']['reasons'].append('changed')
    assert i==before and provenance['reasons']==[]
    with pytest.raises(ValueError,match='DUPLICATE_ALIAS_PAIR'):
        apply_alias_decision(i['task'],i['annotation'],{'action':'upsert_alias',
                             'value':i['annotation']['alias_annotations']['relations'][0]},provenance)
    assert i==before
