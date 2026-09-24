from copy import deepcopy
import json

import pytest

from research.contracts import FIELDS
from research.annotations.tasks import make_tasks
from research.annotations import review_store as store
from research.data.manifest import read_jsonl


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
