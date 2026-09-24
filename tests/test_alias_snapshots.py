import json
from pathlib import Path

import pytest

from alias_fixtures import alias_item
from test_review_store import decision
from research.annotations import review_store as store
from research.data.manifest import digest, json_bytes, read_jsonl


def raw_rows(c):
    return {name:[dict(row) for row in c.execute(f'SELECT * FROM {name} ORDER BY rowid')]
            for name in ('items','decisions','metadata')}


def snapshot(tmp_path):
    i=alias_item(); c=store.open_store(tmp_path/'first.sqlite'); store.import_items(c,[i],'demo')
    rid=i['annotation']['alias_annotations']['relations'][0]['relation_id']
    payload=decision(i['task'],action='accept_alias',target_relation_id=rid)
    result=store.apply_decision(c,payload); before=raw_rows(c)
    store.export_reference(c,tmp_path/'snapshot'); c.close()
    return i,payload,result,before


def test_alias_export_restore_exact_rows_and_idempotence(tmp_path,monkeypatch):
    from research.annotations.snapshots import restore_review_snapshot
    i,payload,result,before=snapshot(tmp_path)
    def no_replay(*args): raise AssertionError('restore must not replay')
    with monkeypatch.context() as patch:
        patch.setattr(store,'apply_decision',no_replay)
        restore_review_snapshot(tmp_path/'snapshot',tmp_path/'second.sqlite')
    c=store.open_store(tmp_path/'second.sqlite')
    assert raw_rows(c)==before
    assert store.apply_decision(c,payload)==result
    with pytest.raises(store.ReviewError,match='DECISION_ID_CONFLICT'):
        store.apply_decision(c,{**payload,'reason':'other'})
    assert len(read_jsonl(tmp_path/'snapshot/alias_groups.jsonl'))==1
    assert read_jsonl(tmp_path/'snapshot/aliases.jsonl')[0]['task_id']==i['task']['task_id']
    manifest=json.loads((tmp_path/'snapshot/manifest.json').read_bytes())
    assert manifest['policy_provenance']=={'kind':'legacy_no_frozen_policy_source'}
    with pytest.raises(FileExistsError): restore_review_snapshot(tmp_path/'snapshot',tmp_path/'second.sqlite')
    c.close()


@pytest.mark.parametrize('corruption',['hash','version','role','sidecar','groups','identity','duplicate_task',
    'duplicate_decision','decision_reference','decision_revision','metadata_role','alias_null','alias_version','missing_exact'])
def test_corrupt_snapshots_fail_before_destination_creation(tmp_path,corruption):
    from research.annotations.snapshots import restore_review_snapshot
    snapshot(tmp_path); bundle=tmp_path/'snapshot'; path=bundle/'manifest.json'; manifest=json.loads(path.read_bytes())
    if corruption=='version': manifest['snapshot_schema_version']='99'
    elif corruption=='role': manifest['role']='test'
    elif corruption=='missing_exact': manifest['files']=[r for r in manifest['files'] if r['path']!='store-items.jsonl']
    else:
        name={'hash':'aliases.jsonl','sidecar':'aliases.jsonl','groups':'alias_groups.jsonl','identity':'store-items.jsonl',
              'duplicate_task':'store-items.jsonl','duplicate_decision':'decision-records.jsonl',
              'decision_reference':'decision-records.jsonl','decision_revision':'decision-records.jsonl',
              'metadata_role':'metadata.jsonl','alias_null':'store-items.jsonl','alias_version':'store-items.jsonl'}[corruption]
        rows=read_jsonl(bundle/name)
        if corruption in ('hash','sidecar'): rows[0]['decision']='not_alias'
        elif corruption=='groups': rows[0]['preferred_name']='invented'
        elif corruption=='identity':
            task=json.loads(rows[0]['task']); task['text_revision']='wrong'; rows[0]['task']=json.dumps(task)
        elif corruption in ('alias_null','alias_version'):
            annotation=json.loads(rows[0]['annotation'])
            if corruption=='alias_null': annotation['alias_annotations']=None
            else: annotation['alias_annotations']['schema_version']='99'
            rows[0]['annotation']=json.dumps(annotation)
        elif corruption=='decision_revision':
            payload=json.loads(rows[0]['payload']); payload['text_revision']='wrong'; rows[0]['payload']=json.dumps(payload)
        elif corruption=='metadata_role': next(r for r in rows if r['key']=='role')['value']='train'
        elif corruption.startswith('duplicate'): rows.append(rows[0])
        else: rows[0]['task_id']='missing'
        (bundle/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        if corruption!='hash':
            next(r for r in manifest['files'] if r['path']==name)['sha256']=digest((bundle/name).read_bytes())
    path.write_bytes(json_bytes(manifest)); destination=tmp_path/'restored.sqlite'
    with pytest.raises(ValueError): restore_review_snapshot(bundle,destination)
    assert not destination.exists()


def test_frozen_policy_and_prompt_bytes_survive_import_export_and_cli_restore(tmp_path):
    from research.cli import main
    from research.annotations.snapshots import read_policy_provenance
    from research.data.bundles import materialize_bundle
    i=alias_item(); doc={'document_id':i['task']['document_id'],'text':i['task']['text'],
        'source':'ecosystems','work_group_id':'g','split':'train','public':True,
        'access_basis':'fixture','text_license':'CC-BY-4.0'}
    materialize_bundle({'role':'train','documents':[doc],'heldout':{}},tmp_path/'train')
    original=tmp_path/'source-policy.md'
    original.write_bytes((Path(__file__).resolve().parents[1]/'annotations/scibert-v2/policy-2.1.md').read_bytes())
    main(['annotate','prepare','--bundle',str(tmp_path/'train'),'--policy',str(original),
          '--output',str(tmp_path/'tasks')])
    task_manifest=json.loads((tmp_path/'tasks/manifest.json').read_bytes())
    expected={r['path']:(tmp_path/'tasks'/r['path']).read_bytes()
              for r in [task_manifest['policy_source'],*task_manifest['prompt_sources'].values()]}
    original.write_text('Changed original policy must never be read again.')
    for record in task_manifest['prompt_sources'].values(): record['source_path']='/missing/mutable-original-prompt.md'
    (tmp_path/'tasks/manifest.json').write_bytes(json_bytes(task_manifest))
    task=read_jsonl(tmp_path/'tasks/tasks.jsonl')[0]; annotation=i['annotation']
    annotation.update({key:task[key] for key in ('task_id','document_id','text_revision','policy_version')})
    annotation['annotator']['prompt_hash']=task_manifest['prompt_sources']['annotate']['sha256']
    replies=tmp_path/'replies'; replies.mkdir(); (replies/'reply.json').write_text(json.dumps(annotation))
    main(['annotate','import','--tasks',str(tmp_path/'tasks'),'--replies',str(replies),
          '--output',str(tmp_path/'reference')])
    reference=tmp_path/'reference'; meta=json.loads((reference/'manifest.json').read_bytes())
    provenance=read_policy_provenance(reference,meta)
    c=store.open_store(tmp_path/'frozen.sqlite')
    store.import_items(c,read_jsonl(reference/'items.jsonl'),'train',provenance)
    before=raw_rows(c); store.export_reference(c,tmp_path/'export'); c.close()
    for name,payload in expected.items():
        assert (reference/name).read_bytes()==payload
        assert (tmp_path/'export'/name).read_bytes()==payload
    assert main(['annotate','restore','--bundle',str(tmp_path/'export'),'--store',str(tmp_path/'restored.sqlite')])==0
    c=store.open_store(tmp_path/'restored.sqlite'); assert raw_rows(c)==before; c.close()
    (tmp_path/'export/policy.md').write_text('corrupt frozen policy')
    with pytest.raises(ValueError,match='missing or changed artifact'):
        main(['annotate','restore','--bundle',str(tmp_path/'export'),'--store',str(tmp_path/'bad.sqlite')])
    assert not (tmp_path/'bad.sqlite').exists()


def test_restore_legacy_without_exact_rows_is_explicitly_unsupported(tmp_path):
    from research.annotations.snapshots import restore_review_snapshot
    bundle=tmp_path/'old'; bundle.mkdir(); (bundle/'manifest.json').write_text(json.dumps({'role':'demo','files':[]}))
    with pytest.raises(ValueError,match='SNAPSHOT_RESTORE_UNSUPPORTED'):
        restore_review_snapshot(bundle,tmp_path/'restored.sqlite')
    assert not (tmp_path/'restored.sqlite').exists()


def test_restore_failed_insert_rolls_back_and_removes_new_database(tmp_path,monkeypatch):
    import sqlite3
    from research.annotations.snapshots import restore_review_snapshot
    snapshot(tmp_path); original_open=store.open_store
    def failing_open(path):
        c=original_open(path)
        c.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON decisions BEGIN SELECT RAISE(ABORT,'disk failure'); END")
        return c
    monkeypatch.setattr(store,'open_store',failing_open)
    destination=tmp_path/'restored.sqlite'
    with pytest.raises(sqlite3.DatabaseError,match='disk failure'):
        restore_review_snapshot(tmp_path/'snapshot',destination)
    assert not destination.exists()
    assert not Path(str(destination)+'-wal').exists()
    assert not Path(str(destination)+'-shm').exists()


def test_legacy_absence_and_raw_json_strings_survive_snapshot(tmp_path):
    from research.annotations.snapshots import restore_review_snapshot
    from test_review_store import item
    i=item(); c=store.open_store(tmp_path/'legacy.sqlite'); store.import_items(c,[i],'demo')
    task_id=i['task']['task_id']
    row=c.execute('SELECT task,annotation FROM items WHERE task_id=?',(task_id,)).fetchone()
    c.execute('UPDATE items SET task=?,annotation=? WHERE task_id=?',(
        json.dumps(json.loads(row['task']),indent=3,ensure_ascii=False),
        json.dumps(json.loads(row['annotation']),indent=4,ensure_ascii=False),task_id))
    c.execute("INSERT INTO metadata VALUES ('custom','untouched metadata')")
    before=raw_rows(c); store.export_reference(c,tmp_path/'snapshot'); c.close()
    restore_review_snapshot(tmp_path/'snapshot',tmp_path/'restored.sqlite')
    c=store.open_store(tmp_path/'restored.sqlite'); assert raw_rows(c)==before
    assert 'alias_annotations' not in store.get_item(c,task_id)['annotation']
    assert read_jsonl(tmp_path/'snapshot/aliases.jsonl')==[]
    assert read_jsonl(tmp_path/'snapshot/alias_groups.jsonl')==[]
    c.close()


@pytest.mark.parametrize('other_task_advanced',[False,True])
@pytest.mark.parametrize('mutation',['annotation','status','json_format'])
def test_snapshot_startup_rejects_unchanged_task_state_drift(tmp_path,other_task_advanced,mutation):
    from test_review_store import item
    from research.annotations.snapshots import restore_review_snapshot, validate_snapshot_store
    first=alias_item(); second=item('Another task with no software.')
    c=store.open_store(tmp_path/'first.sqlite'); store.import_items(c,[first,second],'demo')
    rid=first['annotation']['alias_annotations']['relations'][0]['relation_id']
    store.apply_decision(c,decision(first['task'],action='accept_alias',target_relation_id=rid))
    store.export_reference(c,tmp_path/'snapshot'); c.close()
    restored=tmp_path/'restored.sqlite'; restore_review_snapshot(tmp_path/'snapshot',restored)
    c=store.open_store(restored)
    if other_task_advanced:
        store.apply_decision(c,decision(second['task'],'later-other-task'))
    validate_snapshot_store(tmp_path/'snapshot',restored)
    tid=first['task']['task_id']
    if mutation=='status':
        c.execute("UPDATE items SET status='reviewed' WHERE task_id=?",(tid,))
    else:
        annotation=store.get_item(c,tid)['annotation']
        if mutation=='annotation':
            annotation['alias_annotations']['relations'][0].update(decision='not_alias',preferred_mention_id=None)
        c.execute('UPDATE items SET annotation=? WHERE task_id=?',(
            json.dumps(annotation,indent=2 if mutation=='json_format' else None),tid))
    before=raw_rows(c)
    with pytest.raises(ValueError,match='SNAPSHOT_STORE_STATE_MISMATCH'):
        validate_snapshot_store(tmp_path/'snapshot',restored)
    assert raw_rows(c)==before
    c.close()
