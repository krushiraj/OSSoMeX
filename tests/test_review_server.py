import pytest
import json
from threading import Thread
from urllib.request import Request, urlopen

from alias_fixtures import alias_item
from test_review_store import decision
from research.annotations import review_store as store
from research.data.manifest import digest, json_bytes, write_jsonl

from research.annotations import review_server as server


def test_loopback_only_origin_host_and_csrf():
    for host in ('0.0.0.0','example.org','192.168.1.10'):
        with pytest.raises(ValueError):server.check_bind(host)
    server.check_bind('127.0.0.1')
    good={'Host':'127.0.0.1:8765','Origin':'http://127.0.0.1:8765','X-CSRF-Token':'secret'}
    assert server.authorized_request(good,8765,'secret',write=True)
    for patch in [{'Host':'evil.test:8765'},{'Origin':'https://evil.test'},{'X-CSRF-Token':'wrong'}, {'Origin':None}]:
        assert not server.authorized_request({**good,**patch},8765,'secret',write=True)


def test_alias_http_capability_decision_and_restart(tmp_path):
    i=alias_item(); bundle=tmp_path/'bundle'; bundle.mkdir(); write_jsonl(bundle/'items.jsonl',[i])
    (bundle/'manifest.json').write_bytes(json_bytes({'role':'demo','files':[
        {'path':'items.jsonl','sha256':digest((bundle/'items.jsonl').read_bytes())}]}))
    path=tmp_path/'review.sqlite'; rid=i['annotation']['alias_annotations']['relations'][0]['relation_id']
    payload=decision(i['task'],action='accept_alias',target_relation_id=rid)
    result=None
    for restarted in (False,True):
        http=server.create_server(bundle,path,port=0); thread=Thread(target=http.serve_forever,daemon=True); thread.start()
        base=f'http://127.0.0.1:{http.server_port}'
        try:
            with urlopen(base+'/api/session') as response: session=json.load(response)
            assert session['capabilities']=={'aliases':True} and session['alias_schema_version']=='1.0'
            with urlopen(base+'/api/tasks/'+i['task']['task_id']) as response: current=json.load(response)
            assert len(current['alias_groups'])==1
            if restarted:
                assert current['alias_groups'][0]['quality']=='human_reviewed'
                assert current['annotation']['alias_annotations']['relations'][0]['review']['reviewer']=='Krushi'
            request=Request(base+'/api/decisions',data=json.dumps(payload).encode(),headers={
                'Content-Type':'application/json','Origin':base,'X-CSRF-Token':session['csrf_token']})
            with urlopen(request) as response: saved=json.load(response)
            if restarted: assert saved==result
            else: result=saved
        finally: http.shutdown(); thread.join(); http.server_close()
    c=store.open_store(path)
    assert 'alias_groups' not in store.get_item(c,i['task']['task_id'])
    assert c.execute('SELECT count(*) FROM decisions').fetchone()[0]==1
    c.close()


def test_restored_snapshot_serves_without_original_bundle_and_preserves_history(tmp_path):
    from test_alias_snapshots import raw_rows
    from research.annotations.snapshots import restore_review_snapshot
    i=alias_item(); source_bundle=tmp_path/'source-bundle'; source_bundle.mkdir()
    write_jsonl(source_bundle/'items.jsonl',[i])
    (source_bundle/'manifest.json').write_bytes(json_bytes({'role':'demo','files':[
        {'path':'items.jsonl','sha256':digest((source_bundle/'items.jsonl').read_bytes())}]}))
    initial=server.create_server(source_bundle,tmp_path/'first.sqlite',port=0); initial.server_close()
    c=store.open_store(tmp_path/'first.sqlite')
    rid=i['annotation']['alias_annotations']['relations'][0]['relation_id']
    payload=decision(i['task'],action='accept_alias',target_relation_id=rid)
    result=store.apply_decision(c,payload); before=raw_rows(c)
    bundle=tmp_path/'snapshot'; path=tmp_path/'restored.sqlite'
    store.export_reference(c,bundle); c.close()
    restore_review_snapshot(bundle,path)
    (source_bundle/'items.jsonl').unlink(); (source_bundle/'manifest.json').unlink(); source_bundle.rmdir()
    (tmp_path/'first.sqlite').unlink()
    assert not source_bundle.exists() and not (tmp_path/'first.sqlite').exists()
    for restarted in (False,True):
        http=server.create_server(bundle,path,port=0)
        thread=Thread(target=http.serve_forever,daemon=True); thread.start()
        base=f'http://127.0.0.1:{http.server_port}'
        try:
            with urlopen(base+'/api/session') as response: session=json.load(response)
            with urlopen(base+'/api/tasks/'+i['task']['task_id']) as response: current=json.load(response)
            assert session['role']=='demo'
            assert current['annotation_revision']==(3 if restarted else 2)
            assert len(current['alias_groups'])==(0 if restarted else 1)
            assert current['annotation']['alias_annotations']['relations'][0]['decision']==('not_alias' if restarted else 'alias')
            request=Request(base+'/api/decisions',data=json.dumps(payload).encode(),headers={
                'Content-Type':'application/json','Origin':base,'X-CSRF-Token':session['csrf_token']})
            with urlopen(request) as response: assert json.load(response)==result
            c=store.open_store(path); current_rows=raw_rows(c); c.close()
            if not restarted:
                assert current_rows==before
                request=Request(base+'/api/decisions',data=json.dumps(decision(i['task'],'new-review',2,
                    action='reject_alias',target_relation_id=rid)).encode(),headers={
                    'Content-Type':'application/json','Origin':base,'X-CSRF-Token':session['csrf_token']})
                with urlopen(request) as response: assert json.load(response)['annotation_revision']==3
            else:
                assert current_rows['decisions'][0]==before['decisions'][0]
                assert len(current_rows['decisions'])==2
        finally: http.shutdown(); thread.join(); http.server_close()


@pytest.mark.parametrize('invalid',['missing','empty','role','source','metadata','task','different_history'])
def test_snapshot_startup_rejects_missing_or_mismatched_store_without_writes(tmp_path,invalid):
    from test_alias_snapshots import raw_rows, snapshot
    from research.annotations.snapshots import restore_review_snapshot
    i,payload,result,before=snapshot(tmp_path); bundle=tmp_path/'snapshot'; path=tmp_path/'restored.sqlite'
    if invalid=='missing': pass
    elif invalid=='empty': path.touch()
    else:
        restore_review_snapshot(bundle,path); c=store.open_store(path)
        if invalid=='role': c.execute("UPDATE metadata SET value='train' WHERE key='role'")
        elif invalid=='source': c.execute("UPDATE items SET original_hash=?",('f'*64,))
        elif invalid=='metadata': c.execute("UPDATE metadata SET value='{}' WHERE key='policy_sources'")
        elif invalid=='task':
            task=json.loads(c.execute('SELECT task FROM items').fetchone()[0]); task['source']='other'
            c.execute('UPDATE items SET task=?',(json.dumps(task),))
        else:
            c.execute('DROP TRIGGER decisions_no_update')
            changed={**payload,'reason':'rewritten history'}
            c.execute('UPDATE decisions SET payload=?',(json.dumps(changed),))
        before=raw_rows(c); c.close()
    with pytest.raises(ValueError):
        http=server.create_server(bundle,path,port=0)
        http.server_close()
    if invalid=='missing': assert not path.exists()
    elif invalid=='empty': assert path.read_bytes()==b''
    else:
        c=store.open_store(path); assert raw_rows(c)==before; c.close()


@pytest.mark.parametrize('signal',['version','requested','layer'])
@pytest.mark.parametrize('metadata',['missing','downgraded'])
def test_review_bundle_requires_frozen_policy_before_creating_store(tmp_path,signal,metadata):
    from test_review_store import policy_required_item
    i=policy_required_item(signal); bundle=tmp_path/'bundle'; bundle.mkdir()
    write_jsonl(bundle/'items.jsonl',[i])
    manifest={'role':'train','files':[{'path':'items.jsonl','sha256':digest((bundle/'items.jsonl').read_bytes())}]}
    if metadata=='downgraded':
        manifest.update(policy={'policy_version':'scibert-poc-2.0','policy_hash':'a'*64},
                        policy_provenance={'kind':'legacy_no_frozen_policy_source'})
    (bundle/'manifest.json').write_bytes(json_bytes(manifest)); path=tmp_path/'review.sqlite'
    with pytest.raises(ValueError,match='POLICY_SNAPSHOT_REQUIRED'):
        http=server.create_server(bundle,path,port=0); http.server_close()
    assert not path.exists()
