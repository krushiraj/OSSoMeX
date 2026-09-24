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
