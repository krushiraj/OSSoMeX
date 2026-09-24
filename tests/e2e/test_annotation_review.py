import threading
import json
from pathlib import Path

import pytest

pytest.importorskip('playwright.sync_api')

from research.annotations.tasks import make_tasks
from research.annotations.review_server import create_server
from research.data.manifest import digest, json_bytes, read_jsonl, write_jsonl, write_once


@pytest.fixture
def review_app(tmp_path):
    task=make_tasks({'document_id':'emoji-demo','text':'😀 We used NumPy.', 'source':'synthetic-contract-fixture','split':'demo'},
                    {'policy_version':'scibert-poc-2.0','policy_hash':'a'*64})[0]
    task['whole_passage_audit']=True
    annotation={'occurrences':[],'covered_regions':[],'unresolved_regions':[task['annotation_region']],
                'status':'partial','annotation_revision':1,'review_status':'synthetic_fixture'}
    bundle=tmp_path/'bundle';bundle.mkdir()
    write_jsonl(bundle/'items.jsonl',[{'task':task,'annotation':annotation}])
    write_once(bundle/'manifest.json',json_bytes({'role':'demo','files':[{'path':'items.jsonl','sha256':digest((bundle/'items.jsonl').read_bytes())}]}))
    database=tmp_path/'review.sqlite'
    app=create_server(bundle,database,'127.0.0.1',0)
    thread=threading.Thread(target=app.serve_forever,daemon=True);thread.start()
    yield app, bundle, database
    app.shutdown();thread.join();app.server_close()


def test_browser_unicode_save_restart_export_and_stale_tab(page, browser, review_app):
    app,bundle,database=review_app
    url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url)
    page.get_by_role('button',name='Open passage').first.click()
    stale=browser.new_page();stale.goto(url);stale.get_by_role('button',name='Open passage').first.click()
    # Browser ranges use UTF-16; production UI must store code-point boundaries.
    page.locator('#passage').evaluate("el => { const r=document.createRange(); r.setStart(el.firstChild,11); r.setEnd(el.firstChild,16); const s=window.getSelection(); s.removeAllRanges(); s.addRange(r); }")
    page.get_by_role('button',name='Add selected name').click()
    assert page.locator('#name-start').input_value()=='10'
    assert page.locator('#name-end').input_value()=='15'
    page.locator('#intent-used').check()
    page.locator('#sentiment').select_option('not_expressed')
    page.locator('#version-state').select_option('absent')
    page.locator('#evidence-start').fill('2');page.locator('#evidence-end').fill('16')
    page.locator('#reason').fill('Missed software mention in the whole-passage audit')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    stale.locator('#passage-reason').fill('Checked this passage')
    stale.get_by_role('button',name='Finish name audit').click()
    stale.get_by_text('STALE_REVISION',exact=False).wait_for()
    stale.close()
    # Restart a second listener against the same durable store (not browser storage).
    other=create_server(bundle,database,'127.0.0.1',0)
    thread=threading.Thread(target=other.serve_forever,daemon=True);thread.start()
    try:
        page.goto(f'http://127.0.0.1:{other.server_port}')
        page.get_by_role('button',name='Open passage').first.click()
        page.get_by_role('button',name='Show proposed labels').click()
        assert page.locator('#occurrences').inner_text().find('NumPy')>=0
        page.get_by_role('button',name='Export JSONL').click()
        page.locator('#export-result').filter(has_text='provisional').wait_for()
        exports=list(database.parent.glob('exports/*/occurrences.jsonl'))
        rows=read_jsonl(exports[0]);assert rows[0]['name']=='NumPy'
        assert rows[0]['name_span']=={'start':10,'end':15}
    finally:
        other.shutdown();thread.join();other.server_close()
