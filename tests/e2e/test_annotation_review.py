import threading
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import expect

from alias_fixtures import alias_item
from research.annotations.aliases import alias_relation_id
from research.annotations.tasks import make_tasks, make_region_task
from research.annotations.review_server import create_server
from research.contracts import occurrence_id
from research.data.manifest import digest, json_bytes, read_jsonl, write_jsonl, write_once


def select_passage(page,start,end):
    page.locator('#passage').evaluate("""(el, span) => {
      const nodes=[]; const walk=document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      while (walk.nextNode()) nodes.push(walk.currentNode);
      const position=offset => { let count=0; for (const node of nodes) {
        const length=Array.from(node.textContent).length;
        if (offset <= count+length) return [node, Array.from(node.textContent).slice(0,offset-count).join('').length];
        count+=length;
      }};
      const r=document.createRange(), a=position(span[0]), b=position(span[1]);
      r.setStart(a[0],a[1]); r.setEnd(b[0],b[1]);
      const s=window.getSelection(); s.removeAllRanges(); s.addRange(r);
    }""",[start,end])


@pytest.fixture
def review_app(tmp_path,request):
    variant=getattr(request,'param',None)
    partial=variant=='partial_evidence'
    text='We used X. X was excellent.' if partial else '😀 We used NumPy.'
    task=make_tasks({'document_id':'emoji-demo','text':text, 'source':'synthetic-contract-fixture','split':'demo'},
                    {'policy_version':'scibert-poc-2.0','policy_hash':'a'*64})[0]
    task['whole_passage_audit']=True
    annotation={'occurrences':[],'covered_regions':[],'unresolved_regions':[task['annotation_region']],
                'status':'partial','annotation_revision':1,'review_status':'synthetic_fixture'}
    if partial:
        annotation['occurrences']=[{'schema_version':'2.0','document_id':task['document_id'],'text_revision':task['text_revision'],
             'name':'X','name_span':{'start':8,'end':9},'context_sentence':text,'context_span':task['context_span'],
             'context_kind':'paragraph','version_links':[],'version_status':'absent','intents':['used'],'sentiment':'positive',
             'known':{'software':True,'versions':True,'created':False,'used':True,'shared':False,'sentiment':True},
             'evidence':{'intents':[{'start':0,'end':10}],'sentiment':[{'start':11,'end':26}]},'review':{'status':'unreviewed','reasons':[]}}]
    item={'task':task,'annotation':annotation}
    if variant in ('alias','alias_new','owned_context','fixture-negative','navigation','alias_pair'):
        item=alias_item('design-negative' if variant=='fixture-negative' else 'design-explicit-icekat')
        if variant=='alias_new':
            item['annotation']['alias_annotations']['relations']=[]
        if variant=='owned_context':
            original=deepcopy(item)
            prefix='😀 ContextTool appears nearby. '
            full_text=prefix+original['task']['text']
            shift=len(prefix)
            policy={'policy_version':'scibert-poc-2.1','policy_hash':'a'*64,'alias_schema_version':'1.0'}
            task=make_region_task({'document_id':original['task']['document_id'],'text':full_text,
                                   'source':'synthetic-contract-fixture','split':'demo'},policy,
                                  {'start':shift,'end':len(full_text)},
                                  {'start':0,'end':len(full_text)},region_kind='sentence')
            task['whole_passage_audit']=True
            item['task']=task
            item['annotation'].update(task_id=task['task_id'],text_revision=task['text_revision'],
                                      covered_regions=[{**task['annotation_region'],'status':'complete',
                                                        'fields':dict.fromkeys(('software','versions','created','used','shared','sentiment'),True)}])
            for occurrence in item['annotation']['occurrences']:
                occurrence['text_revision']=task['text_revision']
                occurrence['name_span']={key:value+shift for key,value in occurrence['name_span'].items()}
                occurrence['mention_id']=occurrence_id(task['document_id'],task['text_revision'],
                                                        occurrence['name_span']['start'],occurrence['name_span']['end'])
                occurrence['context_span']={'start':0,'end':len(full_text)}
                occurrence['context_sentence']=full_text
                for spans in occurrence['evidence'].values():
                    for span in spans:
                        span['start']+=shift;span['end']+=shift
            relation=item['annotation']['alias_annotations']['relations'][0]
            relation['text_revision']=task['text_revision']
            relation['member_mention_ids']=sorted(o['mention_id'] for o in item['annotation']['occurrences'])
            relation['preferred_mention_id']=next(o['mention_id'] for o in item['annotation']['occurrences'] if o['name']=='ICEKAT')
            relation['relation_id']=alias_relation_id(task['document_id'],task['text_revision'],relation['member_mention_ids'])
            relation['evidence_spans']=[{'start':shift,'end':len(full_text)}]
    bundle=tmp_path/'bundle';bundle.mkdir()
    items=[item]
    if variant=='navigation':
        items.append(alias_item('design-negative'))
    if variant=='alias_pair':
        items.append(alias_item('design-unknown-repeated-name'))
    write_jsonl(bundle/'items.jsonl',items)
    write_once(bundle/'manifest.json',json_bytes({'role':'demo','files':[{'path':'items.jsonl','sha256':digest((bundle/'items.jsonl').read_bytes())}]}))
    database=tmp_path/'review.sqlite'
    app=create_server(bundle,database,'127.0.0.1',0,ui='legacy')
    thread=threading.Thread(target=app.serve_forever,daemon=True);thread.start()
    yield app, bundle, database
    app.shutdown();thread.join();app.server_close()


@pytest.mark.parametrize('review_app',['alias'],indirect=True)
def test_alias_reveal_accept_and_independent_review(page,review_app):
    app,_,_=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    assert not page.locator('#alias-panel').is_visible()
    page.get_by_role('button',name='Show proposed labels').click()
    page.locator('#alias-panel').wait_for(state='visible')
    assert 'ICEKAT' in page.locator('#alias-panel').inner_text()
    page.get_by_role('button',name='Open alias relation').click()
    page.locator('#alias-reason').fill('Checked the explicit full-name definition')
    page.get_by_role('button',name='Accept alias').click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    item=page.request.get(url+'/api/tasks/'+tid).json()
    assert item['annotation']['alias_annotations']['relations'][0]['review']['status']=='human_reviewed'
    assert 'human_passage_review' not in item['annotation']
    assert all(o['review']['status']!='human_reviewed' for o in item['annotation']['occurrences'])


@pytest.mark.parametrize('review_app',['alias_new'],indirect=True)
def test_alias_relation_create_reject_unresolve_remove_and_export(page,review_app):
    app,_,database=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.locator('#alias-left').select_option(label='Interactive Continuous Enzyme Kinetics Analysis Tool')
    page.locator('#alias-right').select_option(label='ICEKAT')
    select_passage(page,0,75)
    page.get_by_role('button',name='Use selection as alias evidence').click()
    assert page.locator('#alias-evidence-start').input_value()=='0'
    assert page.locator('#alias-evidence-end').input_value()=='75'
    page.locator('#alias-reason').fill('Explicit definition connects the names')
    page.get_by_role('button',name='Save alias relation').click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    relation=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['alias_annotations']['relations'][0]
    assert relation['decision']=='alias'
    assert relation['evidence_spans']==[{'start':0,'end':75}]
    page.get_by_role('button',name='Open alias relation').click()
    page.locator('#alias-reason').fill('These are distinct tools')
    page.get_by_role('button',name='Not aliases').click()
    page.get_by_text('Saved revision 3',exact=True).wait_for()
    relation=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['alias_annotations']['relations'][0]
    assert relation['decision']=='not_alias' and relation['preferred_mention_id'] is None
    page.get_by_role('button',name='Open alias relation').click()
    page.locator('#alias-reason').fill('Need more source evidence')
    page.get_by_role('button',name='Mark alias unresolved').click()
    page.get_by_text('Saved revision 4',exact=True).wait_for()
    relation=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['alias_annotations']['relations'][0]
    assert relation['decision']=='unresolved'
    page.get_by_role('button',name='Export JSONL').click()
    page.locator('#export-result').filter(has_text='provisional').wait_for()
    exports=list(database.parent.glob('exports/*/aliases.jsonl'))
    assert len(exports)==1
    assert read_jsonl(exports[0])[0]['decision']=='unresolved'
    assert read_jsonl(exports[0].with_name('alias_groups.jsonl'))==[]
    page.get_by_role('button',name='Open alias relation').click()
    page.locator('#alias-reason').fill('Remove the uncertain pair')
    page.get_by_role('button',name='Remove alias relation').click()
    page.get_by_text('Saved revision 5',exact=True).wait_for()
    assert page.request.get(url+'/api/tasks/'+tid).json()['annotation']['alias_annotations']['relations']==[]


@pytest.mark.parametrize('review_app',['alias_pair'],indirect=True)
def test_alias_draft_resets_when_opening_another_alias_task(page,review_app):
    app,_,database=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url)
    page.locator('.queue-item').filter(has_text='design-explicit-icekat').get_by_role('button',name='Open passage').click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='Open alias relation').click()
    page.locator('#alias-type').select_option('explicit_alternative_name')
    page.locator('#alias-decision').select_option('not_alias')
    page.locator('#alias-evidence-start').fill('0')
    page.locator('#alias-evidence-end').fill('70')
    page.locator('#alias-reason').fill('Draft from the ICEKAT task')
    page.locator('.queue-item').filter(has_text='design-unknown-repeated-name').get_by_role('button',name='Open passage').click()
    page.get_by_role('button',name='Show proposed labels').click()
    expect(page.locator('#alias-type')).to_have_value('abbreviation')
    expect(page.locator('#alias-decision')).to_have_value('alias')
    expect(page.locator('#alias-evidence-start')).to_have_value('')
    expect(page.locator('#alias-evidence-end')).to_have_value('')
    expect(page.locator('#alias-reason')).to_have_value('')
    with sqlite3.connect(database) as connection:
        before=connection.execute('SELECT count(*) FROM decisions').fetchone()[0]
    page.get_by_role('button',name='Accept alias').click()
    page.get_by_text('Open an alias relation first.',exact=True).wait_for()
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT count(*) FROM decisions').fetchone()[0]==before
    page.get_by_role('button',name='Open alias relation').click()
    expect(page.locator('#alias-type')).to_have_value('explicit_alternative_name')
    expect(page.locator('#alias-decision')).to_have_value('unresolved')
    expect(page.locator('#alias-evidence-start')).to_have_value('0')
    expect(page.locator('#alias-evidence-end')).to_have_value('73')


@pytest.mark.parametrize('review_app',['alias_new'],indirect=True)
def test_keyboard_activation_uses_current_alias_evidence_selection(page,review_app):
    app,_,_=review_app;page.goto(f'http://127.0.0.1:{app.server_port}')
    page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    select_passage(page,0,75)
    page.get_by_role('button',name='Use selection as alias evidence').focus()
    page.keyboard.press('Enter')
    expect(page.locator('#alias-evidence-start')).to_have_value('0')
    expect(page.locator('#alias-evidence-end')).to_have_value('75')


@pytest.mark.parametrize('review_app',['alias'],indirect=True)
def test_linked_span_edit_rejected_no_change_save_keeps_alias_and_stale_tab(page,browser,review_app):
    app,_,_=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    stale=browser.new_page();stale.goto(url);stale.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='ICEKAT (67:73)').click()
    page.locator('#name-start').fill('68')
    page.locator('#reason').fill('Try moving a linked endpoint')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    expect(page.locator('#message')).to_have_text(
        'Remove the alias relations referencing this occurrence before changing its span or deleting it.')
    page.locator('#name-start').fill('67')
    page.locator('#reason').fill('No span change; preserve relation')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    assert len(page.request.get(url+'/api/tasks/'+tid).json()['annotation']['alias_annotations']['relations'])==1
    stale.get_by_role('button',name='Show proposed labels').click()
    stale.get_by_role('button',name='Open alias relation').click()
    stale.locator('#alias-reason').fill('Stale alias review')
    stale.get_by_role('button',name='Accept alias').click()
    stale.get_by_text('STALE_REVISION',exact=False).wait_for()
    stale.close()


@pytest.mark.parametrize('review_app',['fixture-negative'],indirect=True)
def test_negative_owned_sentence_explains_empty_proposals(page,review_app):
    app,_,_=review_app;page.goto(f'http://127.0.0.1:{app.server_port}')
    page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    assert page.locator('#occurrences').inner_text()=='No software mentions proposed in this owned sentence. Check for missed names before finishing the name audit.'


def test_missed_emoji_name_clears_empty_proposals(page,review_app):
    app,_,_=review_app;page.goto(f'http://127.0.0.1:{app.server_port}')
    page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    assert page.locator('#occurrences').inner_text()=='No software mentions proposed in this owned region. Check for missed names before finishing the name audit.'
    select_passage(page,10,15)
    page.get_by_role('button',name='Add selected name').click()
    page.locator('#intent-used').check()
    page.locator('#version-state').select_option('absent')
    page.locator('#evidence-start').fill('2');page.locator('#evidence-end').fill('16')
    page.locator('#reason').fill('Missed NumPy name')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    assert 'No software mentions proposed' not in page.locator('#occurrences').inner_text()


@pytest.mark.parametrize('review_app',['owned_context'],indirect=True)
def test_owned_context_legend_stays_outside_exact_source_text(page,review_app):
    app,_,_=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    source=page.request.get(url+'/api/tasks/'+tid).json()['task']['text']
    legend=page.locator('#passage-legend')
    expect(legend).to_be_visible()
    expect(legend).to_contain_text('Owned text: annotate names here')
    expect(legend).to_contain_text('Surrounding context: evidence only')
    assert page.locator('#passage').text_content()==source
    page.get_by_role('button',name='Show proposed labels').click()
    expect(legend).to_be_visible()
    assert page.locator('#passage').text_content()==source


@pytest.mark.parametrize('review_app',['owned_context'],indirect=True)
def test_owned_context_nested_range_preserves_text_offsets_and_rejects_context_name(page,review_app):
    app,_,database=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    task=page.request.get(url+'/api/tasks/'+tid).json()['task']
    assert page.locator('#passage').text_content()==task['text']
    assert page.locator('#passage .owned').count()>0
    assert page.locator('#passage .context').count()>0
    select_passage(page,43,95)
    page.get_by_role('button',name='Add selected name').click()
    assert page.locator('#name-start').input_value()=='43'
    assert page.locator('#name-end').input_value()=='95'
    page.locator('#passage').evaluate("""(el) => {
      const node=el.querySelector('.context').firstChild, r=document.createRange();
      r.setStart(node,3); r.setEnd(node,14);
      const s=window.getSelection(); s.removeAllRanges(); s.addRange(r);
    }""")
    with sqlite3.connect(database) as connection:
        before=connection.execute('SELECT count(*) FROM decisions').fetchone()[0]
    page.get_by_role('button',name='Add selected name').click()
    page.get_by_text('Select a name inside the owned region; surrounding text is context only.',exact=True).wait_for()
    assert page.request.get(url+'/api/tasks/'+tid).json()['annotation_revision']==1
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT count(*) FROM decisions').fetchone()[0]==before


@pytest.mark.parametrize('review_app',['navigation'],indirect=True)
def test_alt_arrow_navigation_and_keyboard_alias_open(page,review_app):
    app,_,_=review_app;page.goto(f'http://127.0.0.1:{app.server_port}')
    page.get_by_role('button',name='Open passage').first.click()
    expect(page.locator('#source-title')).not_to_have_text('')
    first=page.locator('#source-title').inner_text()
    page.keyboard.press('Alt+ArrowRight')
    expect(page.locator('#source-title')).not_to_have_text(first)
    assert page.locator('#source-title').inner_text()!=first
    page.keyboard.press('Alt+ArrowLeft')
    expect(page.locator('#source-title')).to_have_text(first)
    assert page.locator('#source-title').inner_text()==first
    page.locator('.queue-item').filter(has_text='design-explicit-icekat').get_by_role('button',name='Open passage').click()
    page.get_by_role('heading',name='design-explicit-icekat').wait_for()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='Open alias relation').focus()
    page.keyboard.press('Enter')
    assert page.locator('#alias-form').is_visible()


@pytest.mark.parametrize('review_app',['owned_context'],indirect=True)
def test_synthetic_alias_layout_desktop_and_mobile(page,review_app,tmp_path):
    app,_,_=review_app;page.goto(f'http://127.0.0.1:{app.server_port}')
    page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='Open alias relation').click()
    for width,height,name in ((1365,900,'desktop'),(390,844,'mobile')):
        page.set_viewport_size({'width':width,'height':height})
        expect(page.locator('#alias-form')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        path=tmp_path/f'alias-{name}.png'
        page.screenshot(path=str(path),full_page=True)
        print(f'{name} screenshot: {path}')


def test_browser_unicode_save_restart_export_and_stale_tab(page, browser, review_app):
    app,bundle,database=review_app
    url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url)
    page.get_by_role('button',name='Open passage').first.click()
    stale=browser.new_page();stale.goto(url);stale.get_by_role('button',name='Open passage').first.click()
    # Browser ranges use UTF-16; production UI must store code-point boundaries.
    select_passage(page,10,15)
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
    other=create_server(bundle,database,'127.0.0.1',0,ui='legacy')
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


@pytest.mark.parametrize('review_app',['partial_evidence'],indirect=True)
def test_unrelated_ui_save_preserves_unknown_intents_and_distinct_evidence(page,review_app):
    app,_,_=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='X (8:9)').click()
    page.locator('#reason').fill('Checked name boundaries; preserve unresolved attributes')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    queue=page.request.get(url+'/api/queue').json()
    tid=queue['items'][0]['task_id']
    row=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['occurrences'][0]
    assert row['known']['created'] is False and row['known']['shared'] is False
    assert row['evidence']=={'intents':[{'start':0,'end':10}],'sentiment':[{'start':11,'end':26}]}


@pytest.mark.parametrize('review_app',['partial_evidence'],indirect=True)
def test_review_separate_evidence_arrays_and_explicit_negative_intent(page,review_app):
    app,_,_=review_app;url=f'http://127.0.0.1:{app.server_port}'
    page.goto(url);page.get_by_role('button',name='Open passage').first.click()
    page.get_by_role('button',name='Show proposed labels').click()
    page.get_by_role('button',name='X (8:9)').click()
    page.locator('#known-created').select_option('known')
    page.locator('#evidence-start').fill('3');page.locator('#evidence-end').fill('7')
    page.get_by_role('button',name='Add evidence span',exact=True).click()
    page.locator('#evidence-field').select_option('sentiment')
    page.locator('#evidence-start').fill('17');page.locator('#evidence-end').fill('26')
    page.locator('#reason').fill('Known created-negative; add independent supporting spans')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 2',exact=True).wait_for()
    tid=page.request.get(url+'/api/queue').json()['items'][0]['task_id']
    row=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['occurrences'][0]
    assert row['known']['created'] is True and row['known']['shared'] is False
    assert row['intents']==['used']
    assert row['evidence']=={'intents':[{'start':0,'end':10},{'start':3,'end':7}],
                             'sentiment':[{'start':11,'end':26},{'start':17,'end':26}]}
    page.get_by_role('button',name='X (8:9)').click()
    page.locator('#reason').fill('No further changes')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 3',exact=True).wait_for()
    assert page.request.get(url+'/api/tasks/'+tid).json()['annotation']['occurrences'][0]['evidence']==row['evidence']
    page.get_by_role('button',name='X (8:9)').click()
    page.get_by_role('button',name='Remove sentiment evidence 2',exact=True).click()
    page.locator('#intent-mentioned').check()
    page.locator('#reason').fill('Check explicit mentioned-only and remove one sentiment span')
    page.get_by_role('button',name='Save occurrence',exact=True).click()
    page.get_by_text('Saved revision 4',exact=True).wait_for()
    saved=page.request.get(url+'/api/tasks/'+tid).json()['annotation']['occurrences'][0]
    assert saved['intents']==['mentioned']
    assert all(saved['known'][k] for k in ('created','used','shared'))
    assert saved['evidence']['intents']==row['evidence']['intents']
    assert saved['evidence']['sentiment']==[{'start':11,'end':26}]
