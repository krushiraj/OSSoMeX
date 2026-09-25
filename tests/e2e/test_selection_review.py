import threading
from pathlib import Path

import pytest

pytest.importorskip('playwright.sync_api')

from test_annotation_review import select_passage
from review_workflow_fixtures import selection_item
from research.annotations.review_server import create_server
from research.annotations.aliases import alias_relation_id
from research.data.manifest import digest, json_bytes, write_jsonl, write_once
from playwright.sync_api import expect


@pytest.fixture
def selection_app(tmp_path, request):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    write_jsonl(bundle / 'items.jsonl', [selection_item(getattr(request, 'param', None))])
    write_once(bundle / 'manifest.json', json_bytes({'role': 'demo', 'files': [
        {'path': 'items.jsonl', 'sha256': digest((bundle / 'items.jsonl').read_bytes())}]}))
    database = tmp_path / 'review.sqlite'
    server = create_server(bundle, database, '127.0.0.1', 0, ui='legacy')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, bundle, database
    server.shutdown()
    thread.join()
    server.server_close()


def load(page, selection_app):
    server, _, _ = selection_app
    base = f'http://127.0.0.1:{server.server_port}'
    page.goto(base)
    item = page.request.get(base + '/api/queue').json()['items'][0]
    loaded = page.request.get(base + '/api/tasks/' + item['task_id']).json()
    page.evaluate('''async item => {
      window.modules = await import('/review-draft.js');
      window.selectionModule = await import('/review-selection.js');
      window.item = item; window.draft = modules.createDraft(item);
      window.act = async action => window.draft = await modules.applyDraftAction(draft, action);
    }''', loaded)
    return base, loaded


def save(page, base, ident='browser-batch'):
    envelope = page.evaluate('''id => modules.buildBatch(draft,
        {completion: 'save', reviewer: 'Krushi', decisionId: id})''', ident)
    token = page.request.get(base + '/api/session').json()['csrf_token']
    response = page.request.post(base + '/api/decisions', data=envelope,
                                 headers={'Origin': base, 'X-CSRF-Token': token})
    assert response.ok, response.text()
    return response.json()


def mount_view(page, selection_app):
    base, item = load(page, selection_app)
    page.evaluate('''async () => {
      const {mountReviewView} = await import('/review-view.js');
      document.querySelectorAll('link[rel="stylesheet"]').forEach(node=>node.remove());
      const css = document.createElement('link'); css.rel='stylesheet';css.href='/review-v2.css';document.head.append(css);
      const root = document.createElement('main'); root.id='review-root';document.body.replaceChildren(root);
      window.calls=[]; window.sessionState={saving:false}; window.queue=[];
      window.renderView=()=>view.render(draft,modules.summarizeDraft(draft),queue,sessionState);
      window.view=mountReviewView(root,{
        onAction:async action=>{calls.push(action);try {
          draft=action.type==='undo'?modules.undoDraft(draft):action.type==='redo'?modules.redoDraft(draft):await modules.applyDraftAction(draft,action);
          renderView();
        } catch(error){view.showError(error.message);}},
        onSave:(...args)=>calls.push({save:args}),onNavigate:id=>calls.push({navigate:id}),
        onFilter:value=>calls.push({filter:value}),onExport:()=>calls.push({export:true})});renderView();
    }''')
    return base, item


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_header_labels_focus_and_exact_source(page, selection_app):
    _, item = mount_view(page, selection_app)
    expect(page.locator('#review-summary')).to_contain_text('All displayed labels filled')
    expect(page.locator('#review-summary')).to_contain_text('check for missed names')
    page.get_by_role('button', name='1 software name', exact=True).click()
    expect(page.locator('#software-cards article').first).to_be_focused()
    page.get_by_role('button', name='Used', exact=True).click()
    expect(page.get_by_role('dialog')).to_contain_text('scikit-learn')
    page.get_by_role('button', name='Confirm Used', exact=True).click()
    expect(page.locator('#software-cards')).to_contain_text('Unsaved')
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_exact_name_activates_without_edit_and_proposals_are_compact(page, selection_app):
    _, item = mount_view(page, selection_app)
    expect(page.locator('.proposal-details')).not_to_have_attribute('open', '')
    page.get_by_role('button', name='4 proposals to confirm', exact=True).click()
    expect(page.locator('.proposal-details')).to_have_attribute('open', '')
    expect(page.locator('.proposal-details')).to_contain_text('Approve confirms these unchanged proposals together')
    row = item['annotation']['occurrences'][0]
    select_passage(page, row['name_span']['start'], row['name_span']['end'])
    page.locator('#passage').dispatch_event('mouseup')
    expect(page.locator('#selection-menu')).to_contain_text('Name already identified')
    expect(page.get_by_role('button', name='Link version to scikit-learn', exact=True)).to_be_visible()
    assert page.evaluate('draft.activeMentionId') == row['mention_id']
    assert page.evaluate('draft.operations') == []
    assert page.evaluate('draft.undoStack') == []
    page.keyboard.press('Escape')
    page.get_by_role('button', name='Approval scope', exact=True).click()
    expect(page.get_by_role('dialog')).to_contain_text('including unchanged proposals')
    expect(page.get_by_role('dialog')).to_contain_text('inside the owned region')
    page.keyboard.press('Escape')
    expect(page.get_by_role('button', name='Approval scope', exact=True)).to_be_focused()


def test_view_selection_capture_keyboard_and_unknown(page, selection_app):
    _, item = mount_view(page, selection_app)
    select_passage(page, 8, 20)
    page.locator('#passage').dispatch_event('mouseup')
    expect(page.locator('#selection-menu')).to_be_visible()
    page.get_by_role('button', name='Identify name', exact=True).click()
    expect(page.locator('#software-cards')).to_contain_text('Unknown sentiment')
    assert page.evaluate('draft.view.annotation.occurrences[0].name') == 'scikit-learn'
    select_passage(page, 21, 26)
    page.locator('#passage').dispatch_event('keyup', {'key':'Shift'})
    page.get_by_role('button', name='Link version to scikit-learn', exact=True).focus()
    page.keyboard.press('Escape')
    expect(page.locator('#selection-menu')).to_be_hidden()
    expect(page.locator('#passage')).to_be_focused()
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
def test_view_blind_reconciliation_clicks_and_undo(page, selection_app):
    _, item = mount_view(page, selection_app)
    expect(page.locator('#software-cards article')).to_have_count(0)
    expect(page.locator('#review-summary')).to_contain_text('Proposed names and labels are hidden')
    original = item['annotation']['occurrences'][0]
    select_passage(page, original['name_span']['start'], original['name_span']['end'] - 1)
    page.locator('#passage').dispatch_event('mouseup')
    page.get_by_role('button', name='Identify name', exact=True).click()
    page.get_by_role('button', name='Show proposed labels', exact=True).click()
    expect(page.get_by_role('button', name='Save', exact=True)).to_be_disabled()
    page.get_by_role('button', name='Use ' + original['name'], exact=True).click()
    expect(page.get_by_role('region', name='Discovery reconciliation')).to_have_count(0)
    assert page.evaluate('draft.operations[0].fields') == ['software']
    page.get_by_role('button', name='Undo', exact=True).click()
    expect(page.get_by_role('region', name='Discovery reconciliation')).to_be_visible()
    page.get_by_role('button', name='Discard discovery', exact=True).click()
    expect(page.get_by_role('region', name='Discovery reconciliation')).to_have_count(0)
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_multiple_intents_evidence_unknown_and_reasons(page, selection_app):
    _, item = mount_view(page, selection_app)
    page.get_by_role('button', name='Edit intent', exact=True).click()
    page.get_by_role('button', name='Set Shared', exact=True).click()
    expect(page.get_by_role('button', name='Used', exact=True)).to_be_visible()
    expect(page.get_by_role('button', name='Shared', exact=True)).to_be_visible()
    page.get_by_role('button', name='Shared', exact=True).click()
    expect(page.get_by_role('dialog')).to_contain_text(item['task']['text'])
    page.get_by_role('button', name='Shared unresolved', exact=True).click()
    expect(page.get_by_role('button', name='Shared unresolved', exact=True)).to_be_visible()
    expect(page.get_by_role('button', name='Approve & next', exact=True)).to_be_disabled()
    page.get_by_role('button', name='Correct name', exact=True).click()
    page.get_by_label('Correction reason').select_option('other')
    before = page.evaluate('draft.operations.length')
    page.get_by_role('button', name='Remove name', exact=True).click()
    assert page.evaluate('draft.operations.length') == before
    page.get_by_label('Optional note').fill('The selected word is not software.')
    page.get_by_role('button', name='Remove name', exact=True).click()
    assert page.evaluate('draft.operations.at(-1).reason_code') == 'other'
    expect(page.locator('#software-cards')).to_contain_text('No software names displayed')


@pytest.mark.parametrize('selection_app', ['multi'], indirect=True)
def test_view_three_member_aliases_explicit_evidence_cancel_delete(page, selection_app):
    base, item = mount_view(page, selection_app)
    text = item['task']['text']
    page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:0,end:5}});renderView();
    }''')
    for name in ['A', 'Beta']:
        start = text.index(name, 6)
        select_passage(page, start, start + len(name))
        page.locator('#passage').dispatch_event('mouseup')
        page.get_by_role('button', name='Link alias…', exact=True).click()
        page.get_by_label('Target member').select_option(label='Alpha')
        page.get_by_label('Preferred name').select_option(label='Alpha')
        if name == 'Beta':
            page.get_by_label('Relation type').select_option('explicit_alternative_name')
        expect(page.get_by_role('button', name='Stage alias link')).to_be_disabled()
        before = page.evaluate('draft.operations.length')
        page.get_by_role('button', name='Cancel', exact=True).click()
        assert page.evaluate('draft.operations.length') == before
        expect(page.locator('#passage')).to_be_focused()
        select_passage(page, start, start + len(name))
        page.locator('#passage').dispatch_event('mouseup')
        page.get_by_role('button', name='Link alias…', exact=True).click()
        page.get_by_label('Target member').select_option(label='Alpha')
        if name == 'Beta':
            page.get_by_label('Relation type').select_option('explicit_alternative_name')
        page.get_by_role('button', name='Use displayed evidence', exact=True).click()
        page.get_by_role('button', name='Stage alias link', exact=True).click()
        expect(page.get_by_role('dialog')).to_have_count(0)
    expect(page.get_by_role('region', name='Alias groups')).to_contain_text('Alpha / Beta / A')
    assert len(page.evaluate('modules.aliasGroupsForDraft(draft)[0].members')) == 3
    page.get_by_role('button', name='1 alias group', exact=True).click()
    expect(page.get_by_role('region', name='Alias groups')).to_be_focused()
    page.locator('#software-cards article').filter(has=page.get_by_role('button', name='Alpha', exact=True)).get_by_role('button', name='Correct name').click()
    expect(page.get_by_role('dialog')).to_contain_text('Alpha / A')
    expect(page.get_by_role('dialog')).to_contain_text('Alpha / Beta')
    page.get_by_role('button', name='Close', exact=True).click()
    page.get_by_role('region', name='Alias groups').get_by_role('button').first.click()
    page.get_by_role('button', name='Checked: not an alias', exact=True).click()
    expect(page.get_by_role('region', name='Alias groups')).to_contain_text('Checked non-alias')
    save(page, base, 'view-three-aliases')


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_readonly_source_issue_callbacks_and_task_switch(page, selection_app):
    _, item = mount_view(page, selection_app)
    page.evaluate('''() => {
      sessionState={readOnly:true};renderView();
    }''')
    expect(page.locator('#review-summary')).to_contain_text('Partial source coverage')
    expect(page.locator('#review-footer')).to_contain_text('Read-only review')
    page.get_by_role('button', name='Positive', exact=True).click()
    expect(page.get_by_role('button', name='Negative', exact=True)).to_be_disabled()
    page.keyboard.press('Escape')
    page.evaluate('''() => {
      sessionState={};queue=[{task_id:draft.view.task.task_id},{task_id:'next-task'}];renderView();
    }''')
    page.get_by_role('button', name='Approve & next', exact=True).click()
    assert page.evaluate('calls.at(-1)') == {'save': ['approve', 'next-task']}
    page.evaluate('''() => {
      draft.view.annotation.review_workflow={source_issues:[{issue_id:'fixture-issue',code:'broken_passage',message:'Sentence is truncated.'}]};
      draft.operations=[{action:'record_note',note:'fixture'}];renderView();
    }''')
    expect(page.get_by_role('button', name='Approve & next', exact=True)).to_be_disabled()
    expect(page.get_by_text('Source issue blocks approval')).to_be_visible()
    select_passage(page, 18, 30)
    page.locator('#passage').dispatch_event('mouseup')
    page.evaluate('''() => {draft.view.task.task_id='different';renderView();}''')
    expect(page.locator('#selection-menu')).to_be_hidden()
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('width', [1440, 390])
@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_responsive_contrast_and_screenshots(page, selection_app, width):
    page.set_viewport_size({'width': width, 'height': 1000})
    _, item = mount_view(page, selection_app)
    page.evaluate('''() => {queue=[{task_id:draft.view.task.task_id,review_summary:{workflow_status:'pending'}},{task_id:'next',review_summary:{workflow_status:'in_progress'}}];renderView();}''')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    colors = page.locator('[data-label="used"]').evaluate('''node => {
      const style=getComputedStyle(node); return {fg:style.color,bg:style.backgroundColor};
    }''')
    assert colors == {'fg': 'rgb(30, 64, 175)', 'bg': 'rgb(219, 234, 254)'}
    contrast = page.locator('[data-label="used"], [data-label="positive"]').evaluate_all(r'''nodes => {
      const luminance=color=>color.match(/\d+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=0.04045?x/12.92:((x+0.055)/1.055)**2.4).reduce((s,x,i)=>s+x*[0.2126,0.7152,0.0722][i],0);
      return nodes.map(node=>{const s=getComputedStyle(node),a=luminance(s.color),b=luminance(s.backgroundColor);return (Math.max(a,b)+0.05)/(Math.min(a,b)+0.05);});
    }''')
    assert min(contrast) >= 4.5
    page.get_by_role('button', name='Used', exact=True).focus()
    assert page.locator('[data-label="used"]').evaluate('node => getComputedStyle(node).outlineStyle') != 'none'
    shots = Path('.superpowers/sdd/2026-09-25-scibert-selection-review/screenshots')
    shots.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(shots / f'task5-{width}.png'), full_page=True)
    select_passage(page, 18, 30)
    page.locator('#passage').dispatch_event('mouseup')
    menu = page.locator('#selection-menu').bounding_box()
    assert menu['x'] >= 0 and menu['x'] + menu['width'] <= width
    page.screenshot(path=str(shots / f'task5-selection-{width}.png'), full_page=True)
    page.keyboard.press('Escape')
    expect(page.locator('#passage')).to_be_focused()
    assert page.locator('#passage').text_content() == item['task']['text']
    version = page.get_by_role('button', name='v0.17', exact=True)
    version.focus()
    page.wait_for_function('''() => {
      const focused=document.activeElement.getBoundingClientRect(),footer=document.querySelector('#review-footer').getBoundingClientRect();
      return focused.bottom < footer.top;
    }''')


def test_view_physical_drag_tab_and_escape(page, selection_app):
    _, item = mount_view(page, selection_app)
    box = page.locator('#passage').evaluate('''node => {
      const range=document.createRange();range.setStart(node.firstChild,8);range.setEnd(node.firstChild,20);
      const r=range.getBoundingClientRect();return {left:r.left,right:r.right,y:(r.top+r.bottom)/2};
    }''')
    page.mouse.move(box['left'] + 1, box['y'])
    page.mouse.down()
    page.mouse.move(box['right'] - 1, box['y'], steps=15)
    page.mouse.up()
    expect(page.locator('#selection-menu')).to_be_visible()
    page.keyboard.press('Tab')
    expect(page.get_by_role('button', name='Identify name', exact=True)).to_be_focused()
    page.keyboard.press('Enter')
    expect(page.locator('#software-cards')).to_contain_text('scikit-learn')
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_target_chooser_version_and_live_confirmed_state(page, selection_app):
    base, item = mount_view(page, selection_app)
    start = item['task']['text'].index('v0.17')
    select_passage(page, start, start + 5)
    page.locator('#passage').dispatch_event('mouseup')
    page.get_by_role('button', name='Link version…', exact=True).click()
    expect(page.get_by_role('dialog')).to_contain_text('Choose the software name')
    page.get_by_role('dialog').get_by_role('button', name='scikit-learn', exact=True).click()
    assert page.evaluate('draft.operations') == []  # Existing version edge is a no-op.
    page.get_by_role('button', name='Used', exact=True).click()
    page.get_by_role('button', name='Confirm Used', exact=True).click()
    save(page, base, 'view-confirm-used')
    loaded = page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()
    page.evaluate('''item => {draft=modules.createDraft(item);renderView();}''', loaded)
    expect(page.locator('#software-cards')).to_contain_text('Confirmed')
    expect(page.locator('#software-cards')).not_to_contain_text('Unsaved')


@pytest.mark.parametrize('selection_app', ['context'], indirect=True)
def test_view_context_boundaries_are_visible_without_changing_text(page, selection_app):
    _, item = mount_view(page, selection_app)
    expect(page.locator('.source-context')).to_have_text('ContextTool. ')
    expect(page.locator('.reading-hint')).to_contain_text('surrounding context is for evidence only')
    expect(page.locator('.source-guide')).to_contain_text('Review this region: unshaded text. Context: shaded text')
    expect(page.locator('.source-identity')).to_have_text('Document: ' + item['task']['document_id'])
    assert page.locator('#passage').text_content() == item['task']['text']


@pytest.mark.parametrize('partial', [True, False])
@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_approved_state_and_clean_next(page, selection_app, partial):
    base, item = mount_view(page, selection_app)
    # Use real server approval metadata, then vary source coverage in the render-only fixture.
    envelope = page.evaluate("modules.buildBatch(draft,{completion:'approve',reviewer:'Krushi',decisionId:'view-approval'})")
    token = page.request.get(base + '/api/session').json()['csrf_token']
    response = page.request.post(base + '/api/decisions', data=envelope,
                                 headers={'Origin': base, 'X-CSRF-Token': token})
    assert response.ok, response.text()
    loaded = page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()
    loaded['review_summary']['partial_source_coverage'] = partial
    page.evaluate('''item => {draft=modules.createDraft(item);queue=[{task_id:item.task.task_id},{task_id:'next'}];renderView();}''', loaded)
    expect(page.locator('#review-summary')).to_contain_text('Labels and name check approved')
    if partial:
        expect(page.locator('#review-summary')).to_contain_text('partial source coverage remains')
    expect(page.locator('#review-summary')).not_to_contain_text('check for missed names')
    expect(page.get_by_role('button', name='Save', exact=True)).to_be_disabled()
    page.get_by_role('button', name='Save & next', exact=True).click()
    assert page.evaluate('calls.at(-1)') == {'save': ['save', 'next']}


def test_view_empty_source_blocker_can_skip_without_approval(page, selection_app):
    mount_view(page, selection_app)
    page.evaluate('''() => {
      draft.view.annotation.review_workflow={source_issues:[{issue_id:'source',code:'broken_passage',message:'Missing source text.'}]};
      draft.base.review_summary={...draft.base.review_summary,source_issues:draft.view.annotation.review_workflow.source_issues,can_approve:false,workflow_status:'source_issue'};
      queue=[{task_id:draft.view.task.task_id},{task_id:'next',document_id:'next-document'}];renderView();
    }''')
    expect(page.locator('#review-summary')).to_contain_text('Source issue: repair needed before approval')
    expect(page.locator('#review-summary')).not_to_contain_text('All displayed labels filled')
    expect(page.get_by_role('button', name='Approve & next', exact=True)).to_be_disabled()
    page.get_by_role('button', name='Save & next', exact=True).click()
    assert page.evaluate('calls.at(-1)') == {'save': ['save', 'next']}
    page.get_by_role('button', name='Open passage 2: next-document', exact=True).click()
    assert page.evaluate('calls.at(-1)') == {'navigate': 'next'}


@pytest.mark.parametrize('selection_app', ['complete_proposal'], indirect=True)
def test_view_all_rendered_label_colors_meet_contrast(page, selection_app):
    mount_view(page, selection_app)
    ratios = page.evaluate(r'''async () => {
      const ratios={};
      const luminance=color=>color.match(/\d+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=0.04045?x/12.92:((x+0.055)/1.055)**2.4).reduce((s,x,i)=>s+x*[0.2126,0.7152,0.0722][i],0);
      const measure=label=>{const s=getComputedStyle(document.querySelector(`[data-label="${label}"]`));const a=luminance(s.color),b=luminance(s.backgroundColor);ratios[label]=(Math.max(a,b)+0.05)/(Math.min(a,b)+0.05);};
      const mentionId=draft.view.annotation.occurrences[0].mention_id;
      measure('used');
      for(const field of ['created','shared']){await act({type:'set_field',mentionId,field,value:true});renderView();measure(field);}
      for(const value of ['positive','negative','mixed','not_expressed']){await act({type:'set_field',mentionId,field:'sentiment',value});renderView();measure(value);}
      await act({type:'set_field',mentionId,field:'intents',value:'mentioned'});renderView();measure('mentioned');
      return ratios;
    }''')
    assert len(ratios) == 8
    assert min(ratios.values()) >= 4.5, ratios


def test_view_null_draft_queue_start_and_finish_clear_selection(page, selection_app):
    mount_view(page, selection_app)
    select_passage(page, 8, 20)
    page.locator('#passage').dispatch_event('mouseup')
    page.evaluate('''() => view.render(null,null,[{task_id:'pending-task',document_id:'pending-document'}],{})''')
    expect(page.locator('#review-summary')).to_contain_text('Choose a passage to review')
    expect(page.locator('#passage')).to_be_hidden()
    expect(page.locator('#selection-menu')).to_be_hidden()
    expect(page.locator('#software-cards article')).to_have_count(0)
    page.get_by_role('button', name='Open passage 1: pending-document', exact=True).click()
    assert page.evaluate('calls.at(-1)') == {'navigate': 'pending-task'}
    page.evaluate('''() => view.render(null,null,[],{finished:true,reviewProgress:{total:7,approved:2,pending:2,in_progress:1,source_issue:2}})''')
    expect(page.locator('#review-summary')).to_contain_text('End of the current queue')
    expect(page.locator('#review-summary')).to_contain_text('2 approved of 7. 2 pending, 1 in progress, 2 source issues.')
    expect(page.get_by_role('button', name='Approve & next', exact=True)).to_be_disabled()
    page.evaluate('renderView()')
    expect(page.locator('#passage')).to_be_visible()


def test_identify_unknown_undo_redo_and_real_save(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      const before = JSON.stringify(draft);
      await act({type:'identify_name', span:{start:8,end:20}});
      const after = structuredClone(draft);
      const undone = modules.undoDraft(draft), redone = modules.redoDraft(undone);
      return {before, after, undone, redone};
    }''')
    occurrence = result['after']['view']['annotation']['occurrences'][0]
    assert result['after']['base'] == item
    assert page.evaluate('item') == item
    assert occurrence['name'] == 'scikit-learn'
    assert occurrence['known'] == dict(software=True, versions=False, created=False, used=False, shared=False, sentiment=False)
    assert occurrence['intents'] is None and occurrence['sentiment'] is None
    assert result['undone']['view'] == item and not result['undone']['dirty']
    assert result['redone']['view'] == result['after']['view']
    assert result['redone']['operations'] == result['after']['operations']
    assert len(result['after']['operations']) == 1
    save(page, base)


def test_fields_versions_noops_and_summary_parity_after_save(page, selection_app):
    base, item = load(page, selection_app)
    summary = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:20}});
      const mentionId = draft.activeMentionId;
      await act({type:'link_version',mentionId,span:{start:21,end:26}});
      const count = draft.operations.length;
      await act({type:'link_version',mentionId,span:{start:21,end:26}});
      if (draft.operations.length !== count) throw Error('duplicate edge');
      await act({type:'set_field',mentionId,field:'used',value:true,evidence:[{start:0,end:27}]});
      await act({type:'set_field',mentionId,field:'shared',value:false});
      await act({type:'set_field',mentionId,field:'sentiment',value:'not_expressed'});
      return modules.summarizeDraft(draft);
    }''')
    save(page, base)
    actual = page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']
    assert strip_unsaved(summary) == actual


def strip_unsaved(value):
    if isinstance(value, dict):
        return {key: strip_unsaved(child) for key, child in value.items() if key != 'unsaved'}
    if isinstance(value, list):
        return [strip_unsaved(child) for child in value]
    return value


@pytest.mark.parametrize('selection_app', ['alias', 'partial', 'negative'], indirect=True)
def test_projection_matches_server(page, selection_app):
    _, item = load(page, selection_app)
    assert page.evaluate('modules.summarizeDraft(draft)') == item['review_summary']


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
def test_blind_discovery_no_leak_and_exact_reveal(page, selection_app):
    base, item = load(page, selection_app)
    first = item['annotation']['occurrences'][0]
    result = page.evaluate('''async span => {
      const initial = structuredClone(draft.view);
      await act({type:'identify_name',span});
      const blind = structuredClone(draft);
      let blocked = false;
      try { modules.buildBatch(draft,{completion:'save',reviewer:'Krushi',decisionId:'blind'}); }
      catch { blocked = true; }
      await act({type:'reveal'});
      return {initial,blind,blocked,revealed:draft};
    }''', first['name_span'])
    assert result['initial']['annotation']['occurrences'] == []
    assert result['initial']['annotation'].get('alias_annotations', {}).get('relations', []) == []
    assert result['blind']['view']['annotation']['occurrences'][0]['intents'] is None
    assert result['blind']['operations'] == [] and result['blocked']
    assert len(result['revealed']['view']['annotation']['occurrences']) == len(item['annotation']['occurrences'])
    assert result['revealed']['operations'][0]['action'] == 'accept_fields'
    assert result['revealed']['operations'][0]['fields'] == ['software']
    save(page, base)


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
def test_blind_overlap_reconciliation_blocks_save(page, selection_app):
    _, item = load(page, selection_app)
    span = dict(item['annotation']['occurrences'][0]['name_span'])
    span['end'] -= 1
    result = page.evaluate('''async span => {
      await act({type:'identify_name',span}); await act({type:'reveal'});
      try { modules.buildBatch(draft,{completion:'save',reviewer:'Krushi',decisionId:'conflict'}); }
      catch(error) { return error.message; }
    }''', span)
    assert 'RECONCILIATION_REQUIRED' in result


@pytest.mark.parametrize('selection_app', ['alias'], indirect=True)
def test_alias_id_atomic_undo_and_real_validation(page, selection_app):
    base, item = load(page, selection_app)
    first, second = item['annotation']['occurrences']
    relation = item['annotation']['alias_annotations']['relations'][0]
    expected_id = alias_relation_id(item['task']['document_id'], item['task']['text_revision'], relation['member_mention_ids'])
    result = page.evaluate('''async ({first,second,relation}) => {
      const id = await modules.aliasRelationId(item.task,relation.member_mention_ids.toReversed());
      await act({type:'remove_name',mentionId:second.mention_id,removeRelations:true,reasonCode:'wrong_span'});
      const before = structuredClone(draft);
      await act({type:'link_alias',span:second.name_span,targetMentionId:first.mention_id,
        relationType:relation.relation_type,preferredMentionId:first.mention_id,evidence:relation.evidence_spans});
      const after = structuredClone(draft), undone = modules.undoDraft(draft);
      return {id,before,after,undone};
    }''', dict(first=first, second=second, relation=relation))
    assert result['id'] == expected_id
    assert result['undone']['view'] == result['before']['view']
    assert result['undone']['operations'] == result['before']['operations']
    new = next(o for o in result['after']['view']['annotation']['occurrences'] if o['mention_id'] == second['mention_id'])
    assert new['intents'] is None and new['version_links'] == []
    save(page, base)


@pytest.mark.parametrize('selection_app', ['alias'], indirect=True)
def test_live_alias_group_projection_updates_on_remove_and_undo(page, selection_app):
    _, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      const before=modules.aliasGroupsForDraft(draft);
      await act({type:'remove_alias',relationId:draft.view.annotation.alias_annotations.relations[0].relation_id});
      const removed=modules.aliasGroupsForDraft(draft);
      draft=modules.undoDraft(draft);
      return {before,removed,restored:modules.aliasGroupsForDraft(draft)};
    }''')
    relation = item['annotation']['alias_annotations']['relations'][0]
    assert result['before'][0]['members'] == relation['member_mention_ids']
    assert result['before'][0]['preferredName'] == 'ICEKAT'
    assert result['before'][0]['relationIds'] == [relation['relation_id']]
    assert not result['before'][0]['conflict']
    assert result['removed'] == [] and result['restored'] == result['before']


@pytest.mark.parametrize('selection_app', ['unicode'], indirect=True)
def test_codepoint_backward_cross_node_selection_and_focus(page, selection_app):
    _, item = load(page, selection_app)
    page.evaluate('''text => {
      document.body.innerHTML='<div id="passage"></div><button id="menu">Menu</button>';
      const passage=document.querySelector('#passage');
      passage.append(document.createTextNode(text.slice(0,5)));
      const span=document.createElement('span');span.textContent=text.slice(5);passage.append(span);
    }''', item['task']['text'])
    select_passage(page, 2, 17)
    result = page.evaluate('''() => {
      const s=getSelection(),r=s.getRangeAt(0);s.setBaseAndExtent(r.endContainer,r.endOffset,r.startContainer,r.startOffset);
      window.captured=selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task);
      document.querySelector('#menu').focus();return captured;
    }''')
    assert result['start'] == 2 and result['end'] == 17
    assert result['text'] == item['task']['text'][2:17]
    assert page.evaluate('captured') == result
    assert page.locator('#passage').inner_text() == item['task']['text']


def test_position_clips_to_viewport(page, selection_app):
    load(page, selection_app)
    result = page.evaluate('''selectionModule.positionPopover({left:290,right:310,top:180,bottom:200},
      {width:320,height:210},{width:160,height:80})''')
    assert 0 <= result['left'] <= 160 and 0 <= result['top'] <= 130


@pytest.mark.parametrize('span', [dict(start=7, end=8), dict(start=20, end=8), dict(start=-1, end=5),
                                 dict(start=8, end=90), dict(start=8.5, end=20)])
def test_invalid_names_leave_input_untouched(page, selection_app, span):
    load(page, selection_app)
    result = page.evaluate('''async span => {
      const before=JSON.stringify(draft);let error;
      try { await act({type:'identify_name',span}); } catch(e) { error=e.message; }
      return {error,unchanged:before===JSON.stringify(draft)};
    }''', span)
    assert result['error'] and result['unchanged']


@pytest.mark.parametrize('selection_app', ['context'], indirect=True)
def test_owned_sentence_context_and_context_only_rejection(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      let error;try { await act({type:'identify_name',span:{start:0,end:11}}); } catch(e) { error=e.message; }
      await act({type:'identify_name',span:{start:21,end:33}});
      return {error,row:draft.view.annotation.occurrences[0]};
    }''')
    assert result['error'] == 'OUTSIDE_OWNED_REGION'
    assert result['row']['context_sentence'] == 'We used scikit-learn v0.17.'
    assert result['row']['context_span'] == item['task']['annotation_region']
    assert result['row']['context_kind'] == 'sentence'
    save(page, base)


@pytest.mark.parametrize('selection_app', ['unicode'], indirect=True)
def test_repeated_names_are_distinct_and_overlap_is_actionable(page, selection_app):
    base, item = load(page, selection_app)
    starts = [index for index in range(len(item['task']['text'])) if item['task']['text'].startswith('NumPy', index)]
    result = page.evaluate('''async starts => {
      for(const start of starts) await act({type:'identify_name',span:{start,end:start+5}});
      const before=JSON.stringify(draft);let error;
      try { await act({type:'identify_name',span:{start:starts[0],end:starts[0]+4}}); } catch(e) { error=e.message; }
      return {rows:draft.view.annotation.occurrences,error,unchanged:before===JSON.stringify(draft)};
    }''', starts)
    assert len(result['rows']) == 2 and result['rows'][0]['mention_id'] != result['rows'][1]['mention_id']
    assert result['error'] == 'OVERLAPPING_NAME_REQUIRES_RECONCILIATION' and result['unchanged']
    save(page, base)


def test_mentioned_only_atomic_undo_and_unresolved_version_retains_links(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:20}});
      const mentionId=draft.activeMentionId;
      await act({type:'set_field',mentionId,field:'used',value:true,evidence:[{start:0,end:27}]});
      const before=structuredClone(draft);
      await act({type:'set_field',mentionId,field:'intents',value:'mentioned'});
      const mentioned=structuredClone(draft), undone=modules.undoDraft(draft);
      await act({type:'link_version',mentionId,span:{start:21,end:26}});
      await act({type:'set_field',mentionId,field:'versions',value:{status:'ambiguous',links:draft.view.annotation.occurrences[0].version_links}});
      return {before,mentioned,undone,row:draft.view.annotation.occurrences[0],summary:modules.summarizeDraft(draft)};
    }''')
    assert result['mentioned']['view']['annotation']['occurrences'][0]['intents'] == ['mentioned']
    assert result['mentioned']['operations'][-1]['fields'] == ['created', 'used', 'shared']
    assert result['undone']['view'] == result['before']['view']
    assert result['row']['version_status'] == 'ambiguous' and len(result['row']['version_links']) == 1
    assert not result['row']['known']['versions']
    save(page, base)
    assert strip_unsaved(result['summary']) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']


@pytest.mark.parametrize('selection_app', ['partial'], indirect=True)
def test_shared_evidence_invalidates_only_matching_fingerprints(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      const row=draft.view.annotation.occurrences[0],mentionId=row.mention_id;
      await act({type:'confirm_fields',mentionId,fields:['created','used']});
      await act({type:'set_field',mentionId,field:'shared',value:null,evidence:[row.context_span]});
      return modules.summarizeDraft(draft);
    }''')
    save(page, base)
    assert strip_unsaved(result) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']


@pytest.mark.parametrize('selection_app', ['alias'], indirect=True)
def test_edit_alias_unresolved_preference_and_remove_dependency_guards(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      const relation=draft.view.annotation.alias_annotations.relations[0];
      let blocked;try { await act({type:'remove_name',mentionId:relation.member_mention_ids[0]}); } catch(e) { blocked=e.message; }
      await act({type:'edit_alias',relationId:relation.relation_id,value:{...relation,decision:'unresolved',preferred_mention_id:null},reasonCode:'insufficient_evidence'});
      return {blocked,summary:modules.summarizeDraft(draft)};
    }''')
    assert result['blocked'] == 'REMOVE_REFERENCING_RELATIONS_FIRST'
    assert result['summary']['counts']['alias_groups'] == 0
    save(page, base)
    assert strip_unsaved(result['summary']) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
def test_blind_reconciliation_is_correctable_and_undoable(page, selection_app):
    base, item = load(page, selection_app)
    span = item['annotation']['occurrences'][0]['name_span']
    result = page.evaluate('''async span => {
      await act({type:'identify_name',span:{...span,end:span.end-1}});
      const findingId=draft.activeMentionId;
      await act({type:'reveal'});
      const conflict=structuredClone(draft);
      await act({type:'change_name_span',mentionId:findingId,span});
      const corrected=structuredClone(draft), undone=modules.undoDraft(draft);
      return {conflict,corrected,undone};
    }''', span)
    assert len(result['conflict']['reconciliation']) == 1
    assert result['corrected']['reconciliation'] == []
    assert result['corrected']['operations'][0]['action'] == 'accept_fields'
    assert result['undone']['reconciliation'] == result['conflict']['reconciliation']
    save(page, base)


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
@pytest.mark.parametrize('resolution', ['correct', 'discard'])
def test_repeated_blind_overlap_remains_actionable_through_undo_redo(page, selection_app, resolution):
    base, item = load(page, selection_app)
    span = item['annotation']['occurrences'][0]['name_span']
    result = page.evaluate('''async ({span,resolution}) => {
      await act({type:'identify_name',span:{...span,end:span.end-1}});
      await act({type:'reveal'});
      const first=structuredClone(draft);
      await act({type:'change_name_span',mentionId:draft.reconciliation[0].mentionId,
        span:{...span,end:span.end-2}});
      const second=structuredClone(draft);
      draft=modules.undoDraft(draft);
      const firstRestored=structuredClone(draft);
      draft=modules.redoDraft(draft);
      const secondRestored=structuredClone(draft);
      const mentionId=draft.reconciliation[0].mentionId;
      await act(resolution==='correct' ? {type:'change_name_span',mentionId,span}
        : {type:'remove_name',mentionId});
      const resolved=structuredClone(draft);
      draft=modules.undoDraft(draft);
      const unresolvedAgain=structuredClone(draft);
      draft=modules.redoDraft(draft);
      return {first,second,firstRestored,secondRestored,resolved,unresolvedAgain,
        final:structuredClone(draft),batch:modules.buildBatch(draft,
          {completion:'save',reviewer:'Krushi',decisionId:'repeated-overlap'})};
    }''', dict(span=span, resolution=resolution))
    assert result['second']['reconciliation'][0]['span']['end'] == span['end'] - 2
    assert result['firstRestored']['reconciliation'] == result['first']['reconciliation']
    assert result['secondRestored']['reconciliation'] == result['second']['reconciliation']
    assert result['unresolvedAgain']['reconciliation'] == result['second']['reconciliation']
    assert result['final']['reconciliation'] == result['resolved']['reconciliation'] == []
    identities = [row['mention_id'] for row in item['annotation']['occurrences']]
    for snapshot in ('first', 'second', 'firstRestored', 'secondRestored', 'resolved', 'unresolvedAgain', 'final'):
        assert [row['mention_id'] for row in result[snapshot]['view']['annotation']['occurrences']] == identities
    if resolution == 'correct':
        assert len(result['final']['operations']) == 1
        assert result['final']['operations'][0]['action'] == 'accept_fields'
        assert result['final']['operations'][0]['fields'] == ['software']
        save(page, base)
    else:
        assert result['final']['operations'] == [] and not result['final']['dirty']
        assert result['batch'] is None


@pytest.mark.parametrize('selection_app', ['blind'], indirect=True)
def test_activate_and_blind_note_no_label_leaks(page, selection_app):
    base, item = load(page, selection_app)
    hidden_id = item['annotation']['occurrences'][0]['mention_id']
    result = page.evaluate('''async hiddenId => {
      let blocked;try { await act({type:'activate_name',mentionId:hiddenId}); } catch(e) { blocked=e.message; }
      await act({type:'record_note',note:'Check the definition again.'});
      return {blocked,summary:modules.summarizeDraft(draft),batch:modules.buildBatch(draft,{completion:'save',reviewer:'Krushi',decisionId:'blind-note'})};
    }''', hidden_id)
    assert result['blocked'] == 'UNKNOWN_OCCURRENCE'
    assert result['summary']['counts'] == dict(mentions=0, version_links=0, alias_groups=0)
    assert len(result['batch']['value']['operations']) == 1
    assert not result['batch']['value']['proposals_revealed']
    save(page, base)
    assert page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['annotation']['occurrences'] == item['annotation']['occurrences']


def test_selection_real_mouse_keyboard_and_stale_task(page, selection_app):
    _, item = load(page, selection_app)
    page.evaluate('''text => {
      document.body.innerHTML='<div id="passage" tabindex="0" style="font:20px monospace;white-space:pre"></div><button id="menu">Menu</button>';
      document.querySelector('#passage').textContent=text;
    }''', item['task']['text'])
    points = page.locator('#passage').evaluate('''el => {
      const point = offset => {const r=document.createRange();r.setStart(el.firstChild,offset);r.setEnd(el.firstChild,offset+1);const b=r.getBoundingClientRect();return {x:b.left+1,y:b.top+b.height/2};};
      return [point(8),point(20)];
    }''')
    page.mouse.move(**points[0])
    page.mouse.down()
    page.mouse.move(**points[1], steps=12)
    page.mouse.up()
    captured = page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)")
    assert captured['text'] == 'scikit-learn'
    page.keyboard.press('Shift+ArrowLeft')
    shorter = page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)")
    assert (shorter['start'], shorter['end']) in ((7, 20), (8, 19))
    assert shorter['text'] == item['task']['text'][shorter['start']:shorter['end']]
    page.keyboard.press('Shift+ArrowRight')
    captured = page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)")
    assert captured['start'] == 8 and captured['end'] == 20
    page.evaluate("document.querySelector('#passage').dataset.taskId='another-task'")
    assert page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)") is None


def test_selection_whitespace_outside_and_cleared_range(page, selection_app):
    _, item = load(page, selection_app)
    page.evaluate('''text => {document.body.innerHTML='<div id="passage"></div><p id="outside">Outside</p>';document.querySelector('#passage').textContent=text;}''', item['task']['text'])
    select_passage(page, 7, 8)
    assert page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)") is None
    page.evaluate('''() => {const r=document.createRange();r.setStart(document.querySelector('#passage').firstChild,8);r.setEnd(document.querySelector('#outside').firstChild,3);getSelection().removeAllRanges();getSelection().addRange(r);}''')
    assert page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)") is None
    page.evaluate('getSelection().removeAllRanges()')
    assert page.evaluate("selectionModule.capturePassageSelection(document.querySelector('#passage'),item.task)") is None


@pytest.mark.parametrize('selection_app', ['multi'], indirect=True)
def test_alias_conflicting_preferences_stay_visible_and_match_server(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:0,end:5}});
      const alpha=draft.activeMentionId;
      await act({type:'link_alias',span:{start:7,end:8},targetMentionId:alpha,relationType:'abbreviation',preferredMentionId:alpha,evidence:[{start:0,end:26}]});
      const short=draft.activeMentionId;
      await act({type:'identify_name',span:{start:22,end:26}});
      const beta=draft.activeMentionId;
      await act({type:'link_alias',span:{start:7,end:8},targetMentionId:beta,relationType:'explicit_alternative_name',preferredMentionId:beta,evidence:[{start:0,end:26}]});
      await act({type:'link_alias',span:{start:0,end:5},targetMentionId:beta,relationType:'explicit_alternative_name',preferredMentionId:alpha,evidence:[{start:0,end:26}]});
      const summary=modules.summarizeDraft(draft),before=JSON.stringify(draft);
      let contradiction;
      try { await act({type:'edit_alias',relationId:draft.view.annotation.alias_annotations.relations[2].relation_id,
        value:{member_mention_ids:[alpha,beta],relation_type:'explicit_alternative_name',decision:'not_alias',preferred_mention_id:null,evidence_spans:[{start:0,end:26}]}}); }
      catch(e) { contradiction=e.message; }
      return {summary,contradiction,unchanged:before===JSON.stringify(draft)};
    }''')
    assert result['summary']['counts']['alias_groups'] == 1
    assert any(q['code'] == 'preference_conflict' for q in result['summary']['questions'])
    assert result['contradiction'] == 'ALIAS_CONTRADICTION' and result['unchanged']
    save(page, base)
    assert strip_unsaved(result['summary']) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']


@pytest.mark.parametrize('selection_app', ['unicode'], indirect=True)
def test_unicode_alias_digest_matches_python(page, selection_app):
    base, item = load(page, selection_app)
    starts = [i for i in range(len(item['task']['text'])) if item['task']['text'].startswith('NumPy', i)]
    relation = page.evaluate('''async starts => {
      await act({type:'identify_name',span:{start:starts[0],end:starts[0]+5}});
      const target=draft.activeMentionId;
      await act({type:'link_alias',span:{start:starts[1],end:starts[1]+5},targetMentionId:target,
        relationType:'explicit_alternative_name',preferredMentionId:target,evidence:[item.task.context_span]});
      return draft.view.annotation.alias_annotations.relations[0];
    }''', starts)
    assert relation['relation_id'] == alias_relation_id(item['task']['document_id'], item['task']['text_revision'], relation['member_mention_ids'])
    save(page, base)


def test_failed_compound_alias_never_inserts_name_and_clear_redo_branch(page, selection_app):
    load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:20}});
      const before=JSON.stringify(draft),mentionId=draft.activeMentionId;
      let error;try { await act({type:'link_alias',span:{start:21,end:26},targetMentionId:mentionId,
        relationType:'abbreviation',preferredMentionId:mentionId,evidence:[{start:21,end:26}]}); } catch(e) { error=e.message; }
      const unchanged=before===JSON.stringify(draft);
      await act({type:'set_field',mentionId,field:'sentiment',value:'not_expressed'});
      draft=modules.undoDraft(draft);
      await act({type:'record_note',note:'Inspect again'});
      return {error,unchanged,redo:draft.redoStack.length,blankBatch:modules.buildBatch(modules.createDraft(item),{completion:'save',reviewer:'Krushi',decisionId:'empty'})};
    }''')
    assert result == dict(error='ALIAS_EVIDENCE_INVALID', unchanged=True, redo=0, blankBatch=None)


def test_activate_does_not_create_undo_or_dirty_and_source_bound_snapshots(page, selection_app):
    load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:20}});
      const loaded=structuredClone(item);loaded.annotation=draft.view.annotation;
      draft=modules.createDraft(loaded);
      const before=JSON.stringify(draft);
      const next=await modules.applyDraftAction(draft,{type:'activate_name',mentionId:loaded.annotation.occurrences[0].mention_id});
      return {unchanged:before===JSON.stringify(draft),dirty:next.dirty,operations:next.operations,undo:next.undoStack,active:next.activeMentionId};
    }''')
    assert result['unchanged'] and not result['dirty'] and not result['operations'] and not result['undo']
    assert result['active']


def test_confirmation_rejects_unknown_and_typed_notes_validate(page, selection_app):
    load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:20}});
      const mentionId=draft.activeMentionId,errors=[];
      for (const action of [{type:'confirm_fields',mentionId,fields:['sentiment']},
        {type:'set_field',mentionId,field:'used',value:true},
        {type:'set_field',mentionId,field:'sentiment',value:'positive'},
        {type:'record_note',note:'   '},
        {type:'remove_name',mentionId,reasonCode:'other'},
        {type:'remove_name',mentionId,reasonCode:'link_version'}]) {
        try { await act(action); } catch(e) { errors.push(e.message); }
      }
      return {errors,count:draft.operations.length};
    }''')
    assert result['errors'] == ['BATCH_CANNOT_CONFIRM_UNKNOWN', 'POSITIVE_INTENT_REQUIRES_EVIDENCE',
                                'SENTIMENT_REQUIRES_EVIDENCE', 'BATCH_NOTE_REQUIRED', 'BATCH_NOTE_REQUIRED', 'BATCH_REASON_INVALID']
    assert result['count'] == 1


def test_note_preserves_annotation_and_activation_preserves_history(page, selection_app):
    load(page, selection_app)
    result = page.evaluate('''async () => {
      const before=structuredClone(draft.view.annotation);
      await act({type:'record_note',note:'Passage needs another read.'});
      return {before,after:draft.view.annotation};
    }''')
    assert result['after'] == result['before']


def test_new_alias_identity_preview_does_not_stage_mutation(page, selection_app):
    _, item = load(page, selection_app)
    result = page.evaluate('''() => ({id:modules.mentionIdForSpan(item.task,{start:8,end:20}),
      operations:draft.operations,rows:draft.view.annotation.occurrences})''')
    assert result['id'] == item['task']['document_id'] + '|' + item['task']['text_revision'] + '|8:20'
    assert result['operations'] == [] and result['rows'] == []


def test_name_span_correction_and_removal_keep_atomic_history(page, selection_app):
    base, item = load(page, selection_app)
    result = page.evaluate('''async () => {
      await act({type:'identify_name',span:{start:8,end:19}});
      const previous=draft.activeMentionId;
      await act({type:'change_name_span',mentionId:previous,span:{start:8,end:20}});
      const corrected=structuredClone(draft),mentionId=draft.activeMentionId;
      await act({type:'remove_name',mentionId});
      const removed=structuredClone(draft);
      draft=modules.undoDraft(draft);
      return {previous,corrected,removed,restored:draft,summary:modules.summarizeDraft(draft)};
    }''')
    assert result['previous'] != result['corrected']['activeMentionId']
    assert result['corrected']['view']['annotation']['occurrences'][0]['name'] == 'scikit-learn'
    assert result['corrected']['operations'][-1]['fields'] == ['software']
    assert result['removed']['view']['annotation']['occurrences'] == []
    assert result['restored']['view'] == result['corrected']['view']
    save(page, base)
    assert strip_unsaved(result['summary']) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']


@pytest.mark.parametrize('selection_app', ['negative'], indirect=True)
def test_approval_invalidates_on_note_with_summary_parity(page, selection_app):
    base, item = load(page, selection_app)
    envelope = page.evaluate("modules.buildBatch(draft,{completion:'approve',reviewer:'Krushi',decisionId:'approved'})")
    token = page.request.get(base + '/api/session').json()['csrf_token']
    response = page.request.post(base + '/api/decisions', data=envelope,
                                 headers={'Origin': base, 'X-CSRF-Token': token})
    assert response.ok, response.text()
    current = page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()
    assert current['review_summary']['name_audit'] == 'confirmed'
    summary = page.evaluate('''async item => {draft=modules.createDraft(item);await act({type:'record_note',note:'A follow-up check'});return modules.summarizeDraft(draft);}''', current)
    save(page, base, 'note-after-approval')
    assert strip_unsaved(summary) == page.request.get(base + '/api/tasks/' + item['task']['task_id']).json()['review_summary']
