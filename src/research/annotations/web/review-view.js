import {summarizeDraft, aliasGroupsForDraft, mentionIdForSpan} from './review-draft.js';
import {capturePassageSelection, positionPopover} from './review-selection.js';

const el = (tag, text, attributes = {}) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
};
const button = (text, handler, attributes = {}) => {
  const node = el('button', text, {type:'button', ...attributes});
  node.addEventListener('click', handler);
  return node;
};
const title = value => ({not_expressed:'Not expressed', intents:'Intent', versions:'Version', software:'Software name'}[value]
  || value.charAt(0).toUpperCase() + value.slice(1));
const stateText = value => value?.unsaved ? 'Unsaved' : ({confirmed:'Confirmed', proposed:'Proposed', unresolved:'Unresolved', missing:'Unknown'}[value?.state] || 'Unknown');
const slice = (task, span) => Array.from(task.text).slice(span.start - task.offset_base, span.end - task.offset_base).join('');
const queueReason = code => ({missing:'Needs a decision', unresolved:'Unresolved labels', confirm_proposal:'Proposals to confirm',
  preference_conflict:'Choose a preferred name', broken_passage:'Broken passage', boundary_fragment:'Boundary fragment',
  pending:'Not reviewed', in_progress:'In progress', approved:'Approved', source_issue:'Source issue'}[code] || 'Review requested');

export const mountReviewView = (root, callbacks) => {
  let draft, summary, queue = [], session = {}, selection = null, returnFocus = null, taskKey = null;
  let panel = null, restoreAfterAction = false;
  const header = el('header', undefined, {class:'workspace-header'});
  const message = el('div', '', {id:'review-message', role:'alert'});
  const sessionStatus = el('p', '', {id:'session-status'});
  const sessionChoices = el('section', undefined, {id:'session-choices','aria-label':'Review session choices'});
  const layout = el('div', undefined, {class:'workspace-layout'});
  const sidebar = el('aside', undefined, {class:'queue', 'aria-label':'Passage queue'});
  const content = el('section', undefined, {class:'review-content'});
  const summaryNode = el('section', undefined, {id:'review-summary', 'aria-label':'Review summary'});
  const reading = el('div', undefined, {class:'reading-layout'});
  const paper = el('section', undefined, {class:'paper'});
  const passage = el('div', '', {id:'passage', tabindex:'0', 'aria-label':'Paper passage'});
  const cards = el('section', undefined, {id:'software-cards', 'aria-label':'Software labels'});
  const extras = el('div');
  const footer = el('footer', undefined, {id:'review-footer'});
  const menu = el('section', undefined, {id:'selection-menu', class:'popover', role:'dialog', 'aria-label':'Selection actions', hidden:''});
  paper.append(el('h2', 'Paper passage'),el('p','',{class:'source-identity'}),el('p','',{class:'source-guide'}), passage, el('p','Select text to identify software, link a version, or attach evidence.', {class:'reading-hint'}));
  reading.append(paper, cards); content.append(summaryNode, reading, extras); layout.append(sidebar, content);
  root.replaceChildren(header, sessionStatus, message, sessionChoices, layout, footer, menu);
  const readOnly = () => session.readOnly || session.capabilities?.readOnly || session.capabilities?.editing === false;
  const locked = () => readOnly() || session.busy || session.saving || Boolean(session.uncertainRequest || session.conflict || session.committedReload || session.dirtyNavigation);
  const rows = () => draft.view.annotation.occurrences;
  const relations = () => draft.view.annotation.alias_annotations?.relations || [];
  const byId = id => rows().find(row => row.mention_id === id);
  const active = () => byId(draft.activeMentionId);
  const close = (restore = true) => {
    menu.hidden = true; panel?.remove(); panel = null;
    if (restore) (returnFocus?.isConnected && returnFocus.getClientRects().length ? returnFocus : passage).focus({preventScroll:true});
  };
  const clearSelection = () => { selection = null; close(false); window.getSelection()?.removeAllRanges(); };
  const send = action => { if (locked()) return; restoreAfterAction=true;close(false);callbacks.onAction(action); };
  const focusCard = (id, field) => {
    const card = [...cards.querySelectorAll('article')].find(node => node.dataset.mentionId === id);
    if (card) {
      card.focus(); card.scrollIntoView({block:'nearest'});
      callbacks.onAction({type:'activate_name', mentionId:id});
      if (field) openField(byId(id), field);
    }
  };
  const place = node => {
    const rect = selection?.rect || returnFocus?.getBoundingClientRect() || passage.getBoundingClientRect();
    const pos = positionPopover(rect, {width:innerWidth,height:innerHeight}, node.getBoundingClientRect());
    node.style.left = `${pos.left}px`; node.style.top = `${pos.top}px`;
  };
  const openPanel = heading => {
    close(false); returnFocus = document.activeElement;
    panel = el('section', undefined, {class:'popover', role:'dialog', 'aria-label':heading});
    panel.append(el('h2',heading), button('×',()=>close(),{class:'close','aria-label':'Close'}));
    root.append(panel);
    return panel;
  };
  const ready = node => { place(node); (node.querySelector('button:not(.close), select, input, textarea')||node.querySelector('button.close'))?.focus({preventScroll:true}); };
  const actionButton = (node, text, action, disabled = false) => {
    const b = button(text, () => send(typeof action === 'function' ? action() : action));
    b.disabled = locked() || disabled; node.append(b); return b;
  };
  const evidencePreview = (node, spans) => {
    for (const span of spans) node.append(el('blockquote', slice(draft.view.task, span), {class:'evidence'}));
    if (!spans.length) node.append(el('p','No evidence attached.',{class:'muted'}));
  };
  const reasonControls = (node, options) => {
    const label = el('label','Correction reason');
    const select = el('select',undefined,{'aria-label':'Correction reason'});
    for (const [code,text] of [...options,['other','Other']]) select.append(el('option',text,{value:code}));
    const noteLabel = el('label','Optional note'); const note = el('textarea', '', {'aria-label':'Optional note', rows:'2'});
    noteLabel.append(note); label.append(select); node.append(label,noteLabel);
    select.addEventListener('change',()=>{note.required=select.value==='other';noteLabel.firstChild.textContent=note.required?'Note required for Other':'Optional note';});
    return () => {
      if (select.value==='other' && !note.value.trim()) { note.setCustomValidity('Describe the other reason.');note.reportValidity();return null; }
      note.setCustomValidity(''); return {reasonCode:select.value,...(note.value.trim()?{note:note.value.trim()}:{})};
    };
  };
  const openField = (row, field) => {
    if (!row) return;
    const node = openPanel(`${title(field)} for ${row.name}`);
    const currentField = field==='intents' ? 'used' : field;
    if (field==='software') { openName(row); return; }
    if (field==='versions') {
      for (const edge of row.version_links) {
        evidencePreview(node,[edge.span]);
        actionButton(node,`Remove version ${edge.text}`,{type:'set_field',mentionId:row.mention_id,field:'versions',value:{status:row.version_links.length>1?'explicit':'absent',links:row.version_links.filter(e=>e!==edge)}});
      }
      if (selection) actionButton(node,`Link selected version to ${row.name}`,{type:'link_version',mentionId:row.mention_id,span:selection});
      actionButton(node,'No version expressed',{type:'set_field',mentionId:row.mention_id,field:'versions',value:{status:'absent',links:[]}});
      actionButton(node,'Version unresolved',{type:'set_field',mentionId:row.mention_id,field:'versions',value:{status:'ambiguous',links:row.version_links}});
    } else {
      const evidenceField = field==='sentiment'?'sentiment':'intents';
      const evidence = selection ? [{start:selection.start,end:selection.end}] : row.evidence[evidenceField];
      node.append(el('p',selection?'Selected evidence:':'Current evidence:',{class:'muted'})); evidencePreview(node,evidence);
      if (field==='sentiment') {
        for (const value of ['positive','negative','mixed','not_expressed']) actionButton(node,value==='not_expressed'?'No sentiment expressed':title(value),{type:'set_field',mentionId:row.mention_id,field,value,evidence},value!=='not_expressed'&&!evidence.length);
        actionButton(node,'Unknown sentiment',{type:'set_field',mentionId:row.mention_id,field,value:null});
      } else {
        const bits = field==='intents'?['created','used','shared']:[field];
        for (const bit of bits) {
          actionButton(node,`Set ${title(bit)}`,{type:'set_field',mentionId:row.mention_id,field:bit,value:true,evidence},!evidence.length);
          actionButton(node,`Not ${bit}`,{type:'set_field',mentionId:row.mention_id,field:bit,value:false});
          actionButton(node,`${title(bit)} unresolved`,{type:'set_field',mentionId:row.mention_id,field:bit,value:null});
        }
        actionButton(node,'Mentioned only',{type:'set_field',mentionId:row.mention_id,field:'intents',value:'mentioned'});
      }
      if (!evidence.length) node.append(el('p','Select supporting passage text before assigning an expressed label.',{class:'muted'}));
    }
    if (row.known[currentField]) actionButton(node,`Confirm ${title(currentField)}`,{type:'confirm_fields',mentionId:row.mention_id,fields:[currentField]});
    ready(node);
  };
  const openName = row => {
    const node = openPanel(`Software name: ${row.name}`);
    const linked = relations().filter(rel=>rel.member_mention_ids.includes(row.mention_id));
    if (linked.length) {
      node.append(el('p','Removing or changing this name also removes these relationships:'));
      for (const rel of linked) node.append(el('p',rel.member_mention_ids.map(id=>byId(id)?.name).join(' / ')));
    }
    const reason = reasonControls(node,[['not_software','Not software'],['wrong_span','Wrong span']]);
    const commit = type => {
      const why = reason(); if (!why) return;
      if (type==='change_name_span' && why.reasonCode==='not_software') why.reasonCode='wrong_span';
      send({type,mentionId:row.mention_id,...why,...(type==='change_name_span'?{span:selection}:{}),removeRelations:linked.length>0});
    };
    const remove = button(linked.length?'Remove name and listed relationships':'Remove name',()=>commit('remove_name'));
    remove.disabled=locked();node.append(remove);
    if (selection) { const change=button('Use selection as corrected name',()=>commit('change_name_span'));change.disabled=locked();node.append(change); }
    else node.append(el('p','To correct the span, select the replacement text in the passage.',{class:'muted'}));
    actionButton(node,'Confirm software name',{type:'confirm_fields',mentionId:row.mention_id,fields:['software']},!draft.revealed);
    ready(node);
  };
  const chooseTarget = (heading, apply) => {
    const node = openPanel(heading);
    node.append(el('p','Choose the software name this action applies to.'));
    for (const row of rows()) node.append(button(row.name,()=>apply(row)));
    if (!rows().length) node.append(el('p','Identify a software name first.'));
    ready(node);
  };
  const withTarget = (heading, apply) => active()?apply(active()):chooseTarget(heading,apply);
  const aliasPanel = target => {
    if (!selection) return;
    const captured = structuredClone(selection);
    let selectedId;
    try { selectedId=mentionIdForSpan(draft.view.task,captured); } catch { showError('Select a software name inside the owned passage region.');return; }
    const node = openPanel(`Link ${captured.text} as an alias`);
    const label = el('label','Target member'); const member = el('select',undefined,{'aria-label':'Target member'});
    for (const row of rows().filter(r=>r.mention_id!==selectedId)) member.append(el('option',row.name,{value:row.mention_id}));
    if (target && target.mention_id!==selectedId) member.value=target.mention_id;
    label.append(member);node.append(label);
    const kindLabel = el('label','Relation type'); const kind=el('select',undefined,{'aria-label':'Relation type'});
    kind.append(el('option','Abbreviation',{value:'abbreviation'}),el('option','Explicit alternative name',{value:'explicit_alternative_name'}));kindLabel.append(kind);node.append(kindLabel);
    const prefLabel=el('label','Preferred name');const pref=el('select',undefined,{'aria-label':'Preferred name'});prefLabel.append(pref);node.append(prefLabel);
    const updatePref=()=>{pref.replaceChildren(el('option',byId(member.value)?.name || 'Choose target',{value:member.value}),el('option',captured.text,{value:selectedId}));}; updatePref();member.addEventListener('change',updatePref);
    const evidence = {start:draft.view.task.context_span.start,end:draft.view.task.context_span.end};
    node.append(el('p','Suggested relationship evidence:'));evidencePreview(node,[evidence]);
    let accepted=false;
    const accept=button('Use displayed evidence',()=>{accepted=true;accept.textContent='Displayed evidence selected';stage.disabled=locked()||!member.value;});node.append(accept);
    const stage=button('Stage alias link',()=>{if(accepted)send({type:'link_alias',span:captured,targetMentionId:member.value,relationType:kind.value,preferredMentionId:pref.value,evidence:[evidence]});});stage.disabled=true;node.append(stage,button('Cancel',()=>close()));ready(node);
  };
  const relationPanel = rel => {
    const node = openPanel(`Name relation: ${rel.member_mention_ids.map(id=>byId(id)?.name).join(' / ')}`);
    evidencePreview(node,rel.evidence_spans);
    const kindLabel=el('label','Relation type');const kind=el('select',undefined,{'aria-label':'Relation type'});
    kind.append(el('option','Abbreviation',{value:'abbreviation'}),el('option','Explicit alternative name',{value:'explicit_alternative_name'}));kind.value=rel.relation_type;kindLabel.append(kind);node.append(kindLabel);
    const label=el('label','Preferred name');const pref=el('select',undefined,{'aria-label':'Preferred name'});
    for (const id of rel.member_mention_ids) pref.append(el('option',byId(id)?.name,{value:id}));pref.value=rel.preferred_mention_id||rel.member_mention_ids[0];label.append(pref);node.append(label);
    const edit=decision=>({type:'edit_alias',relationId:rel.relation_id,value:{...rel,relation_type:kind.value,decision,preferred_mention_id:decision==='alias'?pref.value:null}});
    actionButton(node,'Confirm alias',()=>edit('alias'));
    actionButton(node,'Checked: not an alias',()=>({...edit('not_alias'),reasonCode:'wrong_software_link'}));
    actionButton(node,'Relation unresolved',()=>({...edit('unresolved'),reasonCode:'ambiguous_referent'}));
    actionButton(node,'Remove relationship',{type:'remove_alias',relationId:rel.relation_id});ready(node);
  };
  const showSelection = range => {
    if (!draft || !range || range.taskId!==draft.view.task.task_id || range.textRevision!==draft.view.task.text_revision) {clearSelection();return;}
    selection=structuredClone(range);close(false);returnFocus=passage;
    menu.replaceChildren(el('h2',`“${range.text}”`),button('×',()=>{clearSelection();passage.focus();},{class:'close','aria-label':'Close selection'}));
    const exact=rows().find(row=>row.name_span.start===range.start&&row.name_span.end===range.end);
    if (exact) {
      menu.append(el('p','Name already identified'),button(`Correct ${exact.name}`,()=>openName(exact)));
      callbacks.onAction({type:'activate_name',mentionId:exact.mention_id});
    } else actionButton(menu,'Identify name',{type:'identify_name',span:selection});
    for (const finding of draft.reconciliation) actionButton(menu,`Correct discovery “${finding.text}” with selection`,{type:'change_name_span',mentionId:finding.mentionId,span:selection});
    if (draft.revealed) {
      const target=exact||active();
      menu.append(button(target?`Link version to ${target.name}`:'Link version…',()=>withTarget('Link version to software',row=>send({type:'link_version',mentionId:row.mention_id,span:selection}))));
      menu.append(button(target?`Attach intent evidence to ${target.name}`:'Attach intent evidence…',()=>withTarget('Intent evidence for software',row=>openField(row,'intents'))));
      menu.append(button(target?`Attach sentiment evidence to ${target.name}`:'Attach sentiment evidence…',()=>withTarget('Sentiment evidence for software',row=>openField(row,'sentiment'))));
      menu.append(button('Link alias…',()=>aliasPanel(target)));
    }
    menu.hidden=false;place(menu);
  };
  const capture = () => { if(!draft)return;const value=capturePassageSelection(passage,draft.view.task);if(value)showSelection(value); };
  passage.addEventListener('mouseup',capture);passage.addEventListener('keyup',event=>{if(event.key!=='Escape')capture();});
  const onKey = event => {
    if(event.key==='Escape' && (selection || panel || !menu.hidden)){event.preventDefault();selection=null;window.getSelection()?.removeAllRanges();close();}
    if(event.key==='Tab' && !event.shiftKey && !menu.hidden && passage.contains(document.activeElement)){event.preventDefault();menu.querySelector('button:not(.close):not(:disabled)')?.focus();}
    if(event.key==='Tab' && panel){const focusable=[...panel.querySelectorAll('button:not(:disabled),select,textarea,input')];const first=focusable[0],last=focusable.at(-1);if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}}
  };
  root.addEventListener('keydown',onKey);
  const keepFocusVisible = event => {
    const target=event.target;
    if (!(target instanceof HTMLElement) || target.closest('.popover, #review-footer')) return;
    requestAnimationFrame(()=>{
      if(!target.isConnected)return;
      const box=target.getBoundingClientRect(), foot=footer.getBoundingClientRect(), head=header.getBoundingClientRect();
      if(box.bottom>foot.top-12)window.scrollBy(0,box.bottom-foot.top+16);
      else if(box.top<head.bottom+8)window.scrollBy(0,box.top-head.bottom-12);
    });
  };
  root.addEventListener('focusin',keepFocusVisible);
  const showError = text => {message.textContent=text||'';};
  const renderSession = () => {
    sessionStatus.textContent=session.reviewer?`Reviewer: ${session.reviewer}. ${session.role==='demo'?'Synthetic demo. No research labels.':'Local review.'} Decisions record your explicit review actions.`:'';
    sessionChoices.replaceChildren();
    const choice=(label,value)=>{const b=button(label,()=>callbacks.onAction({type:'session_choice',choice:value}));b.disabled=Boolean(session.busy||session.saving);sessionChoices.append(b);};
    if(session.dirtyNavigation){sessionChoices.append(el('p','Unsaved changes. Save before leaving, discard them, or stay here.'));choice('Save and continue','save_leave');choice('Discard and continue','discard');choice('Stay here','stay');}
    if(session.uncertainRequest){sessionChoices.append(el('p','Edits and navigation are paused until the pending save is resolved.'));choice('Retry exact save','retry');}
    if(session.committedReload)choice('Reload saved passage','reload_saved');
    if(session.conflict){
      choice('Reload server state','reload');choice('Keep draft','keep');choice('Review differences','review');
      if(session.conflict.reviewed){const details=el('details');details.open=true;details.append(el('summary','Previous, server and reapplied labels'),el('pre',session.conflict.differences));sessionChoices.append(details);choice('Reapply reviewed changes','reapply');}
    }
  };
  const renderQueue = () => {
    sidebar.replaceChildren();const details=el('details');details.open=innerWidth>800;details.append(el('summary',`Passages (${queue.length})`));
    const filter=el('select',undefined,{'aria-label':'Filter passages'});for(const [value,label] of [['all','All passages'],['pending','Not reviewed'],['in_progress','In progress'],['approved','Approved'],['source_issue','Source issues']])filter.append(el('option',label,{value}));filter.value=session.filter||'all';filter.disabled=Boolean(locked());filter.addEventListener('change',()=>callbacks.onFilter(filter.value));details.append(filter);
    const list=el('nav',undefined,{class:'queue-list','aria-label':'Passages'});
    for(const [index,item] of queue.entries()){const id=item.task_id||item.task?.task_id;const identity=item.document_id||item.task?.document_id||item.title||`Passage ${index+1}`;const b=button('',()=>{clearSelection();callbacks.onNavigate(id);},{'aria-current':String(id===draft?.view.task.task_id),'aria-label':`Open passage ${index+1}: ${identity}`});b.disabled=Boolean(locked());b.append(el('span',identity),el('small',queueReason(item.review_summary?.workflow_status||item.workflow_status||item.reason)));list.append(b);}details.append(list);sidebar.append(details);
  };
  const render = (next, ignoredSummary, nextQueue=[], nextSession={}) => {
    queue=Array.isArray(nextQueue)?nextQueue:nextQueue.items||[];session=nextSession;
    renderSession();
    if(!next){
      clearSelection();taskKey=null;draft=null;restoreAfterAction=false;
      header.replaceChildren(el('h1','Software annotation review'),button('Export',()=>callbacks.onExport()));
      showError(session.error||'');renderQueue();
      summaryNode.replaceChildren(el('h2',session.finished?'End of the current queue.':queue.length?'Choose a passage to review.':'No passages available.'));
      const progress=session.reviewProgress;
      if(progress)summaryNode.append(el('p',`${progress.approved||0} approved of ${progress.total||0}. ${progress.pending||0} pending, ${progress.in_progress||0} in progress, ${progress.source_issue||0} source issues.`));
      if(session.finished)summaryNode.append(el('p','Use the queue or filters to revisit remaining passages.',{class:'muted'}));
      reading.hidden=true;passage.replaceChildren();cards.replaceChildren();extras.replaceChildren();footer.replaceChildren();
      const controls=el('div',undefined,{class:'toolbar'});for(const label of ['Save','Save & next','Approve & next']){const b=button(label,()=>{});b.disabled=true;controls.append(b);}footer.append(controls,el('p','Choose a passage to continue.',{class:'footer-note'}));return;
    }
    reading.hidden=false;
    const key=`${next.view.task.task_id}:${next.view.task.text_revision}`;
    if(taskKey!==key)clearSelection();
    taskKey=key;draft=next;summary=summarizeDraft(draft);
    const focusedCard=document.activeElement?.closest?.('article')?.dataset.mentionId;
    header.replaceChildren(el('h1','Software annotation review'));
    const tools=el('div',undefined,{class:'toolbar'});
    for(const [name,type,disabled] of [['Undo','undo',!draft.undoStack.length],['Redo','redo',!draft.redoStack.length]]){const b=button(name,()=>send({type}));b.disabled=disabled||locked();tools.append(b);}
    tools.append(button('Export',()=>callbacks.onExport()));header.append(tools);
    showError(session.error||'');
    renderQueue();
    summaryNode.replaceChildren();
    const counts=el('div',undefined,{class:'summary-counts'});
    const countButton=(label,handler)=>counts.append(button(label,handler));
    countButton(`${summary.counts.mentions} software name${summary.counts.mentions===1?'':'s'}`,()=>{if(rows()[0])focusCard(rows()[0].mention_id);else passage.focus();});
    countButton(`${summary.counts.version_links} version link${summary.counts.version_links===1?'':'s'}`,()=>{const row=rows().find(r=>r.version_links.length)||rows()[0];if(row)focusCard(row.mention_id,'versions');});
    countButton(`${summary.counts.alias_groups} alias group${summary.counts.alias_groups===1?'':'s'}`,()=>extras.querySelector('.alias-groups')?.focus());summaryNode.append(counts);
    const summaryMessage=!draft.revealed?'Find software names first, then show proposed labels.'
      :summary.source_issues.length?'Source issue: repair needed before approval.'
      :summary.workflow_status==='approved'?(summary.partial_source_coverage?'Labels and name check approved; partial source coverage remains.':'Labels and name check approved.')
      :summary.needs_decisions?`${summary.needs_decisions} decision${summary.needs_decisions===1?'':'s'} needed`
      :!rows().length?'No software names displayed; check for missed names.'
      :!summary.proposals_to_confirm?(summary.name_audit==='confirmed'?'Displayed labels and name check confirmed.':'Displayed labels confirmed; check for missed names.')
      :'All displayed labels filled; check for missed names.';
    summaryNode.append(el('p',summaryMessage));
    if(draft.revealed){const pending=el('div',undefined,{class:'row'});pending.append(button(`${summary.proposals_to_confirm} proposals to confirm`,()=>{const disclosure=extras.querySelector('.proposal-details');if(disclosure){disclosure.open=true;disclosure.querySelector('summary').focus();disclosure.scrollIntoView({block:'nearest'});}}),el('small',`Name audit: ${summary.name_audit==='confirmed'?'confirmed':'pending'}.`));summaryNode.append(pending);}
    else summaryNode.append(el('small','Proposed names and labels are hidden.'));
    if(summary.partial_source_coverage)summaryNode.append(el('small','Partial source coverage. Displayed labels do not establish complete extraction.'));
    if(!draft.revealed)actionButton(summaryNode,'Show proposed labels',{type:'reveal'});
    passage.dataset.taskId=draft.view.task.task_id;passage.dataset.textRevision=draft.view.task.text_revision;
    passage.replaceChildren();let cursor=draft.view.task.offset_base;
    const sourceEnd=draft.view.task.offset_base+Array.from(draft.view.task.text).length;
    const owned=draft.view.task.annotation_region;
    paper.querySelector('.source-identity').textContent=`Document: ${draft.view.task.document_id}`;
    paper.querySelector('.source-guide').textContent=owned.start>draft.view.task.offset_base||owned.end<sourceEnd?'Review this region: unshaded text. Context: shaded text, evidence only.':'Review this region: the full passage below.';
    paper.querySelector('.reading-hint').textContent=owned.start>draft.view.task.offset_base||owned.end<sourceEnd
      ? 'Select names in the owned passage. Shaded surrounding context is for evidence only.'
      : 'Select text to identify software, link a version, or attach evidence.';
    const appendSource=(start,end)=>{
      const breaks=[start,...[owned.start,owned.end].filter(x=>x>start&&x<end),end];
      for(let index=0;index<breaks.length-1;index++){
        const a=breaks[index],b=breaks[index+1],text=slice(draft.view.task,{start:a,end:b});
        passage.append(a<owned.start||a>=owned.end?el('span',text,{class:'source-context'}):document.createTextNode(text));
      }
    };
    for(const row of [...rows()].sort((a,b)=>a.name_span.start-b.name_span.start)){
      appendSource(cursor,row.name_span.start);
      const mark=el('mark',slice(draft.view.task,row.name_span),{tabindex:'0',role:'button','aria-label':`Review ${row.name}`,'data-active':String(row.mention_id===draft.activeMentionId)});
      mark.addEventListener('click',()=>{if(!window.getSelection()?.toString())focusCard(row.mention_id);});mark.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();focusCard(row.mention_id);}});passage.append(mark);cursor=row.name_span.end;
    }appendSource(cursor,sourceEnd);
    cards.replaceChildren(el('h2','Software labels'));
    if(!rows().length)cards.append(el('p',draft.revealed?'No software names displayed. Select a missed name in the passage.':'Select names in the passage to record your discoveries.',{class:'muted'}));
    for(const row of rows()){
      const card=el('article',undefined,{class:'software-card',tabindex:'-1','data-mention-id':row.mention_id,'data-active':String(row.mention_id===draft.activeMentionId),'aria-label':`Labels for ${row.name}`});
      const states=summary.fields[row.mention_id]||{};
      card.append(button(row.name,()=>focusCard(row.mention_id),{class:'name-button'}),el('span',stateText(states.software),{class:'state'}));
      const pillRow=(label,items)=>{const group=el('div',undefined,{class:'label-row'});group.append(el('small',label));const pills=el('div',undefined,{class:'pills'});for(const [text,field,labelKey] of items){const b=button(text,()=>openField(row,field),{class:'pill','data-label':labelKey||field,'data-state':states[field]?.state||'missing'});b.disabled=!draft.revealed;pills.append(b,el('span',stateText(states[field]),{class:'state'}));}group.append(pills);card.append(group);};
      const intents=[];
      if(row.intents?.includes('mentioned'))intents.push(['Mentioned','used','mentioned']);
      for(const bit of ['created','used','shared'])if(row.intents?.includes(bit))intents.push([title(bit),bit,bit]);
      for(const bit of ['created','used','shared'])if(!row.known[bit])intents.push([`${title(bit)} unresolved`,bit,bit]);
      if(!intents.length)intents.push(['Review intent','intents','mentioned']);
      pillRow('Intent',intents);
      const editIntent=button('Edit intent',()=>openField(row,'intents'),{class:'pill'});editIntent.disabled=!draft.revealed;card.append(editIntent);
      pillRow('Sentiment',[[row.known.sentiment?title(row.sentiment):'Unknown sentiment','sentiment',row.sentiment||'unknown']]);
      pillRow('Version',[[row.version_links.map(e=>e.text).join(', ')||(row.version_status==='absent'?'No version expressed':'Version unresolved'),'versions','version']]);
      const actions=el('div',undefined,{class:'card-actions'});actions.append(button('Correct name',()=>openName(row)));card.append(actions);cards.append(card);
    }
    extras.replaceChildren();
    const groups=aliasGroupsForDraft(draft);if(groups.length||relations().length){const section=el('section',undefined,{class:'alias-groups',tabindex:'-1','aria-label':'Alias groups'});section.append(el('h2','Aliases'));
      for(const group of groups){const block=el('div');block.append(el('p',group.members.map(id=>byId(id)?.name).join(' / ')),el('small',group.conflict?'Preferred name needs a decision':`Preferred name: ${group.preferredName}`));section.append(block);}
      for(const rel of relations())section.append(button(`${rel.member_mention_ids.map(id=>byId(id)?.name).join(' / ')}: ${rel.decision==='not_alias'?'Checked non-alias':rel.decision==='unresolved'?'Unresolved relation':'Alias'} (${stateText(summary.relations[rel.relation_id])})`,()=>relationPanel(rel)));extras.append(section);}
    if(draft.reconciliation.length){const section=el('section',undefined,{class:'questions','aria-label':'Discovery reconciliation'});section.append(el('h2','Reconcile discoveries'));
      for(const finding of draft.reconciliation){const block=el('div',undefined,{class:'question'});block.append(el('p',`“${finding.text}” overlaps a proposed name. Choose the correct span or discard this discovery.`));const actions=el('div',undefined,{class:'row'});
        for(const id of finding.proposalMentionIds){const row=byId(id);actionButton(actions,`Use ${row.name}`,{type:'change_name_span',mentionId:finding.mentionId,span:row.name_span});}
        actionButton(actions,'Use selected corrected span',()=>({type:'change_name_span',mentionId:finding.mentionId,span:selection}),!selection);
        actionButton(actions,'Discard discovery',{type:'remove_name',mentionId:finding.mentionId});block.append(actions);section.append(block);}extras.append(section);}
    const questions=summary.questions.filter(q=>!draft.reconciliation.some(f=>f.mentionId===q.target_id));
    const questionButton=q=>{const row=byId(q.target_id);const rel=relations().find(r=>r.relation_id===q.target_id);return button(`${row?.name||'Name relation'}: ${title(q.field)}. ${queueReason(q.code)}`,()=>{if(row)focusCard(row.mention_id,q.field);else if(rel)relationPanel(rel);else extras.querySelector('.alias-groups')?.focus();});};
    const required=questions.filter(q=>q.code!=='confirm_proposal'), proposed=questions.filter(q=>q.code==='confirm_proposal');
    if(required.length){const section=el('section',undefined,{class:'questions','aria-label':'Review questions'});section.append(el('h2','Review questions'),...required.map(questionButton));extras.append(section);}
    if(proposed.length){const disclosure=el('details',undefined,{class:'proposal-details advanced'});disclosure.append(el('summary','Inspect proposed labels'),el('p','Approve confirms these unchanged proposals together. Open a field only to inspect or correct it.',{class:'muted'}));const items=el('div',undefined,{class:'questions'});items.append(...proposed.map(questionButton));disclosure.append(items);extras.append(disclosure);}
    if(summary.source_issues.length){const section=el('section',undefined,{class:'source-issues'});section.append(el('h2','Source issue blocks approval'));for(const issue of summary.source_issues)section.append(el('p',`${queueReason(issue.code)}: ${issue.message}`));extras.append(section);}
    const advanced=el('details',undefined,{class:'advanced'});advanced.append(el('summary','Advanced details'),el('pre',JSON.stringify({task_id:draft.view.task.task_id,text_revision:draft.view.task.text_revision,annotation_revision:draft.view.annotation_revision,owned_region:draft.view.task.annotation_region,source:draft.view.task.source,review_workflow:draft.view.annotation.review_workflow},null,2)));
    for(const row of rows()){advanced.append(el('h3',row.name),el('pre',JSON.stringify({name_span:row.name_span,evidence:row.evidence,review:row.review},null,2)));}
    const startLabel=el('label','Selection start');const start=el('input',undefined,{type:'number','aria-label':'Selection start'});const endLabel=el('label','Selection end');const end=el('input',undefined,{type:'number','aria-label':'Selection end'});startLabel.append(start);endLabel.append(end);advanced.append(startLabel,endLabel,button('Preview offsets',()=>{const span={start:Number(start.value),end:Number(end.value)};if(span.start<span.end&&span.start>=draft.view.task.offset_base&&span.end<=draft.view.task.offset_base+Array.from(draft.view.task.text).length)showSelection({...span,taskId:draft.view.task.task_id,textRevision:draft.view.task.text_revision,text:slice(draft.view.task,span),rect:passage.getBoundingClientRect()});else showError('Choose valid offsets within this passage.');}));extras.append(advanced);
    footer.replaceChildren();const controls=el('div',undefined,{class:'toolbar'});
    const nextIndex=queue.findIndex(item=>(item.task_id||item.task?.task_id)===draft.view.task.task_id)+1;
    const destination=queue[nextIndex]?.task_id||queue[nextIndex]?.task?.task_id||null;
    const previous=button('Previous',()=>callbacks.onNavigate('previous'));previous.disabled=Boolean(locked())||nextIndex<2;controls.append(previous);
    const blocker=readOnly()?'Read-only review':session.saving?'Saving…':session.busy?'Loading…':session.uncertainRequest?'Resolve the pending save before continuing.':session.conflict?'Reload or resolve the conflicting revision.':session.committedReload?'Reload the saved passage before continuing.':session.dirtyNavigation?'Choose what to do with unsaved changes.':draft.reconciliation.length?'Resolve overlapping discoveries before saving.':!draft.revealed&&draft.blindFindings.length?'Reveal proposals to reconcile discoveries.':'';
    for(const [label,completion,dest] of [['Save','save',null],['Save & next','save',destination],['Approve & next','approve',destination]]){const b=button(label,()=>callbacks.onSave(completion,dest,...(label==='Save'?[]:[{advance:true}])),{class:completion==='approve'?'primary':'',...(completion==='approve'?{'aria-describedby':'approval-scope'}:{})});b.disabled=Boolean(blocker)||(completion==='approve'&&(!draft.revealed||!summary.can_approve))||(label==='Save'&&!draft.dirty);controls.append(b);}
    footer.append(controls,el('p',blocker||(!draft.revealed?'Reveal proposals before approval.':!summary.can_approve?'Resolve unknown labels and source issues before approval.':draft.dirty?'Unsaved changes. Approval also confirms the name audit.':'Approval confirms displayed labels and the name audit.'),{class:'footer-note'}));
    footer.append(button('Approval scope',()=>{const node=openPanel('Approval scope');node.append(el('p','Approve confirms all displayed software names, versions, intent, sentiment and alias decisions, including unchanged proposals. It also records that you checked for missed software names inside the owned region. Surrounding context is evidence only and is not included in the name audit.',{id:'approval-scope-expanded'}));ready(node);},{'aria-describedby':'approval-scope'}),el('span','Approve confirms all displayed software names, versions, intent, sentiment and alias decisions, including unchanged proposals, and a missed-name check in the owned region only. Surrounding context is excluded from the name audit.',{id:'approval-scope',class:'visually-hidden'}));
    if(focusedCard){const node=[...cards.querySelectorAll('article')].find(n=>n.dataset.mentionId===focusedCard);node?.focus({preventScroll:true});}
    if(restoreAfterAction){restoreAfterAction=false;const node=[...cards.querySelectorAll('article')].find(n=>n.dataset.mentionId===draft.activeMentionId);(node||passage).focus({preventScroll:true});}
  };
  return {render,showSelection,clearSelection,showError,destroy:()=>{root.removeEventListener('keydown',onKey);root.removeEventListener('focusin',keepFocusVisible);clearSelection();root.replaceChildren();}};
};
