import {createDraft, applyDraftAction, undoDraft, redoDraft, summarizeDraft, buildBatch} from './review-draft.js';

const clone = value => structuredClone(value);
const freeze = value => {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
};
const messageFor = error => ({
  REVEAL_REQUIRED:'Show proposed labels before saving discoveries or approving.',
  RECONCILIATION_REQUIRED:'Resolve overlapping discoveries before saving.',
  STALE_SELECTION:'This selection belongs to an older passage. Select the text again.',
  BATCH_APPROVAL_NOT_READY:'Resolve unknown labels and source issues before approving.',
  DECISION_ID_CONFLICT:'The decision identity was already used for different content. Reload server state before continuing.',
}[error.code || error.message] || error.message || 'The request failed.');

export const createReviewSession = ({request, view, reviewer}) => {
  let draft = null, queue = {items:[]}, chain = Promise.resolve(), destroyed = false;
  let initialized = false, journal = [], redoJournal = [], initialRevealed = false;
  let pending = null, generation = 0;
  const state = {filter:'all', reviewer, saving:false, busy:false, error:'', finished:false};
  const render = () => { if (!destroyed) view.render(draft, draft ? summarizeDraft(draft) : null, queue, {...state, reviewProgress:queue.review_progress}); };
  const serial = action => {
    const result = chain.then(() => destroyed ? undefined : action());
    chain = result.catch(error => { if (!destroyed) {state.error = messageFor(error);render();} });
    return chain;
  };
  const locked = () => state.busy || state.saving || state.uncertainRequest || state.conflict || state.committedReload || state.dirtyNavigation;
  const blocked = () => { render(); return Promise.resolve(); };
  const initialize = async () => {
    if (initialized) return;
    const session = await request('/api/session');
    state.role = session.role;
    state.readOnly = session.capabilities?.review_workflow !== true || session.review_workflow_schema_version !== '1.0';
    initialized = true;
    await refreshQueue();
  };
  const refreshQueue = async () => {
    queue = await request('/api/queue' + (state.filter === 'all' ? '' : `?workflow_status=${encodeURIComponent(state.filter)}`));
  };
  const freshDraft = async (item, preserveReveal = false) => {
    let next = createDraft(item);
    if (preserveReveal && !next.revealed) next = await applyDraftAction(next, {type:'reveal'});
    next.undoStack = []; next.redoStack = [];
    return next;
  };
  const install = async (item, preserveReveal = false) => {
    const next = item ? await freshDraft(item, preserveReveal) : null;
    if (destroyed) return;
    generation += 1; view.clearSelection();draft = next;journal = [];redoJournal = [];
    initialRevealed = Boolean(next?.revealed);state.finished = !item;
  };
  const getTask = id => request(`/api/tasks/${encodeURIComponent(id)}`);
  const go = async destination => {
    if (destination.kind === 'filter') {
      const nextQueue = await request('/api/queue' + (destination.value === 'all' ? '' : `?workflow_status=${encodeURIComponent(destination.value)}`));
      state.filter = destination.value;queue = nextQueue;await install(null);state.finished = false;
    } else if (destination.id) await install(await getTask(destination.id));
    else await install(null);
  };
  const navigateTo = destination => {
    if (locked()) return blocked();
    state.busy = true;
    return serial(async () => {
      try {
        await initialize();
        if (draft?.dirty) state.dirtyNavigation = destination;
        else {state.error = '';await go(destination);}
      } finally {state.busy = false;render();}
    });
  };
  const openTask = id => {
    if (id) return navigateTo({kind:'task',id});
    if (locked()) return blocked();
    state.busy = true;
    return serial(async () => {try {await initialize();} finally {state.busy = false;render();}});
  };
  const navigate = target => {
    if (target === 'previous' || target === 'next') {
      const index = queue.items.findIndex(item => item.task_id === draft?.base.task.task_id);
      const next = queue.items[index + (target === 'next' ? 1 : -1)];
      if (!next && target === 'previous') return Promise.resolve();
      return navigateTo({kind:'task',id:next?.task_id || null});
    }
    return navigateTo({kind:'task',id:target});
  };
  const finishSaved = async submission => {
    if (destroyed || submission.generation !== generation) return;
    try {
      const item = await getTask(submission.envelope.task_id);
      await install(item, submission.revealed);
      state.committedReload = null;
    } catch (error) {
      state.committedReload = submission;
      state.error = `Saved; reload failed. ${messageFor(error)}`;
      return;
    }
    state.error = 'Saved.';
    try {await refreshQueue();} catch (error) {state.error = `Saved; queue refresh failed. ${messageFor(error)}`;}
    if (submission.destination) {
      try {await go(submission.destination);} catch (error) {state.error = `Saved; navigation failed. ${messageFor(error)}`;}
    }
  };
  const submit = async submission => {
    try {
      await request('/api/decisions', submission.envelope);
    } catch (error) {
      if (destroyed || submission.generation !== generation) return;
      if (error.code === 'STALE_REVISION' || error.code === 'DECISION_ID_CONFLICT') {
        state.conflict = {code:error.code, reviewed:false};state.error = 'Another decision changed this passage. Your draft is intact.';
        pending = null;state.uncertainRequest = null;
      } else if (!error.status || error.status >= 500) {
        pending = submission;state.uncertainRequest = true;
        state.error = 'Save outcome unknown. Retry the exact request to determine whether it was committed.';
      } else {pending = null;state.uncertainRequest = null;state.error = messageFor(error);}
      return;
    }
    if (destroyed || submission.generation !== generation) return;
    pending = null;state.uncertainRequest = null;
    await finishSaved(submission);
  };
  const saveCurrent = async (completion, destination) => {
    if (!draft || state.readOnly) return;
    const envelope = buildEnvelope(completion);
    if (!envelope) {
      state.error = 'No changes to save.';
      if (destination) await go(destination);
      return;
    }
    const submission = freeze({envelope, destination, generation, revealed:draft.revealed});
    await submit(submission);
  };
  const buildEnvelope = completion => buildBatch(draft, {completion,reviewer,decisionId:crypto.randomUUID()});
  const save = (completion, destinationTaskId = null, {advance = false} = {}) => {
    if (locked()) return blocked();
    state.saving = true;state.error = '';render();
    const destination = destinationTaskId || advance ? {kind:'task',id:destinationTaskId} : null;
    return serial(async () => {try {await saveCurrent(completion, destination);} finally {state.saving = false;render();}});
  };
  const reviewDifferences = async () => {
    const latest = await getTask(draft.base.task.task_id);
    if (JSON.stringify(latest.task) !== JSON.stringify(draft.base.task)) throw new Error('Passage source changed. Reload server state and review the source again.');
    let reapplied = await freshDraft(latest, initialRevealed);
    const replayJournal = [];
    for (const action of journal) {
      const previous = reapplied;
      reapplied = await applyDraftAction(reapplied, action);
      if (reapplied.undoStack.length > previous.undoStack.length) replayJournal.push(action);
    }
    const visibleAnnotation = item => draft.revealed ? item.annotation : {notice:'Proposed labels remain hidden.'};
    state.conflict = {...state.conflict, reviewed:true, latest, reapplied, replayJournal,
      differences:JSON.stringify({previous:visibleAnnotation(draft.base), server:visibleAnnotation(latest), reapplied:reapplied.view.annotation}, null, 2)};
    state.error = 'Review the previous, server and reapplied labels below. Reapplication stages changes for a new save.';
  };
  const resolve = choice => {
    if (state.saving || state.busy) return blocked();
    state.busy = true;
    return serial(async () => {
      try {
        if (choice === 'retry' && pending) {state.saving = true;await submit(pending);}
        else if (choice === 'reload_saved' && state.committedReload) await finishSaved(state.committedReload);
        else if (state.dirtyNavigation) {
          const destination = state.dirtyNavigation;state.dirtyNavigation = null;
          if (choice === 'discard') {await go(destination);state.error = '';}
          else if (choice === 'save_leave') {state.saving = true;await saveCurrent('save',destination);}
        } else if (state.conflict) {
          if (choice === 'reload') {await install(await getTask(draft.base.task.task_id),draft.revealed);state.conflict = null;state.error = 'Loaded server state.';}
          else if (choice === 'keep') {state.conflict = {code:state.conflict.code,reviewed:false};state.error = 'Draft kept. Review differences before reapplying, or reload server state.';}
          else if (choice === 'review') await reviewDifferences();
          else if (choice === 'reapply' && state.conflict.reviewed) {
            const latest = await getTask(draft.base.task.task_id);
            if (latest.annotation_revision !== state.conflict.latest.annotation_revision || JSON.stringify(latest.task) !== JSON.stringify(state.conflict.latest.task)) {
              state.conflict = {code:'STALE_REVISION',reviewed:false};state.error = 'Server state changed again. Review the new differences before reapplying.';
            } else {
              draft = state.conflict.reapplied;journal = state.conflict.replayJournal;redoJournal = [];state.conflict = null;
              state.error = 'Reviewed changes reapplied. Inspect the draft, then Save.';
            }
          }
        }
      } finally {state.saving = false;state.busy = false;render();}
    });
  };
  const dispatch = action => {
    if (action.type === 'session_choice') return resolve(action.choice);
    if (locked() || state.readOnly) return blocked();
    const captured = clone(action);
    return serial(async () => {
      if (!draft) return;
      const previous = draft;
      if (captured.type === 'undo') {draft = undoDraft(draft);if (draft !== previous && journal.length) redoJournal.push(journal.pop());}
      else if (captured.type === 'redo') {draft = redoDraft(draft);if (draft !== previous && redoJournal.length) journal.push(redoJournal.pop());}
      else {
        draft = await applyDraftAction(draft, captured);
        if (draft.undoStack.length > previous.undoStack.length) {journal.push(captured);redoJournal = [];}
      }
      state.error = '';render();
    });
  };
  const exportReference = () => {
    if (locked()) return blocked();
    state.busy = true;
    return serial(async () => {
      try {
        if (draft?.dirty) {state.error = 'Save or discard your draft before exporting the saved reference.';return;}
        const result = await request('/api/export', {});state.error = `${result.quality} export: ${result.path}`;
      } finally {state.busy = false;render();}
    });
  };
  const beforeUnload = event => {
    if (draft?.dirty || state.saving || state.uncertainRequest || state.committedReload) {event.preventDefault();event.returnValue = '';}
  };
  const onKey = event => {
    if (event.altKey && ['ArrowLeft','ArrowRight'].includes(event.key)) {event.preventDefault();navigate(event.key === 'ArrowLeft' ? 'previous' : 'next');}
  };
  window.addEventListener('beforeunload', beforeUnload);window.addEventListener('keydown', onKey);
  return {openTask, dispatch, save, navigate, setFilters:value => navigateTo({kind:'filter',value}), exportReference,
    destroy:() => {destroyed = true;generation += 1;window.removeEventListener('beforeunload',beforeUnload);window.removeEventListener('keydown',onKey);view.destroy();}};
};
