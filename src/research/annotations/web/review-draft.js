const FIELDS = ['software', 'versions', 'created', 'used', 'shared', 'sentiment'];
const BITS = ['created', 'used', 'shared'];
const copy = value => structuredClone(value);
const fail = message => { throw new Error(message); };
const stable = value => JSON.stringify(value, (_, child) => child && !Array.isArray(child) && typeof child === 'object'
  ? Object.fromEntries(Object.keys(child).sort().map(key => [key, child[key]])) : child);
const equal = (left, right) => stable(left) === stable(right);
const sameSpan = (left, right) => left.start === right.start && left.end === right.end;
const overlaps = (left, right) => left.start < right.end && right.start < left.end;
const mentionId = (task, span) => `${task.document_id}|${task.text_revision}|${span.start}:${span.end}`;
const sourceSlice = (task, span) => Array.from(task.text).slice(span.start - task.offset_base, span.end - task.offset_base).join('');
const spanIn = (span, bounds) => Number.isInteger(span?.start) && Number.isInteger(span?.end)
  && bounds.start <= span.start && span.start < span.end && span.end <= bounds.end;
const checkedSpan = (task, span, owned = false) => {
  if (span?.taskId !== undefined && span.taskId !== task.task_id
      || span?.textRevision !== undefined && span.textRevision !== task.text_revision) fail('STALE_SELECTION');
  if (!spanIn(span, owned ? task.annotation_region : task.context_span)) fail(owned ? 'OUTSIDE_OWNED_REGION' : 'INVALID_SPAN');
  const result = {start: span.start, end: span.end};
  const text = sourceSlice(task, result);
  if (!text.trim() || span.text !== undefined && span.text !== text) fail('INVALID_SPAN_TEXT');
  return result;
};
export const mentionIdForSpan = (task, span) => mentionId(task, checkedSpan(task, span, true));
const evidenceSpans = (task, evidence) => {
  if (!Array.isArray(evidence)) fail('EVIDENCE_REQUIRED');
  return evidence.map(span => checkedSpan(task, span));
};
const digestMembers = async (task, members) => {
  const bytes = new TextEncoder().encode(JSON.stringify([task.document_id, task.text_revision, [...members].sort()], null, 2) + '\n');
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(hash)].map(value => value.toString(16).padStart(2, '0')).join('');
};
export const aliasRelationId = async (task, memberMentionIds) => 'alias:' + await digestMembers(task, memberMentionIds);

const relations = draft => draft.view.annotation.alias_annotations?.relations || [];
const occurrence = (draft, id) => draft.view.annotation.occurrences.find(row => row.mention_id === id) || fail('UNKNOWN_OCCURRENCE');
const fieldSignature = (task, row, field) => {
  const value = field === 'software' ? {name: row.name, name_span: row.name_span}
    : field === 'versions' ? {version_status: row.version_status, version_links: row.version_links}
    : BITS.includes(field) ? {positive: (row.intents || []).includes(field), evidence: row.evidence.intents}
    : {sentiment: row.sentiment, evidence: row.evidence.sentiment};
  return stable([[task.task_id, task.document_id, task.text_revision, row.mention_id, row.name_span], field, row.known[field], value]);
};
const freshName = (task, input) => {
  const span = checkedSpan(task, input, true);
  const sentence = task.annotation_region_kind === 'sentence';
  const context = copy(sentence ? task.annotation_region : task.context_span);
  return {schema_version: '2.0', document_id: task.document_id, text_revision: task.text_revision,
    mention_id: mentionId(task, span), name: sourceSlice(task, span), name_span: span,
    context_sentence: sourceSlice(task, context), context_span: context, context_kind: sentence ? 'sentence' : 'paragraph',
    version_links: [], version_status: 'unannotated', intents: null, sentiment: null,
    known: Object.fromEntries(FIELDS.map(field => [field, field === 'software'])),
    evidence: {intents: [], sentiment: []}, review: {status: 'pending', reasons: []}};
};
const stateKeys = ['view', 'operations', 'revealed', 'activeMentionId', 'blindFindings', 'dirty', 'reconciliation', 'fieldReviews', 'relationReviews', 'groupIds'];
const snapshot = draft => Object.fromEntries(stateKeys.map(key => [key, copy(draft[key])]));

export const createDraft = item => {
  const base = copy(item);
  const revealed = !item.task.whole_passage_audit;
  const view = revealed ? copy(item) : {
    task: {...copy(item.task), candidate_system_predictions: null}, annotation_revision: item.annotation_revision,
    status: 'unreviewed', annotation: {occurrences: [], covered_regions: [],
      unresolved_regions: [copy(item.task.annotation_region)], status: 'partial'}};
  return {base, view, operations: [], undoStack: [], redoStack: [], revealed,
    activeMentionId: null, blindFindings: [], dirty: false, reconciliation: [], fieldReviews: {}, relationReviews: {}, groupIds: {}};
};

const reasons = {
  upsert_occurrence: ['identify_name', 'link_version', 'set_intent', 'set_sentiment', 'attach_evidence', 'accept_proposal', 'wrong_span', 'insufficient_evidence', 'ambiguous_referent'],
  accept_fields: ['accept_proposal'], remove_occurrence: ['wrong_span', 'not_software'],
  upsert_alias: ['link_alias', 'accept_proposal', 'wrong_software_link', 'insufficient_evidence', 'ambiguous_referent', 'attach_evidence'],
  remove_alias: ['wrong_software_link', 'not_software'], record_note: ['note'],
};
const addOperation = (draft, action, payload, input, defaultReason) => {
  const reason = input.reasonCode || defaultReason;
  if (!reasons[action].includes(reason) && reason !== 'other') fail('BATCH_REASON_INVALID');
  if ((reason === 'other' || action === 'record_note') && !input.note?.trim()) fail('BATCH_NOTE_REQUIRED');
  if (input.note !== undefined && typeof input.note !== 'string') fail('BATCH_NOTE_INVALID');
  const operation = {operation_id: `op:${draft.operations.length + 1}`, action, ...copy(payload), reason_code: reason};
  if (input.note !== undefined) operation.note = input.note;
  draft.operations.push(operation);
};
const markFields = (draft, row, fields) => {
  const reviews = draft.fieldReviews[row.mention_id] ||= {};
  for (const field of fields) reviews[field] = {state: row.known[field] ? 'confirmed' : 'unresolved',
    signature: fieldSignature(draft.view.task, row, field)};
  row.review = {status: fields.some(field => !row.known[field]) ? 'unresolved' : 'pending', reasons: []};
};
const upsert = (draft, row, old, fields, action, reason) => {
  addOperation(draft, 'upsert_occurrence', {target_name_span: old?.name_span || null, value: row, fields}, action, reason);
  const rows = draft.view.annotation.occurrences;
  if (old) rows.splice(rows.indexOf(old), 1);
  rows.push(row);
  rows.sort((a, b) => a.name_span.start - b.name_span.start);
  markFields(draft, row, fields);
  draft.activeMentionId = row.mention_id;
};
const confirm = (draft, row, fields, action) => {
  if (!Array.isArray(fields) || !fields.length || new Set(fields).size !== fields.length
      || fields.some(field => !FIELDS.includes(field))) fail('BATCH_FIELDS_INVALID');
  if (fields.some(field => !row.known[field])) fail('BATCH_CANNOT_CONFIRM_UNKNOWN');
  if (fields.every(field => fieldState(draft, row, field).state === 'confirmed')) return;
  addOperation(draft, 'accept_fields', {target_name_span: row.name_span, fields}, action, 'accept_proposal');
  markFields(draft, row, fields);
};
const identify = (draft, span, action) => {
  const row = freshName(draft.view.task, span);
  const existing = draft.view.annotation.occurrences.find(other => sameSpan(other.name_span, row.name_span));
  if (existing) { draft.activeMentionId = existing.mention_id; return existing; }
  if (draft.view.annotation.occurrences.some(other => overlaps(other.name_span, row.name_span))) fail('OVERLAPPING_NAME_REQUIRES_RECONCILIATION');
  if (draft.revealed) upsert(draft, row, null, ['software'], action, 'identify_name');
  else {
    draft.blindFindings.push({mentionId: row.mention_id, span: row.name_span, text: row.name});
    draft.view.annotation.occurrences.push(row);
    draft.activeMentionId = row.mention_id;
  }
  return row;
};
const setField = (draft, old, action) => {
  const row = copy(old), {field, value} = action, task = draft.view.task;
  let fields = [field], reason;
  if (BITS.includes(field) || field === 'intents') {
    reason = 'set_intent';
    if (field === 'intents') {
      if (value !== 'mentioned') fail('INVALID_INTENT_VALUE');
      fields = BITS;
      for (const bit of BITS) row.known[bit] = true;
      row.intents = ['mentioned'];
    } else {
      if (![true, false, null].includes(value)) fail('INVALID_INTENT_VALUE');
      row.known[field] = value !== null;
      row.intents = BITS.filter(bit => bit === field ? value === true : (old.intents || []).includes(bit));
      if (!BITS.some(bit => row.known[bit])) row.intents = null;
      else if (BITS.every(bit => row.known[bit]) && !row.intents.length) row.intents = ['mentioned'];
    }
    if (action.evidence !== undefined) row.evidence.intents = evidenceSpans(task, action.evidence);
    if (BITS.some(bit => (row.intents || []).includes(bit)) && !row.evidence.intents.length) fail('POSITIVE_INTENT_REQUIRES_EVIDENCE');
  } else if (field === 'sentiment') {
    reason = 'set_sentiment';
    if (![null, 'positive', 'negative', 'mixed', 'not_expressed'].includes(value)) fail('INVALID_SENTIMENT');
    row.sentiment = value;
    row.known.sentiment = value !== null;
    if (action.evidence !== undefined) row.evidence.sentiment = evidenceSpans(task, action.evidence);
    if (value === null) row.evidence.sentiment = [];
    if (['positive', 'negative', 'mixed'].includes(value) && !row.evidence.sentiment.length) fail('SENTIMENT_REQUIRES_EVIDENCE');
  } else if (field === 'versions') {
    reason = 'link_version';
    if (!value || !['explicit', 'absent', 'ambiguous', 'unannotated'].includes(value.status) || !Array.isArray(value.links)) fail('INVALID_VERSION_STATE');
    row.version_status = value.status;
    row.version_links = value.links.map(edge => {
      const span = checkedSpan(task, edge.span);
      if (!['explicit_local', 'explicit_remote'].includes(edge.status) || edge.text !== sourceSlice(task, span)
          || ['null', 'na', 'n/a', 'none'].includes(edge.text.toLowerCase().trim())) fail('INVALID_VERSION_LINK');
      return {text: edge.text, span, status: edge.status};
    });
    if (new Set(row.version_links.map(edge => stable(edge.span))).size !== row.version_links.length) fail('DUPLICATE_VERSION_LINK');
    if (value.status === 'explicit' && !value.links.length || ['absent', 'unannotated'].includes(value.status) && value.links.length) fail('INVALID_VERSION_STATE');
    row.known.versions = ['explicit', 'absent'].includes(value.status);
  } else fail('UNKNOWN_FIELD');
  if (fields.every(field => fieldSignature(task, row, field) === fieldSignature(task, old, field))
      && fields.every(field => fieldState(draft, old, field).state === (old.known[field] ? 'confirmed' : 'unresolved'))) return;
  upsert(draft, row, old, fields, action, reason);
};
const removeRelation = (draft, id, action) => {
  const rows = relations(draft), row = rows.find(relation => relation.relation_id === id);
  if (!row) fail('UNKNOWN_ALIAS_RELATION');
  addOperation(draft, 'remove_alias', {target_relation_id: id}, action, 'wrong_software_link');
  rows.splice(rows.indexOf(row), 1);
  delete draft.relationReviews[id];
};
const removeDependencies = (draft, row, action) => {
  const linked = relations(draft).filter(relation => relation.member_mention_ids.includes(row.mention_id));
  if (linked.length && action.removeRelations !== true) fail('REMOVE_REFERENCING_RELATIONS_FIRST');
  for (const relation of linked) removeRelation(draft, relation.relation_id, {reasonCode: 'wrong_software_link'});
};
const aliasGroups = draft => {
  const groups = [], visited = new Set(), rows = relations(draft), adjacency = new Map();
  for (const row of rows.filter(row => row.decision === 'alias')) {
    const [a,b] = row.member_mention_ids;
    if (!adjacency.has(a)) adjacency.set(a, new Set());
    if (!adjacency.has(b)) adjacency.set(b, new Set());
    adjacency.get(a).add(b); adjacency.get(b).add(a);
  }
  for (const root of [...adjacency.keys()].sort()) {
    if (visited.has(root)) continue;
    const members = [], stack = [root];
    while (stack.length) {
      const id = stack.pop();
      if (visited.has(id)) continue;
      visited.add(id); members.push(id);
      stack.push(...adjacency.get(id));
    }
    members.sort();
    const within = rows.filter(row => row.member_mention_ids.every(id => members.includes(id)));
    if (within.some(row => row.decision === 'not_alias')) fail('ALIAS_CONTRADICTION');
    const positive = within.filter(row => row.decision === 'alias');
    const names = new Set(positive.map(row => occurrence(draft, row.preferred_mention_id).name));
    groups.push({members, preferredName: names.size === 1 ? [...names][0] : null,
      conflict: names.size > 1, relationIds: positive.map(row => row.relation_id)});
  }
  return groups;
};
export const aliasGroupsForDraft = draft => aliasGroups(draft);
const upsertAlias = async (draft, value, targetId, action) => {
  const task = draft.view.task;
  if (task.policy_version !== 'scibert-poc-2.1' || !task.requested_fields.includes('aliases')) fail('ALIAS_NOT_REQUESTED');
  const members = value.member_mention_ids;
  if (!Array.isArray(members) || members.length !== 2 || members[0] === members[1]) fail('ALIAS_ENDPOINT_INVALID');
  const endpoints = members.map(id => occurrence(draft, id));
  for (const endpoint of endpoints) checkedSpan(task, endpoint.name_span, true);
  if (overlaps(endpoints[0].name_span, endpoints[1].name_span)) fail('ALIAS_ENDPOINT_INVALID');
  if (!['abbreviation', 'explicit_alternative_name'].includes(value.relation_type)
      || !['alias', 'not_alias', 'unresolved'].includes(value.decision)) fail('INVALID_ALIAS_RECORD');
  if (value.decision === 'alias' ? !members.includes(value.preferred_mention_id) : value.preferred_mention_id !== null) fail('ALIAS_PREFERENCE_INVALID');
  const evidence = evidenceSpans(task, value.evidence_spans);
  if (!evidence.some(span => endpoints.every(endpoint => span.start <= endpoint.name_span.start && endpoint.name_span.end <= span.end))) fail('ALIAS_EVIDENCE_INVALID');
  const id = await aliasRelationId(task, members);
  const row = {relation_id: id, document_id: task.document_id, text_revision: task.text_revision,
    member_mention_ids: [...members].sort(), relation_type: value.relation_type, decision: value.decision,
    preferred_mention_id: value.preferred_mention_id, evidence_spans: evidence, review: {status: 'pending', reasons: []}};
  const layer = draft.view.annotation.alias_annotations ||= {schema_version: '1.0', relations: []};
  const old = layer.relations.find(relation => relation.relation_id === targetId);
  const duplicate = layer.relations.find(relation => relation.relation_id === id && relation !== old);
  if (duplicate) fail('DUPLICATE_ALIAS_PAIR');
  const content = relation => Object.fromEntries(Object.entries(relation).filter(([key]) => key !== 'review'));
  if (old && equal(content(old), content(row)) && draft.relationReviews[id]) return;
  if (targetId && !old) fail('UNKNOWN_ALIAS_RELATION');
  addOperation(draft, 'upsert_alias', {target_relation_id: targetId || null, value: row}, action, 'link_alias');
  if (old) layer.relations.splice(layer.relations.indexOf(old), 1);
  layer.relations.push(row);
  draft.relationReviews[id] = {state: row.decision === 'unresolved' ? 'unresolved' : 'confirmed', signature: stable(content(row))};
  aliasGroups(draft);
};
const recomputeCoverage = draft => {
  const annotation = draft.view.annotation, covered = annotation.covered_regions || [];
  for (const region of covered) region.status = Object.values(region.fields).every(Boolean)
    && annotation.occurrences.filter(row => overlaps(region, row.name_span)).every(row => Object.values(row.known).every(Boolean)) ? 'complete' : 'partial';
  annotation.status = covered.length && !annotation.unresolved_regions?.length && covered.every(region => region.status === 'complete') ? 'complete' : 'partial';
  annotation.complete_negative_regions = covered.filter(region => region.status === 'complete'
    && Object.values(region.fields).every(Boolean) && !annotation.occurrences.some(row => overlaps(region, row.name_span))).map(copy);
};
const reconcileFinding = (draft, finding, action) => {
  const exact = draft.view.annotation.occurrences.find(row => sameSpan(row.name_span, finding.span));
  if (exact) { confirm(draft, exact, ['software'], action); draft.activeMentionId = exact.mention_id; return; }
  const matches = draft.view.annotation.occurrences.filter(row => overlaps(row.name_span, finding.span));
  if (matches.length) {
    draft.reconciliation.push({mentionId: finding.mentionId, span: copy(finding.span), text: finding.text,
      proposalMentionIds: matches.map(row => row.mention_id)});
  } else identify(draft, finding.span, action);
};

export const applyDraftAction = async (draft, action) => {
  if (!action || typeof action.type !== 'string') fail('UNKNOWN_ACTION');
  const next = {...copy(draft), undoStack: draft.undoStack, redoStack: draft.redoStack};
  const task = next.view.task;
  if (action.type === 'activate_name') {
    occurrence(next, action.mentionId);
    next.activeMentionId = action.mentionId;
    return next;
  }
  if (action.type === 'reveal') {
    if (next.revealed) return draft;
    next.revealed = true;
    next.view = copy(next.base);
    for (const finding of next.blindFindings) reconcileFinding(next, finding, {});
  } else if (action.type === 'identify_name') identify(next, action.span, action);
  else if (action.type === 'record_note') addOperation(next, 'record_note', {}, action, 'note');
  else if (['remove_name', 'change_name_span'].includes(action.type)
      && ((!next.revealed && next.blindFindings.some(row => row.mentionId === action.mentionId))
        || next.reconciliation.some(row => row.mentionId === action.mentionId))) {
    next.blindFindings = next.blindFindings.filter(row => row.mentionId !== action.mentionId);
    next.reconciliation = next.reconciliation.filter(row => row.mentionId !== action.mentionId);
    if (!next.revealed) next.view.annotation.occurrences = next.view.annotation.occurrences.filter(row => row.mention_id !== action.mentionId);
    next.activeMentionId = null;
    if (action.type === 'change_name_span') {
      const span = checkedSpan(task, action.span, true);
      const finding = {mentionId: mentionId(task, span), span, text: sourceSlice(task, span)};
      if (next.revealed) reconcileFinding(next, finding, action);
      else identify(next, span, action);
    }
  } else {
    if (!next.revealed) fail('REVEAL_REQUIRED');
    if (action.type === 'set_field') setField(next, occurrence(next, action.mentionId), action);
    else if (action.type === 'confirm_fields') confirm(next, occurrence(next, action.mentionId), action.fields, action);
    else if (action.type === 'link_version') {
      const row = occurrence(next, action.mentionId), span = checkedSpan(task, action.span);
      if (row.version_links.some(edge => sameSpan(edge.span, span))) return draft;
      const edge = {span, text: sourceSlice(task, span), status: spanIn(span, row.context_span) ? 'explicit_local' : 'explicit_remote'};
      setField(next, row, {...action, field: 'versions', value: {status: 'explicit', links: [...row.version_links, edge]}});
    } else if (action.type === 'remove_name' || action.type === 'change_name_span') {
      const old = occurrence(next, action.mentionId);
      if (action.type === 'change_name_span') {
        const span = checkedSpan(task, action.span, true);
        if (sameSpan(old.name_span, span)) return draft;
        if (!spanIn(span, old.context_span)) fail('OUTSIDE_OCCURRENCE_CONTEXT');
        if (next.view.annotation.occurrences.some(row => row !== old && overlaps(row.name_span, span))) fail('OVERLAPPING_NAME_REQUIRES_RECONCILIATION');
        removeDependencies(next, old, action);
        upsert(next, {...copy(old), name: sourceSlice(task, span), name_span: span, mention_id: mentionId(task, span)}, old, ['software'], action, 'wrong_span');
      } else {
        removeDependencies(next, old, action);
        addOperation(next, 'remove_occurrence', {target_name_span: old.name_span}, action, 'not_software');
        next.view.annotation.occurrences.splice(next.view.annotation.occurrences.indexOf(old), 1);
        if (next.activeMentionId === old.mention_id) next.activeMentionId = null;
      }
    } else if (action.type === 'remove_alias') removeRelation(next, action.relationId, action);
    else if (action.type === 'edit_alias') await upsertAlias(next, action.value, action.relationId, action);
    else if (action.type === 'link_alias') {
      occurrence(next, action.targetMentionId);
      const row = identify(next, action.span, {});
      await upsertAlias(next, {member_mention_ids: [row.mention_id, action.targetMentionId],
        relation_type: action.relationType, decision: 'alias', preferred_mention_id: action.preferredMentionId,
        evidence_spans: action.evidence}, null, action);
    } else fail('UNKNOWN_ACTION');
  }
  if (equal(snapshot(next), snapshot(draft))) return draft;
  if (next.operations.length > 200) fail('BATCH_OPERATIONS_LIMIT');
  if (next.operations.slice(draft.operations.length).some(op => ['upsert_occurrence', 'remove_occurrence'].includes(op.action))) recomputeCoverage(next);
  for (const group of aliasGroups(next)) next.groupIds[stable(group.members)] = 'local-software:' + await digestMembers(task, group.members);
  next.dirty = Boolean(next.operations.length || next.reconciliation.length || !next.revealed && next.blindFindings.length);
  next.undoStack = [...draft.undoStack, snapshot(draft)];
  next.redoStack = [];
  return next;
};

export const undoDraft = draft => draft.undoStack.length ? {
  ...draft, ...copy(draft.undoStack.at(-1)), undoStack: draft.undoStack.slice(0, -1),
  redoStack: [...draft.redoStack, snapshot(draft)]} : draft;
export const redoDraft = draft => draft.redoStack.length ? {
  ...draft, ...copy(draft.redoStack.at(-1)), redoStack: draft.redoStack.slice(0, -1),
  undoStack: [...draft.undoStack, snapshot(draft)]} : draft;

const fieldState = (draft, row, field) => {
  const local = draft.fieldReviews[row.mention_id]?.[field];
  const signature = fieldSignature(draft.view.task, row, field);
  if (local?.signature === signature) return {state: local.state, unsaved: true};
  const baseRow = draft.base.annotation.occurrences.find(other => other.mention_id === row.mention_id);
  const original = draft.base.review_summary?.fields[row.mention_id]?.[field];
  if (draft.revealed && original && baseRow && signature === fieldSignature(draft.base.task, baseRow, field)) return copy(original);
  return {state: row.known[field] ? 'proposed' : row.version_status === 'ambiguous' && field === 'versions'
    || row.review?.status === 'unresolved' ? 'unresolved' : 'missing'};
};
const requiredFields = task => new Set(task.requested_fields.flatMap(field => ({software: ['software'], version: ['versions'],
  version_links: ['versions'], intents: BITS, sentiment: ['sentiment']}[field] || [])));

export const summarizeDraft = draft => {
  if (draft.revealed && !draft.operations.length && !draft.reconciliation.length && draft.base.review_summary) return copy(draft.base.review_summary);
  const {task, annotation} = draft.view, requested = requiredFields(task), fields = {}, relationStates = {}, questions = [];
  const ask = (target_id, field, code, message) => questions.push({target_id, field, code, message});
  for (const row of annotation.occurrences) {
    const states = fields[row.mention_id] = Object.fromEntries(FIELDS.map(field => [field, fieldState(draft, row, field)]));
    for (const [unit, members] of [['software', ['software']], ['versions', ['versions']], ['intents', BITS], ['sentiment', ['sentiment']]]) {
      const relevant = members.filter(field => requested.has(field));
      if (!relevant.length) continue;
      const values = new Set(relevant.map(field => states[field].state));
      if (unit === 'intents' && relevant.some(field => row.intents?.includes(field)) && !row.evidence.intents.length
          || unit === 'sentiment' && ['positive', 'negative', 'mixed'].includes(row.sentiment) && !row.evidence.sentiment.length) values.add('missing');
      const code = values.has('unresolved') ? 'unresolved' : values.has('missing') ? 'missing' : values.has('proposed') ? 'confirm_proposal' : null;
      if (code) ask(row.mention_id, unit, code, {unresolved: 'Resolve this field.', missing: 'Supply a supported decision.', confirm_proposal: 'Confirm the proposed value.'}[code]);
    }
  }
  for (const relation of relations(draft)) {
    const local = draft.relationReviews[relation.relation_id];
    const state = relation.decision === 'unresolved' ? 'unresolved' : local ? 'confirmed'
      : draft.base.review_summary?.relations[relation.relation_id]?.state || 'proposed';
    relationStates[relation.relation_id] = {state, ...(local ? {unsaved: true} : {})};
    if (state !== 'confirmed') ask(relation.relation_id, 'aliases', state === 'proposed' ? 'confirm_proposal' : 'unresolved',
      state === 'proposed' ? 'Confirm the proposed name relation.' : 'Resolve this name relation.');
  }
  const groups = aliasGroups(draft);
  for (const group of groups.filter(group => group.conflict)) {
    const id = draft.groupIds[stable(group.members)];
    if (!id) fail('ALIAS_GROUP_ID_UNPREPARED');
    ask(id, 'aliases', 'preference_conflict', 'Choose one preferred name.');
  }
  for (const finding of draft.reconciliation) ask(finding.mentionId, 'software', 'unresolved', 'Reconcile this discovery with overlapping proposals.');
  const issues = copy(annotation.review_workflow?.source_issues || []);
  const blockers = questions.filter(question => ['missing', 'unresolved', 'preference_conflict'].includes(question.code));
  blockers.push(...issues.map(issue => ({target_id: issue.issue_id, field: 'source', code: issue.code, message: issue.message})));
  const baseSummary = draft.revealed ? draft.base.review_summary : null;
  const nameChanged = draft.operations.some(op => op.action === 'remove_occurrence' || op.action === 'upsert_occurrence' && op.fields.includes('software'));
  const approvalInvalidated = draft.operations.length && draft.base.annotation.review_workflow?.approval;
  const nameAudit = nameChanged || approvalInvalidated ? 'pending' : baseSummary?.name_audit || 'pending';
  return {schema_version: '1.0', counts: {mentions: annotation.occurrences.length,
    version_links: annotation.occurrences.reduce((count,row) => count + row.version_links.length, 0), alias_groups: groups.length},
    needs_decisions: questions.filter(question => ['missing', 'unresolved', 'preference_conflict'].includes(question.code)).length,
    proposals_to_confirm: questions.filter(question => question.code === 'confirm_proposal').length,
    source_issues: issues, name_audit: nameAudit,
    workflow_status: issues.length ? 'source_issue' : draft.operations.length || draft.view.status !== 'unreviewed'
      || nameAudit === 'confirmed' || Object.values(fields).some(row => Object.values(row).some(value => value.state === 'confirmed')) ? 'in_progress' : 'pending',
    partial_source_coverage: annotation.status !== 'complete' || Boolean(annotation.unresolved_regions?.length),
    can_approve: !blockers.length, approval_blockers: blockers, fields, relations: relationStates, questions};
};

export const buildBatch = (draft, {completion, reviewer, decisionId}) => {
  if (!['save', 'approve'].includes(completion)) fail('BATCH_COMPLETION_INVALID');
  if (!reviewer?.trim() || !decisionId?.trim()) fail('REVIEW_IDENTITY_REQUIRED');
  if (draft.reconciliation.length) fail('RECONCILIATION_REQUIRED');
  if (!draft.revealed && (completion === 'approve' || draft.blindFindings.length)) fail('REVEAL_REQUIRED');
  if (completion === 'approve' && !summarizeDraft(draft).can_approve) fail('BATCH_APPROVAL_NOT_READY');
  if (completion === 'save' && !draft.operations.length) return null;
  const task = draft.base.task;
  return {decision_id: decisionId, task_id: task.task_id, document_id: task.document_id, text_revision: task.text_revision,
    base_annotation_revision: draft.base.annotation_revision, reviewer, actor_kind: 'human', action: 'apply_review_batch',
    reason: 'Reviewer saved scoped decisions.', value: {schema_version: '1.0', completion,
      proposals_revealed: draft.revealed, operations: copy(draft.operations)}};
};
