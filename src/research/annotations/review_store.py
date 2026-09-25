"""Transactional review snapshots backed by an append-only decision log."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from ..data.manifest import digest, json_bytes, write_jsonl, write_once
from .validation import validate_task_occurrence
from .alias_decisions import ALIAS_ACTIONS, validate_annotation_aliases
from .review_batches import apply_batch, result_metadata
from .review_mutations import reduce_mutation
from .review_workflow import validate_workflow


class ReviewError(ValueError):
    pass


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(path,timeout=10,isolation_level=None)
    c.row_factory=sqlite3.Row
    c.executescript('''
      PRAGMA journal_mode=WAL;
      PRAGMA foreign_keys=ON;
      CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS items (task_id TEXT PRIMARY KEY, original_hash TEXT NOT NULL,
        task TEXT NOT NULL, annotation TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES items(task_id),
        payload TEXT NOT NULL, result TEXT NOT NULL, recorded_at TEXT NOT NULL);
      CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
        BEGIN SELECT RAISE(ABORT,'append-only decisions'); END;
      CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
        BEGIN SELECT RAISE(ABORT,'append-only decisions'); END;
    ''')
    return c


def import_items(c, items: list[dict], role: str, policy_provenance: dict | None = None) -> None:
    from .snapshots import LEGACY_PROVENANCE, validate_item_policy
    provenance=policy_provenance if policy_provenance is not None else dict(LEGACY_PROVENANCE)
    try: validate_item_policy(items,provenance,role)
    except ValueError as exc: raise ReviewError(str(exc)) from exc
    if role not in ('train','dev','demo'):
        raise ReviewError('SPLIT_NOT_PERMITTED')
    c.execute('BEGIN IMMEDIATE')
    try:
        previous=c.execute("SELECT value FROM metadata WHERE key='role'").fetchone()
        if previous and previous['value']!=role: raise ReviewError('SPLIT_CONFLICT')
        c.execute("INSERT OR IGNORE INTO metadata VALUES ('role',?)",(role,))
        previous_sources=c.execute("SELECT value FROM metadata WHERE key='policy_sources'").fetchone()
        if previous_sources and json.loads(previous_sources['value'])!=provenance:
            raise ReviewError('POLICY_SOURCE_CONFLICT')
        c.execute("INSERT OR IGNORE INTO metadata VALUES ('policy_sources',?)",(json.dumps(provenance),))
        for item in items:
            task=item['task']; original=digest(json_bytes(item)); tid=task['task_id']
            if task.get('split')!=role: raise ReviewError('SPLIT_CONFLICT')
            annotation=deepcopy(item['annotation'])
            annotation['occurrences']=[validate_task_occurrence(task,o) for o in annotation['occurrences']]
            validate_annotation_aliases(task,annotation)
            before=c.execute('SELECT original_hash FROM items WHERE task_id=?',(tid,)).fetchone()
            if before:
                if before['original_hash']!=original: raise ReviewError('SOURCE_CONFLICT')
                continue
            task={**task,'occurrence_audit_ids':item.get('occurrence_audit_ids',[])}
            c.execute('INSERT INTO items VALUES (?,?,?,?,?,?)',
                      (tid,original,json.dumps(task),json.dumps(annotation),1,'unreviewed'))
        c.commit()
    except ValueError as exc:
        c.rollback();raise ReviewError(str(exc)) from exc
    except Exception:
        c.rollback();raise


def get_item(c, task_id: str) -> dict:
    row=c.execute('SELECT * FROM items WHERE task_id=?',(task_id,)).fetchone()
    if not row: raise ReviewError('UNKNOWN_TASK')
    return {'task':json.loads(row['task']),'annotation':json.loads(row['annotation']),
            'annotation_revision':row['revision'],'status':row['status']}


def queue(c, filters=None) -> dict:
    from .routing import alias_review_reasons, review_reasons
    filters=filters or {}; rows=[]; reviewed=0
    for row in c.execute('SELECT task_id,status FROM items ORDER BY task_id'):
        item=get_item(c,row['task_id']); task=item['task']; annotation=item['annotation']
        reviewed+=item['status']=='reviewed'
        reasons=sorted(set(r for o in annotation['occurrences'] for r in review_reasons(o,None)))
        reasons.extend(alias_review_reasons(annotation))
        if task.get('whole_passage_audit'): reasons.append('whole_passage_audit')
        if task.get('occurrence_audit_ids'): reasons.append('random_occurrence_audit')
        if annotation.get('status')=='partial': reasons.append('partial_annotation')
        entry={'task_id':task['task_id'],'document_id':task['document_id'],'source':task.get('source'),
               'split':task.get('split'),'status':item['status'],'reasons':sorted(set(reasons)),
               'annotation_revision':item['annotation_revision']}
        field=filters.get('field')
        if field and not any(not o.get('known',{}).get(field,False) for o in annotation['occurrences']): continue
        if any(filters.get(k) and filters[k] not in ('all',str(entry.get(k))) for k in ('status','source')): continue
        if filters.get('reason') and filters['reason'] not in entry['reasons']: continue
        rows.append(entry)
    return {'items':rows,'progress':{'total':c.execute('SELECT count(*) FROM items').fetchone()[0],'reviewed':reviewed},
            'quality':'provisional'}


def decision_history(c, task_id: str) -> list[dict]:
    return [{'payload': json.loads(row['payload']), 'result': json.loads(row['result']),
             'recorded_at': row['recorded_at']} for row in c.execute(
        'SELECT payload,result,recorded_at FROM decisions WHERE task_id=? ORDER BY rowid', (task_id,))]


def apply_decision(c, decision: dict) -> dict:
    if not isinstance(decision, dict):
        raise ReviewError('INVALID_DECISION')
    payload=json.dumps(decision,sort_keys=True,ensure_ascii=False)
    if any(not isinstance(decision.get(k),str) or not decision[k].strip() for k in ('decision_id','task_id','reviewer','reason')):
        raise ReviewError('IDENTITY_AND_REASON_REQUIRED')
    c.execute('BEGIN IMMEDIATE')
    try:
        prior=c.execute('SELECT payload,result FROM decisions WHERE decision_id=?',(decision['decision_id'],)).fetchone()
        if prior:
            if prior['payload']!=payload: raise ReviewError('DECISION_ID_CONFLICT')
            c.commit();return json.loads(prior['result'])
        item=get_item(c,decision['task_id']);task=item['task'];annotation=item['annotation']
        validate_annotation_aliases(task,annotation)
        if any(decision.get(k)!=task[k] for k in ('document_id','text_revision')):
            raise ReviewError('SOURCE_REVISION_MISMATCH')
        if type(decision.get('base_annotation_revision')) is not int or decision['base_annotation_revision']!=item['annotation_revision']:
            raise ReviewError('STALE_REVISION')
        action=decision.get('action')
        history=decision_history(c,task['task_id'])
        validate_workflow(item,history)
        now=datetime.now(timezone.utc).isoformat()
        provenance={'status':'human_reviewed','reviewer':decision['reviewer'],'decision_id':decision['decision_id'],
                    'recorded_at_utc':now,'reasons':[]}
        status=item['status']
        if action=='apply_review_batch':
            annotation,status,operation_results=apply_batch(item,decision,history,now)
        else:
            if decision.get('actor_kind','human')!='human': raise ReviewError('ACTOR_FORBIDDEN')
            annotation=reduce_mutation(task,annotation,decision,provenance)
            if action=='accept_passage': status='reviewed'
            elif action not in ALIAS_ACTIONS or status!='reviewed': status='in_progress'
            if item['annotation'].get('review_workflow',{}).get('approval') and action!='accept_passage':
                status='in_progress'
        validate_annotation_aliases(task,annotation)
        annotation['annotation_revision']=item['annotation_revision']+1
        result={'decision_id':decision['decision_id'],'task_id':task['task_id'],
                'annotation_revision':annotation['annotation_revision'],'recorded_at_utc':now}
        if action=='apply_review_batch': result.update(result_metadata(annotation,operation_results))
        pending={'payload':decision,'result':result,'recorded_at':now}
        validate_workflow({**item,'annotation':annotation,'annotation_revision':annotation['annotation_revision']},history+[pending])
        c.execute('UPDATE items SET annotation=?,revision=?,status=? WHERE task_id=?',
                  (json.dumps(annotation),annotation['annotation_revision'],status,task['task_id']))
        c.execute('INSERT INTO decisions VALUES (?,?,?,?,?)',(decision['decision_id'],task['task_id'],payload,json.dumps(result),now))
        c.commit();return result
    except ValueError as exc:
        c.rollback();raise ReviewError(str(exc)) from exc
    except Exception:
        c.rollback();raise


def export_reference(c, destination: Path) -> dict:
    from .snapshots import LEGACY_PROVENANCE, SNAPSHOT_VERSION, reference_rows, validate_item_policy, write_policy_provenance
    if destination.exists(): raise FileExistsError(destination)
    c.execute('BEGIN')
    try:
        items=[get_item(c,r[0]) for r in c.execute('SELECT task_id FROM items ORDER BY task_id')]
        role=c.execute("SELECT value FROM metadata WHERE key='role'").fetchone()[0]
        decision_records=[dict(r) for r in c.execute('SELECT * FROM decisions ORDER BY rowid')]
        store_items=[dict(r) for r in c.execute('SELECT * FROM items ORDER BY task_id')]
        metadata=[dict(r) for r in c.execute('SELECT * FROM metadata ORDER BY rowid')]
        c.commit()
    except Exception:
        c.rollback();raise
    provenance=json.loads(next((row['value'] for row in metadata if row['key']=='policy_sources'),json.dumps(LEGACY_PROVENANCE)))
    validate_item_policy(items,provenance,role)
    files=reference_rows(items,decision_records)
    files.update({'store-items.jsonl':store_items,'decision-records.jsonl':decision_records,'metadata.jsonl':metadata})
    destination.mkdir(parents=True)
    for name,rows in files.items():write_jsonl(destination/name,rows)
    sources=write_policy_provenance(destination,provenance)
    manifest={**sources,'role':role,'quality':'provisional','task_count':len(items), 'decision_count':len(decision_records),
              'snapshot_schema_version':SNAPSHOT_VERSION,'alias_schema_version':'1.0',
              'capabilities':{'aliases':True,'exact_store_restore':True},
              'files':[{'path':n,'sha256':digest((destination/n).read_bytes())} for n in files]+sources['files']}
    write_once(destination/'manifest.json',json_bytes(manifest))
    return manifest
