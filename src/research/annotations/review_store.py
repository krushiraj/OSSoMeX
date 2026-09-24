"""Transactional review snapshots backed by an append-only decision log."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from ..contracts import FIELDS, INTENT_BITS
from ..data.manifest import digest, json_bytes, write_jsonl, write_once
from .validation import validate_task_occurrence


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


def import_items(c, items: list[dict], role: str) -> None:
    if role not in ('train','dev','demo'):
        raise ReviewError('SPLIT_NOT_PERMITTED')
    c.execute('BEGIN IMMEDIATE')
    try:
        previous=c.execute("SELECT value FROM metadata WHERE key='role'").fetchone()
        if previous and previous['value']!=role: raise ReviewError('SPLIT_CONFLICT')
        c.execute("INSERT OR IGNORE INTO metadata VALUES ('role',?)",(role,))
        for item in items:
            task=item['task']; original=digest(json_bytes(item)); tid=task['task_id']
            if task.get('split')!=role: raise ReviewError('SPLIT_CONFLICT')
            before=c.execute('SELECT original_hash FROM items WHERE task_id=?',(tid,)).fetchone()
            if before:
                if before['original_hash']!=original: raise ReviewError('SOURCE_CONFLICT')
                continue
            annotation=deepcopy(item['annotation'])
            annotation['occurrences']=[validate_task_occurrence(task,o) for o in annotation['occurrences']]
            task={**task,'occurrence_audit_ids':item.get('occurrence_audit_ids',[])}
            c.execute('INSERT INTO items VALUES (?,?,?,?,?,?)',
                      (tid,original,json.dumps(task),json.dumps(annotation),1,'unreviewed'))
        c.commit()
    except Exception:
        c.rollback();raise


def get_item(c, task_id: str) -> dict:
    row=c.execute('SELECT * FROM items WHERE task_id=?',(task_id,)).fetchone()
    if not row: raise ReviewError('UNKNOWN_TASK')
    return {'task':json.loads(row['task']),'annotation':json.loads(row['annotation']),
            'annotation_revision':row['revision'],'status':row['status']}


def queue(c, filters=None) -> dict:
    from .routing import review_reasons
    filters=filters or {}; rows=[]; reviewed=0
    for row in c.execute('SELECT task_id,status FROM items ORDER BY task_id'):
        item=get_item(c,row['task_id']); task=item['task']; annotation=item['annotation']
        reviewed+=item['status']=='reviewed'
        reasons=sorted(set(r for o in annotation['occurrences'] for r in review_reasons(o,None)))
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


def apply_decision(c, decision: dict) -> dict:
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
        if any(decision.get(k)!=task[k] for k in ('document_id','text_revision')):
            raise ReviewError('SOURCE_REVISION_MISMATCH')
        if type(decision.get('base_annotation_revision')) is not int or decision['base_annotation_revision']!=item['annotation_revision']:
            raise ReviewError('STALE_REVISION')
        action=decision.get('action'); value=decision.get('value'); target=decision.get('target_name_span')
        occurrences=annotation['occurrences']
        existing=next((o for o in occurrences if o['name_span']==target),None)
        now=datetime.now(timezone.utc).isoformat()
        provenance={'status':'human_reviewed','reviewer':decision['reviewer'],'decision_id':decision['decision_id'],
                    'recorded_at_utc':now,'reasons':[]}
        status=item['status']
        if action=='upsert_occurrence':
            try: new=validate_task_occurrence(task,value)
            except (ValueError,KeyError,TypeError) as exc: raise ReviewError('INVALID_OCCURRENCE:'+str(exc)) from exc
            if target is not None and existing is None: raise ReviewError('UNKNOWN_OCCURRENCE')
            if any(o is not existing and o['name_span']==new['name_span'] for o in occurrences):
                raise ReviewError('DUPLICATE_OCCURRENCE')
            if existing: occurrences.remove(existing)
            new['review']=provenance;occurrences.append(new)
            status='in_progress'
        elif action=='accept_passage':
            annotation['human_passage_review']={'fields':['software'],'reviewer':decision['reviewer'],
                                                 'decision_id':decision['decision_id'],'recorded_at_utc':now}
            # Human name coverage supplements, rather than replaces, agent field coverage.
            regions=annotation.get('covered_regions',[])+[
                {**r,'fields':dict.fromkeys(FIELDS,False)} for r in annotation.get('unresolved_regions',[])]
            for region in regions:
                region['fields']['software']=True
                region['human_reviewed_fields']=sorted(set(region.get('human_reviewed_fields',[]))|{'software'})
                region['status']='complete' if all(region['fields'].values()) else 'partial'
            annotation['covered_regions']=sorted(regions,key=lambda r:r['start'])
            annotation['unresolved_regions']=[]
            status='reviewed'
        elif action in ('accept_occurrence','remove_occurrence','mark_field_unresolved','link_version','unlink_version'):
            if existing is None: raise ReviewError('UNKNOWN_OCCURRENCE')
            if action=='remove_occurrence': occurrences.remove(existing)
            else:
                existing['review']=provenance
                if action=='mark_field_unresolved':
                    field=decision.get('field')
                    if field=='sentiment': existing['sentiment']=None;existing['known']['sentiment']=False;existing['evidence']['sentiment']=[]
                    elif field=='versions': existing['version_status']='ambiguous';existing['known']['versions']=False
                    elif field=='intents':
                        existing['intents']=None;existing['evidence']['intents']=[]
                        for key in INTENT_BITS: existing['known'][key]=False
                    else: raise ReviewError('UNKNOWN_FIELD')
                    existing['review']['status']='unresolved';existing['review']['reasons']=['unresolved_field']
                elif action=='link_version':
                    existing['version_links'].append(value);existing['version_status']='explicit'
                elif action=='unlink_version':
                    if value not in existing['version_links']: raise ReviewError('UNKNOWN_VERSION_LINK')
                    existing['version_links'].remove(value)
                    if not existing['version_links']: existing['version_status']='absent' if existing['known']['versions'] else 'unannotated'
                validate_task_occurrence(task,existing)
            status='in_progress'
        else: raise ReviewError('UNKNOWN_ACTION')
        annotation['occurrences']=sorted(occurrences,key=lambda o:o['name_span']['start'])
        covered=annotation.get('covered_regions',[])
        annotation['status']='complete' if covered and not annotation.get('unresolved_regions') and all(
            r['status']=='complete' and all(r['fields'].values()) for r in covered) else 'partial'
        annotation['complete_negative_regions']=[deepcopy(r) for r in covered
            if r['status']=='complete' and all(r['fields'].values())
            and not any(r['start'] < o['name_span']['end'] and o['name_span']['start'] < r['end'] for o in occurrences)]
        annotation['annotation_revision']=item['annotation_revision']+1
        result={'decision_id':decision['decision_id'],'task_id':task['task_id'],
                'annotation_revision':annotation['annotation_revision'],'recorded_at_utc':now}
        c.execute('UPDATE items SET annotation=?,revision=?,status=? WHERE task_id=?',
                  (json.dumps(annotation),annotation['annotation_revision'],status,task['task_id']))
        c.execute('INSERT INTO decisions VALUES (?,?,?,?,?)',(decision['decision_id'],task['task_id'],payload,json.dumps(result),now))
        c.commit();return result
    except Exception:
        c.rollback();raise


def export_reference(c, destination: Path) -> dict:
    if destination.exists(): raise FileExistsError(destination)
    c.execute('BEGIN')
    try:
        items=[get_item(c,r[0]) for r in c.execute('SELECT task_id FROM items ORDER BY task_id')]
        role=c.execute("SELECT value FROM metadata WHERE key='role'").fetchone()[0]
        decisions=[{**json.loads(r['payload']),'recorded_at_utc':r['recorded_at']} for r in c.execute('SELECT * FROM decisions ORDER BY rowid')]
        c.commit()
    except Exception:
        c.rollback();raise
    destination.mkdir(parents=True)
    coverage=[{'document_id':i['task']['document_id'],'text_revision':i['task']['text_revision'],**r}
              for i in items for r in i['annotation'].get('covered_regions',[])]
    files={'items.jsonl':items,'occurrences.jsonl':[o for i in items for o in i['annotation']['occurrences']],
           'coverage.jsonl':coverage,'decisions.jsonl':decisions}
    for name,rows in files.items():write_jsonl(destination/name,rows)
    manifest={'role':role,'quality':'provisional','task_count':len(items), 'decision_count':len(decisions),
              'files':[{'path':n,'sha256':digest((destination/n).read_bytes())} for n in files]}
    write_once(destination/'manifest.json',json_bytes(manifest))
    return manifest
