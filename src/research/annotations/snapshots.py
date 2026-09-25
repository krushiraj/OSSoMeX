"""Verified review artifacts and lossless, fresh-database snapshot restoration."""

import base64
import json
from pathlib import Path
import re
import sqlite3

from ..data.manifest import digest, json_bytes, read_jsonl, verified_path, write_once
from .alias_decisions import validate_annotation_aliases
from .aliases import build_alias_groups
from .policies import load_policy
from .validation import validate_task_occurrence
from .review_workflow import validate_workflow

LEGACY_PROVENANCE={'kind':'legacy_no_frozen_policy_source'}
SNAPSHOT_VERSION='1.0'
WORKFLOW_SNAPSHOT_VERSION='1.1'


def snapshot_format(items: list[dict]) -> dict:
    workflow=any('review_workflow' in item['annotation'] for item in items)
    result={'snapshot_schema_version':WORKFLOW_SNAPSHOT_VERSION if workflow else SNAPSHOT_VERSION,
            'capabilities':{'aliases':True,'exact_store_restore':True}}
    if workflow:
        result['capabilities']['review_workflow']=True
        result['review_workflow_schema_version']='1.0'
    return result


def validate_review_history(items: list[dict], decision_records: list[dict]) -> None:
    _validate_decision_rows(decision_records,items)
    histories={item['task']['task_id']:[] for item in items}
    for row in decision_records:
        if row['task_id'] not in histories: raise ValueError('SNAPSHOT_DECISION_REFERENCE')
        payload=json.loads(row['payload']); result=json.loads(row['result'])
        if payload.get('action')=='apply_review_batch':
            _validate_batch_record(payload,result)
        histories[row['task_id']].append({'payload':payload,'result':result,'recorded_at':row['recorded_at']})
    for item in items:
        history=histories[item['task']['task_id']]
        if 'review_workflow' not in item['annotation'] and any(
                row['payload'].get('action')=='apply_review_batch' for row in history):
            raise ValueError('SNAPSHOT_WORKFLOW_REQUIRED')
        validate_workflow(item,history)


def _validate_batch_record(payload: dict, result: dict) -> None:
    from .review_batches import _validate_operation
    value=payload.get('value'); actor=payload.get('actor_kind')
    if (not isinstance(value,dict) or set(value)!={'schema_version','completion','proposals_revealed','operations'}
            or value['schema_version']!='1.0' or value['completion'] not in ('save','approve')
            or type(value['proposals_revealed']) is not bool or actor not in ('human','system')
            or not isinstance(value['operations'],list) or len(value['operations'])>200):
        raise ValueError('SNAPSHOT_BATCH_INVALID')
    generated=result.get('approval_operations'); outcomes=result.get('operation_results')
    if not isinstance(generated,list) or not isinstance(outcomes,list) or not isinstance(result.get('field_hashes'),dict):
        raise ValueError('SNAPSHOT_BATCH_RESULT_INVALID')
    operations=value['operations']+generated
    for operation in operations: _validate_operation(operation)
    if (len({op['operation_id'] for op in operations})!=len(operations)
            or len(outcomes)!=len(operations)
            or any(not isinstance(outcome,dict) or outcome.get('operation')!=operation
                   or outcome.get('operation_id')!=operation['operation_id']
                   or outcome.get('action')!=operation['action']
                   or outcome.get('generated') is not (index>=len(value['operations']))
                   or not {'before','after'}<=outcome.keys()
                   for index,(operation,outcome) in enumerate(zip(operations,outcomes)))):
        raise ValueError('SNAPSHOT_BATCH_RESULT_INVALID')
    if (actor=='system' and (value['completion']!='save' or any(
            op['action'] not in ('record_note','record_source_issue') for op in operations))
            or value['completion']=='approve' and not value['proposals_revealed']
            or generated and (actor!='human' or value['completion']!='approve' or any(
                op['action'] not in ('accept_fields','accept_alias','reject_alias') for op in generated))):
        raise ValueError('SNAPSHOT_BATCH_ACTOR_INVALID')


def read_policy_provenance(bundle: Path, manifest: dict) -> dict:
    if 'policy_source' not in manifest:
        if 'prompt_sources' in manifest or manifest.get('policy_provenance',LEGACY_PROVENANCE)!=LEGACY_PROVENANCE:
            raise ValueError('POLICY_SNAPSHOT_REQUIRED')
        return dict(LEGACY_PROVENANCE)
    source=manifest['policy_source']; prompts=manifest.get('prompt_sources'); policy=manifest.get('policy')
    if (not isinstance(source,dict) or not isinstance(policy,dict)
            or not isinstance(prompts,dict) or set(prompts)!={'annotate','check'}
            or source.get('sha256')!=policy.get('policy_hash')):
        raise ValueError('POLICY_SNAPSHOT_MISMATCH')
    records=[source,*prompts.values()]; snapshots=[]; seen=set()
    for record in records:
        if (not isinstance(record,dict) or record.get('path') in seen
                or not any(row=={'path':record.get('path'),'sha256':record.get('sha256')}
                           for row in manifest['files'])):
            raise ValueError('POLICY_SNAPSHOT_MISMATCH')
        path=verified_path(bundle,record); seen.add(record['path'])
        snapshots.append({'path':record['path'],'sha256':record['sha256'],
                          'content_base64':base64.b64encode(path.read_bytes()).decode('ascii')})
    if load_policy(verified_path(bundle,source))!=policy:
        raise ValueError('POLICY_SNAPSHOT_MISMATCH')
    return {'kind':'frozen_policy_sources','policy':policy,'policy_source':source,
            'prompt_sources':prompts,'snapshots':snapshots}


def write_policy_provenance(destination: Path, provenance: dict) -> dict:
    if provenance==LEGACY_PROVENANCE:
        return {'policy_provenance':dict(LEGACY_PROVENANCE),'files':[]}
    files=[]
    for record in provenance['snapshots']:
        relative=Path(record['path']); path=(destination/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(destination.resolve()):
            raise ValueError('unsafe artifact path')
        payload=base64.b64decode(record['content_base64'],validate=True)
        if digest(payload)!=record['sha256']: raise ValueError('POLICY_SNAPSHOT_MISMATCH')
        write_once(path,payload)
        files.append({'path':record['path'],'sha256':record['sha256']})
    return {key:provenance[key] for key in ('policy','policy_source','prompt_sources')} | {
        'policy_provenance':{'kind':'frozen_policy_sources'},'files':files}


def validate_item_policy(items: list[dict], provenance: dict, role: str) -> None:
    if role not in ('train','dev','demo'): raise ValueError('SPLIT_NOT_PERMITTED')
    legacy=provenance==LEGACY_PROVENANCE
    if not legacy and (not isinstance(provenance,dict) or provenance.get('kind')!='frozen_policy_sources'):
        raise ValueError('POLICY_SNAPSHOT_REQUIRED')
    for item in items:
        task=item['task']
        if task.get('split')!=role: raise ValueError('SPLIT_CONFLICT')
        requires_alias_policy=(task.get('policy_version')=='scibert-poc-2.1'
                               or 'aliases' in task.get('requested_fields',[])
                               or 'alias_annotations' in item['annotation'])
        synthetic=role=='demo' and task.get('source')=='synthetic-contract-fixture'
        if legacy:
            if requires_alias_policy and not synthetic: raise ValueError('POLICY_SNAPSHOT_REQUIRED')
            continue
        policy=provenance['policy']
        if requires_alias_policy and not synthetic and policy.get('policy_version')!='scibert-poc-2.1':
            raise ValueError('TASK_POLICY_MISMATCH')
        if any(task.get(key)!=policy[key] for key in ('policy_version','policy_hash')):
            raise ValueError('TASK_POLICY_MISMATCH')
        prompt_hashes={r['sha256'] for r in provenance['prompt_sources'].values()}
        if item['annotation'].get('annotator',{}).get('prompt_hash') not in prompt_hashes:
            raise ValueError('PROMPT_HASH_MISMATCH')


def reference_rows(items: list[dict], decision_records: list[dict]) -> dict:
    aliases=[]; groups=[]
    for item in items:
        task=item['task']; annotation=item['annotation']
        validate_annotation_aliases(task,annotation)
        aliases.extend({**r,'task_id':task['task_id']} for r in annotation.get('alias_annotations',{}).get('relations',[]))
        groups.extend({**g,'task_id':task['task_id']} for g in build_alias_groups(
            annotation['occurrences'],annotation.get('alias_annotations')))
    return {'items.jsonl':items,
            'occurrences.jsonl':[o for i in items for o in i['annotation']['occurrences']],
            'coverage.jsonl':[{'document_id':i['task']['document_id'],'text_revision':i['task']['text_revision'],**r}
                              for i in items for r in i['annotation'].get('covered_regions',[])],
            'decisions.jsonl':[{**json.loads(r['payload']),'recorded_at_utc':r['recorded_at']} for r in decision_records],
            'aliases.jsonl':aliases,'alias_groups.jsonl':groups}


def _validate_store_rows(rows: list[dict], role: str) -> list[dict]:
    items=[]; seen=set()
    for row in rows:
        if (set(row)!={'task_id','original_hash','task','annotation','revision','status'}
                or any(not isinstance(row[k],str) for k in ('task_id','original_hash','task','annotation','status'))
                or not re.fullmatch('[0-9a-f]{64}',row['original_hash'])
                or type(row['revision']) is not int or row['revision']<1
                or row['status'] not in ('unreviewed','in_progress','reviewed')):
            raise ValueError('SNAPSHOT_ITEM_INVALID')
        task=json.loads(row['task']); annotation=json.loads(row['annotation']); tid=row['task_id']
        if tid in seen: raise ValueError('SNAPSHOT_DUPLICATE_TASK')
        seen.add(tid)
        owned=task['annotation_region']; context=task['context_span']
        identity=[task['document_id'],task['text_revision'],task['policy_version'],task['policy_hash'],owned['start'],owned['end']]
        if (task['task_id']!=tid or tid!='task:'+digest(json_bytes(identity))[:32]
                or task.get('split')!=role or annotation.get('annotation_revision')!=row['revision']
                or any(key in annotation and annotation[key]!=task[key]
                       for key in ('task_id','document_id','text_revision','policy_version'))
                or task['offset_base']!=context['start']
                or len(task['text'])!=context['end']-context['start']
                or not 0<=context['start']<=owned['start']<owned['end']<=context['end']):
            raise ValueError('SNAPSHOT_ITEM_IDENTITY_MISMATCH')
        occurrences=[validate_task_occurrence(task,o) for o in annotation['occurrences']]
        if occurrences!=annotation['occurrences'] or len({o['mention_id'] for o in occurrences})!=len(occurrences):
            raise ValueError('SNAPSHOT_OCCURRENCE_INVALID')
        validate_annotation_aliases(task,annotation)
        items.append({'task':task,'annotation':annotation,'annotation_revision':row['revision'],'status':row['status']})
    return items


def _validate_decision_rows(rows: list[dict], items: list[dict]) -> None:
    tasks={i['task']['task_id']:i for i in items}; revisions=dict.fromkeys(tasks,1); seen=set()
    for row in rows:
        if (set(row)!={'decision_id','task_id','payload','result','recorded_at'}
                or any(not isinstance(value,str) or not value for value in row.values())):
            raise ValueError('SNAPSHOT_DECISION_INVALID')
        if row['decision_id'] in seen: raise ValueError('SNAPSHOT_DUPLICATE_DECISION')
        seen.add(row['decision_id'])
        if row['task_id'] not in tasks: raise ValueError('SNAPSHOT_DECISION_REFERENCE')
        task=tasks[row['task_id']]['task']; payload=json.loads(row['payload']); result=json.loads(row['result'])
        if (any(payload.get(k)!=row[k] or result.get(k)!=row[k] for k in ('decision_id','task_id'))
                or any(payload.get(k)!=task[k] for k in ('document_id','text_revision'))
                or type(payload.get('base_annotation_revision')) is not int
                or payload['base_annotation_revision']!=revisions[row['task_id']]
                or type(result.get('annotation_revision')) is not int
                or result['annotation_revision']!=revisions[row['task_id']]+1
                or result.get('recorded_at_utc')!=row['recorded_at']
                or any(not isinstance(payload.get(k),str) or not payload[k].strip() for k in ('reviewer','reason'))):
            raise ValueError('SNAPSHOT_DECISION_IDENTITY_MISMATCH')
        revisions[row['task_id']]+=1
    if any(i['annotation_revision']!=revisions[i['task']['task_id']] for i in items):
        raise ValueError('SNAPSHOT_REVISION_MISMATCH')


def read_review_snapshot(bundle: Path) -> tuple[dict, dict]:
    manifest=json.loads((bundle/'manifest.json').read_bytes())
    required={'store-items.jsonl','decision-records.jsonl','metadata.jsonl','items.jsonl',
              'occurrences.jsonl','coverage.jsonl','decisions.jsonl','aliases.jsonl','alias_groups.jsonl'}
    records=manifest.get('files',[]); paths=[r['path'] for r in records]
    version=manifest.get('snapshot_schema_version')
    if version not in (SNAPSHOT_VERSION,WORKFLOW_SNAPSHOT_VERSION) or not required<=set(paths):
        raise ValueError('SNAPSHOT_RESTORE_UNSUPPORTED')
    if len(paths)!=len(set(paths)): raise ValueError('SNAPSHOT_DUPLICATE_FILE')
    if manifest.get('role') not in ('train','dev','demo'): raise ValueError('SPLIT_NOT_PERMITTED')
    capabilities={'aliases':True,'exact_store_restore':True}
    if version==WORKFLOW_SNAPSHOT_VERSION:
        capabilities['review_workflow']=True
        if manifest.get('review_workflow_schema_version')!='1.0': raise ValueError('SNAPSHOT_RESTORE_UNSUPPORTED')
    elif 'review_workflow_schema_version' in manifest: raise ValueError('SNAPSHOT_RESTORE_UNSUPPORTED')
    if (manifest.get('alias_schema_version')!='1.0' or manifest.get('capabilities')!=capabilities
            or any(value is not True for value in manifest['capabilities'].values())):
        raise ValueError('SNAPSHOT_RESTORE_UNSUPPORTED')
    for record in records: verified_path(bundle,record)
    data={name:read_jsonl(bundle/name) for name in required}
    metadata=data['metadata.jsonl']; values={}
    for row in metadata:
        if (set(row)!={'key','value'} or any(not isinstance(v,str) for v in row.values())
                or row['key'] in values): raise ValueError('SNAPSHOT_METADATA_INVALID')
        values[row['key']]=row['value']
    if values.get('role')!=manifest['role']: raise ValueError('SPLIT_CONFLICT')
    provenance=read_policy_provenance(bundle,manifest)
    if json.loads(values.get('policy_sources',json.dumps(LEGACY_PROVENANCE)))!=provenance:
        raise ValueError('POLICY_SNAPSHOT_MISMATCH')
    try:
        items=_validate_store_rows(data['store-items.jsonl'],manifest['role'])
        validate_review_history(items,data['decision-records.jsonl'])
        if snapshot_format(items)['snapshot_schema_version']!=version:
            raise ValueError('SNAPSHOT_WORKFLOW_FORMAT_MISMATCH')
        validate_item_policy(items,provenance,manifest['role'])
        if any(data[name]!=rows for name,rows in reference_rows(items,data['decision-records.jsonl']).items()):
            raise ValueError('SNAPSHOT_SIDECAR_MISMATCH')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('SNAPSHOT_INVALID') from exc
    if manifest.get('task_count')!=len(items) or manifest.get('decision_count')!=len(data['decision-records.jsonl']):
        raise ValueError('SNAPSHOT_COUNT_MISMATCH')
    return manifest,data


def validate_snapshot_store(bundle: Path, store: Path) -> None:
    manifest,data=read_review_snapshot(bundle)
    if not store.is_file(): raise ValueError('SNAPSHOT_STORE_REQUIRED: restore the snapshot first')
    c=None
    try:
        c=sqlite3.connect(store.resolve().as_uri()+'?mode=ro',uri=True)
        c.row_factory=sqlite3.Row; c.execute('BEGIN')
        rows=[dict(row) for row in c.execute('SELECT * FROM items ORDER BY task_id')]
        decisions=[dict(row) for row in c.execute('SELECT * FROM decisions ORDER BY rowid')]
        metadata=[dict(row) for row in c.execute('SELECT * FROM metadata ORDER BY rowid')]
        if metadata!=data['metadata.jsonl']: raise ValueError('SNAPSHOT_STORE_METADATA_MISMATCH')
        source_keys=('task_id','original_hash','task')
        sources=[tuple(row[key] for key in source_keys) for row in rows]
        expected=[tuple(row[key] for key in source_keys) for row in data['store-items.jsonl']]
        if sources!=expected: raise ValueError('SNAPSHOT_STORE_SOURCE_MISMATCH')
        items=_validate_store_rows(rows,manifest['role'])
        validate_review_history(items,decisions)
        snapshot_decisions=data['decision-records.jsonl']
        if decisions[:len(snapshot_decisions)]!=snapshot_decisions:
            raise ValueError('SNAPSHOT_STORE_HISTORY_MISMATCH')
        advanced_tasks={row['task_id'] for row in decisions[len(snapshot_decisions):]}
        baseline={row['task_id']:row for row in data['store-items.jsonl']}
        if any(row['task_id'] not in advanced_tasks and row!=baseline[row['task_id']] for row in rows):
            raise ValueError('SNAPSHOT_STORE_STATE_MISMATCH')
    except (sqlite3.DatabaseError,KeyError,TypeError,AttributeError) as exc:
        raise ValueError('SNAPSHOT_STORE_INVALID') from exc
    finally:
        if c is not None: c.close()


def restore_review_snapshot(bundle: Path, destination: Path) -> dict:
    from .review_store import open_store
    if destination.exists(): raise FileExistsError(destination)
    manifest,data=read_review_snapshot(bundle)
    metadata=data['metadata.jsonl']; items=data['items.jsonl']
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb'): pass
    c=None
    try:
        c=open_store(destination); c.execute('BEGIN IMMEDIATE')
        for row in metadata: c.execute('INSERT INTO metadata VALUES (?,?)',(row['key'],row['value']))
        for row in data['store-items.jsonl']:
            c.execute('INSERT INTO items VALUES (?,?,?,?,?,?)',tuple(row[k] for k in
                      ('task_id','original_hash','task','annotation','revision','status')))
        for row in data['decision-records.jsonl']:
            c.execute('INSERT INTO decisions VALUES (?,?,?,?,?)',tuple(row[k] for k in
                      ('decision_id','task_id','payload','result','recorded_at')))
        c.commit()
    except Exception:
        if c is not None: c.rollback(); c.close(); c=None
        for path in (destination,Path(str(destination)+'-wal'),Path(str(destination)+'-shm')):
            path.unlink(missing_ok=True)
        raise
    finally:
        if c is not None: c.close()
    return {'role':manifest['role'],'task_count':len(items),'decision_count':len(data['decision-records.jsonl'])}
