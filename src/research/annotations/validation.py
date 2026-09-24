from copy import deepcopy
import re
from pathlib import Path

from ..contracts import FIELDS, occurrence_id, text_revision, validate_occurrence
from ..data.manifest import digest, json_bytes, write_once


def _inside(span, lo, hi):
    return (isinstance(span,dict) and type(span.get('start')) is int and type(span.get('end')) is int
            and lo <= span['start'] < span['end'] <= hi)


def validate_task_occurrence(task: dict, occurrence: dict) -> dict:
    row = deepcopy(occurrence)
    if row.get('document_id') != task['document_id'] or row.get('text_revision') != task['text_revision']:
        raise ValueError('OCCURRENCE_IDENTITY_MISMATCH')
    owned=task['annotation_region']; span=row.get('name_span',{})
    if not _inside(span,owned['start'],owned['end']):
        raise ValueError('OUTSIDE_OWNED_REGION')
    base=task['offset_base']; local=deepcopy(row)
    positions=[local['name_span'],local['context_span']]+[e['span'] for e in local['version_links']]
    positions += [s for spans in local.get('evidence',{}).values() for s in spans]
    for s in positions:
        if not _inside(s,base,base+len(task['text'])):
            raise ValueError('OUTSIDE_CONTEXT')
        s['start']-=base; s['end']-=base
    local['text_revision']=text_revision(task['text'])
    local['mention_id']=occurrence_id(local['document_id'],local['text_revision'],local['name_span']['start'],local['name_span']['end'])
    validate_occurrence(local,task['text'])
    mid=occurrence_id(task['document_id'],task['text_revision'],span['start'],span['end'])
    if 'mention_id' in row and row['mention_id'] != mid:
        raise ValueError('MENTION_ID_MISMATCH')
    row['mention_id']=mid
    return row


def validate_reply(task: dict, reply: dict) -> dict:
    for key in ('task_id','document_id','text_revision','policy_version'):
        if reply.get(key) != task[key]:
            raise ValueError('REPLY_IDENTITY_MISMATCH:'+key)
    if reply.get('status') not in ('complete','partial'):
        raise ValueError('INVALID_REPLY_STATUS')
    if not isinstance(reply.get('attempt_id'),str) or not reply['attempt_id'].strip():
        raise ValueError('ATTEMPT_ID_REQUIRED')
    actor=reply.get('annotator',{})
    if (actor.get('runtime') != 'current_codex_session' or not actor.get('run_identifier')
            or not re.fullmatch('[0-9a-f]{64}',actor.get('prompt_hash',''))
            or 'model_identifier' not in actor):
        raise ValueError('ANNOTATOR_PROVENANCE_REQUIRED')
    owned=task['annotation_region']; lo,hi=owned['start'],owned['end']
    covered=reply.get('covered_regions'); unresolved=reply.get('unresolved_regions')
    if not isinstance(covered,list) or not isinstance(unresolved,list) or not isinstance(reply.get('occurrences'),list):
        raise ValueError('REPLY_ARRAYS_REQUIRED')
    regions=[]
    for r in covered:
        if (not _inside(r,lo,hi) or r.get('status') not in ('complete','partial')
                or not isinstance(r.get('fields'),dict) or set(r['fields']) != set(FIELDS)
                or any(type(v) is not bool for v in r['fields'].values())):
            raise ValueError('INVALID_COVERAGE')
        if reply['status']=='complete' and (r['status']!='complete' or not all(r['fields'].values())):
            raise ValueError('INCOMPLETE_COVERAGE')
        regions.append(r)
    for r in unresolved:
        if not _inside(r,lo,hi):
            raise ValueError('INVALID_UNRESOLVED_REGION')
        regions.append(r)
    cursor=lo
    for r in sorted(regions,key=lambda r:r['start']):
        if r['start'] != cursor:
            raise ValueError('COVERAGE_GAP_OR_OVERLAP')
        cursor=r['end']
    if cursor!=hi or (reply['status']=='complete' and unresolved):
        raise ValueError('COVERAGE_GAP')
    occurrences=[validate_task_occurrence(task,r) for r in reply['occurrences']]
    if len({r['mention_id'] for r in occurrences}) != len(occurrences):
        raise ValueError('DUPLICATE_OCCURRENCE')
    result=deepcopy(reply); result['occurrences']=occurrences
    result['review_status']='agent_provisional'
    result['complete_negative_regions']=[deepcopy(r) for r in covered if r['status']=='complete' and all(r['fields'].values())
                                         and not any(r['start'] < o['name_span']['end'] and o['name_span']['start'] < r['end'] for o in occurrences)]
    result['annotation_revision']=1
    return result


def store_reply(destination: Path, task: dict, reply: dict) -> dict:
    result=validate_reply(task,reply)
    path=destination/(digest(json_bytes([task['task_id'],reply['attempt_id']]))+'.json')
    write_once(path,json_bytes(result))
    return {'path':path.name,'sha256':digest(path.read_bytes()), 'task_id':task['task_id'], 'attempt_id':reply['attempt_id']}
