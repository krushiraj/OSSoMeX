import json
from pathlib import Path

from ..data.bundles import load_bundle
from ..data.manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from .tasks import check_authorization, make_tasks
from .tasks import make_region_task
from .policies import load_policy, snapshot_prompts
from .validation import store_reply, validate_reply
from .routing import audit_passages, freeze_occurrence_audit, review_reasons

ROOT=Path(__file__).resolve().parents[3]


def register(subparsers):
    parser=subparsers.add_parser('annotate')
    sub=parser.add_subparsers(dest='annotation_command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--policy',type=Path); p.add_argument('--regions',type=Path)
    p.add_argument('--output',type=Path,required=True); p.set_defaults(func=run)
    p=sub.add_parser('import')
    for name in ('tasks','replies','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection',type=Path); p.set_defaults(func=run)
    p=sub.add_parser('review',description='Review an import bundle, or resume a restored snapshot store.',
                     epilog='Snapshot recovery: research annotate restore --bundle SNAPSHOT --store NEW_DB; '
                            'then research annotate review --bundle SNAPSHOT --store NEW_DB. '
                            'Snapshot review requires an existing restored store and never reimports its items.')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--store',type=Path,required=True); p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8765); p.set_defaults(func=run)
    p=sub.add_parser('restore'); p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--store',type=Path,required=True); p.set_defaults(func=run)
    p=sub.add_parser('demo'); p.add_argument('--output',type=Path,required=True); p.set_defaults(func=run)


def run(args):
    if args.annotation_command=='restore':
        from .snapshots import restore_review_snapshot
        print(json.dumps(restore_review_snapshot(args.bundle,args.store)))
        return 0
    if args.annotation_command=='review':
        from .review_server import serve_review
        serve_review(args.bundle,args.store,args.host,args.port)
        return 0
    if args.output.exists(): raise FileExistsError(args.output)
    if args.annotation_command=='demo':
        return build_demo(args.output)
    if args.annotation_command=='prepare':
        manifest,docs=load_bundle(args.bundle)
        explicit=args.policy is not None or args.regions is not None
        policy_path=args.policy or ROOT/'annotations/scibert-v2/policy.md'
        policy=load_policy(policy_path) if explicit else {'policy_version':'scibert-poc-2.0',
               'policy_hash':digest(policy_path.read_bytes()),'max_chars':6000}
        approvals=json.loads((ROOT/'approvals.json').read_bytes())
        if args.regions is None:
            tasks=[t for d in docs for t in make_tasks(d,policy)]
        else:
            documents={d['document_id']:d for d in docs}
            requests=read_jsonl(args.regions)
            spans={}
            tasks=[]
            for row in requests:
                document=documents.get(row.get('document_id'))
                if document is None:
                    raise ValueError('UNKNOWN_REGION_DOCUMENT')
                if row.get('text_revision')!=document['text_revision']:
                    raise ValueError('REGION_REVISION_MISMATCH')
                task=make_region_task(document,policy,row.get('annotation_region'),row.get('context_span'),
                                      region_kind=row.get('region_kind'))
                owned=task['annotation_region']; key=task['document_id']
                if any(owned['start']<other['end'] and other['start']<owned['end'] for other in spans.get(key,[])):
                    raise ValueError('OVERLAPPING_OWNED_REGIONS')
                spans.setdefault(key,[]).append(owned)
                tasks.append(task)
        for task in tasks: check_authorization(task,approvals)
        audit=audit_passages(tasks,42,3)
        for task in tasks: task['whole_passage_audit']=task['task_id'] in audit
        args.output.mkdir(parents=True)
        write_jsonl(args.output/'tasks.jsonl',tasks)
        files=[{'path':'tasks.jsonl','sha256':digest((args.output/'tasks.jsonl').read_bytes())}]
        if explicit:
            write_once(args.output/'policy.md',policy_path.read_bytes())
            files.append({'path':'policy.md','sha256':policy['policy_hash']})
        meta={'role':manifest['role'],'policy':policy,'source_bundle_sha256':digest((args.bundle/'manifest.json').read_bytes()),
              'files':files,
              'whole_passage_audit_ids':audit, 'task_count':len(tasks)}
        if explicit:
            meta['policy_source']={'path':'policy.md','sha256':policy['policy_hash'],
                                   'source_path':str(policy_path.resolve())}
            prompts=snapshot_prompts(args.output,policy)
            meta['prompt_sources']=prompts
            files.extend({'path':record['path'],'sha256':record['sha256']} for record in prompts.values())
        write_once(args.output/'manifest.json',json_bytes(meta))
        print(json.dumps({'task_count':len(tasks),'whole_passage_audits':len(audit)}))
    else:
        meta=json.loads((args.tasks/'manifest.json').read_bytes())
        if meta['role'] not in ('train','dev'): raise ValueError('SPLIT_NOT_PERMITTED')
        for row in meta['files']: verified_path(args.tasks,row)
        tasks={t['task_id']:t for t in read_jsonl(args.tasks/'tasks.jsonl')}
        policy=meta.get('policy')
        if (not isinstance(policy,dict)
                or any(t.get('policy_version')!=policy.get('policy_version')
                       or t.get('policy_hash')!=policy.get('policy_hash') for t in tasks.values())):
            raise ValueError('TASK_POLICY_MISMATCH')
        snapshot_required=(policy.get('policy_version')=='scibert-poc-2.1'
                           or 'prompt_sources' in meta
                           or any(row.get('path')=='policy.md' for row in meta['files']))
        if snapshot_required and not isinstance(meta.get('policy_source'),dict):
            raise ValueError('POLICY_SNAPSHOT_REQUIRED')
        if 'policy_source' in meta:
            source=meta['policy_source']
            if (not isinstance(source,dict)
                    or source.get('sha256')!=policy['policy_hash']
                    or source.get('path')!='policy.md'
                    or not any(row=={'path':'policy.md','sha256':source['sha256']} for row in meta['files'])):
                raise ValueError('POLICY_SNAPSHOT_MISMATCH')
            snapshot=verified_path(args.tasks,source)
            if load_policy(snapshot)!=policy:
                raise ValueError('POLICY_SNAPSHOT_MISMATCH')
            prompts=meta.get('prompt_sources')
            if not isinstance(prompts,dict) or set(prompts)!={'annotate','check'}:
                raise ValueError('PROMPT_SNAPSHOT_MISMATCH')
            for record in prompts.values():
                if (not isinstance(record,dict) or
                        not any(row=={'path':record.get('path'),'sha256':record.get('sha256')}
                                for row in meta['files'])):
                    raise ValueError('PROMPT_SNAPSHOT_MISMATCH')
                verified_path(args.tasks,record)
        candidates={}; seen=set()
        for path in sorted(args.replies.glob('*.json')):
            reply=json.loads(path.read_bytes()); key=(reply['task_id'],reply['attempt_id'])
            if key in seen: raise ValueError('DUPLICATE_ATTEMPT')
            seen.add(key)
            if reply['task_id'] not in tasks: raise ValueError('UNKNOWN_TASK')
            if 'policy_source' in meta and reply.get('annotator',{}).get('prompt_hash') not in {
                    row['sha256'] for row in meta['prompt_sources'].values()}:
                raise ValueError('PROMPT_HASH_MISMATCH')
            valid=validate_reply(tasks[reply['task_id']],reply)
            candidates.setdefault(reply['task_id'],[]).append((reply,valid))
        selection=json.loads(args.selection.read_bytes()) if args.selection else {}
        items=[]; selected=[]
        for tid,task in tasks.items():
            attempts=candidates.get(tid,[])
            if tid in selection: attempts=[a for a in attempts if a[0]['attempt_id']==selection[tid]]
            if len(attempts)!=1: raise ValueError('EXPLICIT_SINGLE_ATTEMPT_REQUIRED:'+tid)
            original,annotation=attempts[0]
            for o in annotation['occurrences']:
                o.update(source=task['source'],split=task['split'])
                o['review']={'status':'agent_provisional','reasons':review_reasons(o,None)}
            items.append({'task':task,'annotation':annotation})
            selected.append((task,original))
        args.output.mkdir(parents=True)
        for task,reply in selected: store_reply(args.output/'attempts',task,reply)
        audit=freeze_occurrence_audit([o for item in items for o in item['annotation']['occurrences']],args.output/'occurrence-audit.json',42)
        for item in items:
            item['occurrence_audit_ids']=[o['mention_id'] for o in item['annotation']['occurrences'] if o['mention_id'] in audit['selected_ids']]
        write_jsonl(args.output/'items.jsonl',items)
        from .snapshots import read_policy_provenance, write_policy_provenance
        sources=write_policy_provenance(args.output,read_policy_provenance(args.tasks,meta))
        write_once(args.output/'manifest.json',json_bytes({**sources,'role':meta['role'],'status':'provisional',
                   'task_count':len(items),'task_manifest_sha256':digest((args.tasks/'manifest.json').read_bytes()),
                   'files':[{'path':'items.jsonl','sha256':digest((args.output/'items.jsonl').read_bytes())}]+sources['files']}))
        print(json.dumps({'status':'provisional','tasks':len(items)}))
    return 0


def build_demo(output: Path) -> int:
    from copy import deepcopy
    from ..contracts import FIELDS, occurrence_id
    pack=json.loads((ROOT/'docs/plans/scibert-contract-examples.json').read_bytes())
    policy={'policy_version':'scibert-poc-2.0','policy_hash':digest((ROOT/'annotations/scibert-v2/policy.md').read_bytes())}
    items=[]
    cases=deepcopy(pack['extraction_cases'])
    cases.append({'input':{'document_id':'fixture-missed-emoji','text':'😀 We used NumPy. The software made the analysis easier.'},'expected_occurrences':[]})
    for case in cases:
        doc={**case['input'],'source':'synthetic-contract-fixture','split':'demo'}
        for task in make_tasks(doc,policy):
            region=task['annotation_region']
            occurrences=[o for o in case['expected_occurrences'] if region['start']<=o['name_span']['start']<o['name_span']['end']<=region['end']]
            for o in occurrences:
                o['mention_id']=occurrence_id(o['document_id'],o['text_revision'],o['name_span']['start'],o['name_span']['end'])
                o['review']={'status':'synthetic_fixture','reasons':review_reasons(o,None)}
            task['whole_passage_audit']=True
            items.append({'task':task,'annotation':{'occurrences':occurrences,'status':'partial',
                         'covered_regions':[],'unresolved_regions':[region],
                         'annotation_revision':1,'review_status':'synthetic_fixture'}})
    output.mkdir(parents=True)
    write_jsonl(output/'items.jsonl',items)
    write_once(output/'manifest.json',json_bytes({'role':'demo','quality':'synthetic_not_research',
               'files':[{'path':'items.jsonl','sha256':digest((output/'items.jsonl').read_bytes())}]}))
    print(json.dumps({'status':'synthetic_demo','tasks':len(items)}))
    return 0
