import math
import random
from pathlib import Path

from ..data.manifest import digest, json_bytes, write_once
from .aliases import build_alias_groups


def alias_review_reasons(annotation: dict) -> list[str]:
    layer=annotation.get('alias_annotations')
    if layer is None: return []
    reasons=set()
    for relation in layer['relations']:
        reasons.update(relation['review']['reasons'])
        if relation['decision']=='unresolved': reasons.add('unresolved_alias')
        elif relation['review']['status']!='human_reviewed': reasons.add('alias_needs_review')
    for group in build_alias_groups(annotation['occurrences'],layer):
        reasons.update(group.get('review_reasons',[]))
    return sorted(reasons)


def review_reasons(occurrence: dict, disagreement: dict | None) -> list[str]:
    reasons=set((occurrence.get('review') or {}).get('reasons',[]))
    if disagreement: reasons.add('evidence_disagreement')
    if occurrence.get('sentiment') in ('positive','negative','mixed'): reasons.add('expressed_sentiment')
    if set(occurrence.get('intents') or []) & {'created','shared'}: reasons.add('created_or_shared')
    if len(occurrence.get('version_links',[]))>1: reasons.add('multiple_versions')
    if occurrence.get('version_status')=='ambiguous': reasons.add('ambiguous_version')
    if occurrence.get('context_kind') in ('table','reference','references'): reasons.add('table_or_reference')
    if occurrence.get('cross_sentence'): reasons.add('cross_sentence')
    if any(not v for v in occurrence.get('known',{}).values()): reasons.add('unresolved_field')
    reasons.update(occurrence.get('review_reasons',[]))
    return sorted(reasons)


def audit_passages(tasks: list[dict], seed: int, count_per_doc: int) -> list[str]:
    rng=random.Random(seed); docs={}
    for t in tasks: docs.setdefault(t['document_id'],[]).append(t['task_id'])
    result=[]
    for doc in sorted(docs):
        ids=sorted(docs[doc]); result.extend(rng.sample(ids,min(count_per_doc,len(ids))))
    return sorted(result)


def freeze_occurrence_audit(occurrences: list[dict], path: Path, seed: int) -> dict:
    pools={}
    for o in occurrences:
        if not review_reasons(o,None): pools.setdefault((o['source'],o['split']),[]).append(o['mention_id'])
    rng=random.Random(seed); selected=[]
    for key, ids in sorted(pools.items()):
        selected.extend(rng.sample(sorted(ids),math.ceil(len(ids)*.1)))
    result={'pool_digest':digest(json_bytes(sorted(o['mention_id'] for o in occurrences))),
            'seed':seed,'selected_ids':sorted(selected),'skipped_ids':[]}
    write_once(path,json_bytes(result))
    return result
