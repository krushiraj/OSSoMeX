import re

from ..contracts import validate_document
from ..data.manifest import digest, json_bytes


def make_tasks(document: dict, policy: dict) -> list[dict]:
    doc = validate_document(document)
    text = doc['text']; limit = min(policy.get('max_chars',6000),6000)
    if limit < 1:
        raise ValueError('INVALID_TASK_LIMIT')
    tasks = []
    starts = [0] + [m.end() for m in re.finditer(r'\n\s*\n',text)]
    for para_start, para_end in zip(starts, starts[1:]+[len(text)]):
        boundaries = [para_start] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+',text[para_start:para_end])]
        # Regex positions above are paragraph-local.
        boundaries = [para_start] + [para_start+b for b in boundaries[1:]] + [para_end]
        start = para_start
        while start < para_end:
            candidates = [b for b in boundaries if start < b <= start+limit]
            end = max(candidates) if candidates else min(start+limit,para_end)
            hard = start not in boundaries or end not in boundaries
            lo = max([b for b in boundaries if b < start],default=para_start)
            hi = min([b for b in boundaries if b > end],default=para_end)
            # A single huge sentence cannot be repeated as unbounded context.
            lo = max(lo,start-limit); hi = min(hi,end+limit)
            identity = [doc['document_id'],doc['text_revision'],policy['policy_version'],policy['policy_hash'],start,end]
            tasks.append({'task_id':'task:'+digest(json_bytes(identity))[:32], 'document_id':doc['document_id'],
                          'text_revision':doc['text_revision'], 'policy_version':policy['policy_version'],
                          'policy_hash':policy['policy_hash'], 'text':text[lo:hi], 'context_span':{'start':lo,'end':hi},
                          'annotation_region':{'start':start,'end':end}, 'offset_base':lo,
                          'requested_fields':['software','version','version_links','intents','sentiment'],
                          'candidate_system_predictions':None, 'hard_split':hard,
                          'source':doc.get('source'), 'split':doc.get('split'), 'public':doc.get('public',False),
                          'access_basis':doc.get('access_basis'), 'text_license':doc.get('text_license')})
            start=end
    return tasks


def check_authorization(task: dict, approvals: dict) -> None:
    permission = approvals.get('scoped_authorizations',{}).get('codex_public_text_annotation',{})
    if permission.get('approved') is not True or permission.get('runtime') != 'current_codex_session':
        raise ValueError('ANNOTATION_NOT_AUTHORIZED')
    if task.get('public') is not True or not task.get('access_basis') or not task.get('text_license'):
        raise ValueError('PUBLIC_PROVENANCE_REQUIRED')
    if task.get('split') not in ('train','dev'):
        raise ValueError('ISOLATED_TEST_CONTEXT_REQUIRED')
