"""Copy-on-write alias review, independent from occurrence and passage labels."""

from copy import deepcopy

from .aliases import alias_relation_id, validate_alias_annotations

ALIAS_ACTIONS={'upsert_alias','accept_alias','reject_alias','unresolve_alias','remove_alias'}


def validate_annotation_aliases(task: dict, annotation: dict) -> None:
    if 'alias_annotations' in annotation:
        if annotation['alias_annotations'] is None:
            raise ValueError('ALIAS_SCHEMA_UNSUPPORTED')
        validate_alias_annotations(task,annotation['occurrences'],annotation['alias_annotations'])


def assert_alias_members_unchanged(annotation: dict, old_mention_id: str, new_mention_id: str | None) -> None:
    if old_mention_id==new_mention_id:
        return
    if any(old_mention_id in relation['member_mention_ids']
           for relation in annotation.get('alias_annotations',{}).get('relations',[])):
        raise ValueError('ALIAS_MEMBER_IN_USE')


def apply_alias_decision(task: dict, annotation: dict, decision: dict, provenance: dict) -> dict:
    validate_annotation_aliases(task,annotation)
    if task.get('policy_version')!='scibert-poc-2.1' or 'aliases' not in task.get('requested_fields',[]):
        raise ValueError('ALIAS_TASK_UNSUPPORTED')
    result=deepcopy(annotation)
    layer=result.setdefault('alias_annotations',{'schema_version':'1.0','relations':[]})
    relations=layer['relations']; action=decision['action']; target=decision.get('target_relation_id')
    existing=next((r for r in relations if r['relation_id']==target),None)
    if (target is not None or action!='upsert_alias') and existing is None:
        raise ValueError('UNKNOWN_ALIAS_RELATION')
    review=deepcopy(provenance)
    if action=='upsert_alias':
        value=decision.get('value')
        fields=('member_mention_ids','relation_type','decision','preferred_mention_id','evidence_spans')
        if not isinstance(value,dict) or any(key not in value for key in fields):
            raise ValueError('INVALID_ALIAS_RECORD')
        members=value['member_mention_ids']
        if not isinstance(members,list) or any(not isinstance(mid,str) for mid in members):
            raise ValueError('INVALID_ALIAS_RECORD')
        relation={key:deepcopy(value[key]) for key in fields}
        relation.update(member_mention_ids=sorted(members),document_id=task['document_id'],
                        text_revision=task['text_revision'],review=review,
                        relation_id=alias_relation_id(task['document_id'],task['text_revision'],members))
        if relation['decision']=='unresolved':
            review.update(status='unresolved',reasons=['unresolved_alias'])
        if existing is not None: relations.remove(existing)
        relations.append(relation)
    elif action=='remove_alias':
        relations.remove(existing)
    elif action=='accept_alias':
        if existing['decision']!='alias': raise ValueError('ALIAS_NOT_POSITIVE')
        existing['review']=review
    elif action in ('reject_alias','unresolve_alias'):
        existing.update(decision='not_alias' if action=='reject_alias' else 'unresolved',
                        preferred_mention_id=None,review=review)
        if action=='unresolve_alias': review.update(status='unresolved',reasons=['unresolved_alias'])
    else:
        raise ValueError('UNKNOWN_ACTION')
    validate_annotation_aliases(task,result)
    return result
