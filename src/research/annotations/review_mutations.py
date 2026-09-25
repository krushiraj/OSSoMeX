"""Copy-on-write label mutations shared by legacy decisions and scoped batches."""

from copy import deepcopy

from ..contracts import FIELDS, INTENT_BITS
from .alias_decisions import ALIAS_ACTIONS, apply_alias_decision, assert_alias_members_unchanged, validate_annotation_aliases
from .validation import validate_task_occurrence


def recompute_coverage(annotation: dict) -> None:
    occurrences = annotation['occurrences']
    covered = annotation.get('covered_regions', [])
    for region in covered:
        overlapping = [o for o in occurrences
                       if region['start'] < o['name_span']['end'] and o['name_span']['start'] < region['end']]
        region['status'] = 'complete' if all(region['fields'].values()) and all(
            all(o['known'].values()) for o in overlapping) else 'partial'
    annotation['status'] = 'complete' if covered and not annotation.get('unresolved_regions') and all(
        r['status'] == 'complete' for r in covered) else 'partial'
    annotation['complete_negative_regions'] = [deepcopy(r) for r in covered
        if r['status'] == 'complete' and all(r['fields'].values())
        and not any(r['start'] < o['name_span']['end'] and o['name_span']['start'] < r['end'] for o in occurrences)]


def reduce_mutation(task: dict, annotation: dict, operation: dict, provenance: dict) -> dict:
    result = deepcopy(annotation)
    action = operation.get('action')
    value = operation.get('value')
    target = operation.get('target_name_span')
    occurrences = result['occurrences']
    existing = next((o for o in occurrences if o['name_span'] == target), None)
    if action in ALIAS_ACTIONS:
        result = apply_alias_decision(task, result, operation, provenance)
    elif action == 'upsert_occurrence':
        try:
            new = validate_task_occurrence(task, value)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('INVALID_OCCURRENCE:' + str(exc)) from exc
        if target is not None and existing is None:
            raise ValueError('UNKNOWN_OCCURRENCE')
        if any(o is not existing and o['name_span'] == new['name_span'] for o in occurrences):
            raise ValueError('DUPLICATE_OCCURRENCE')
        if existing:
            assert_alias_members_unchanged(result, existing['mention_id'], new['mention_id'])
            occurrences.remove(existing)
            new = {**existing, **new}
        new['review'] = deepcopy(provenance)
        occurrences.append(new)
    elif action == 'accept_passage':
        if (result.get('review_workflow') or {}).get('source_issues'):
            raise ValueError('SOURCE_ISSUE_BLOCKS_APPROVAL')
        result['human_passage_review'] = {key: deepcopy(provenance[key]) for key in
            ('reviewer', 'decision_id', 'recorded_at_utc')}
        result['human_passage_review']['fields'] = ['software']
        regions = result.get('covered_regions', []) + [
            {**r, 'fields': dict.fromkeys(FIELDS, False)} for r in result.get('unresolved_regions', [])]
        for region in regions:
            region['fields']['software'] = True
            region['human_reviewed_fields'] = sorted(set(region.get('human_reviewed_fields', [])) | {'software'})
        result['covered_regions'] = sorted(regions, key=lambda r: r['start'])
        result['unresolved_regions'] = []
    elif action in ('accept_occurrence', 'remove_occurrence', 'mark_field_unresolved', 'link_version', 'unlink_version'):
        if existing is None:
            raise ValueError('UNKNOWN_OCCURRENCE')
        if action == 'remove_occurrence':
            assert_alias_members_unchanged(result, existing['mention_id'], None)
            occurrences.remove(existing)
        else:
            existing['review'] = deepcopy(provenance)
            if action == 'mark_field_unresolved':
                field = operation.get('field')
                if field == 'sentiment':
                    existing['sentiment'] = None
                    existing['known']['sentiment'] = False
                    existing['evidence']['sentiment'] = []
                elif field == 'versions':
                    existing['version_status'] = 'ambiguous'
                    existing['known']['versions'] = False
                elif field == 'intents':
                    existing['intents'] = None
                    existing['evidence']['intents'] = []
                    for key in INTENT_BITS:
                        existing['known'][key] = False
                else:
                    raise ValueError('UNKNOWN_FIELD')
                existing['review'].update(status='unresolved', reasons=['unresolved_field'])
            elif action == 'link_version':
                existing['version_links'].append(deepcopy(value))
                existing['version_status'] = 'explicit'
                existing['known']['versions'] = True
            elif action == 'unlink_version':
                if value not in existing['version_links']:
                    raise ValueError('UNKNOWN_VERSION_LINK')
                existing['version_links'].remove(value)
                if not existing['version_links']:
                    existing['version_status'] = 'absent' if existing['known']['versions'] else 'unannotated'
            validate_task_occurrence(task, existing)
    else:
        raise ValueError('UNKNOWN_ACTION')
    if action not in ALIAS_ACTIONS:
        result['occurrences'] = sorted(occurrences, key=lambda o: o['name_span']['start'])
        recompute_coverage(result)
    workflow = result.get('review_workflow')
    if isinstance(workflow, dict):
        workflow['approval'] = None
        current = {o['mention_id']: o for o in result['occurrences']}
        workflow['field_reviews'] = {mid: reviews for mid, reviews in workflow['field_reviews'].items() if mid in current}
    validate_annotation_aliases(task, result)
    return result
