from copy import deepcopy
import json
from pathlib import Path

from research.annotations.tasks import make_tasks
from research.contracts import FIELDS


def alias_item(case_id='design-explicit-icekat'):
    path = Path(__file__).resolve().parents[1] / 'docs/plans/scibert-alias-examples.json'
    case = next(c for c in json.loads(path.read_bytes())['cases'] if c['case_id'] == case_id)
    policy = {'policy_version': 'scibert-poc-2.1', 'policy_hash': 'a' * 64,
              'alias_schema_version': '1.0'}
    task = make_tasks({**case['input'], 'source': 'synthetic-contract-fixture', 'split': 'demo'}, policy)[0]
    task.update(whole_passage_audit=True, annotation_region_kind='sentence')
    if 'aliases' not in task['requested_fields']:
        task['requested_fields'].append('aliases')
    annotation = {key: task[key] for key in ('task_id', 'document_id', 'text_revision', 'policy_version')}
    annotation.update(attempt_id='test-1', status='complete', annotation_revision=1,
                      occurrences=deepcopy(case['expected_occurrences']), review_status='synthetic_fixture',
                      covered_regions=[{**task['annotation_region'], 'status': 'complete',
                                        'fields': dict.fromkeys(FIELDS, True)}],
                      unresolved_regions=[], alias_annotations=deepcopy(case['proposed_alias_annotations']),
                      annotator={'runtime': 'current_codex_session', 'model_identifier': None,
                                 'prompt_hash': 'b' * 64, 'run_identifier': 'synthetic-test'})
    return {'task': task, 'annotation': annotation}
