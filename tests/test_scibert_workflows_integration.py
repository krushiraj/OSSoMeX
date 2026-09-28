import json
from urllib.parse import parse_qs, urlsplit

from research.comparison.backends import capabilities, make_backend
from research.comparison.report import build_report
from research.comparison.runner import run_comparison
from research.comparison.windows import freeze_windows
from research.contracts import text_revision
from research.data import openalex_transport
from research.data.exposure import build_exposures, exposure_reasons, load_exposures
from research.data.manifest import digest, read_jsonl
from research.data.openalex_snippets import collect_snippets


class SyntheticResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


class SyntheticSession:
    def __init__(self, payload):
        self.payload = payload
        self.headers, self.proxies, self.params, self.cookies = {}, {}, {}, {}

    def get(self, url, **kwargs):
        query = parse_qs(urlsplit(url).query)['search'][0]
        return SyntheticResponse(self.payload if query == 'java' else {'results': []})

    def close(self):
        pass


def codepoint_tokenizer(text, **kwargs):
    return {'input_ids': list(range(len(text))),
            'offset_mapping': [(index, index + 1) for index in range(len(text))]}


class SyntheticModel:
    def __init__(self, arm, measured_inputs):
        self.arm_id = arm['arm_id']
        self.measured_inputs = measured_inputs

    def load(self, arm):
        return {'status': 'ready', 'reason': None, 'capabilities': capabilities(),
                'identity': {'synthetic': True}}

    def predict(self, window):
        if not window.get('synthetic'):
            self.measured_inputs[self.arm_id].append(
                (window['window_id'], window['text'].encode('utf-8')))
        start = window['text'].find('NumPy')
        spans = [] if start < 0 else [
            {'label': 'SOFTWARE', 'text': 'NumPy', 'start': start, 'end': start + 5,
             'score': None, 'score_kind': None, 'alignment_method': 'native_codepoint'}]
        return {'window_id': window['window_id'], 'status': 'success' if spans else 'no_mentions',
                'spans': spans, 'unresolved': [],
                'raw': {'input_text': window['text'], 'native_spans': spans}, 'error': None}

    def close(self):
        pass


def test_diagnostic_parent_cannot_enter_training(tmp_path, monkeypatch):
    payload = {'results': [
        {'id': 'https://openalex.org/W1001', 'doi': 'https://doi.org/10.1000/diagnostic',
         'relevance_score': 1.0,
         'snippets': ['🧪 We used <em>NumPy</em> 1.24.\r\nNumPy stays.']},
        {'id': 'https://openalex.org/W1002', 'doi': 'https://doi.org/10.1000/second',
         'relevance_score': 0.5, 'snippets': ['We use R &amp; Python.\n']},
    ]}
    monkeypatch.setattr(openalex_transport.requests, 'Session', lambda: SyntheticSession(payload))
    diagnostic = tmp_path / 'diagnostic'
    collected = collect_snippets({
        'endpoint': 'https://api.openalex.org/funder-search',
        'queries': ['java', 'Python', 'R', 'ImageJ', 'scikit-learn', 'NumPy',
                    'GROMACS', 'MATLAB', 'SPSS', 'BLAST'],
        'page': 1, 'per_page': 5, 'snippets_per_work': 2, 'max_candidates': 100,
        'timeout_seconds': 35, 'max_retries': 0, 'max_bytes': 16777216, 'max_concurrency': 1,
    }, diagnostic)
    assert collected['status'] == 'completed'
    assert collected['role'] == 'diagnostic'
    assert collected['training_eligible'] is collected['future_untouched_test_eligible'] is False
    documents = read_jsonl(diagnostic / 'inputs.jsonl')
    assert [document['text'] for document in documents] == [
        '🧪 We used NumPy 1.24.\r\nNumPy stays.', 'We use R & Python.\n']
    assert all(document['training_eligible'] is document['future_untouched_test_eligible'] is False
               for document in documents)

    exposure_output = tmp_path / 'exposures'
    exposure_manifest = build_exposures({'inputs': [{
        'kind': 'bundle_manifest', 'path': str(diagnostic / 'manifest.json'),
        'sha256': digest((diagnostic / 'manifest.json').read_bytes()), 'role': 'diagnostic',
    }]}, exposure_output)
    exposures = load_exposures(exposure_output)
    assert exposure_manifest['status'] == 'complete'
    assert exposure_manifest['document_count'] == 2
    assert exposure_manifest['identity_count'] == 2

    windows = freeze_windows(documents, codepoint_tokenizer,
                             max_content_tokens=24, overlap_tokens=8)
    assert [(window['start'], window['end']) for window in windows] == [(0, 24), (16, 35), (0, 19)]
    expected_bytes = [
        '🧪 We used NumPy 1.24.\r\nN'.encode('utf-8'),
        b'1.24.\r\nNumPy stays.', b'We use R & Python.\n',
    ]
    assert [window['text'].encode('utf-8') for window in windows] == expected_bytes
    measured_inputs = {'encoder-fixture': [], 'llm-fixture': []}
    arms = [{'arm_id': arm_id, 'backend': 'synthetic-model', 'config': {}}
            for arm_id in measured_inputs]
    arms.append({'arm_id': 'scibert-base', 'backend': 'base', 'config': {}})

    def model_boundary(arm):
        return (make_backend(arm) if arm['arm_id'] == 'scibert-base'
                else SyntheticModel(arm, measured_inputs))

    run = tmp_path / 'comparison'
    manifest = run_comparison(documents, windows, arms, run, adapter_factory=model_boundary)
    results = read_jsonl(run / 'results.jsonl')
    arm_count, document_count = len(arms), len(documents)
    assert len(results) == arm_count * document_count == manifest['result_count'] == 6
    assert {(row['arm_id'], row['document_id']) for row in results} == {
        (arm['arm_id'], document['document_id']) for arm in arms for document in documents}
    assert manifest['status'] == 'complete' and manifest['exit_code'] == 0
    assert read_jsonl(run / 'inputs.jsonl') == documents
    assert read_jsonl(run / 'windows.jsonl') == windows
    for captured in measured_inputs.values():
        assert captured == list(zip([window['window_id'] for window in windows], expected_bytes))
    for row in results:
        raw_index = json.loads((run / row['raw_artifact']).read_bytes())
        if row['arm_id'] == 'scibert-base':
            assert row['status'] == 'unsupported' and row['reason'] == 'no_task_head'
            assert not any(row['capabilities'].values())
            assert row['spans'] == [] and raw_index['windows'] == []
        else:
            assert row['status'] == ('success' if row['document_id'] == documents[0]['document_id']
                                     else 'no_mentions')
            assert [(span['start'], span['end']) for span in row['spans']] == (
                [(10, 15), (23, 28)] if row['status'] == 'success' else [])
            for item in raw_index['windows']:
                raw = json.loads((run / item['path']).read_bytes())
                expected = next(window for window in windows if window['window_id'] == item['window_id'])
                assert raw['window']['text'].encode('utf-8') == expected['text'].encode('utf-8')
                assert raw['raw']['input_text'].encode('utf-8') == expected['text'].encode('utf-8')

    report = build_report(run, tmp_path / 'agreement-report')
    assert report['status'] == 'reported' and report['mode'] == 'unlabelled_agreement'
    assert report['heldout_quality_evaluated'] is False and 'references' not in report
    assert [document['text'] for document in report['documents']] == [document['text'] for document in documents]
    assert all(len(document['arms']) == arm_count for document in report['documents'])
    assert not any(f'"{metric}"' in json.dumps(report) for metric in ('precision', 'recall', 'f1'))
    for pair in report['agreement']['pairs']:
        if 'scibert-base' in (pair['left_arm_id'], pair['right_arm_id']):
            assert pair['jointly_valid_documents'] == 0
            assert all(metric['jaccard'] is None for metric in pair['labels'].values())

    candidate_text = 'An independently extracted complete methods section from another provider.'
    candidate = {'document_id': 'europepmc:PMC9001', 'source': 'europepmc',
                 'source_ids': {'pmcid': 'PMC9001', 'doi': 'https://dx.doi.org/10.1000/DIAGNOSTIC'},
                 'text': candidate_text, 'text_revision': text_revision(candidate_text),
                 'fulltext_eligible': True, 'training_eligible': True}
    assert candidate['document_id'] not in {document['document_id'] for document in documents}
    assert exposure_reasons(candidate, exposures, purpose='training')
    reasons = exposure_reasons(candidate, exposures, purpose='training')
    assert all(reason['role'] == 'diagnostic' and reason['reason'] == 'identifier_overlap' for reason in reasons)
    assert exposure_reasons({**candidate, 'source_ids': {'pmcid': 'PMC9001', 'doi': '10.1000/unseen'}},
                            exposures, purpose='training') == []
