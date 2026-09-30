"""Guard population joins and example selection in the public brief."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_openalex_progress_report.py"
    spec = importlib.util.spec_from_file_location("progress_brief", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reports():
    document = {"document_id": "paper-1", "text": "We used NumPy.",
                "text_revision": "sha256:fixed", "arms": []}
    original = {"provenance": {"reference_sha256": "gold-fixed"},
                "documents": [document], "models": [], "field_scores": {}, "timing_rows": []}
    extra = deepcopy(original)
    extra["documents"][0]["arms"] = [{"arm_id": "softcite-scibert", "status": "success", "spans": []}]
    extra["models"] = [{"arm_id": "softcite-scibert", "kind": "softcite"}]
    extra["field_scores"] = {"softcite-scibert": {"mention_detection": {"tp": 1, "fp": 0, "fn": 0}}}
    return original, extra


def test_merge_keeps_frozen_report_unchanged_and_adds_new_arm():
    original, extra = reports()
    before = deepcopy(original)
    result = builder().merge_comparison(original, extra, "softcite-scibert")
    assert original == before
    assert result["field_scores"]["softcite-scibert"]["mention_detection"]["tp"] == 1
    assert result["documents"][0]["arms"][0]["arm_id"] == "softcite-scibert"


@pytest.mark.parametrize("change", ["references", "text", "revision", "missing_document", "duplicate_document", "missing_arm"])
def test_merge_rejects_different_population_or_reference(change):
    original, extra = reports()
    if change == "references":
        extra["provenance"]["reference_sha256"] = "different-gold"
    elif change == "text":
        extra["documents"][0]["text"] = "Different input"
    elif change == "revision":
        extra["documents"][0]["text_revision"] = "sha256:changed"
    elif change == "missing_document":
        extra["documents"] = []
    elif change == "duplicate_document":
        extra["documents"] *= 2
    else:
        extra["documents"][0]["arms"] = []
    with pytest.raises(ValueError):
        builder().merge_comparison(original, extra, "softcite-scibert")


def test_examples_require_a_clean_winner_and_keep_failure_status():
    fn = builder().clean_name_difference
    reference = [{"label": "SOFTWARE", "start": 8, "end": 13, "text": "NumPy"}]
    good = {"status": "success", "spans": reference}
    missed = {"status": "success", "spans": []}
    extra = {"status": "success", "spans": reference + [
        {"label": "SOFTWARE", "start": 0, "end": 2, "text": "We"}]}
    failed = {"status": "failure", "spans": []}
    assert fn(reference, [good, missed]) is True
    assert fn(reference, [missed, extra]) is False
    assert fn(reference, [good, good]) is False
    assert fn(reference, [good, failed]) is False


def test_merge_rejects_overwriting_an_existing_arm():
    original, extra = reports()
    original['field_scores']['softcite-scibert'] = {'original': True}
    with pytest.raises(ValueError, match='replace'):
        builder().merge_comparison(original, extra, 'softcite-scibert')


def test_merge_does_not_mix_timing_for_other_models():
    original, extra = reports()
    extra['timing_rows'] = [{'arm_id': 'softcite-scibert', 'device': 'cpu'},
                            {'arm_id': 'unrelated-model', 'device': 'cpu'}]
    result = builder().merge_comparison(original, extra, 'softcite-scibert')
    assert result['timing_rows'] == [{'arm_id': 'softcite-scibert', 'device': 'cpu'}]


@pytest.mark.parametrize('field', ['reference_spans', 'coverage', 'ignored_software', 'reference_links'])
def test_merge_checks_embedded_reference_and_mask_content(field):
    original, extra = reports()
    original['documents'][0][field] = [{'original': True}]
    extra['documents'][0][field] = [{'changed': True}]
    with pytest.raises(ValueError, match='reference'):
        builder().merge_comparison(original, extra, 'softcite-scibert')


def test_public_brief_uses_brand_and_has_no_checkpoint_history():
    module = builder()
    original, extra = reports()
    document = original['documents'][0]
    document.update({'reference_spans': [], 'source_ids': {'doi': 'https://doi.org/example'}})
    original['documents'] = []
    for prefix in ('1072bf', '12827f'):
        doc = deepcopy(document)
        doc['document_id'] = 'snippet:' + prefix
        doc['arms'] = [{'arm_id': arm, 'status': 'success', 'spans': []} for arm, _ in module.ARMS]
        original['documents'].append(doc)
    for arm, _ in module.ARMS:
        original['field_scores'][arm] = {
            'mention_detection': {'tp': 1, 'fp': 1, 'fn': 1, 'precision': .5, 'recall': .5, 'f1': .5},
            'intents': {'exact_set_accuracy': .5},
        }
        original['timing_rows'].append({'arm_id': arm, 'device': 'cpu', 'summary': {
            'successful': 90, 'scheduled': 90, 'failed': 0,
            'latency_median_seconds': .1, 'latency_p95_seconds': .2, 'successful_per_second': 5.0}})
    pages = module.make_pages(original)
    html = ''.join(module.html_block(block) for page in pages for block in page)
    assert len(pages) == 2
    assert 'OSSoMeX' in html and 'Softcite (SciBERT)' in html
    for forbidden in ('006', '007', 'Selection trade-off', 'Our ', 'our model', 'our setup'):
        assert forbidden not in html


def test_name_projection_is_separate_from_native_fields_and_checks_revisions():
    module = builder()
    report, extra = reports()
    report = module.merge_comparison(report, extra, 'softcite-scibert')
    projection = {'reference_sha256': 'gold-fixed', 'arms': {'softcite-scibert': {
        'native_name_score': {'tp': 1, 'fp': 0, 'fn': 0},
        'projected_name_score': {'tp': 2, 'fp': 0, 'fn': 0},
        'per_document': [{'document_id': 'paper-1', 'text_revision': 'sha256:fixed',
                          'projected_names': []}]}}}
    before = deepcopy(report)
    result = module.with_name_projection(report, projection)
    assert report == before
    assert result['field_scores'] == report['field_scores']
    assert module.name_score(result, 'softcite-scibert')['tp'] == 2
    projection['arms']['softcite-scibert']['per_document'][0]['text_revision'] = 'different'
    with pytest.raises(ValueError, match='revision'):
        module.with_name_projection(report, projection)
