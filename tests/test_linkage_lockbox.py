import pytest

from research.linkage import lockbox

JATS = b'''<article xml:lang="en"><front><article-meta>
<article-id pub-id-type="pmcid">PMC123</article-id><article-id pub-id-type="doi">10.1/X</article-id>
<title-group><article-title>A study</article-title></title-group>
<permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/" xmlns:xlink="http://www.w3.org/1999/xlink"/></permissions>
</article-meta></front><body>
<sec><title>Introduction</title><p>Background without tools.</p></sec>
<sec sec-type="methods"><title>Materials and methods</title><p>Reads were aligned with STAR.</p>
<sec><title>Statistics</title><p>Analyses used R 4.3.</p></sec></sec>
<sec><title>Statistical analysis</title><p>SPSS 25 was used.</p></sec>
<sec><title>Results</title><p>It worked.</p></sec>
</body></article>'''


def test_methods_text_keeps_methods_and_analysis_sections_only():
    parsed = lockbox.methods_document(JATS)
    assert parsed['text'] == 'Reads were aligned with STAR.\n\nAnalyses used R 4.3.\n\nSPSS 25 was used.'
    assert parsed['source_ids'] == {'pmcid': 'PMC123', 'doi': '10.1/x'} and parsed['title'] == 'A study'
    assert parsed['text_license'] == 'CC-BY-4.0'


def test_methods_document_rejects_unlicensed_or_methodless_articles():
    with pytest.raises(ValueError, match='TEXT_LICENSE_UNSUPPORTED'):
        lockbox.methods_document(JATS.replace(b'licenses/by/4.0', b'licenses/by-nc/4.0'))
    with pytest.raises(ValueError, match='NO_METHODS_SECTION'):
        lockbox.methods_document(JATS.replace(b'sec-type="methods"', b'').replace(b'Materials and methods', b'Intro2')
                                 .replace(b'Statistical analysis', b'Discussion').replace(b'Statistics', b'Extra'))


def test_sample_spreads_journals_and_is_seeded():
    rows = [{'pmcid': f'PMC{i}', 'journalTitle': f'J{i % 3}'} for i in range(30)]
    first = lockbox.sample_candidates(rows, n=5, per_journal=2, seed=4)
    assert first == lockbox.sample_candidates(list(reversed(rows)), n=5, per_journal=2, seed=4)
    assert len(first) == 5 and max(sum(r['journalTitle'] == j for r in first) for j in ('J0', 'J1', 'J2')) <= 2


def test_paper_level_end_to_end_metrics():
    gold = [{'paper_id': 'p1', 'entity_ids': ['a', 'a2']}, {'paper_id': 'p1', 'entity_ids': ['b']},
            {'paper_id': 'p1', 'entity_ids': []}, {'paper_id': 'p2', 'entity_ids': ['c']}]
    predicted = [('p1', 'a2'), ('p1', 'x'), ('p2', 'c'), ('p2', 'c')]
    m = lockbox.paper_link_metrics(gold, predicted)
    assert (m['predicted_links'], m['correct_links'], m['gold_links'], m['recalled_links']) == (3, 2, 3, 2)
    assert m['precision'] == pytest.approx(2 / 3) and m['recall'] == pytest.approx(2 / 3)


def test_prediction_occurrences_become_resolver_tasks():
    text = 'Reads were aligned with STAR 2.7. We reran star later. Plots used R.'
    prediction = {'document_id': 'd1', 'status': 'success', 'occurrences': [
        {'name': 'STAR', 'name_span': {'start': 24, 'end': 28}},
        {'name': 'star', 'name_span': {'start': 43, 'end': 47}},
        {'name': 'R', 'name_span': {'start': 66, 'end': 67}}]}
    tasks = lockbox.tasks_from_prediction({'document_id': 'd1', 'text': text}, prediction)
    assert [t['name'] for t in tasks] == ['STAR', 'R']
    star = tasks[0]
    assert star['task_id'] == 'd1#star' and star['paper_id'] == 'd1' and star['surface_forms'] == ['STAR', 'star']
    assert '[[STAR]]' in star['contexts'][0] and star['urls'] == []


def test_failed_predictions_are_not_silently_empty():
    with pytest.raises(ValueError, match='PREDICTION_FAILED'):
        lockbox.tasks_from_prediction({'document_id': 'd1', 'text': 'x'}, {'document_id': 'd1', 'status': 'failed'})


def test_gold_rows_become_tasks_with_located_contexts():
    doc = {'document_id': 'd1', 'text': 'Reads were aligned with STAR. Later STAR ran again.'}
    tasks = lockbox.tasks_from_gold(doc, [{'paper_id': 'd1', 'name': 'STAR', 'surface_forms': ['STAR']}])
    assert tasks[0]['task_id'] == 'd1#star' and len(tasks[0]['contexts']) == 2
    with pytest.raises(ValueError, match='SURFACE_FORM_NOT_FOUND'):
        lockbox.tasks_from_gold(doc, [{'paper_id': 'd1', 'name': 'X', 'surface_forms': ['Xyz']}])


def test_extraction_recall_matches_gold_names_case_insensitively():
    gold = [{'paper_id': 'd1', 'surface_forms': ['STAR', 'Star']}, {'paper_id': 'd1', 'surface_forms': ['R']},
            {'paper_id': 'd2', 'surface_forms': ['SPSS']}]
    tasks = [{'paper_id': 'd1', 'surface_forms': ['star']}, {'paper_id': 'd2', 'surface_forms': ['Excel']}]
    assert lockbox.extraction_recall(gold, tasks) == {'gold_software': 3, 'found': 1, 'recall': 1 / 3,
                                                      'predicted_names': 2, 'unmatched_predicted_names': 1}


def test_paper_bootstrap_is_deterministic_and_bounded():
    gold = [{'paper_id': 'p1', 'entity_ids': ['a']}, {'paper_id': 'p2', 'entity_ids': ['b']}]
    ci = lockbox._paper_bootstrap(gold, [('p1', 'a'), ('p2', 'x')], samples=300, seed=2)
    assert ci == lockbox._paper_bootstrap(gold, [('p1', 'a'), ('p2', 'x')], samples=300, seed=2)
    assert 0.0 <= ci['precision'][0] <= ci['precision'][1] <= 1.0


def test_partial_attribute_stages_still_yield_detector_occurrences():
    doc = {'document_id': 'd1', 'text': 'Used STAR.'}
    prediction = {'document_id': 'd1', 'status': 'partial', 'stage_status': {'detector': {'status': 'success'}},
                  'occurrences': [{'name': 'STAR', 'name_span': {'start': 5, 'end': 9}}]}
    assert lockbox.tasks_from_prediction(doc, prediction)[0]['name'] == 'STAR'
    prediction['stage_status']['detector']['status'] = 'partial'
    with pytest.raises(ValueError, match='PREDICTION_FAILED'):
        lockbox.tasks_from_prediction(doc, prediction)
