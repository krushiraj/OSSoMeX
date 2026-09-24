import pytest

from research.data import jats


ARTICLE = '''<article xml:lang="en" xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><article-id pub-id-type="doi">10.1234/example</article-id>
<article-id pub-id-type="pmcid">PMC123</article-id><title-group><article-title>A study</article-title></title-group>
<permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license></permissions>
</article-meta></front><body><sec><title>Methods</title><p>😀 We used <italic>NumPy</italic> 1.24 and SciPy.</p>
<p>No tools here. This is another sentence.</p></sec></body><back><ref-list><ref><mixed-citation>Not body text.</mixed-citation></ref></ref-list></back></article>'''


def test_jats_inline_text_license_identity_and_sentence_offsets():
    parsed = jats.read_jats(ARTICLE.encode(), expected_doi='10.1234/example', expected_pmcid='PMC123')
    assert parsed['text'] == 'A study\n\n😀 We used NumPy 1.24 and SciPy.\n\nNo tools here. This is another sentence.'
    assert parsed['source_ids'] == {'doi':'10.1234/example','pmcid':'PMC123'}
    assert parsed['text_license'] == 'CC-BY-4.0'
    assert parsed['supplied_text_scope'] == 'title_abstract_body_prose'
    assert parsed['fulltext_eligible'] is False
    sentences = jats.sentence_regions(parsed)
    assert [r['text'] for r in sentences] == ['😀 We used NumPy 1.24 and SciPy.', 'No tools here.', 'This is another sentence.']
    assert sentences[0]['start'] == 9 and sentences[0]['end'] == 40
    assert all(parsed['text'][r['start']:r['end']] == r['text'] for r in sentences)


@pytest.mark.parametrize('change,error', [
    (lambda x:x.replace('10.1234/example','10.1234/wrong'),'ARTICLE_IDENTITY_MISMATCH'),
    (lambda x:x.replace('PMC123','PMC456'),'ARTICLE_IDENTITY_MISMATCH'),
    (lambda x:x.replace('/licenses/by/4.0/','/licenses/by-nc/4.0/'),'TEXT_LICENSE_UNSUPPORTED'),
    (lambda x:x.replace('xml:lang="en"','xml:lang="fr"'),'NON_ENGLISH_ARTICLE'),
    (lambda x:x.replace('<body>','<missing>').replace('</body>','</missing>'),'MISSING_ARTICLE_BODY')])
def test_unusable_articles_fail_instead_of_becoming_empty_examples(change,error):
    with pytest.raises(ValueError,match=error):
        jats.read_jats(change(ARTICLE).encode(), expected_doi='10.1234/example', expected_pmcid='PMC123')


@pytest.mark.parametrize('encoding',['utf-8','utf-16'])
def test_jats_rejects_internal_entities_without_expansion(encoding):
    xml='<!DOCTYPE article [<!ENTITY secret "fabricated">]>'+ARTICLE.replace('A study','&secret;')
    with pytest.raises(ValueError,match='UNSAFE_XML'):
        jats.read_jats(xml.encode(encoding),expected_doi='10.1234/example',expected_pmcid='PMC123')


def test_external_jats_doctype_is_not_fetched():
    xml='<!DOCTYPE article SYSTEM "https://example.invalid/unavailable.dtd">'+ARTICLE
    assert jats.read_jats(xml.encode(),expected_doi='10.1234/example',expected_pmcid='PMC123')['title']=='A study'


def test_sentence_selection_is_bounded_deterministic_and_not_gold():
    parsed=jats.read_jats(ARTICLE.encode(),expected_doi='10.1234/example',expected_pmcid='PMC123')
    selected=jats.select_sentences(parsed,['numpy'],max_matches=1,seed=42)
    assert selected==jats.select_sentences(parsed,['numpy'],max_matches=1,seed=42)
    assert len(selected)==2
    assert {r['selection_reason'] for r in selected}=={'literal_candidate_match','random_nonmatching_audit'}
    assert selected[0]['candidate_aliases']==['numpy']
    assert not any('occurrences' in r or 'negative' in r for r in selected)


def test_alias_boundaries_and_abbreviations_do_not_split_versions_or_match_substrings():
    xml=ARTICLE.replace('😀 We used <italic>NumPy</italic> 1.24 and SciPy.',
                        'We used NumPython v. 1.24. We used R and NumPy 1.24.2.')
    parsed=jats.read_jats(xml.encode(),expected_doi='10.1234/example',expected_pmcid='PMC123')
    selected=jats.select_sentences(parsed,['NumPy','R'],max_matches=3,seed=42)
    matches=[r for r in selected if r['selection_reason']=='literal_candidate_match']
    assert [r['text'] for r in matches]==['We used R and NumPy 1.24.2.']
    assert jats.sentence_regions(parsed)[0]['text']=='We used NumPython v. 1.24.'


def test_scientific_citations_and_enzyme_abbreviations_keep_real_sentence_boundaries():
    xml=ARTICLE.replace('😀 We used <italic>NumPy</italic> 1.24 and SciPy.',
        'An observation.30,31 We used scipy.55 Graphs used matplotlib.56')
    xml=xml.replace('No tools here. This is another sentence.',
        'The enzyme (ADI; E.C. 3.5.3.6) was measured (Shirai et al. 2001). Another sentence.')
    parsed=jats.read_jats(xml.encode(),expected_doi='10.1234/example',expected_pmcid='PMC123')
    assert [r['text'] for r in jats.sentence_regions(parsed)]==[
        'An observation.30,31','We used scipy.55','Graphs used matplotlib.56',
        'The enzyme (ADI; E.C. 3.5.3.6) was measured (Shirai et al. 2001).','Another sentence.']
