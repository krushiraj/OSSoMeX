import pytest

from research.linkage import evaluate, resolve
from research.linkage.gold import CandidateIndex
from research.registry import build


def entity(entity_id, names, description='', aliases=(), package_ids=None, repo_keys=(), sources=('wikidata',),
           homepages=(), czi=()):
    return {'entity_id': entity_id, 'names': list(names), 'aliases': list(aliases), 'description': description,
            'package_ids': package_ids or {}, 'repo_keys': list(repo_keys), 'sources': list(sources),
            'czi': [{'czi_name': n, 'mentions_count': m} for n, m in czi], 'members': [entity_id],
            'homepages': list(homepages), 'rrids': [], 'publication_dois': [], 'topics': [], 'classes': []}


ENTITIES = [
    entity('wikidata:Q2', ['STAR'], 'ultrafast universal RNA-seq aligner for sequencing reads',
           repo_keys=['github.com/alexdobin/star'], sources=('wikidata', 'biotools')),
    entity('ecosystems:cran/STAR', ['STAR'], 'Spike Train Analysis with R', package_ids={'cran': ['star']},
           sources=('ecosystems',), czi=[('STAR', 5759)]),
    entity('ecosystems:cran/glmnet', ['glmnet'], 'Lasso and elastic-net regularized generalized linear models',
           package_ids={'cran': ['glmnet']}, sources=('ecosystems',), czi=[('glmnet', 1607)]),
    entity('ecosystems:pypi/glmnet', ['glmnet'], 'Python wrapper for glmnet', package_ids={'pypi': ['glmnet']},
           sources=('ecosystems',), czi=[('glmnet', 1607)]),
    entity('wikidata:Q3', ['Foo Tool'], 'thing', homepages=['https://foo.example.org/'],
           repo_keys=['github.com/acme/foo'])]


def task(task_id, name, contexts, urls=(), paper='p1'):
    return {'task_id': task_id, 'paper_id': paper, 'name': name, 'surface_forms': [name], 'contexts': contexts,
            'urls': list(urls), 'versions': [], 'publishers': [], 'languages': [], 'domain': 'Bio'}


@pytest.fixture
def index():
    return CandidateIndex(ENTITIES, build.surface_forms(ENTITIES))


def test_fast_text_match_picks_most_mentioned_exact_match_and_abstains_otherwise(index):
    baseline = resolve.FastTextMatch(index)
    assert baseline.resolve(task('t1', 'STAR', ['reads were aligned with [[STAR]]']))['entity_id'] == 'ecosystems:cran/STAR'
    assert baseline.resolve(task('t2', 'Unknown', ['[[Unknown]] was used']))['entity_id'] is None


def test_context_resolver_uses_context_ecosystem_and_url_evidence(index):
    resolver = resolve.ContextResolver(index)
    star = resolver.resolve(task('t1', 'STAR', ['RNA-seq reads were aligned to the genome with [[STAR]] aligner']))
    assert star['entity_id'] == 'wikidata:Q2'
    glm = resolver.resolve(task('t2', 'glmnet', ['we fitted lasso models with the R package [[glmnet]]']))
    assert glm['entity_id'] == 'ecosystems:cran/glmnet'
    url = resolver.resolve(task('t3', 'FooTool2', ['[[FooTool2]] (https://github.com/acme/foo)'],
                                urls=['https://github.com/acme/foo']))
    assert url['entity_id'] == 'wikidata:Q3'
    assert resolver.resolve(task('t4', 'Unknown', ['[[Unknown]] was used']))['entity_id'] is None
    assert [c['entity_id'] for c in star['candidates']][:1] == ['wikidata:Q2']


def test_threshold_controls_abstention(index):
    strict = resolve.ContextResolver(index, threshold=99)
    assert strict.resolve(task('t1', 'STAR', ['aligned with [[STAR]]']))['entity_id'] is None


GOLD = [
    {'task_id': 'a', 'paper_id': 'p1', 'entity_ids': ['e1', 'e1b'], 'nil_reason': None},
    {'task_id': 'b', 'paper_id': 'p1', 'entity_ids': ['e2'], 'nil_reason': None},
    {'task_id': 'c', 'paper_id': 'p2', 'entity_ids': [], 'nil_reason': 'not_in_registry'},
    {'task_id': 'd', 'paper_id': 'p2', 'entity_ids': [], 'nil_reason': 'not_software'},
    {'task_id': 'e', 'paper_id': 'p2', 'entity_ids': ['e3'], 'nil_reason': None}]


def test_link_metrics_accept_equivalent_ids_and_penalize_linking_nil():
    predictions = [{'task_id': 'a', 'entity_id': 'e1b'}, {'task_id': 'b', 'entity_id': 'wrong'},
                   {'task_id': 'c', 'entity_id': 'e9'}, {'task_id': 'd', 'entity_id': None},
                   {'task_id': 'e', 'entity_id': None}]
    m = evaluate.link_metrics(GOLD, predictions)
    assert (m['predicted_links'], m['correct_links'], m['gold_links']) == (3, 1, 3)
    assert m['precision'] == pytest.approx(1 / 3) and m['recall'] == pytest.approx(1 / 3)
    assert m['nil']['gold'] == 2 and m['nil']['abstained'] == 1
    assert m['registry_coverage'] == pytest.approx(3 / 4)
    assert m['errors'] == {'wrong_entity': 1, 'linked_nil': 1, 'missed': 1}


def test_paper_bootstrap_is_deterministic():
    predictions = [{'task_id': t['task_id'], 'entity_id': (t['entity_ids'] or [None])[0]} for t in GOLD]
    first = evaluate.bootstrap(GOLD, predictions, samples=200, seed=1)
    assert first == evaluate.bootstrap(GOLD, predictions, samples=200, seed=1)
    assert first['precision'] == [1.0, 1.0]


def test_metrics_reject_missing_predictions():
    with pytest.raises(ValueError, match='MISSING_PREDICTIONS'):
        evaluate.link_metrics(GOLD, [{'task_id': 'a', 'entity_id': None}])
