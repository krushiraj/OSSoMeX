import json
from pathlib import Path

import pytest

from research.data.manifest import digest, read_jsonl
from research.registry import audit, build, normalize, sources


ECOSYSTEMS_ROW = {
    'id': 1, 'czi_id': 'SM1', 'ecosystem': 'cran', 'name': 'STAR', 'mentions_count': 5759,
    'project_url': 'https://papers.ecosyste.ms/api/v1/projects/cran/STAR',
    'mentions_url': 'https://papers.ecosyste.ms/api/v1/projects/cran/STAR/mentions',
    'package': {'name': 'STAR', 'ecosystem': 'cran', 'description': 'Spike Train Analysis with R',
                'homepage': None, 'repository_url': None, 'keywords_array': ['spike']}}
BIOTOOLS_TOOL = {
    'name': 'struct', 'biotoolsID': 'struct', 'homepage': 'https://bioconductor.org/packages/structToolbox',
    'description': 'Metabolomics toolbox',
    'link': [{'url': 'http://bioconductor.org/packages/structToolbox', 'type': ['Repository']},
             {'url': 'https://github.com/computational-metabolomics', 'type': ['Repository']}],
    'download': [{'url': 'https://github.com/computational-metabolomics/structToolbox', 'type': 'Source code'}],
    'publication': [{'doi': '10.1093/Bioinformatics/btaa1031'}, {'doi': None}],
    'otherID': [{'value': 'RRID:SCR_000001', 'type': 'rrid'}],
    'topic': [{'term': 'Metabolomics'}]}


def binding(qid, key, value):
    return {'s': {'value': 'http://www.wikidata.org/entity/' + qid}, 'k': {'value': key}, 'v': {'value': value}}


WIKIDATA_BINDINGS = [
    binding('Q1659584', 'label', 'ImageJ'), binding('Q1659584', 'alias', 'NIH Image'),
    binding('Q1659584', 'description', 'image processing software'),
    binding('Q1659584', 'repo', 'https://github.com/imagej/ImageJ'),
    binding('Q1659584', 'website', 'https://imagej.net/ij/'), binding('Q1659584', 'rrid', 'SCR_003070'),
    binding('Q1659584', 'doi', '10.1038/NMETH.2089'),
    binding('Q1659584', 'instance_of', 'http://www.wikidata.org/entity/Q341'),
    binding('Q2', 'label', 'STAR'), binding('Q2', 'description', 'RNA-seq aligner'),
    binding('Q2', 'repo', 'https://github.com/alexdobin/STAR.git')]


def test_forms_are_casefolded_and_loose_forms_drop_punctuation():
    assert normalize.form_exact('  Image J ') == 'image j'
    assert normalize.form_loose('Image-J') == 'imagej'
    assert normalize.form_loose('---') is None


@pytest.mark.parametrize('url,key', [
    ('https://github.com/imagej/ImageJ.git', 'github.com/imagej/imagej'),
    ('git+https://www.github.com/a/b/tree/main/src', 'github.com/a/b'),
    ('git@github.com:a/b.git', 'github.com/a/b'),
    ('https://gitlab.com/group/sub/proj/-/tree/x', 'gitlab.com/group/sub/proj'),
    ('https://sourceforge.net/projects/mrbayes/files', 'sourceforge.net/mrbayes'),
    ('https://github.com/computational-metabolomics', None),
    ('https://imagej.net/ij/', None), (None, None)])
def test_repository_keys_identify_projects_not_organisations(url, key):
    assert normalize.repo_key(url) == key


def test_package_ids_come_from_registry_urls():
    assert normalize.package_ids_from_url('https://pypi.org/project/Scikit_Learn/') == ('pypi', 'scikit-learn')
    assert normalize.package_ids_from_url('https://cran.r-project.org/package=lme4') == ('cran', 'lme4')
    assert normalize.package_ids_from_url('https://cran.r-project.org/web/packages/vegan/index.html') == ('cran', 'vegan')
    assert normalize.package_ids_from_url(
        'http://bioconductor.org/packages/release/bioc/html/DESeq2.html') == ('bioconductor', 'deseq2')
    assert normalize.package_ids_from_url('https://bioconductor.org/packages/structToolbox') == ('bioconductor', 'structtoolbox')
    assert normalize.package_ids_from_url('https://imagej.net/ij/') is None


def test_ecosystems_record_keeps_czi_link_separate_from_package_identity():
    record = sources.parse_ecosystems(ECOSYSTEMS_ROW)
    assert record['key'] == 'ecosystems:cran/STAR'
    assert record['package_ids'] == {'cran': ['star']}
    assert record['czi'] == {'czi_id': 'SM1', 'czi_name': 'STAR', 'mentions_count': 5759,
                             'project_url': ECOSYSTEMS_ROW['project_url']}
    assert record['description'] == 'Spike Train Analysis with R' and record['topics'] == ['spike']


def test_biotools_record_collects_ids_dois_and_project_repositories():
    record = sources.parse_biotools(BIOTOOLS_TOOL)
    assert record['key'] == 'biotools:struct'
    assert record['package_ids'] == {'bioconductor': ['structtoolbox']}
    assert record['repo_keys'] == ['github.com/computational-metabolomics/structtoolbox']
    assert record['publication_dois'] == ['10.1093/bioinformatics/btaa1031']
    assert record['rrids'] == ['SCR_000001'] and record['topics'] == ['Metabolomics']


def test_wikidata_bindings_group_into_records():
    records = {r['key']: r for r in sources.parse_wikidata(WIKIDATA_BINDINGS)}
    imagej = records['wikidata:Q1659584']
    assert imagej['name'] == 'ImageJ' and imagej['aliases'] == ['NIH Image']
    assert imagej['repo_keys'] == ['github.com/imagej/imagej'] and imagej['homepages'] == ['https://imagej.net/ij/']
    assert imagej['rrids'] == ['SCR_003070'] and imagej['publication_dois'] == ['10.1038/nmeth.2089']
    assert imagej['classes'] == ['Q341']
    assert records['wikidata:Q2']['repo_keys'] == ['github.com/alexdobin/star']


def record(source, source_id, name, **fields):
    base = {'source': source, 'source_id': source_id, 'key': f'{source}:{source_id}', 'name': name, 'aliases': [],
            'description': None, 'homepages': [], 'repo_urls': [], 'repo_keys': [], 'package_ids': {}, 'rrids': [],
            'publication_dois': [], 'topics': [], 'classes': []}
    return {**base, **fields}


def test_merge_uses_identifiers_never_names_and_refuses_monorepo_package_conflicts():
    records = [
        record('wikidata', 'Q1', 'scikit-learn', package_ids={'pypi': ['scikit-learn']}),
        record('ecosystems', 'pypi/scikit-learn', 'scikit-learn', package_ids={'pypi': ['scikit-learn']},
               repo_keys=['github.com/scikit-learn/scikit-learn']),
        record('biotools', 'star', 'STAR', repo_keys=['github.com/alexdobin/star']),
        record('wikidata', 'Q2', 'STAR', repo_keys=['github.com/alexdobin/star']),
        record('ecosystems', 'cran/STAR', 'STAR', package_ids={'cran': ['star']}),
        record('ecosystems', 'pypi/a', 'a', package_ids={'pypi': ['a']}, repo_keys=['github.com/org/mono']),
        record('ecosystems', 'pypi/b', 'b', package_ids={'pypi': ['b']}, repo_keys=['github.com/org/mono']),
        record('biotools', 'plugin', 'plugin', package_ids={'pypi': ['plugin-x']}, repo_keys=['github.com/org/tool']),
        record('ecosystems', 'pypi/tool', 'tool', package_ids={'pypi': ['tool']}, repo_keys=['github.com/org/tool'])]
    entities, conflicts = build.merge(records)
    members = {e['entity_id']: e['members'] for e in entities}
    assert members['wikidata:Q1'] == ['ecosystems:pypi/scikit-learn', 'wikidata:Q1']
    assert members['wikidata:Q2'] == ['biotools:star', 'wikidata:Q2']
    assert 'ecosystems:cran/STAR' in members
    assert 'ecosystems:pypi/a' in members and 'ecosystems:pypi/b' in members
    assert conflicts == [{'edge': 'repo:github.com/org/mono', 'left': 'ecosystems:pypi/a', 'right': 'ecosystems:pypi/b',
                          'reason': 'same_source_ecosystems'},
                         {'edge': 'repo:github.com/org/tool', 'left': 'biotools:plugin', 'right': 'ecosystems:pypi/tool',
                          'reason': 'distinct_pypi_packages'}]


def test_surface_forms_expose_collisions_and_flag_czi_links():
    records = [
        record('wikidata', 'Q2', 'STAR', aliases=['Spliced Transcripts Alignment to a Reference'],
               description='RNA-seq aligner'),
        record('ecosystems', 'cran/STAR', 'STAR', package_ids={'cran': ['star']}, description='Spike Train Analysis',
               czi={'czi_id': 'SM1', 'czi_name': 'STAR', 'mentions_count': 9, 'project_url': 'u1'}),
        record('ecosystems', 'pypi/excel', 'excel', package_ids={'pypi': ['excel']},
               description='This package name is reserved by Microsoft Corporation',
               czi={'czi_id': 'SM2', 'czi_name': 'excel', 'mentions_count': 3, 'project_url': 'u2'}),
        record('ecosystems', 'pypi/numpy', 'numpy', package_ids={'pypi': ['numpy']}, description='Arrays',
               czi={'czi_id': 'SM3', 'czi_name': 'numpy', 'mentions_count': 1, 'project_url': 'u3'})]
    entities, _ = build.merge(records)
    forms = build.surface_forms(entities)
    star = next(f for f in forms if f['key_type'] == 'exact' and f['form'] == 'star')
    assert [e['entity_id'] for e in star['entities']] == ['ecosystems:cran/STAR', 'wikidata:Q2']
    links = {link['czi_id']: link for link in build.czi_links(entities, forms)}
    assert links['SM1']['record'] == 'ecosystems:cran/STAR'
    assert links['SM1']['flags'] == ['name_collision'] and links['SM1']['trust'] == 'low_trust'
    assert links['SM2']['flags'] == ['placeholder_description']
    assert links['SM3']['flags'] == [] and links['SM3']['trust'] == 'unaudited'


def test_audit_sample_is_deterministic_and_stratified():
    links = [{'record': f'ecosystems:pypi/p{i:02d}', 'czi_id': f'SM{i // 2}', 'mentions_count': i} for i in range(50)]
    first = audit.sample(links, top=5, random_n=10, seed=7)
    assert first == audit.sample(list(reversed(links)), top=5, random_n=10, seed=7)
    top = [r['record'] for r in first if r['stratum'] == 'top']
    assert top == [f'ecosystems:pypi/p{i}' for i in (49, 48, 47, 46, 45)]
    randoms = [r['record'] for r in first if r['stratum'] == 'random']
    assert len(randoms) == 10 and not set(randoms) & set(top)


def test_wilson_interval_and_report_exclude_unsure():
    low, high = audit.wilson(2, 10)
    assert round(low, 3) == 0.057 and round(high, 3) == 0.510
    tasks = [{'task_id': 't1', 'stratum': 'top', 'mentions_count': 90, 'flags': ['name_collision']},
             {'task_id': 't2', 'stratum': 'top', 'mentions_count': 10, 'flags': []},
             {'task_id': 't3', 'stratum': 'top', 'mentions_count': 5, 'flags': []}]
    labels = [{'task_id': 't1', 'verdict': 'wrong'}, {'task_id': 't2', 'verdict': 'correct'},
              {'task_id': 't3', 'verdict': 'unsure'}]
    report = audit.report(tasks, labels)
    top = report['strata']['top']
    assert (top['judged'], top['wrong'], top['unsure']) == (2, 1, 1)
    assert top['mention_weighted_wrong_rate'] == 0.9
    assert report['by_flag']['flagged']['wrong'] == 1 and report['by_flag']['unflagged']['wrong'] == 0
    with pytest.raises(ValueError, match='UNKNOWN_AUDIT_TASK'):
        audit.report(tasks, [{'task_id': 'nope', 'verdict': 'wrong'}])
    with pytest.raises(ValueError, match='INVALID_VERDICT'):
        audit.report(tasks, [{'task_id': 't1', 'verdict': 'maybe'}])


def fake_fetch(responses):
    def fetch(url, destination, policy):
        payload = json.dumps(responses[url]).encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {'url': url, 'sha256': digest(payload), 'status': 'success'}
    return fetch


def test_pull_then_build_runs_end_to_end(tmp_path, monkeypatch):
    from research.registry import pull
    responses = {
        sources.ecosystems_page_url(1): [ECOSYSTEMS_ROW], sources.ecosystems_page_url(2): [],
        sources.biotools_page_url(1): {'list': [BIOTOOLS_TOOL], 'next': None},
        sources.wikidata_list_url(): {'results': {'bindings': [
            {'s': {'value': 'http://www.wikidata.org/entity/Q1659584'}},
            {'s': {'value': 'http://www.wikidata.org/entity/Q2'}}]}},
        sources.wikidata_detail_url(['Q2', 'Q1659584']): {'results': {'bindings': WIKIDATA_BINDINGS}}}
    monkeypatch.setattr(pull, 'fetch_public', fake_fetch(responses))
    run = tmp_path / 'run'
    config = {'user_agent': 'test', 'delay_seconds': 0, 'wikidata_batch': 250}
    manifest = pull.pull(config, run)
    assert manifest['status'] == 'success'
    assert {s: manifest['sources'][s]['pages'] for s in manifest['sources']} == {
        'ecosystems': 2, 'biotools': 1, 'wikidata': 2}
    stats = build.build_run(run)
    assert stats['records'] == {'biotools': 1, 'ecosystems': 1, 'wikidata': 2}
    assert stats['entities'] == 4 and stats['ambiguous_exact_forms'] == 1
    assert stats['czi_links'] == {'total': 1, 'low_trust': 1, 'unaudited': 0}
    entities = read_jsonl(run / 'entities.jsonl')
    assert {e['entity_id'] for e in entities} == {
        'wikidata:Q1659584', 'wikidata:Q2', 'ecosystems:cran/STAR', 'biotools:struct'}
    assert build.build_run(run) == stats
    (run / 'raw/biotools/page-0001.json').write_text('{}')
    with pytest.raises(ValueError, match='missing or changed artifact'):
        build.build_run(run)


def test_coverage_reports_exact_loose_and_ambiguous_matches(tmp_path):
    forms = [{'form': 'star', 'key_type': 'exact', 'entities': [{'entity_id': 'a'}, {'entity_id': 'b'}]},
             {'form': 'imagej', 'key_type': 'exact', 'entities': [{'entity_id': 'c'}]},
             {'form': 'imagej', 'key_type': 'loose', 'entities': [{'entity_id': 'c'}]},
             {'form': 'star', 'key_type': 'loose', 'entities': [{'entity_id': 'a'}, {'entity_id': 'b'}]}]
    result = build.coverage(forms, ['STAR', 'Image-J', 'GATK'])
    assert result == {'names': 3, 'exact': 1, 'loose': 2, 'ambiguous_exact': 1, 'missing_examples': ['GATK']}


def test_jsonl_round_trips_unicode_line_separators_inside_strings(tmp_path):
    from research.data.manifest import write_jsonl
    rows = [{'description': 'first second\x85third'}, {'description': 'plain'}]
    write_jsonl(tmp_path / 'rows.jsonl', rows)
    assert read_jsonl(tmp_path / 'rows.jsonl') == rows


def test_audit_records_unreachable_project_evidence_instead_of_aborting(tmp_path, monkeypatch):
    from research.data.acquire import AcquisitionError
    from research.data.manifest import write_jsonl
    run = tmp_path / 'run'
    entity = {'entity_id': 'ecosystems:cran/lm.br', 'package_ids': {'cran': ['lm.br']}, 'description': 'd',
              'repo_keys': [], 'names': ['lm.br']}
    write_jsonl(run / 'entities.jsonl', [entity])
    write_jsonl(run / 'czi_links.jsonl', [{
        'record': 'ecosystems:cran/lm.br', 'czi_id': 'SM9', 'czi_name': 'lm.br', 'mentions_count': 1,
        'project_url': 'https://papers.ecosyste.ms/api/v1/projects/cran/lm.br', 'entity_id': 'ecosystems:cran/lm.br',
        'flags': [], 'same_name_entities': []}])

    def missing(url, destination, policy):
        raise AcquisitionError('http_404')
    monkeypatch.setattr(audit, 'fetch_public', missing)
    assert audit.sample_run(run, top=1, random_n=0, delay=0)['tasks'] == 1
    task = read_jsonl(run / 'audit/tasks.jsonl')[0]
    assert task['papers'] == [] and task['evidence_error'] == 'http_404'

    def offline(url, destination, policy):
        raise AcquisitionError('network_failed')
    monkeypatch.setattr(audit, 'fetch_public', offline)
    with pytest.raises(AcquisitionError):
        audit._paper_evidence({'project_url': 'https://papers.ecosyste.ms/x'}, tmp_path, {}, 5, 0)


@pytest.mark.parametrize('url,expected', [
    ('https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html', ('bioconductor', 'deseq2')),
    ('https://bioconductor.org/packages/3.18/bioc/html/limma.html', ('bioconductor', 'limma')),
    ('https://bioconductor.org/packages/release/bioc/', None),
    ('https://bioconductor.org/packages/devel/', None)])
def test_bioconductor_release_paths_never_become_package_ids(url, expected):
    assert normalize.package_ids_from_url(url) == expected


def test_repository_edges_never_merge_items_of_one_source_or_disjoint_packages():
    records = [
        record('wikidata', 'Q10', '@babel/core', repo_keys=['github.com/babel/babel']),
        record('wikidata', 'Q11', '@babel/parser', repo_keys=['github.com/babel/babel']),
        record('wikidata', 'Q12', 'scikit-learn', package_ids={'pypi': ['scikit-learn', 'sklearn']},
               repo_keys=['github.com/scikit-learn/scikit-learn']),
        record('ecosystems', 'pypi/scikit-learn', 'scikit-learn', package_ids={'pypi': ['scikit-learn']},
               repo_keys=['github.com/scikit-learn/scikit-learn'])]
    entities, conflicts = build.merge(records)
    members = {e['entity_id']: e['members'] for e in entities}
    assert members['wikidata:Q10'] == ['wikidata:Q10'] and members['wikidata:Q11'] == ['wikidata:Q11']
    assert members['wikidata:Q12'] == ['ecosystems:pypi/scikit-learn', 'wikidata:Q12']
    assert conflicts == [{'edge': 'repo:github.com/babel/babel', 'left': 'wikidata:Q10', 'right': 'wikidata:Q11',
                          'reason': 'same_source_wikidata'}]


def test_multi_package_records_do_not_bridge_distinct_packages():
    records = [
        record('biotools', 'qpcr', 'qpcr', package_ids={'cran': ['pcr', 'qpcr']}),
        record('ecosystems', 'cran/pcr', 'pcr', package_ids={'cran': ['pcr']}),
        record('ecosystems', 'cran/qpcR', 'qpcR', package_ids={'cran': ['qpcr']})]
    entities, _ = build.merge(records)
    assert sorted(len(e['members']) for e in entities) == [1, 1, 1]


def test_shared_identifiers_never_merge_two_records_of_one_source():
    records = [
        record('biotools', 'muscle', 'MUSCLE', rrids=['SCR_011812']),
        record('biotools', 'berkeley_group', 'Berkeley Phylogenomics Group', rrids=['SCR_011812']),
        record('wikidata', 'Q1', 'MUSCLE', rrids=['SCR_011812'])]
    entities, conflicts = build.merge(records)
    assert len(entities) == 3
    assert conflicts == [{'edge': 'rrid:SCR_011812', 'left': 'biotools:berkeley_group', 'right': 'biotools:muscle',
                          'reason': 'same_source_biotools'}]


def test_wikidata_accepts_multilingual_labels_but_prefers_english():
    def lit(qid, key, value, lang):
        return {'s': {'value': 'http://www.wikidata.org/entity/' + qid}, 'k': {'value': key},
                'v': {'value': value, 'xml:lang': lang}}
    records = {r['key']: r for r in sources.parse_wikidata([
        lit('Q165277', 'label', 'OpenCV', 'mul'), lit('Q165277', 'alias', 'Open Source Computer Vision', 'en'),
        lit('Q5', 'label', 'Thing (mul)', 'mul'), lit('Q5', 'label', 'Thing', 'en')])}
    assert records['wikidata:Q165277']['name'] == 'OpenCV'
    assert records['wikidata:Q5']['name'] == 'Thing' and records['wikidata:Q5']['aliases'] == ['Thing (mul)']
    from urllib.parse import unquote_plus
    assert 'LANG(?v) IN ("en", "mul")' in unquote_plus(sources.wikidata_detail_url(['Q1']))


def test_wikidata_sitelinks_are_kept_and_entities_take_the_maximum():
    rows = WIKIDATA_BINDINGS + [binding('Q1659584', 'sitelinks', '41')]
    records = {r['key']: r for r in sources.parse_wikidata(rows)}
    assert records['wikidata:Q1659584']['sitelinks'] == 41 and records['wikidata:Q2']['sitelinks'] == 0
    assert 'wikibase:sitelinks' in __import__('urllib.parse').parse.unquote_plus(sources.wikidata_detail_url(['Q1']))
    entities, _ = build.merge([record('wikidata', 'Q1', 'A', rrids=['X'], sitelinks=7),
                               record('biotools', 'a', 'A', rrids=['X'])])
    assert entities[0]['sitelinks'] == 7


def test_extra_wikidata_layers_append_new_items_only(tmp_path, monkeypatch):
    from research.registry import pull
    layer = {'name': 'languages', 'classes': ['Q9143'], 'require': ['P856', 'P1324']}
    lang_list = sources.wikidata_layer_list_url(layer)
    from urllib.parse import unquote_plus
    assert 'wd:Q9143' in unquote_plus(lang_list) and 'wdt:P856' in unquote_plus(lang_list)
    responses = {
        sources.ecosystems_page_url(1): [], sources.biotools_page_url(1): {'list': [], 'next': None},
        sources.wikidata_list_url(): {'results': {'bindings': [{'s': {'value': 'http://www.wikidata.org/entity/Q2'}}]}},
        sources.wikidata_detail_url(['Q2']): {'results': {'bindings': WIKIDATA_BINDINGS[-3:]}},
        lang_list: {'results': {'bindings': [{'s': {'value': 'http://www.wikidata.org/entity/Q2'}},
                                             {'s': {'value': 'http://www.wikidata.org/entity/Q28865'}}]}},
        sources.wikidata_detail_url(['Q28865']): {'results': {'bindings': [binding('Q28865', 'label', 'Python')]}}}
    monkeypatch.setattr(pull, 'fetch_public', fake_fetch(responses))
    manifest = pull.pull({'user_agent': 't', 'delay_seconds': 0, 'wikidata_layers': [layer]}, tmp_path / 'run')
    files = manifest['sources']['wikidata']['files']
    assert [f['path'] for f in files] == ['raw/wikidata/list.json', 'raw/wikidata/page-0001.json',
                                          'raw/wikidata/layer-languages-list.json',
                                          'raw/wikidata/layer-languages-0001.json']
    stats = build.build_run(tmp_path / 'run')
    assert stats['records']['wikidata'] == 2
