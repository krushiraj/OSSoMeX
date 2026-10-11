import io
import zipfile

from research.linkage import gold
from research.registry import build


TEI = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body>
<p>Reads were aligned with <rs xml:id="s1" type="software">STAR</rs> version <rs xml:id="v1" type="version" corresp="#s1">2.7</rs>.
Figures used <rs xml:id="s2" type="software" subtype="environment">R</rs> and <rs xml:id="s3" type="software">ImageJ</rs>
(<rs xml:id="u3" type="url" corresp="#s3">https://imagej.net/ij/</rs>) by <rs xml:id="p3" type="publisher" corresp="#s3">NIH</rs>.</p>
<p>We reran <rs xml:id="s4" type="software">star</rs> on new samples using <rs xml:id="s5" type="software" subtype="implicit">the software</rs>.</p>
</body></text></TEI>'''


def archive(members):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return data.getvalue()


def entity(entity_id, names, description='', aliases=(), package_ids=None, repo_keys=(), sources=('wikidata',)):
    return {'entity_id': entity_id, 'names': list(names), 'aliases': list(aliases), 'description': description,
            'package_ids': package_ids or {}, 'repo_keys': list(repo_keys), 'sources': list(sources), 'czi': [],
            'members': [entity_id], 'homepages': [], 'rrids': [], 'publication_dois': [], 'topics': [], 'classes': []}


ENTITIES = [entity('wikidata:Q2', ['STAR'], 'RNA-seq aligner', repo_keys=['github.com/alexdobin/star']),
            entity('ecosystems:cran/STAR', ['STAR'], 'Spike Train Analysis', package_ids={'cran': ['star']},
                   sources=('ecosystems',)),
            entity('wikidata:Q1659584', ['ImageJ'], 'image processing software', aliases=['NIH Image']),
            entity('wikidata:Q206904', ['R'], 'programming language for statistics'),
            entity('biotools:imagej-fiji', ['Fiji'], 'ImageJ distribution', aliases=['Fiji is just ImageJ'])]


def test_paper_tasks_group_mentions_and_attach_attributes():
    tasks = gold.paper_tasks('sofair:Dom/1.xml', 'Dom', TEI)
    by_name = {t['name']: t for t in tasks}
    assert sorted(by_name) == ['ImageJ', 'R', 'STAR']
    star = by_name['STAR']
    assert star['task_id'] == 'sofair:Dom/1.xml#star' and star['mention_count'] == 2
    assert star['surface_forms'] == ['STAR', 'star'] and star['versions'] == ['2.7']
    assert '[[STAR]]' in star['contexts'][0]
    imagej = by_name['ImageJ']
    assert imagej['urls'] == ['https://imagej.net/ij/'] and imagej['publishers'] == ['NIH']
    assert by_name['R']['subtypes'] == ['environment']


def test_candidates_include_exact_alias_and_token_matches_with_summaries():
    forms = build.surface_forms(ENTITIES)
    index = gold.CandidateIndex(ENTITIES, forms)
    star = [c['entity_id'] for c in index.candidates(['STAR', 'star'])]
    assert star[:2] == ['ecosystems:cran/STAR', 'wikidata:Q2']
    imagej = index.candidates(['ImageJ'])
    assert imagej[0]['entity_id'] == 'wikidata:Q1659584' and imagej[0]['match'] == 'exact'
    assert 'biotools:imagej-fiji' in [c['entity_id'] for c in imagej]
    assert index.candidates(['NIH Image'])[0]['entity_id'] == 'wikidata:Q1659584'
    assert index.search('spike')[0]['entity_id'] == 'ecosystems:cran/STAR'
    assert index.search('github.com/alexdobin')[0]['entity_id'] == 'wikidata:Q2'


def test_sample_papers_is_stratified_seeded_and_excludes_ids():
    names = [f'tei/D{d}/{i}.training.tei.xml' for d in range(3) for i in range(5)]
    picked = gold.sample_papers(names, per_domain=2, seed=1, exclude={'tei/D0/0.training.tei.xml'})
    assert picked == gold.sample_papers(list(reversed(names)), per_domain=2, seed=1,
                                        exclude={'tei/D0/0.training.tei.xml'})
    assert len(picked) == 6 and 'tei/D0/0.training.tei.xml' not in picked
    assert sorted({p.split('/')[1] for p in picked}) == ['D0', 'D1', 'D2']


def test_build_tasks_from_archive(tmp_path):
    from research.data.manifest import read_jsonl, write_jsonl
    run = tmp_path / 'registry'
    write_jsonl(run / 'entities.jsonl', ENTITIES)
    write_jsonl(run / 'surface_forms.jsonl', build.surface_forms(ENTITIES))
    zip_path = tmp_path / 'sofair.zip'
    zip_path.write_bytes(archive({'tei/Dom/1.training.tei.xml': TEI, 'tei/Dom/2.training.tei.xml': TEI,
                                  'tei/README': b'x'}))
    summary = gold.build_sofair_tasks(zip_path, run, tmp_path / 'gold', per_domain=1, seed=3, exclude=set())
    assert summary['papers'] == 1 and summary['tasks'] == 3
    tasks = read_jsonl(tmp_path / 'gold/tasks.jsonl')
    assert all(t['candidates'] for t in tasks if t['name'] != 'R') and tasks[0]['domain'] == 'Dom'
