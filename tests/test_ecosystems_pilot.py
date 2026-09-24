import json
from pathlib import Path

import pytest

from research.data.manifest import read_jsonl, write_jsonl
from research.contracts import validate_document
from test_ecosystems_sentences import ARTICLE


def test_real_pipeline_binds_ids_extracts_tasks_and_keeps_associations_weak(tmp_path,monkeypatch):
    from research.data import ecosystems, acquire
    from test_acquisition import Session, Response
    project='https://papers.ecosyste.ms/api/v1/projects/pypi/numpy'
    paper='https://papers.ecosyste.ms/api/v1/papers/10.1234%2Fexample'
    search='https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI%3A%2210.1234%2Fexample%22&format=json&resultType=core&pageSize=10'
    responses={project:{'name':'numpy','project_url':project,'mentions_url':project+'/mentions'},
               project+'/mentions?per_page=20&page=1':[{'id':1,'project_url':project,'paper_url':paper}],
               paper:{'doi':'10.1234/example','openalex_id':'W1','title':'A study'},
               search:{'resultList':{'result':[{'doi':'10.1234/example','pmcid':'PMC123','isOpenAccess':'Y','language':'eng'}]}}}
    original=acquire.fetch_public
    def fetch(url,path,policy):
        data=ARTICLE.encode() if url=='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML' else json.dumps(responses[url]).encode()
        return original(url,path,{**policy,'session':Session([Response(data)])})
    monkeypatch.setattr(ecosystems,'fetch_public',fetch)
    config={'projects':[{'url':project}],'max_papers':1,'max_candidates_per_project':4,'max_sentences_per_paper':2,'seed':42,
            'exclusion_documents':[]}
    report=ecosystems.collect_pilot(config,tmp_path/'pilot')
    assert report['status']=='ready_for_annotation' and report['paper_count']==1
    docs=read_jsonl(tmp_path/'pilot/bundle/documents.jsonl')
    assert docs[0]['source_ids']=={'doi':'10.1234/example','pmcid':'PMC123','openalex':'W1'}
    assert docs[0]['development_exposed'] is True and docs[0]['split']=='train'
    sentences=read_jsonl(tmp_path/'pilot/sentences.jsonl')
    assert len(sentences)==2
    tasks=read_jsonl(tmp_path/'pilot/tasks/tasks.jsonl')
    assert all(t['candidate_system_predictions'] is None and t['split']=='train' for t in tasks)
    assert all(t['text'][t['annotation_region']['start']-t['offset_base']:t['annotation_region']['end']-t['offset_base']]==s['text'] for t,s in zip(tasks,sentences))
    assert report['training_ready'] is False and report['test_eligible'] is False
    assert report['selection_bias']=='software-project-selected sentences; not a recall benchmark'
    assert ecosystems.collect_pilot(config,tmp_path/'pilot')==report


def test_native_alias_or_excerpt_is_excluded_before_annotation():
    from research.data.ecosystems import ExclusionIndex
    old=validate_document({'document_id':'native-test','source_ids':{'pmcid':'PMC123'},
                           'text':'The same distinctive passage describing a particular scientific experiment and its software.'})
    index=ExclusionIndex([old])
    alias=validate_document({'document_id':'new','source_ids':{'pmcid':'https://example.org/PMC123'},'text':'Different extraction.'})
    assert index.reasons(alias)[0]['reason']=='identifier_overlap'
    excerpt=validate_document({'document_id':'excerpt','text':'A heading. '+old['text']+' More results.'})
    assert index.reasons(excerpt)[0]['reason']=='text_overlap'


def test_pilot_never_converts_acquisition_errors_to_zero_mentions(tmp_path,monkeypatch):
    from research.data import ecosystems
    from research.data.acquire import AcquisitionError
    def fail(*args):raise AcquisitionError('http_403')
    monkeypatch.setattr(ecosystems,'fetch_public',fail)
    result=ecosystems.collect_pilot({'projects':[{'url':'https://papers.ecosyste.ms/api/v1/projects/pypi/numpy'}],
                                    'max_papers':1,'exclusion_documents':[]},tmp_path/'pilot')
    assert result['status']=='blocked' and result['paper_count']==0
    assert read_jsonl(tmp_path/'pilot/issues.jsonl')[0]['code']=='http_403'
    assert not (tmp_path/'pilot/bundle').exists()


def test_exposed_alias_can_only_be_assigned_to_training(tmp_path):
    from research.data.splits import group_works,assign_splits,SplitError
    a=validate_document({'document_id':'exposed','source':'ecosystems','source_ids':{'doi':'10.1/x'},
                         'text':'Pilot version.','development_exposed':True,'fulltext_eligible':False})
    b=validate_document({'document_id':'fresh-id','source':'ecosystems','source_ids':{'doi':'10.1/x'},
                         'text':'New extraction.','fulltext_eligible':True})
    groups=group_works([a,b])['groups']
    for role in ('dev','test'):
        with pytest.raises(SplitError,match='QUOTA_SHORTAGE'):
            assign_splits(groups,{'ecosystems':{role:1}},42)
    assert len(assign_splits(groups,{'ecosystems':{'train':1}},42)['documents'])==1


def test_cli_loads_exposure_reservations_and_checks_their_hashes(tmp_path):
    from research.cli import main
    from research.data.bundles import materialize_bundle
    from research.data.manifest import digest,json_bytes,write_once
    docs=[validate_document({'document_id':str(i),'source':'ecosystems','source_record_id':str(i),
                            'source_ids':{'doi':f'10.1234/{i}'},'text':f'Paper {i}.','fulltext_eligible':True}) for i in range(3)]
    corpus=tmp_path/'corpus';write_jsonl(corpus/'documents.jsonl',docs)
    write_once(corpus/'manifest.json',json_bytes({'files':[{'path':'documents.jsonl','sha256':digest((corpus/'documents.jsonl').read_bytes())}],'sources':[]}))
    reserved=[{**d,'document_id':'exposed-'+d['document_id'],'split':'train'} for d in docs]
    materialize_bundle({'role':'train','documents':reserved,'heldout':{}},tmp_path/'reserved')
    config={'historical_documents':[],'training_reservations':[str(tmp_path/'reserved')],
            'quotas':{'ecosystems':{'train':1,'dev':1,'test':1}},'seed':42}
    (tmp_path/'config.json').write_text(json.dumps(config))
    args=['data','split','--corpus',str(corpus),'--config',str(tmp_path/'config.json'),
          '--output',str(tmp_path/'split'),'--private-output',str(tmp_path/'private')]
    assert main(args)==2
    report=json.loads((tmp_path/'split/blocked.json').read_bytes())
    assert {r['split'] for r in report['shortages']}=={'dev','test'}
    (tmp_path/'reserved/documents.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='missing or changed artifact'):
        main(args)


def test_ecosystems_cli_is_available_and_reports_blocked(tmp_path,monkeypatch):
    from research.cli import main
    from research.data import ecosystems
    from research.data.acquire import AcquisitionError
    def fail(*args):raise AcquisitionError('http_403')
    monkeypatch.setattr(ecosystems,'fetch_public',fail)
    (tmp_path/'config.json').write_text(json.dumps({'projects':[{'url':'https://papers.ecosyste.ms/api/v1/projects/pypi/numpy'}],
                                                  'max_papers':1,'exclusion_documents':[]}))
    assert main(['data','ecosystems','--config',str(tmp_path/'config.json'),'--output',str(tmp_path/'pilot')])==2
    assert json.loads((tmp_path/'pilot/manifest.json').read_bytes())['status']=='blocked'


@pytest.fixture
def pilot_inputs(tmp_path, monkeypatch):
    from research.data import ecosystems, acquire
    from test_acquisition import Session, Response
    project='https://papers.ecosyste.ms/api/v1/projects/pypi/numpy'
    paper='https://papers.ecosyste.ms/api/v1/papers/10.1234%2Fexample'
    search='https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI%3A%2210.1234%2Fexample%22&format=json&resultType=core&pageSize=10'
    urls={'project':project,'mentions':project+'/mentions?per_page=20&page=1','paper':paper,'search':search}
    responses={project:{'name':'numpy','project_url':project},
               urls['mentions']:[{'project_url':project,'paper_url':paper}],
               paper:{'doi':'10.1234/example'},
               search:{'resultList':{'result':[{'doi':'10.1234/example','pmcid':'PMC123','isOpenAccess':'Y','language':'eng'}]}}}
    original=acquire.fetch_public
    def fetch(url,path,policy):
        data=ARTICLE.encode() if url.endswith('/fullTextXML') else json.dumps(responses[url]).encode()
        return original(url,path,{**policy,'session':Session([Response(data)])})
    monkeypatch.setattr(ecosystems,'fetch_public',fetch)
    excluded=tmp_path/'excluded.jsonl'
    write_jsonl(excluded,[{'document_id':'excluded','text':'A different article with no shared content.'}])
    config={'projects':[{'url':project}],'max_papers':1,'exclusion_documents':[str(excluded)]}
    return config, tmp_path/'pilot', urls, responses


@pytest.mark.parametrize('change',['missing','changed'])
def test_cached_pilot_rechecks_external_exclusions(pilot_inputs,change):
    from research.data.ecosystems import collect_pilot
    config,output,_,_=pilot_inputs
    collect_pilot(config,output)
    excluded=Path(config['exclusion_documents'][0])
    if change=='missing': excluded.unlink()
    else: excluded.write_text('{"document_id":"new","text":"Changed corpus."}\n')
    with pytest.raises(ValueError,match='STALE_EXCLUSION_INPUTS'):
        collect_pilot(config,output)


@pytest.mark.parametrize('stage,payload',[
    ('project',None), ('mentions',[None]), ('mentions',[{'paper_url':None}]),
    ('paper',[]), ('search',None), ('search',{'resultList':None}),
    ('search',{'resultList':{'result':None}}), ('search',{'resultList':{'result':[None]}}),
    ('search',{'resultList':{'result':[{'doi':'10.1234/example','isOpenAccess':'Y','pmcid':None}]}}),
])
def test_malformed_remote_shapes_become_acquisition_issues(pilot_inputs,stage,payload):
    from research.data.ecosystems import collect_pilot
    config,output,urls,responses=pilot_inputs
    responses[urls[stage]]=payload
    report=collect_pilot(config,output)
    assert report['status']=='blocked' and report['issue_count']==1
    assert read_jsonl(output/'issues.jsonl')[0]['code']
    assert (output/'manifest.json').is_file()


def test_bad_candidate_does_not_discard_later_valid_paper(pilot_inputs):
    from research.data.ecosystems import collect_pilot
    config,output,urls,responses=pilot_inputs
    responses[urls['mentions']].insert(0,None)
    report=collect_pilot(config,output)
    assert report['paper_count']==1 and report['issue_count']==1


@pytest.mark.parametrize('stage',['bundle/manifest.json','tasks/manifest.json'])
def test_interrupted_finalization_resumes_identical_files(pilot_inputs,monkeypatch,stage):
    from research.data import ecosystems, bundles
    config,output,_,_=pilot_inputs
    module=bundles if stage.startswith('bundle/') else ecosystems
    original=module.write_once
    def interrupted(path,payload):
        if path==output/stage: raise RuntimeError('interrupted')
        return original(path,payload)
    monkeypatch.setattr(module,'write_once',interrupted)
    with pytest.raises(RuntimeError,match='interrupted'):
        ecosystems.collect_pilot(config,output)
    before={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    monkeypatch.setattr(module,'write_once',original)
    assert ecosystems.collect_pilot(config,output)['status']=='ready_for_annotation'
    assert all(p.read_bytes()==payload for p,payload in before.items())


def test_incomplete_bundle_bytes_are_not_overwritten(pilot_inputs,monkeypatch):
    from research.data import ecosystems
    config,output,_,_=pilot_inputs
    (output/'bundle').mkdir(parents=True)
    damaged=output/'bundle/documents.jsonl'
    damaged.write_bytes(b'{"partial":')
    with pytest.raises(ValueError,match='artifact conflict'):
        ecosystems.collect_pilot(config,output)
    assert damaged.read_bytes()==b'{"partial":'
    assert not (output/'manifest.json').exists()


def test_versioned_collector_policy_snapshot_and_alias_tasks(pilot_inputs):
    from research.data.ecosystems import collect_pilot
    from research.annotations.policies import load_policy
    config,output,_,_=pilot_inputs
    policy_path=Path(__file__).resolve().parents[1]/'annotations/scibert-v2/policy-2.1.md'
    config['annotation_policy']=str(policy_path)
    report=collect_pilot(config,output)
    assert report['status']=='ready_for_annotation'
    meta=json.loads((output/'tasks/manifest.json').read_bytes())
    policy=load_policy(policy_path)
    assert meta['policy']==policy
    assert meta['policy_source']['sha256']==policy['policy_hash']
    assert set(meta['prompt_sources'])=={'annotate','check'}
    tasks=read_jsonl(output/'tasks/tasks.jsonl')
    assert tasks and all('aliases' in task['requested_fields'] and task['policy_hash']==policy['policy_hash'] for task in tasks)
    assert (output/'tasks/policy.md').read_bytes()==policy_path.read_bytes()


def test_frozen_collector_cache_replay_retains_existing_hashes_without_fetch(monkeypatch):
    from research.data import ecosystems
    root=Path(__file__).resolve().parents[1]
    output=root/'data/scibert-v2/ecosystems-pilot-002'
    config=json.loads((output/'config.json').read_bytes())
    before=(output/'manifest.json').read_bytes()
    task_manifest=(output/'tasks/manifest.json').read_bytes()
    def no_fetch(*args,**kwargs):
        raise AssertionError('cached replay attempted network access')
    monkeypatch.setattr(ecosystems,'fetch_public',no_fetch)
    assert ecosystems.collect_pilot(config,output)==json.loads(before)
    assert (output/'manifest.json').read_bytes()==before
    assert (output/'tasks/manifest.json').read_bytes()==task_manifest
