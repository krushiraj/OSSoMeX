"""Small, source-attributed ecosystems sentence pilots. Associations are not labels."""

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlencode, urlsplit

from ..contracts import validate_document
from ..annotations.tasks import check_authorization
from .acquire import AcquisitionError, component_url, fetch_public
from .bundles import materialize_bundle
from .jats import normalize_doi, read_jats, select_sentences
from .manifest import digest, json_bytes, read_jsonl, verified_path, write_jsonl, write_once
from .splits import identifiers

ROOT = Path(__file__).resolve().parents[3]
API = 'https://papers.ecosyste.ms/api/v1'
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest'


def shingles(text):
    words = re.findall(r'\w+',text.lower())
    return set(tuple(words[i:i+5]) for i in range(len(words)-4))


class ExclusionIndex:
    def __init__(self, documents):
        self.rows = [(d['document_id'],identifiers(d),shingles(d['text']),digest(re.sub(r'\s+',' ',d['text']).strip().encode())) for d in documents]

    def reasons(self, document):
        keys, tokens = identifiers(document), shingles(document['text'])
        text_hash = digest(re.sub(r'\s+',' ',document['text']).strip().encode())
        result = []
        for ident,other_keys,other_tokens,other_hash in self.rows:
            same_text = text_hash == other_hash
            overlap = bool(tokens and other_tokens) and (len(tokens & other_tokens)/len(tokens | other_tokens)>=.85
                       or min(len(tokens),len(other_tokens))>=6 and (tokens<=other_tokens or other_tokens<=tokens))
            if keys & other_keys or same_text or overlap:
                result.append({'document_id':ident,'reason':'identifier_overlap' if keys & other_keys else 'text_overlap'})
        return result


def ecosystem_url(url, kind):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.netloc != 'papers.ecosyste.ms' or parsed.query or parsed.fragment
            or not parsed.path.startswith('/api/v1/'+kind+'/')):
        raise ValueError('INVALID_ECOSYSTEMS_URL')
    return url


def collect_pilot(config: dict, output: Path) -> dict:
    write_once(output/'config.json',json_bytes(config))
    if (output/'manifest.json').exists():
        report = json.loads((output/'manifest.json').read_bytes())
        for row in report['files']: verified_path(output,row)
        return report
    max_papers = config.get('max_papers',5)
    candidate_limit = config.get('max_candidates_per_project',10)
    if (type(max_papers) is not int or not 1<=max_papers<=20 or type(candidate_limit) is not int
            or not 1<=candidate_limit<=40 or not 1<=len(config['projects'])<=10):
        raise ValueError('INVALID_PILOT_LIMITS')
    exclusions = [d for path in config['exclusion_documents'] for d in read_jsonl(Path(path))]
    index = ExclusionIndex(exclusions)
    source_inputs = [{'path':str(Path(p).resolve()),'sha256':digest(Path(p).read_bytes())} for p in config['exclusion_documents']]
    approvals = json.loads((ROOT/'approvals.json').read_bytes())
    policy_hash = digest((ROOT/'annotations/scibert-v2/policy.md').read_bytes())
    policy_version = 'scibert-poc-2.0'
    docs, tasks, sentences, associations, issues = [], [], [], [], []
    seen = set()

    def fetch(url, *, is_json=True):
        path = output/'requests'/digest(url.encode())
        record = fetch_public(url,path,{'max_bytes':16*1024*1024,**({'format':'json'} if is_json else {})})
        return (json.loads(path.read_bytes()) if is_json else path.read_bytes()), {**record,'path':str(path.relative_to(output))}

    for spec in config['projects']:
        if len(docs)>=max_papers: break
        project_url = ecosystem_url(spec['url'],'projects')
        try:
            project, project_record = fetch(project_url)
            if project.get('project_url') != project_url or not project.get('name'):
                raise ValueError('PROJECT_IDENTITY_MISMATCH')
            candidates=[]
            for page in range(1,3):
                rows,_=fetch(project_url+'/mentions?'+urlencode({'per_page':20,'page':page}))
                if not isinstance(rows,list): raise ValueError('INVALID_MENTIONS_RESPONSE')
                candidates.extend(rows)
                if len(candidates)>=candidate_limit or len(rows)<20: break
        except (AcquisitionError,ValueError) as exc:
            issues.append({'project_url':project_url,'stage':'project','code':str(exc)})
            continue
        for candidate in candidates[:candidate_limit]:
            paper_url = candidate.get('paper_url')
            try:
                ecosystem_url(paper_url,'papers')
                if candidate.get('project_url') != project_url:
                    raise ValueError('ASSOCIATION_PROJECT_MISMATCH')
                doi = normalize_doi(unquote(urlsplit(paper_url).path.removeprefix('/api/v1/papers/')))
                if not re.fullmatch(r'10\.\d{4,9}/[^\s"<>]+',doi): raise ValueError('INVALID_DOI')
                if doi in seen: continue
                paper, paper_record = fetch(component_url(API+'/papers',doi))
                if normalize_doi(paper.get('doi')) != doi: raise ValueError('PAPER_IDENTITY_MISMATCH')
                search_url = EPMC+'/search?'+urlencode({'query':'DOI:"'+doi+'"','format':'json','resultType':'core','pageSize':10})
                result, search_record = fetch(search_url)
                found = [r for r in result.get('resultList',{}).get('result',[])
                         if normalize_doi(r.get('doi'))==doi and r.get('isOpenAccess')=='Y' and re.fullmatch(r'PMC\d+',r.get('pmcid',''))]
                if len(found)!=1: raise ValueError('NO_UNIQUE_OPEN_PMC_ARTICLE')
                pmc = found[0]['pmcid']; xml_url = EPMC+'/'+pmc+'/fullTextXML'
                raw, raw_record = fetch(xml_url,is_json=False)
                parsed = read_jats(raw,expected_doi=doi,expected_pmcid=pmc,fallback_language=found[0].get('language'))
                source_ids = {**parsed['source_ids'],**({'openalex':paper['openalex_id']} if paper.get('openalex_id') else {})}
                doc = validate_document({**parsed,'document_id':'ecosystems:'+doi,'source':'ecosystems','source_ids':source_ids,
                       'source_record_id':doi,'split':'train','development_exposed':True,'public':True,
                       'access_basis':{'article_url':xml_url,'license_url':parsed['license_url'],'xml_sha256':raw_record['sha256']},
                       'raw_source':raw_record,'paper_metadata':paper_record,'resolution_metadata':search_record,
                       'metadata_license':'CC-BY-SA-4.0','native_split':None,'annotation_status':'unannotated',
                       'work_group_id':'work:doi:'+doi})
                conflicts = index.reasons(doc)+ExclusionIndex(docs).reasons(doc)
                if conflicts:
                    issues.append({'paper_url':paper_url,'code':'EXCLUDED_OVERLAP','conflicts':conflicts})
                    continue
                chosen = select_sentences(parsed,[project['name']],max_matches=config.get('max_sentences_per_paper',4),seed=config.get('seed',42))
                if not chosen: raise ValueError('NO_LITERAL_PROJECT_MENTION')
                if any(r['paragraph_span']['end']-r['paragraph_span']['start']>6000 for r in chosen):
                    raise ValueError('SELECTED_CONTEXT_EXCEEDS_LIMIT')
                for row in chosen:
                    context=row['paragraph_span']; owned={'start':row['start'],'end':row['end']}
                    identity=[doc['document_id'],doc['text_revision'],policy_version,policy_hash,row['start'],row['end']]
                    task={'task_id':'task:'+digest(json_bytes(identity))[:32],'document_id':doc['document_id'],
                          'text_revision':doc['text_revision'],'policy_version':policy_version,'policy_hash':policy_hash,
                          'text':doc['text'][context['start']:context['end']],'context_span':context,
                          'annotation_region':owned,'offset_base':context['start'],
                          'requested_fields':['software','version','version_links','intents','sentiment'],
                          'candidate_system_predictions':None,'hard_split':False,'source':'ecosystems','split':'train',
                          'public':True,'access_basis':doc['access_basis'],'text_license':doc['text_license'],
                          'whole_passage_audit':True}
                    check_authorization(task,approvals)
                    tasks.append(task)
                    sentences.append({**row,'task_id':task['task_id'],'document_id':doc['document_id'],
                                      'text_revision':doc['text_revision'],'source_ids':source_ids,'split':'train'})
                associations.append({'document_id':doc['document_id'],'association':candidate,'project_url':project_url,
                                     'project_metadata':project_record,'candidate_alias':project['name'],'status':'weak_candidate_not_gold'})
                docs.append(doc);seen.add(doi)
                print(json.dumps({'acquired_papers':len(docs),'pmcid':pmc,'project':project['name'],'selected_sentences':len(chosen)}),flush=True)
                break
            except (AcquisitionError,ValueError) as exc:
                issues.append({'project_url':project_url,'paper_url':paper_url,'stage':'paper','code':str(exc)})
    write_jsonl(output/'issues.jsonl',issues)
    write_jsonl(output/'associations.jsonl',associations)
    write_jsonl(output/'sentences.jsonl',sentences)
    if docs:
        materialize_bundle({'documents':docs,'role':'train','heldout':{},'split_digest':digest(json_bytes([d['source_ids'] for d in docs]))},output/'bundle')
        write_jsonl(output/'tasks/tasks.jsonl',tasks)
        write_once(output/'tasks/manifest.json',json_bytes({'role':'train','task_count':len(tasks),'policy':{'policy_version':policy_version,'policy_hash':policy_hash},
                   'source_bundle_sha256':digest((output/'bundle/manifest.json').read_bytes()),
                   'files':[{'path':'tasks.jsonl','sha256':digest((output/'tasks/tasks.jsonl').read_bytes())}],
                   'whole_passage_audit_ids':[t['task_id'] for t in tasks]}))
    files=[{'path':str(p.relative_to(output)),'sha256':digest(p.read_bytes())} for p in sorted(output.rglob('*')) if p.is_file()]
    report={'status':'ready_for_annotation' if len(docs)==max_papers else 'partial' if docs else 'blocked',
            'paper_count':len(docs),'target_papers':max_papers,'sentence_count':len(sentences),'issue_count':len(issues),
            'training_ready':False,'test_eligible':False,'requires':['human_review','global_split_recheck'],
            'selection_bias':'software-project-selected sentences; not a recall benchmark','exclusion_inputs':source_inputs,'files':files}
    write_once(output/'manifest.json',json_bytes(report))
    return report
