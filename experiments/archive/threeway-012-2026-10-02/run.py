"""Fresh matched-window inference; resume only within this frozen run directory."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
OUT=Path(__file__).resolve().parent
from research.contracts import validate_document

def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lines(p,x): Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
DATA={
 'fresh30':('data/scibert-v2/focused-evaluation-002/inputs.jsonl','annotations/scibert-v2/focused-fresh-004/references.jsonl'),
 'exposed50':('data/scibert-v2/focused-evaluation-001/inputs.jsonl','annotations/scibert-v2/focused-evaluation-003/references.jsonl'),
 'reserve27':('data/scibert-v2/focused-evaluation-001/reserve.jsonl','annotations/scibert-v2/focused-reserve-001/references.jsonl'),
 'regression7':('data/scibert-v2/cli-diagnostic-001/inputs.jsonl','data/scibert-v2/cli-diagnostic-001/references.jsonl'),
 'softcite-gold-paragraphs':('reports/scibert-v2/softcite-gold-001/inputs-para.jsonl','reports/scibert-v2/softcite-gold-001/references-para.jsonl'),
 'softcite-gold-documents':('reports/scibert-v2/softcite-gold-001/inputs.jsonl','reports/scibert-v2/softcite-gold-001/references.jsonl'),
 'somesci10':('inputs/somesci/somesci_docs.jsonl','inputs/somesci/somesci_gold.jsonl'),
 'sofair6':('inputs/sofair/sofair_docs.jsonl','inputs/sofair/sofair_gold.jsonl'),
 'fulltext20':('reports/scibert-v2/fulltext-001/inputs.jsonl',None),
 'diverse10-fulltext':('reports/scibert-v2/diverse-001/inputs.jsonl',None),
 'dev190':('inputs/dev/openalex_snippets_dev.jsonl',None),
 'reference5':('inputs/dev/reference_docs.jsonl',None),
}

def prepare():
 from transformers import AutoTokenizer
 from research.comparison.windows import freeze_windows
 if (OUT/'freeze.json').exists(): raise FileExistsError('Already frozen')
 tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/scibert-detector-012-softcite-wordpiece',local_files_only=True)
 manifests={}
 for name,(inp,ref) in DATA.items():
  docs=[validate_document(r) for r in rows(ROOT/inp)]
  if name=='dev190':
   for i,d in enumerate(docs):
    d['source_document_id']=d['document_id'];d['document_id']+=f'|row{i}'
  target=OUT/name;target.mkdir(exist_ok=True)
  lines(target/'inputs.jsonl',docs)
  if ref: (target/'references-source.jsonl').write_bytes((ROOT/ref).read_bytes())
  windows=freeze_windows(docs,tok)
  lines(target/'windows.jsonl',windows)
  manifests[name]={'input':inp,'input_sha256':sha(ROOT/inp),'reference':ref,'reference_sha256':sha(ROOT/ref) if ref else None,'documents':len(docs),'windows':len(windows),'characters':sum(len(d['text']) for d in docs)}
 docs=[validate_document({'document_id':f"diverse-excerpt:{r['n']}",'text':r['excerpt'],'metadata':r}) for r in rows(ROOT/'reports/scibert-v2/diverse-001/excerpts.jsonl')]
 target=OUT/'diverse10-excerpts';target.mkdir(exist_ok=True);lines(target/'inputs.jsonl',docs);windows=freeze_windows(docs,tok);lines(target/'windows.jsonl',windows)
 manifests[target.name]={'input':'reports/scibert-v2/diverse-001/excerpts.jsonl','input_sha256':sha(ROOT/'reports/scibert-v2/diverse-001/excerpts.jsonl'),'reference':None,'documents':len(docs),'windows':len(windows),'characters':sum(len(d['text']) for d in docs)}
 save(OUT/'freeze.json',{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'datasets':manifests,'checkpoint':'scibert-full-label-012','checkpoint_sha256':sha(ROOT/'checkpoints/scibert-full-label-012/manifest.json'),'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/research').rglob('*.py')},'runner_sha256':sha(__file__),'policy':{'windows':'480 content tokens,64 overlap; same exact text for all arms; full 5-stage OSSoMeX per window','merge':'deduplicate exact spans; choose occurrence from window with largest minimum distance to boundary; do not create links across windows','softcite_native':'software-name only','softcite_explicit':'software-name excluding implicit + language','offsets':'strict UTF16 to Unicode; no fuzzy snapping; retain malformed predictions separately','reference_status':'OpenAlex/regression provisional and exposed; legacy imports diagnostic; gold paragraphs and documents overlap; catalogue sets lack exhaustive labels','execution':'each arm sequential, no concurrent benchmark inference; warmup excluded; MPS synchronization included'}})
 print(json.dumps(manifests,indent=2))

def run(arm):
 import torch, requests
 torch.set_num_threads(4)
 device='mps'
 started=time.perf_counter();pipe=None
 if arm=='ossomex012':
  from research.training.full_label import FullLabelPipeline
  pipe=FullLabelPipeline(ROOT/'checkpoints/scibert-full-label-012',device)
 else:
  port=8060 if arm=='wapiti' else 8062
  session=requests.Session();session.trust_env=False
  endpoint=f'http://127.0.0.1:{port}/service/processSoftwareText'
  health=session.get(f'http://127.0.0.1:{port}/service/isalive',timeout=10)
  health.raise_for_status()
 def predict(text,identifier):
  if pipe:return pipe.predict(validate_document({'document_id':identifier,'text':text}))
  response=session.post(endpoint,data={'text':text},timeout=180)
  result={'status_code':response.status_code,'body':response.text}
  try:result['native']=response.json()
  except ValueError:pass
  return result
 load=time.perf_counter()-started
 warm=predict('We used NumPy version 1.24 for statistical analysis.','warmup')
 if pipe:torch.mps.synchronize()
 save(OUT/f'{arm}-environment.json',{'arm':arm,'load_seconds':load,'platform':platform.platform(),'torch':torch.__version__,'device':device if pipe else 'amd64 Docker CPU on ARM host','threads':torch.get_num_threads(),'warmup':warm})
 frozen=json.loads((OUT/'freeze.json').read_text())
 for name in frozen['datasets']:
  windows=rows(OUT/name/'windows.jsonl');path=OUT/name/f'raw-{arm}.jsonl'
  done={r['window_id'] for r in rows(path)} if path.exists() else set()
  with path.open('a') as fp:
   for n,w in enumerate(windows):
    if w['window_id'] in done:continue
    if pipe:torch.mps.synchronize()
    t=time.perf_counter();record={'window_id':w['window_id'],'document_id':w['document_id'],'arm':arm}
    try:record['result']=predict(w['text'],w['window_id'])
    except Exception as e:record['error']=repr(e)
    if pipe:torch.mps.synchronize()
    record['seconds']=time.perf_counter()-t
    fp.write(json.dumps(record,ensure_ascii=False)+'\n');fp.flush()
    if n%20==0 or n==len(windows)-1:print(arm,name,n+1,'/',len(windows),round(record['seconds'],3),record.get('error',''),flush=True)
 print('COMPLETE',arm,flush=True)

if __name__=='__main__':
 os.chdir(ROOT)
 arg=argparse.ArgumentParser();arg.add_argument('action',choices=['prepare','ossomex012','wapiti','scibert']);a=arg.parse_args()
 prepare() if a.action=='prepare' else run(a.action)
