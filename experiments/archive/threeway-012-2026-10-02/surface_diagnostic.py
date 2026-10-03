"""Position-free distinct names per document: separate from occurrence F1."""
import collections,json
from pathlib import Path
from run import ROOT,OUT,save,lines
from jsonl_io import rows
from research.evaluation.metrics import prf

def collect(base,name,arm,policy):
 path=base/name/f'raw-{arm}.jsonl'
 if not path.exists():return None
 payload=path.read_text()
 if not payload.endswith('\n'):return None
 rs=[json.loads(s) for s in payload.split('\n') if s.strip()]
 if len(rs)!=len(rows(base/name/'windows.jsonl')):return None
 names=collections.defaultdict(set)
 for r in rs:
  result=r.get('result',{})
  if arm=='ossomex012':
   names[r['document_id']].update(f['name'] for f in result.get('field_predictions',[]))
  elif result.get('status_code')==200:
   body=json.loads(result['body'].encode('latin1').decode('utf8'))
   for m in body.get('mentions',body.get('software',[])):
    for field in ['software-name'] if policy=='native' else ['software-name','language']:
     if field=='software-name' and policy=='explicit' and m.get('software-type')=='implicit':continue
     if m.get(field):names[r['document_id']].add(m[field]['rawForm'])
 lines(base/name/f'surfaces-{arm}-{policy}.jsonl',[{'document_id':d,'names':sorted(ns)} for d,ns in sorted(names.items())])
 return names

def main():
 out=[]
 for base in [OUT,OUT/'corrected']:
  if not (base/'freeze.json').exists():continue
  for name in json.loads((base/'freeze.json').read_text())['datasets']:
   ref=base/name/'references-source.jsonl';source=rows(ref) if ref.exists() else []
   gold=collections.defaultdict(set)
   for r in source:
    if 'spans' in r:gold[r['document_id']].update(s['text'] for s in r['spans'] if s['label']=='SOFTWARE')
    else:gold[r['document_id']].add(r['name'])
   for arm in ['ossomex012','wapiti','scibert']:
    for policy in ['native'] if arm=='ossomex012' else ['native','explicit']:
     names=collect(base,name,arm,policy)
     if names is None or not gold:continue
     gs={(d,s) for d,ns in gold.items() for s in ns};ps={(d,s) for d,ns in names.items() for s in ns};m=prf(len(gs&ps),len(ps-gs),len(gs-ps))
     if name in ['somesci10','sofair6'] or name.endswith('corrected'):m.update(precision=None,f1=None)
     out.append({'dataset':name,'arm':arm,'policy':policy,**m,'metric':'case-sensitive distinct surface forms per document; positions, mention multiplicity and links ignored; not occurrence F1'})
 save(OUT/'surface-diagnostics.json',out)
 print('surface diagnostics written')
if __name__=='__main__':main()
