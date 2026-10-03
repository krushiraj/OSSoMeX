"""Illustrative cases selected after scoring; never independent validation."""
import json,sys
from pathlib import Path
from run import OUT,save
from jsonl_io import rows
candidates=[]
for dataset in ['fresh30','exposed50','reserve27']:
 docs={d['document_id']:d for d in rows(OUT/dataset/'inputs.jsonl')};ms={}
 for arm,policy in [('ossomex012','native'),('wapiti','explicit'),('scibert','explicit')]:
  p=OUT/dataset/f'metrics-{arm}-{policy}.json'
  if p.exists():ms[arm]={r['document_id']:r['counts']['mentions'] for r in json.loads(p.read_text())['per_document']}
 if len(ms)!=3:continue
 refs={r['document_id']:r for r in rows(OUT/dataset/'references-source.jsonl')}
 for did,d in docs.items():
  delta=ms['ossomex012'][did]['tp']-ms['scibert'][did]['tp']
  if delta:candidates.append({'dataset':dataset,'document':d,'counts':{a:m[did] for a,m in ms.items()},'delta':delta,'reference_names':[s['text'] for s in refs[did]['spans'] if s['label']=='SOFTWARE']})
selected=[]
for sign in [1,-1]:
 selected+=sorted([c for c in candidates if c['delta']*sign>0],key=lambda c:abs(c['delta']),reverse=True)[:2]
sections=['# Side-by-side source cases','These examples were selected after seeing the results to show both directions. They reuse exposed/provisional benchmark references and are not an independent validation set. Counts respect the frozen software-coverage masks.']
for i,c in enumerate(selected,1):
 d=c['document'];dataset=c['dataset'];did=d['document_id'];sections += [f"## {i}. {dataset}: {'OSSoMeX' if c['delta']>0 else 'Softcite SciBERT'} recovers more reference names",'Document: `'+did+'`','> '+d['text'].replace('\n','\n> '),'Reference names: '+', '.join('`'+n+'`' for n in c['reference_names'])]
 sections.append('| Model | TP / FP / FN | Returned names |\n|---|---|---|')
 c['predictions']={}
 for arm,policy in [('ossomex012','native'),('wapiti','explicit'),('scibert','explicit')]:
  ps=[p['name'] for p in rows(OUT/dataset/f'predictions-{arm}-{policy}.jsonl') if p['document_id']==did];c['predictions'][arm]=ps;m=c['counts'][arm]
  sections[-1]+=f"\n| {arm} | {m['tp']} / {m['fp']} / {m['fn']} | "+', '.join('`'+p.replace('|','\\|')+'`' for p in ps)+' |'
sections += ['## Boundary and attribute checks','The fresh30 false positives for012 include `ImageJ®`, `Scikit-LearnThe`, `Scikit Learn-h`, and `scikit-learn 12`, where the selected span differs from the reference boundary. The searchable error viewer includes the original contexts. Name detection and exact boundary recovery should be tracked separately during error analysis, while the primary metric remains exact span F1.','The seven regression excerpts have28 reference version links.012 recovered19 with one false positive; SciBERT recovered5 with three false positives; Wapiti recovered3 with two false positives. This is stronger evidence for version association on these examples than the zero-version fresh30 set.','[Open the searchable review page](errors.html) · [Return to the full report](report.html)']
text='\n\n'.join(sections).replace('for012','for 012').replace('have28','have 28').replace('links.012','links. 012').replace('recovered19','recovered 19').replace('recovered5','recovered 5').replace('recovered3','recovered 3')+'\n'
(OUT/'examples.md').write_text(text);save(OUT/'examples.json',selected)
sys.path.insert(0,'/tmp/ossomex-article-deps');import markdown
(OUT/'examples.html').write_text('<!doctype html><meta charset="utf-8"><title>Benchmark source cases</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 25px;color:#19313a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}blockquote{border-left:3px solid #176b5a;padding-left:18px;font-family:Georgia}code{font-size:90%;overflow-wrap:anywhere}</style>'+markdown.markdown(text,extensions=['tables']))
print('source cases',len(selected))
