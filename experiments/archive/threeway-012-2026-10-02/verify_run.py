"""Final completeness and evidence checks. Does not invoke any model."""
import csv,hashlib,json,math
from pathlib import Path
from run import ROOT,OUT,sha,save
from jsonl_io import rows
assert 'COMPLETE ALL' in (OUT/'queue.log').read_text()
assert 'COMPLETE WHITESPACE CONTROL' in (OUT/'whitespace-control.log').read_text()
freeze=json.loads((OUT/'freeze.json').read_text())
assert sha(ROOT/'checkpoints/scibert-full-label-012/manifest.json')==freeze['checkpoint_sha256']
assert sha(OUT/'run.py')==freeze['runner_sha256']
for path,digest in freeze['source_hashes'].items():assert sha(ROOT/path)==digest,path
for entry in freeze['datasets'].values():
 assert sha(ROOT/entry['input'])==entry['input_sha256']
 if entry.get('reference'):assert sha(ROOT/entry['reference'])==entry['reference_sha256']
for path,digest in json.loads((OUT/'corrected/freeze.json').read_text())['sources'].items():assert sha(ROOT/path)==digest,path
inventory=[];http_failures=[]
groups=[('main',OUT),('corrected',OUT/'corrected'),('timing',OUT/'timing'),('whitespace-control',OUT/'whitespace-control')]
if (OUT/'whitespace-canonical/freeze.json').exists():groups.append(('whitespace-canonical',OUT/'whitespace-canonical'))
for group,base in groups:
 frozen=json.loads((base/'freeze.json').read_text())
 for dataset in frozen['datasets']:
  entry=frozen['datasets'][dataset]
  if group in ['whitespace-control','whitespace-canonical']:
   assert sha(base/dataset/'inputs.jsonl')==entry['input_sha256']
   assert sha(base/dataset/'windows.jsonl')==entry['windows_sha256']
  docs={d['document_id']:d for d in rows(base/dataset/'inputs.jsonl')};windows=rows(base/dataset/'windows.jsonl')
  for w in windows:assert docs[w['document_id']]['text'][w['start']:w['end']]==w['text']
  for arm in ['ossomex012','wapiti','scibert']:
   raw=rows(base/dataset/f'raw-{arm}.jsonl');assert len(raw)==len(windows)
   assert {r['window_id'] for r in raw}=={w['window_id'] for w in windows}
   assert all(r['seconds']>=0 and math.isfinite(r['seconds']) for r in raw)
   if arm=='ossomex012':assert all(r.get('result') for r in raw)
   else:
    http_failures.extend({'experiment':group,'dataset':dataset,'arm':arm,'window_id':r['window_id'],'status_code':r.get('result',{}).get('status_code')} for r in raw if r.get('result',{}).get('status_code')!=200)
   for policy in ['native'] if arm=='ossomex012' else ['native','explicit']:
    p=base/dataset/f'predictions-{arm}-{policy}.jsonl'
    if group!='timing':
     assert p.exists(),p
     for r in rows(p):
      d=docs[r['document_id']];s=r['name_span'];assert d['text'][s['start']:s['end']]==r['name']
      for v in r['version_links']:
       s=v['span'];assert d['text'][s['start']:s['end']]==v['text']
   inventory.append({'experiment':group,'dataset':dataset,'arm':arm,'documents':len(docs),'windows':len(raw),'reused_fresh_windows':sum('reused_fresh_response' in r for r in raw)})
summary=json.loads((OUT/'summary.json').read_text())
errors=list(csv.DictReader((OUT/'errors.csv').open()))
for r in summary:
 if r.get('f1') is None:continue
 for kind,key in [('FP','fp'),('FN','fn')]:
  assert sum(e['dataset']==r['dataset'] and e['arm']==r['arm'] and e['policy']==r['policy'] and e['kind']==kind for e in errors)==r[key],(r['dataset'],r['arm'],kind)
for b in json.loads((OUT/'paired-bootstrap.json').read_text()):
 other=b['comparison'].split()[-1]
 scores={r['arm']:r['f1'] for r in summary if r['dataset']==b['dataset'] and r['policy']==('native' if r['arm']=='ossomex012' else 'explicit')}
 assert abs(b['delta_f1']-(scores['ossomex012']-scores[other]))<1e-12
assert len(rows(OUT/'timing/cpu-control-raw.jsonl'))==90
assert all(r['observations']==90 for r in json.loads((OUT/'timing-summary.json').read_text()))
(OUT/'STATUS.md').write_text('# Benchmark complete\n\nFresh inference, controlled whitespace replay, repeated timing, scoring, and structural evidence checks are complete. Model/service failures remain recorded in verification.json and the report.\n\nRead report.html for results, errors.html for source-context review, and next-experiments.md for the proposed follow-up plan. Raw outputs, input/reference manifests, and scripts remain in this directory.\n')
files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['verification.json','artifact-manifest.json'] and p.suffix!='.log']
save(OUT/'artifact-manifest.json',[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)])
result={'status':'passed','frozen_sources_unchanged':True,'inference_runner_unchanged':True,'all_expected_windows_present':True,'raw_http_failures':http_failures,'canonical_prediction_offsets_valid':True,'paired_bootstrap_matches_main_scores':True,'note':'Operational HTTP completeness is distinct from native offset integrity and full-field quality. Native alignment issues remain recorded, and the controlled replay is separate.','inventory':inventory,'artifact_count':len(files)}
save(OUT/'verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))
