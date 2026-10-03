"""Catalogue/candidate coverage, paired bootstrap, timing and error summaries."""
import ast, collections, csv, json, random, re, statistics, unicodedata
from pathlib import Path
from run import ROOT, OUT, save
from jsonl_io import rows
ARMS=['ossomex012','wapiti','scibert']
def norm(s):return re.sub('[^a-z0-9]+','',unicodedata.normalize('NFKD',s).lower())
def pred(name,arm,policy=None):
 p=OUT/name/f'predictions-{arm}-{policy or ("native" if arm=="ossomex012" else "explicit")}.jsonl'
 return rows(p) if p.exists() else None

def main():
 coverage=[]
 papers=json.loads((ROOT/'reports/scibert-v2/fulltext-001/corpus-manifest.json').read_text())['papers']
 for arm in ARMS:
  for policy in ['native'] if arm=='ossomex012' else ['native','explicit']:
   predictions=pred('fulltext20',arm,policy)
   if predictions is None:continue
   per=[]
   for p in papers:
    surface_path=OUT/'fulltext20'/f'surfaces-{arm}-{policy}.jsonl'
    names={norm(s) for r in rows(surface_path) if r['document_id']==p['document_id'] for s in r['names']} if surface_path.exists() else {norm(r['name']) for r in predictions if r['document_id']==p['document_id']}
    projects=p['ecosystems_projects'];hit=[s for s in projects if norm(s.rsplit('/',1)[-1]) in names]
    per.append({'document_id':p['document_id'],'hits':len(hit),'total':len(projects),'matched_projects':hit})
   coverage.append({'dataset':'fulltext20','arm':arm,'policy':policy,'hits':sum(p['hits'] for p in per),'total':sum(p['total'] for p in per),'per_document':per})
 tree=ast.parse((ROOT/'reports/scibert-v2/diverse-001/scripts/make_annotate_html.py').read_text());candidates=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CANDIDATES' for t in n.targets))
 for arm in ARMS:
  predictions=pred('diverse10-excerpts',arm)
  if predictions is None:continue
  per=[]
  for d in rows(OUT/'diverse10-excerpts/inputs.jsonl'):
   meta=d['metadata'];names=candidates[meta['ecosystem']];rx=re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(n) for n in sorted(set(names),key=len,reverse=True))+r')(?![A-Za-z0-9])');gold={(m.start(),m.end()) for m in rx.finditer(d['text'])};ps={(p['name_span']['start'],p['name_span']['end']) for p in predictions if p['document_id']==d['document_id']}
   per.append({'document_id':d['document_id'],'topic':meta['ecosystem'],'hits':len(ps&gold),'total':len(gold),'predictions':len(ps)})
  coverage.append({'dataset':'diverse10-excerpts','arm':arm,'policy':'native' if arm=='ossomex012' else 'explicit','hits':sum(p['hits'] for p in per),'total':sum(p['total'] for p in per),'per_document':per})
 for arm in ARMS:
  predictions=pred('dev190',arm)
  if predictions is None:continue
  per=[]
  for d in rows(OUT/'dev190/inputs.jsonl'):
   seed=d['metadata']['query'];names={norm(r['name']) for r in predictions if r['document_id']==d['document_id']}
   per.append({'document_id':d['document_id'],'seed':seed,'hit':norm(seed) in names})
  coverage.append({'dataset':'dev190-query-seed','arm':arm,'hits':sum(p['hit'] for p in per),'total':len(per),'per_document':per})
 save(OUT/'coverage-diagnostics.json',coverage)
 comparisons=[]
 for dataset in ['fresh30','exposed50','reserve27','regression7','softcite-gold-paragraphs']:
  predictions={a:pred(dataset,a) for a in ARMS}
  if any(p is None for p in predictions.values()):continue
  src=rows(OUT/dataset/'references-source.jsonl');gold=collections.defaultdict(set)
  if 'spans' in src[0]:
   for r in src:gold[r['document_id']]={(s['start'],s['end']) for s in r['spans'] if s['label']=='SOFTWARE'}
  else:
   for r in src:gold[r['document_id']].add((r['name_span']['start'],r['name_span']['end']))
  dids=[d['document_id'] for d in rows(OUT/dataset/'inputs.jsonl')];counts={}
  for arm,ps in predictions.items():
   pmap=collections.defaultdict(set)
   for p in ps:pmap[p['document_id']].add((p['name_span']['start'],p['name_span']['end']))
   policy='native' if arm=='ossomex012' else 'explicit'
   metric=json.loads((OUT/dataset/f'metrics-{arm}-{policy}.json').read_text())
   counted={r['document_id']:r['counts']['mentions'] for r in metric['per_document']}
   counts[arm]=[tuple(counted[d][k] for k in ['tp','fp','fn']) for d in dids]
  if dataset=='softcite-gold-paragraphs':
   papers=collections.defaultdict(list)
   for i,d in enumerate(rows(OUT/dataset/'inputs.jsonl')):papers[d['metadata']['article']].append(i)
   counts={arm:[tuple(sum(cs[i][k] for i in indices) for k in range(3)) for indices in papers.values()] for arm,cs in counts.items()}
  units=len(counts['ossomex012'])
  def f1(cs):
   t,f,n=map(sum,zip(*cs));return 2*t/(2*t+f+n) if 2*t+f+n else 0
  for other in ['wapiti','scibert']:
   rng=random.Random(1234);deltas=[]
   for _ in range(5000):
    ix=[rng.randrange(units) for _ in range(units)];deltas.append(f1([counts['ossomex012'][i] for i in ix])-f1([counts[other][i] for i in ix]))
   deltas.sort();comparisons.append({'dataset':dataset,'comparison':'ossomex012 minus '+other,'delta_f1':f1(counts['ossomex012'])-f1(counts[other]),'bootstrap_95_percent_interval':[deltas[125],deltas[4874]],'resamples':5000,'units':units,'unit':'source article' if dataset=='softcite-gold-paragraphs' else 'input document','limitations':'provisional labels on local snippet sets; no correction for multiple comparisons'})
 save(OUT/'paired-bootstrap.json',comparisons)
 timing=[]
 base=OUT/'timing'
 if (base/'freeze.json').exists():
  for arm in ARMS:
   records=[];per=[]
   for name in json.loads((base/'freeze.json').read_text())['datasets']:
    p=base/name/f'raw-{arm}.jsonl'
    if not p.exists():continue
    rs=rows(p);records+=rs;ts=[r['seconds'] for r in rs];per.append({'repeat':name,'documents':len(rs),'seconds':sum(ts),'median_ms':1000*statistics.median(ts)})
   if not records:continue
   empty=0
   for r in records:
    if arm=='ossomex012':has_name=bool(r['result'].get('field_predictions'))
    else:
     result=r.get('result',{});body=json.loads(result['body'].encode('latin1').decode('utf8')) if result.get('status_code')==200 else {}
     has_name=any(m.get('language') or (m.get('software-name') and m.get('software-type')!='implicit') for m in body.get('mentions',body.get('software',[])))
    empty+=not has_name
   ts=[r['seconds'] for r in records];timing.append({'arm':arm,'observations':len(ts),'responses_without_explicit_names':empty,'seconds':sum(ts),'median_ms':1000*statistics.median(ts),'p95_ms':1000*sorted(ts)[int(.95*len(ts))],'windows_per_second':len(ts)/sum(ts),'repeats':per})
 save(OUT/'timing-summary.json',timing)
 if (base/'cpu-control-raw.jsonl').exists():
  def decisions(result):
   fields=[{'name':f['name'],'span':f['name_span'],**{k:{'status':f[k]['status'],'value':f[k]['value']} for k in ['versions','intents','sentiment']}} for f in result['field_predictions']]
   return json.dumps({'status':result['status'],'fields':sorted(fields,key=lambda f:(f['span']['start'],f['span']['end']))},sort_keys=True)
  mps={(repeat,r['document_id']):r for repeat in range(1,4) for r in rows(base/f'fresh30-repeat{repeat}/raw-ossomex012.jsonl')}
  cpu=rows(base/'cpu-control-raw.jsonl');differences=[{'repeat':r['repeat'],'document_id':r['document_id']} for r in cpu if decisions(r['result'])!=decisions(mps[(r['repeat'],r['document_id'])]['result'])]
  save(OUT/'timing/device-decision-comparison.json',{'observations':len(cpu),'different_field_decisions':differences,'scope':'Names, spans, linked-version values, intent, sentiment and completion status; excludes floating-point scores, aliases and run/mention identifiers.'})
 errors=list(csv.DictReader((OUT/'errors.csv').open())) if (OUT/'errors.csv').exists() else []
 freq=[]
 for dataset in ['fresh30','exposed50','reserve27','softcite-gold-paragraphs']:
  for kind in ['FN','FP']:
   counter=collections.Counter(r['name'] for r in errors if r['arm']=='ossomex012' and r['dataset']==dataset and r['kind']==kind)
   freq.append({'dataset':dataset,'kind':kind,'top_names':counter.most_common(20)})
 save(OUT/'ossomex-error-frequency.json',freq)
 boundary=[]
 for dataset in ['fresh30','exposed50','reserve27','regression7','softcite-gold-paragraphs']:
  for arm in ARMS:
   policy='native' if arm=='ossomex012' else 'explicit'
   es=[r for r in errors if r['dataset']==dataset and r['arm']==arm and r['policy']==policy]
   fps=[r for r in es if r['kind']=='FP'];fns=[r for r in es if r['kind']=='FN']
   def overlap(a,b):return a['document_id']==b['document_id'] and max(int(a['start']),int(b['start']))<min(int(a['end']),int(b['end']))
   boundary.append({'dataset':dataset,'arm':arm,'false_positives':len(fps),'false_negatives':len(fns),'fp_overlapping_a_missed_reference':sum(any(overlap(p,n) for n in fns) for p in fps),'fn_overlapping_an_inexact_prediction':sum(any(overlap(n,p) for p in fps) for n in fns),'note':'Covered exact-span errors only. Overlap is a boundary-error candidate, not proof of correct entity recognition; counts are not one-to-one matches.'})
 save(OUT/'boundary-diagnostics.json',boundary)
 print('diagnostics written')
if __name__=='__main__':main()
