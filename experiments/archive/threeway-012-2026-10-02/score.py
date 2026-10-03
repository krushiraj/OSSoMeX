"""Score saved fresh inference with strict offsets and explicit reference coverage."""
import collections, copy, csv, json, re, statistics, sys, unicodedata
from pathlib import Path
from run import ROOT, OUT, save, lines
from jsonl_io import rows
from research.comparison.alignment import native_span
from research.comparison.attributes import pipeline_occurrences
from research.contracts import validate_document, FIELDS
from research.evaluation.metrics import evaluate_v2, prf, _covered
ARMS=['ossomex012','wapiti','scibert']

def project(base,name,arm,policy):
 docs=rows(base/name/'inputs.jsonl');windows=rows(base/name/'windows.jsonl');wp={w['window_id']:w for w in windows}
 path=base/name/f'raw-{arm}.jsonl'
 if not path.exists():return None
 payload=path.read_text()
 if not payload.endswith('\n'):return None
 raw=[json.loads(line) for line in payload.split('\n') if line.strip()]
 if len(raw)!=len(windows):return None
 assert len({r['window_id'] for r in raw})==len(windows)
 order={w['window_id']:i for i,w in enumerate(windows)};raw.sort(key=lambda r:order[r['window_id']])
 selected={};version_spans=set();problems=[];states=collections.defaultdict(collections.Counter)
 for r in raw:
  w=wp[r['window_id']];d=next(d for d in docs if d['document_id']==w['document_id']);result=r.get('result',{});occ=[]
  if arm=='ossomex012':
   if not result:states[d['document_id']]['failure']+=1;continue
   states[d['document_id']][result['status']]+=1
   local=validate_document({'document_id':w['window_id'],'text':w['text']})
   occ=pipeline_occurrences(local,result)
   for o in occ:
    for key in ['start','end']:o['name_span'][key]+=w['start']
    for v in o['version_links']:
     for key in ['start','end']:v['span'][key]+=w['start']
   for s in (result.get('detector_diagnostics') or {}).get('spans',[]):
    if s['label']=='VERSION':version_spans.add((d['document_id'],w['start']+s['start'],w['start']+s['end']))
  else:
   body=json.loads(result['body'].encode('latin1').decode('utf-8')) if result.get('status_code')==200 else result.get('native',{});mentions=body.get('mentions',body.get('software')) if isinstance(body,dict) else None
   if result.get('status_code')!=200 or not isinstance(mentions,list):
    states[d['document_id']]['failure']+=1;problems.append({'window_id':w['window_id'],'error':r.get('error',str(result)[:500])});continue
   states[d['document_id']]['success']+=1
   for mention in mentions:
    for field in (['software-name'] if policy=='native' else ['software-name','language']):
     item=mention.get(field)
     if not item or (policy=='explicit' and field=='software-name' and mention.get('software-type')=='implicit'):continue
     try:
      s=native_span(w['text'],{'text':item['rawForm'],'start':item['offsetStart'],'end':item['offsetEnd']},unit='utf16',offset_base=w['start'])
     except Exception as e:
      problems.append({'window_id':w['window_id'],'field':field,'native':item,'error':str(e)});continue
     o={'name':s['text'],'name_span':{'start':s['start'],'end':s['end']},'version_links':[],'intents':None,'sentiment':None,'invalid_fields':[]}
     v=mention.get('version')
     if v and field=='software-name':
      try:
       vs=native_span(w['text'],{'text':v['rawForm'],'start':v['offsetStart'],'end':v['offsetEnd']},unit='utf16',offset_base=w['start'])
       o['version_links']=[{'text':vs['text'],'span':{'start':vs['start'],'end':vs['end']}}];version_spans.add((d['document_id'],vs['start'],vs['end']))
      except Exception as e:problems.append({'window_id':w['window_id'],'field':'version','native':v,'error':str(e)})
     ctx=mention.get('mentionContextAttributes',{})
     if all(isinstance(ctx.get(k),dict) and type(ctx[k].get('value'))==bool for k in ['used','created','shared']):o['intents']=[k for k in ['created','used','shared'] if ctx[k]['value']] or ['mentioned']
     occ.append(o)
  for o in occ:
   o.update(document_id=d['document_id'],text_revision=d['text_revision'])
   s=o['name_span'];assert d['text'][s['start']:s['end']]==o['name']
   key=(d['document_id'],s['start'],s['end']);margin=min(s['start']-w['start'],w['end']-s['end'])
   if key not in selected or margin>selected[key][0]:selected[key]=(margin,o)
 preds=[v[1] for v in selected.values()]
 statuses=[{'document_id':d['document_id'],'status':'failure' if states[d['document_id']]['failure'] else 'partial' if states[d['document_id']]['partial'] else 'success','window_statuses':dict(states[d['document_id']])} for d in docs]
 return docs,preds,statuses,problems,version_spans,[r['seconds'] for r in raw]

def reference(base,name,docs):
 path=base/name/'references-source.jsonl'
 if not path.exists():return [],[],[],[],[]
 source=rows(path);gold=[];coverage=[];versions=[];invalid=[];byid={d['document_id']:d for d in docs}
 if 'spans' in source[0]:
  for r in source:
   d=byid[r['document_id']]
   if r.get('attribute_occurrences'):
    gold.extend(r['attribute_occurrences']);coverage.extend(r['attribute_coverage'])
   else:
    for s in r['spans']:
     if s['label']=='SOFTWARE':gold.append({'record_kind':'legacy_evaluation_only','document_id':d['document_id'],'text_revision':d['text_revision'],'name':s['text'],'name_span':{'start':s['start'],'end':s['end']},'known':{k:k=='software' for k in FIELDS},'version_links':[],'intents':None,'sentiment':None})
    coverage.extend({'document_id':d['document_id'],'text_revision':d['text_revision'],'start':c['start'],'end':c['end'],'fields':{k:k=='software' for k in FIELDS}} for c in r['coverage'] if c['label']=='SOFTWARE' and c['complete'])
   versions.extend((r['document_id'],s['start'],s['end']) for s in r['spans'] if s['label']=='VERSION')
 elif 'known' in source[0]:
  gold=source
  cp=base/name/'coverage-source.jsonl'
  if cp.exists():coverage=rows(cp)
  else:
   suffix='-para' if name.endswith('paragraphs') else ''
   coverage=rows(ROOT/f'reports/scibert-v2/softcite-gold-001/coverage{suffix}.jsonl')
 else:
  # Legacy imports have broken offsets. Retain invalids; no F1 from these files.
  for r in source:
   d=byid[r['document_id']];s=r['name_span']
   if d['text'][s['start']:s['end']]!=r['name']:invalid.append(r)
   else:gold.append({'record_kind':'legacy_evaluation_only','document_id':d['document_id'],'text_revision':d['text_revision'],'name':r['name'],'name_span':s,'known':{k:k=='software' for k in FIELDS},'version_links':[],'intents':None,'sentiment':None})
 return gold,coverage,source,versions,invalid

def main(base=OUT):
 summary=[];errors=[]
 frozen=json.loads((base/'freeze.json').read_text())
 for name in frozen['datasets']:
  for arm in ARMS:
   for policy in (['native'] if arm=='ossomex012' else ['native','explicit']):
    data=project(base,name,arm,policy)
    if data is None:continue
    docs,pred,statuses,problems,vpred,timing=data
    gold,coverage,source,vgold,invalid=reference(base,name,docs)
    target=base/name;tag=f'{arm}-{policy}'
    lines(target/f'predictions-{tag}.jsonl',pred);save(target/f'projection-issues-{tag}.json',problems)
    out={'dataset':name,'arm':arm,'policy':policy,'documents':len(docs),'windows':len(timing),'failed_documents':sum(s['status']=='failure' for s in statuses),'partial_documents':sum(s['status']=='partial' for s in statuses),'projection_issues':len(problems),'invalid_reference_offsets':len(invalid),'predicted_names':len(pred),'seconds':sum(timing),'median_window_seconds':statistics.median(timing),'p95_window_seconds':sorted(timing)[min(len(timing)-1,int(.95*len(timing)))]}
    if gold and not invalid:
     metric=evaluate_v2(docs,copy.deepcopy(gold),pred,coverage,statuses,{'software':True,'versions':True,'version_offsets':True,'intents':True,'sentiment':arm=='ossomex012'})
     save(target/f'metrics-{tag}.json',metric);out.update(metric['mention_detection']);out['gold_names']=len(gold);out['predictions_outside_software_coverage']=metric['exclusions'].get('prediction_outside_software_coverage',0)
     if not any(c['fields'].get('software') for c in coverage):
      out.update(precision=None,f1=None,reference_scope='positive_only; recall only')
     gs={(r['document_id'],r['name_span']['start'],r['name_span']['end']):r for r in gold};ps={(r['document_id'],r['name_span']['start'],r['name_span']['end']):r for r in pred};docmap={d['document_id']:d for d in docs}
     for kind,keys,records in [('FN',gs.keys()-ps.keys(),gs),('FP',ps.keys()-gs.keys(),ps)]:
      for k in sorted(keys):
       d=docmap[k[0]];covered=_covered([c for c in coverage if c['document_id']==k[0]],(k[1],k[2]),'software',len(d['text']))
       if kind=='FP' and not covered:continue
       errors.append({'dataset':name,'arm':arm,'policy':policy,'kind':kind,'document_id':k[0],'name':records[k]['name'],'start':k[1],'end':k[2],'context':d['text'][max(0,k[1]-100):k[2]+100]})
     if source and 'spans' in source[0]:
      g=set(vgold);out['version_span_score']=prf(len(g&vpred),len(vpred-g),len(g-vpred))
      ge={(r['document_id'],e['software']['start'],e['software']['end'],e['version']['start'],e['version']['end']) for r in source for e in r.get('version_links',[])}
      pe={(p['document_id'],p['name_span']['start'],p['name_span']['end'],e['span']['start'],e['span']['end']) for p in pred for e in p['version_links'] if e.get('span')}
      out['version_link_score']=prf(len(ge&pe),len(pe-ge),len(ge-pe))
    name_issues=[p for p in problems if p.get('field') in ('software-name','language')]
    out['name_alignment_issues']=len(name_issues)
    out['alignment_invalid_documents']=len({wp['document_id'] for wp in rows(target/'windows.jsonl') if wp['window_id'] in {p['window_id'] for p in name_issues}})
    if name_issues and 'f1' in out:
     out['aligned_subset_score']={k:out.get(k) for k in ('tp','fp','fn','precision','recall','f1')}
     out.update(precision=None,recall=None,f1=None,score_status='not_ranked_unresolved_native_name_offsets')
    if invalid:save(target/'invalid-reference-offsets.json',invalid)
    summary.append(out)
 save(base/'summary.json',summary)
 keys=['dataset','arm','policy','documents','windows','gold_names','predicted_names','predictions_outside_software_coverage','tp','fp','fn','precision','recall','f1','failed_documents','partial_documents','projection_issues','name_alignment_issues','alignment_invalid_documents','invalid_reference_offsets','seconds','median_window_seconds','p95_window_seconds']
 with (base/'scores.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(summary)
 with (base/'errors.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['dataset','arm','policy','kind','document_id','name','start','end','context']);w.writeheader();w.writerows(errors)
 print(json.dumps([{k:r.get(k) for k in ['dataset','arm','policy','f1','tp','fp','fn','projection_issues']} for r in summary],indent=2))
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else OUT)
