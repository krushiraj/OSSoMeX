"""Standalone, read-only search of saved exact-span error evidence."""
import csv,json
from pathlib import Path
from run import OUT
from research.evaluation.metrics import _covered
from jsonl_io import rows
records=[]
for group,base in [('main',OUT),('corrected',OUT/'corrected'),('whitespace-control',OUT/'whitespace-control'),('whitespace-canonical',OUT/'whitespace-canonical')]:
 p=base/'errors.csv'
 if p.exists():records += [{**r,'experiment':group} for r in csv.DictReader(p.open())]
unjudged=[]
for dataset in ['fresh30','exposed50','reserve27']:
 docs={d['document_id']:d for d in rows(OUT/dataset/'inputs.jsonl')}
 refs=rows(OUT/dataset/'references-source.jsonl');cov=[c for r in refs for c in r.get('attribute_coverage',[])]
 gold={(r['document_id'],s['name_span']['start'],s['name_span']['end']) for r in refs for s in r.get('attribute_occurrences',[])}
 for arm,policy in [('ossomex012','native'),('wapiti','explicit'),('scibert','explicit')]:
  path=OUT/dataset/f'predictions-{arm}-{policy}.jsonl'
  if not path.exists():continue
  for p in rows(path):
   span=p['name_span'];key=(p['document_id'],span['start'],span['end'])
   covered=_covered([c for c in cov if c['document_id']==p['document_id']],(span['start'],span['end']),'software',len(docs[p['document_id']]['text']))
   if key not in gold and not covered:
    d=docs[p['document_id']];unjudged.append({'experiment':'main','dataset':dataset,'arm':arm,'policy':policy,'kind':'UNJUDGED','document_id':p['document_id'],'name':p['name'],'start':span['start'],'end':span['end'],'context':d['text'][max(0,span['start']-100):span['end']+100]})
records+=unjudged
with (OUT/'unjudged-predictions.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['experiment','dataset','arm','policy','kind','document_id','name','start','end','context']);writer.writeheader();writer.writerows(unjudged)
payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OSSoMeX benchmark error review</title><style>body{font:16px/1.55 system-ui;margin:36px auto;max-width:1080px;padding:0 24px;color:#1b2c36;background:#fafbf9}h1{line-height:1.2}a{color:#146c5d}nav{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}select,input,button{font:inherit;padding:8px;border:1px solid #bbc9c5;border-radius:5px;background:white}input{flex:1;min-width:240px}article{background:white;border:1px solid #dbe3df;border-radius:7px;padding:16px;margin:12px 0}h3{margin:0 0 8px}.meta{font-size:13px;color:#5a6d74;overflow-wrap:anywhere}pre{white-space:pre-wrap;font:15px/1.65 Georgia,serif}#summary{font-weight:600}.fp{color:#a64532}.fn{color:#146c5d}</style><a href="report.html">← Comparison report</a><h1>Review predictions and errors</h1><p>Search the saved source context behind an exact-span false positive or false negative. References are provisional on OpenAlex/regression and positive-only on repaired imports. Main document-gold offsets are unreliable for Softcite; use the separate whitespace-control results when available. UNJUDGED candidates have no adjudicated reference label and are excluded from scoring. The same papers and predictions may appear in several experiments or name policies.</p><nav><select id="experiment"></select><select id="dataset"></select><select id="arm"></select><select id="policy"></select><select id="kind"></select><input id="search" placeholder="Search software name or context" aria-label="Search errors"></nav><p id="summary"></p><main id="results"></main><button id="more">Show 100 more</button><script id="data" type="application/json">DATA</script><script>
const rows=JSON.parse(document.querySelector('#data').textContent);let limit=100;
const fields=['experiment','dataset','arm','policy','kind'];
for(const field of fields){const el=document.getElementById(field);const all=document.createElement('option');all.value='';all.textContent='All '+({experiment:'experiments',dataset:'datasets',arm:'models',policy:'name policies',kind:'record types'}[field]);el.append(all);for(const value of [...new Set(rows.map(r=>r[field]))].sort()){const option=document.createElement('option');option.value=value;option.textContent=value;el.append(option)}el.addEventListener('change',()=>{limit=100;render()})}
document.querySelector('#search').addEventListener('input',()=>{limit=100;render()});document.querySelector('#more').addEventListener('click',()=>{limit+=100;render()});
const render=()=>{const q=document.querySelector('#search').value.toLowerCase();const selected=rows.filter(r=>fields.every(f=>!document.getElementById(f).value||r[f]===document.getElementById(f).value)&&(!q||(r.name+' '+r.context+' '+r.document_id).toLowerCase().includes(q)));document.querySelector('#summary').textContent=`${selected.length} matching records; showing ${Math.min(limit,selected.length)}.`;const target=document.querySelector('#results');target.replaceChildren();for(const r of selected.slice(0,limit)){const card=document.createElement('article');const title=document.createElement('h3');title.className=r.kind.toLowerCase();title.textContent=`${r.kind}: ${r.name}`;const meta=document.createElement('div');meta.className='meta';meta.textContent=`${r.experiment} · ${r.dataset} · ${r.arm} · ${r.policy} · offsets ${r.start}–${r.end}`;const doc=document.createElement('div');doc.className='meta';doc.textContent=r.document_id;const context=document.createElement('pre');context.textContent=r.context;card.append(title,meta,doc,context);target.append(card)}document.querySelector('#more').hidden=selected.length<=limit};render();
</script></html>'''.replace('DATA',payload)
(OUT/'errors.html').write_text(page)
print('error viewer records',len(records))
