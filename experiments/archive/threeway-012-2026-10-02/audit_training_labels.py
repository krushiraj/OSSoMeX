import collections,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from run import ROOT,OUT,save
from jsonl_io import rows
m=json.loads((ROOT/'checkpoints/scibert-detector-012-softcite-wordpiece/manifest.json').read_text())
t='{http://www.tei-c.org/ns/1.0}'
hold={(i.text or '').strip().lower() for i in ET.parse(ROOT/'reports/scibert-v2/softcite-gold-001/softcite-holdout.tei.xml').getroot().iter(t+'idno') if (i.text or '').strip().lower().startswith('10.')}
dev=set(m['provenance']['dev_article_ids']);train=[r for r in rows(ROOT/'reports/scibert-v2/softcite-gold-001/gold-documents.jsonl') if (r['ids'].get('DOI') or '').strip().lower() not in hold|dev]
assert len(train)==900
out=[]
for name in ['Java','Python','MATLAB','SPSS','NumPy','R','C']:
 matched=labelled=0;examples=[]
 for d in train:
  for match in re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',d['text']):
   matched+=1;covered=any(n['start']<=match.start() and n['end']>=match.end() for n in d['software_names']);labelled+=covered
   if not covered and len(examples)<6:examples.append({'document_id':d['document_id'],'context':d['text'][max(0,match.start()-100):match.end()+100]})
 out.append({'surface':name,'literal_text_occurrences':matched,'covered_by_software_name':labelled,'outside_software_name':matched-labelled,'unlabelled_examples':examples})
train_ids={r['document_id'] for r in train}
paragraphs=rows(ROOT/'reports/scibert-v2/softcite-gold-001/paragraphs.jsonl')
train_paragraphs=[p for p in paragraphs if p['document_id'].rsplit('#p',1)[0] in train_ids]
test=rows(OUT/'softcite-gold-paragraphs/inputs.jsonl')
normalize=lambda text:' '.join(text.split())
train_texts={normalize(p['text']) for p in train_paragraphs}
overlap=[p['document_id'] for p in test if normalize(p['text']) in train_texts]
save(OUT/'train-gold-overlap.json',{'detector_training_records':len(train),'mapped_training_paragraphs':len(train_paragraphs),'test_paragraphs':len(test),'same_normalized_paragraphs':overlap,'same_article_ids':sorted(train_ids & {p['document_id'].rsplit('#p',1)[0] for p in test}),'normalization':'Unicode whitespace collapse only; exact text, not a near-duplicate audit','scope':'OSSoMeX012 reconstructed training pool only. The packaged Softcite models training membership was not independently audited.'})
save(OUT/'training-label-audit.json',{'train_records':len(train),'note':'Case-sensitive literal words inside training source; outside software spans are O-supervised. Not all literal matches are software (especially R and C). Manual context adjudication required. Counts exclude duplicated ablation copies.','surfaces':out})
names=[s['name'] for d in train for s in d['software_names']]
normalize_name=lambda s:' '.join(s.casefold().split())
vocabulary={normalize_name(n) for n in names};seen=[]
for dataset in ['fresh30','exposed50','reserve27','regression7','softcite-gold-paragraphs']:
 refs=rows(OUT/dataset/'references-source.jsonl')
 gold=[(r['document_id'],s['start'],s['end'],s['text']) for r in refs for s in r.get('spans',[]) if s['label']=='SOFTWARE'] if 'spans' in refs[0] else [(r['document_id'],r['name_span']['start'],r['name_span']['end'],r['name']) for r in refs]
 predicted={(r['document_id'],r['name_span']['start'],r['name_span']['end']) for r in rows(OUT/dataset/'predictions-ossomex012-native.jsonl')}
 for known in [True,False]:
  subset=[r for r in gold if (normalize_name(r[3]) in vocabulary)==known];hits=sum(r[:3] in predicted for r in subset)
  seen.append({'dataset':dataset,'name_seen_in_detector_training':known,'gold_mentions':len(subset),'exact_hits':hits,'recall':hits/len(subset) if subset else None})
draws=m['training']['draw_counts'];drawn_names=[s['name'] for d in train if draws.get(d['document_id'],0)>0 for s in d['software_names']]
save(OUT/'training-vocabulary-diagnostic.json',{'training_records':len(train),'training_articles':m['provenance']['train_articles'],'software_name_annotations':len(names),'unique_exact_names':len(set(names)),'unique_casefold_whitespace_names':len(vocabulary),'drawn_original_record_ids':sum(not k.endswith('#ablated') for k in draws),'drawn_ablated_record_ids':sum(k.endswith('#ablated') for k in draws),'total_sampled_draws':sum(draws.values()),'ablated_sampled_draws':sum(v for k,v in draws.items() if k.endswith('#ablated')),'unique_exact_names_in_drawn_source_records':len(set(drawn_names)),'selection':m['training']['selection'],'normalization':'Unicode casefold and whitespace collapse; no alias or project resolution','scope':'OSSoMeX012 detector only. Name counts describe source records, not a token exposure trace of sampled windows. Packaged baseline training membership unavailable. Descriptive exposed-set recall, not a controlled unseen-name experiment.','recall_by_name_exposure':seen})
print(json.dumps([{k:v for k,v in x.items() if k!='unlabelled_examples'} for x in out],indent=2))
