from run import OUT,save
from jsonl_io import rows
out=[]
for dataset in ['fresh30','exposed50','reserve27','regression7']:
 source=rows(OUT/dataset/'references-source.jsonl');known={}
 for r in source:
  for pair in r.get('alias_pairs',[]):
   if pair.get('known'):
    spans=tuple(sorted((m['span']['start'],m['span']['end']) for m in pair['members']))
    known[(r['document_id'],spans)]=pair['decision']
 windows={w['window_id']:w for w in rows(OUT/dataset/'windows.jsonl')};positive=set()
 for r in rows(OUT/dataset/'raw-ossomex012.jsonl'):
  w=windows[r['window_id']]
  for p in r['result']['alias_predictions']['pairs']:
   if p['status']=='success' and p['label']=='alias':positive.add((r['document_id'],tuple(sorted((p[k]['start']+w['start'],p[k]['end']+w['start']) for k in ['first_span','second_span']))))
 gold={k for k,v in known.items() if v=='alias'}
 out.append({'dataset':dataset,'arm':'ossomex012','known_pairs':len(known),'positive_gold_pairs':len(gold),'true_positive_pairs':len(gold&positive),'false_negative_pairs':len(gold-positive),'predicted_positive_reviewed_negative_pairs':len({k for k in positive if known.get(k)=='not_alias'}),'unreviewed_positive_predictions':len(positive-known.keys()),'positive_pair_recall':len(gold&positive)/len(gold) if gold else None,'scope':'exact endpoints; provisional selected pair labels; no exhaustive pair precision; Softcite aliases unsupported'})
save(OUT/'alias-diagnostics.json',out)
for r in out:print(r)
