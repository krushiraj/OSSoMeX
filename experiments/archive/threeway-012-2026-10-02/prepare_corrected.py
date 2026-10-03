import json, re
from pathlib import Path
from run import ROOT, OUT, rows, save, lines, sha
from research.comparison.windows import freeze_windows
from transformers import AutoTokenizer
base=OUT/'corrected';base.mkdir(exist_ok=True)
tok=AutoTokenizer.from_pretrained(ROOT/'checkpoints/scibert-detector-012-softcite-wordpiece',local_files_only=True)
source=ROOT/'data/scibert-v2/corpus-001'
docs=rows(source/'documents.jsonl');refs=rows(source/'occurrences.jsonl');cov=rows(source/'coverage.jsonl');manifest={}
for dataset,oldfile in [('somesci10-corrected','inputs/somesci/somesci_docs.jsonl'),('sofair6-corrected','inputs/sofair/sofair_docs.jsonl')]:
 ids=[d['document_id'].split('_',1)[1] for d in rows(ROOT/oldfile)]
 selected=[];mapping={}
 for identifier in ids:
  found=[d for d in docs if re.search(r'(?:/|:)'+re.escape(identifier)+r'(?:\.|/|$)',d['document_id'])]
  if len(found)!=1:raise ValueError((identifier,len(found)))
  selected+=found;mapping[identifier]=found[0]['document_id']
 dids={d['document_id'] for d in selected};references=[r for r in refs if r['document_id'] in dids];coverage=[c for c in cov if c['document_id'] in dids]
 for r in references:
  d=next(d for d in selected if d['document_id']==r['document_id']);s=r['name_span'];assert d['text'][s['start']:s['end']]==r['name']
 target=base/dataset;target.mkdir(exist_ok=True)
 lines(target/'inputs.jsonl',selected);lines(target/'references-source.jsonl',references);lines(target/'coverage-source.jsonl',coverage)
 windows=freeze_windows(selected,tok);lines(target/'windows.jsonl',windows)
 manifest[dataset]={'documents':len(selected),'windows':len(windows),'gold_names':len(references),'mapping':mapping,'coverage_complete_regions':sum(c['fields'].get('software',False) for c in coverage)}
save(base/'freeze.json',{'datasets':manifest,'sources':{str(p.relative_to(ROOT)):sha(p) for p in [source/'documents.jsonl',source/'occurrences.jsonl',source/'coverage.jsonl']},'policy':'Repaired native imports for the same selected source papers. Partial software coverage: recall only; do not treat unlabelled predictions as false positives.'})
print(json.dumps(manifest,indent=2))
