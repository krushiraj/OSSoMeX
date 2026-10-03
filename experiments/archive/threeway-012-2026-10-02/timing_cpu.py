"""Additional native ARM CPU control; run only after the benchmark queue finishes."""
import json,random,statistics,time
from pathlib import Path
import torch
from run import ROOT,OUT,save,lines
from jsonl_io import rows
from research.training.full_label import FullLabelPipeline
torch.set_num_threads(4)
assert 'COMPLETE ALL' in (OUT/'queue.log').read_text()
assert 'COMPLETE WHITESPACE CONTROL' in (OUT/'whitespace-control.log').read_text()
start=time.perf_counter();pipe=FullLabelPipeline(ROOT/'checkpoints/scibert-full-label-012','cpu');load=time.perf_counter()-start
warm=pipe.predict({'document_id':'warmup','text':'We used NumPy version 1.24 for statistical analysis.'})
records=[]
for repeat in range(3):
 docs=rows(OUT/'fresh30/inputs.jsonl');random.Random(1234+repeat).shuffle(docs)
 for i,d in enumerate(docs):
  t=time.perf_counter();result=pipe.predict(d);elapsed=time.perf_counter()-t
  records.append({'repeat':repeat+1,'document_id':d['document_id'],'seconds':elapsed,'result':result})
 print('CPU repeat',repeat+1,'complete',flush=True)
lines(OUT/'timing/cpu-control-raw.jsonl',records);ts=[r['seconds'] for r in records]
save(OUT/'timing/cpu-control-summary.json',{'arm':'ossomex012','device':'native ARM CPU','threads':4,'load_seconds':load,'observations':len(ts),'median_ms':1000*statistics.median(ts),'p95_ms':1000*sorted(ts)[int(.95*len(ts))],'windows_per_second':len(ts)/sum(ts),'seconds':sum(ts),'status_counts':{s:sum(r['result']['status']==s for r in records) for s in set(r['result']['status'] for r in records)},'note':'No simultaneous benchmark inference. Native ARM CPU versus amd64 Docker still differs in architecture/runtime.'})
print('COMPLETE CPU CONTROL',flush=True)
