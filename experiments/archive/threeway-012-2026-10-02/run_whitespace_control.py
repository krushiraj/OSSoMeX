"""Replay only leading-whitespace windows; reuse identical fresh-run windows."""
import hashlib,json,shutil,time
from pathlib import Path
import run
from jsonl_io import rows
BASE=run.OUT
while 'COMPLETE ALL' not in (BASE/'queue.log').read_text():
 if 'Traceback (most recent call last)' in (BASE/'queue.log').read_text():raise RuntimeError('Parent queue failed')
 time.sleep(5)
OUT=BASE/'whitespace-control';OUT.mkdir(exist_ok=True);datasets={}
for parent,name in [(BASE,'softcite-gold-documents'),(BASE,'sofair6'),(BASE/'corrected','sofair6-corrected')]:
 target=OUT/name;target.mkdir(exist_ok=True)
 for filename in ['inputs.jsonl','references-source.jsonl','coverage-source.jsonl']:
  if (parent/name/filename).exists():shutil.copyfile(parent/name/filename,target/filename)
 original=rows(parent/name/'windows.jsonl');windows=[];changed=set()
 for w in original:
  text=w['text'].lstrip();removed=len(w['text'])-len(text);new=dict(w)
  if removed:
   assert text
   changed.add(w['window_id']);new.update(start=w['start']+removed,text=text,window_id=w['window_id']+f'|lstrip:{removed}',window_text_revision=run.validate_document({'document_id':'unused','text':text})['text_revision'])
  windows.append(new)
 run.lines(target/'windows.jsonl',windows)
 for arm in ['ossomex012','wapiti','scibert']:
  source=parent/name/f'raw-{arm}.jsonl';digest=run.sha(source)
  reuse=[{**r,'reused_fresh_response':{'path':str(source.relative_to(BASE)),'file_sha256':digest}} for r in rows(source) if r['window_id'] not in changed]
  run.lines(target/f'raw-{arm}.jsonl',reuse)
 datasets[name]={'documents':len(rows(target/'inputs.jsonl')),'windows':len(windows),'replayed_windows_per_arm':len(changed),'unchanged_fresh_windows_reused_per_arm':len(windows)-len(changed),'parent':str(parent.relative_to(BASE)),'input_sha256':run.sha(target/'inputs.jsonl'),'windows_sha256':run.sha(target/'windows.jsonl')}
run.save(OUT/'freeze.json',{'datasets':datasets,'policy':'Leading whitespace only removed before all three arms; source starts advanced by exact removed codepoint count. Predictions map back to original source. Only changed windows rerun; identical unchanged windows reuse raw responses from this same fresh experiment, never historical reports. Not an independent additional population; timing is not compared for this mixed replay.'})
run.OUT=OUT;run.rows=rows
for arm in ['ossomex012','wapiti','scibert']:
 print('CONTROL START',arm,flush=True);run.run(arm)
print('COMPLETE WHITESPACE CONTROL',flush=True)
