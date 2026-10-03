"""Sequential benchmark queue. Never overlaps timed inference arms."""
import json, os, random, subprocess, sys, time
from pathlib import Path
from run import ROOT, OUT, rows, save, lines
log=OUT/'progress-ossomex012.log'
while 'COMPLETE ossomex012' not in log.read_text():
 if 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Initial OSSoMeX process failed')
 time.sleep(5)
base=OUT/'timing';base.mkdir(exist_ok=True);datasets={}
for repeat in range(3):
 name=f'fresh30-repeat{repeat+1}';target=base/name;target.mkdir(exist_ok=True)
 ds=rows(OUT/'fresh30/inputs.jsonl');ws=rows(OUT/'fresh30/windows.jsonl');random.Random(1234+repeat).shuffle(ws)
 lines(target/'inputs.jsonl',ds);lines(target/'windows.jsonl',ws);datasets[name]={'documents':30,'windows':30,'order_seed':1234+repeat}
save(base/'freeze.json',{'datasets':datasets,'policy':'three warm repeats of the same fresh30 inputs; same randomized order in each arm; arms sequential; load and warmup excluded'})
for group in ['main','corrected','timing']:
 for arm in ['ossomex012','wapiti','scibert']:
  if group=='main' and arm=='ossomex012':continue
  target=OUT if group=='main' else OUT/group
  print('START',group,arm,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
  code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run;from jsonl_io import rows;run.rows=rows;run.OUT=Path(sys.argv[2]);run.run(sys.argv[3])"
  with (target/f'progress-{arm}.log').open('w') as f:
   result=subprocess.run([sys.executable,'-c',code,str(OUT),str(target),arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{group} {arm} failed: {result.returncode}')
  print('DONE',group,arm,flush=True)
print('COMPLETE ALL',flush=True)
