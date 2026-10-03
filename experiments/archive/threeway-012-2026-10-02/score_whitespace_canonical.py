"""Posthoc whitespace-only sensitivity. No inference or string searching."""
import hashlib,json
from pathlib import Path
import score
from run import OUT,save,sha

SOURCE=OUT/'whitespace-control'
TARGET=OUT/'whitespace-canonical'
STRICT=score.native_span
AUDIT=[]

def canonical_span(text,span,*,unit,offset_base=0):
 try:return STRICT(text,span,unit=unit,offset_base=offset_base)
 except ValueError as exc:
  if str(exc)!='native span does not match exact source slice':raise
  boundaries={0:0};units=0
  for i,c in enumerate(text):
   units+=2 if ord(c)>0xFFFF else 1;boundaries[units]=i+1
  a,b=span['start'],span['end']
  if unit=='utf16':
   if a not in boundaries or b not in boundaries:raise exc
   a,b=boundaries[a],boundaries[b]
  elif unit!='codepoint':raise exc
  if not 0<=a<b<=len(text):raise exc
  raw=text[a:b]
  if not raw.strip() or ' '.join(raw.split())!=' '.join(span['text'].split()):raise exc
  a+=len(raw)-len(raw.lstrip());b-=len(raw)-len(raw.rstrip())
  result={'text':text[a:b],'start':a+offset_base,'end':b+offset_base,'alignment_method':'native_range_whitespace_canonicalization'}
  AUDIT.append({'source_window_sha256':hashlib.sha256(text.encode()).hexdigest(),'native':span,'canonical':result})
  return result

def main():
 assert 'COMPLETE WHITESPACE CONTROL' in (OUT/'whitespace-control.log').read_text()
 frozen=json.loads((SOURCE/'freeze.json').read_text());TARGET.mkdir(exist_ok=True)
 for name in frozen['datasets']:
  target=TARGET/name;target.mkdir(exist_ok=True)
  for filename in ['inputs.jsonl','windows.jsonl','references-source.jsonl','coverage-source.jsonl',*[f'raw-{a}.jsonl' for a in score.ARMS]]:
   source=SOURCE/name/filename
   if source.exists() and not (target/filename).exists():(target/filename).symlink_to(Path('../../whitespace-control')/name/filename)
  for r in score.rows(SOURCE/name/'predictions-ossomex012-native.jsonl'):
   assert r['name']==r['name'].strip()
   assert all(v['text']==v['text'].strip() for v in r['version_links'])
 save(TARGET/'freeze.json',{'datasets':frozen['datasets'],'policy':'Posthoc analysis-only sensitivity on the whitespace replay. Keep native numeric ranges, strip only boundary whitespace, and accept internal whitespace differences only when the entire supplied source range and rawForm have identical non-whitespace content and word separation. Retain original source characters inside accepted spans. No searching, snapping, non-whitespace boundary edits, reference-label access, or new inference. OSSoMeX outputs verified to have no boundary whitespace. Strict primary results remain separate.','source_raw_hashes':{f'{name}/{arm}':sha(SOURCE/name/f'raw-{arm}.jsonl') for name in frozen['datasets'] for arm in score.ARMS}})
 score.native_span=canonical_span
 try:score.main(TARGET)
 finally:score.native_span=STRICT
 save(TARGET/'normalization-audit.json',AUDIT)
 print('Whitespace canonical sensitivity complete; accepted adjustments',len(AUDIT))

if __name__=='__main__':main()
