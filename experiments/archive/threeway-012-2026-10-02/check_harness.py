"""Focused assertions for the new benchmark harness, independent of predictions."""
import json,tempfile
from pathlib import Path
from score import project
from jsonl_io import rows
from run import lines
from research.contracts import text_revision
from research.evaluation.metrics import _covered
from score_whitespace_canonical import canonical_span

assert _covered([{'start':0,'end':5,'fields':{'software':True}},{'start':5,'end':10,'fields':{'software':True}}],(3,8),'software',10)
assert not _covered([{'start':0,'end':5,'fields':{'software':True}},{'start':6,'end':10,'fields':{'software':True}}],(3,8),'software',10)
normalized=canonical_span('😀 Foo\n  Bar  1',{'text':'Foo   Bar','start':3,'end':14},unit='utf16')
assert normalized['text']=='Foo\n  Bar' and (normalized['start'],normalized['end'])==(2,11)
try:canonical_span('Foo Bar Foo',{'text':'Foo','start':4,'end':7},unit='utf16')
except ValueError:pass
else:raise AssertionError('Whitespace sensitivity must not search for another occurrence')

with tempfile.TemporaryDirectory() as tmp:
 base=Path(tmp);target=base/'synthetic';target.mkdir()
 text='😀 Python 3.12 café\u0085 test';d={'document_id':'fixture','text':text,'text_revision':text_revision(text)}
 w={'window_id':'w1','document_id':'fixture','text':text,'text_revision':d['text_revision'],'start':0,'end':len(text)}
 lines(target/'inputs.jsonl',[d]);lines(target/'windows.jsonl',[w])
 mention={'software-name':{'rawForm':'Python','offsetStart':3,'offsetEnd':9},'version':{'rawForm':'3.12','offsetStart':10,'offsetEnd':14},'language':{'rawForm':'Python','offsetStart':3,'offsetEnd':9},'software-type':'implicit','mentionContextAttributes':{k:{'value':k=='used'} for k in ['used','created','shared']}}
 body=json.dumps({'mentions':[mention]},ensure_ascii=False).encode('utf-8').decode('latin1')
 lines(target/'raw-wapiti.jsonl',[{'window_id':'w1','document_id':'fixture','seconds':1,'result':{'status_code':200,'body':body}}])
 native=project(base,'synthetic','wapiti','native');explicit=project(base,'synthetic','wapiti','explicit')
 assert native[1][0]['name_span']=={'start':2,'end':8}
 assert native[1][0]['version_links'][0]['span']=={'start':9,'end':13}
 assert explicit[1][0]['name']=='Python' and explicit[1][0]['version_links']==[]
 assert not native[3] and not explicit[3]
 assert rows(target/'inputs.jsonl')[0]['text']==text
 # Equal spans from overlapping windows are merged exactly once.
 w2={**w,'window_id':'w2'};lines(target/'windows.jsonl',[w,w2]);r=rows(target/'raw-wapiti.jsonl')[0];lines(target/'raw-wapiti.jsonl',[r,{**r,'window_id':'w2'}])
 assert len(project(base,'synthetic','wapiti','native')[1])==1
 # A failed response remains operational failure.
 lines(target/'raw-wapiti.jsonl',[{**r,'result':{'status_code':500,'body':'error'}},{**r,'window_id':'w2','result':{'status_code':500,'body':'error'}}])
 assert project(base,'synthetic','wapiti','native')[2][0]['status']=='failure'
print('PASS: strict UTF16, UTF8 byte recovery, LF-only JSONL, field projection, deduplication, contiguous coverage, HTTP failure accounting')
