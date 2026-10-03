import json,requests
from run import OUT,save
s=requests.Session();s.trust_env=False
records=[]
for arm,port in [('wapiti',8060),('scibert',8062)]:
 for prefix in ['', ' ', '  ', '         ', '\n        ']:
  text=prefix+'We used SPSS version 13 for analysis.'
  response=s.post(f'http://127.0.0.1:{port}/service/processSoftwareText',data={'text':text},timeout=180)
  body=json.loads(response.content.decode('utf8'))
  names=[m['software-name'] for m in body.get('mentions',[]) if m.get('software-name')]
  records.append({'arm':arm,'text':text,'expected_span':[text.index('SPSS'),text.index('SPSS')+4],'headers':dict(response.headers),'native_names':names,'native_exact':all(text[n['offsetStart']:n['offsetEnd']]==n['rawForm'] for n in names),'response':body})
  print(arm,repr(prefix),[(n['rawForm'],n['offsetStart'],n['offsetEnd']) for n in names],flush=True)
save(OUT/'whitespace-offset-probe.json',records)
