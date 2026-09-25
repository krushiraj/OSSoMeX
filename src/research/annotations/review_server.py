"""Loopback-only annotation review; source text never leaves this server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from ..data.manifest import read_jsonl, verified_path
from .review_store import ReviewError, apply_decision, decision_history, export_reference, get_item, import_items, open_store, queue
from .review_workflow import project_review
from .aliases import build_alias_groups
from .snapshots import read_policy_provenance, validate_item_policy, validate_snapshot_store


def check_bind(host):
    if host!='127.0.0.1': raise ValueError('LOOPBACK_127_0_0_1_REQUIRED')


def authorized_request(headers, port, token, write=False):
    host=f'127.0.0.1:{port}'; origin=f'http://{host}'
    if headers.get('Host')!=host: return False
    supplied=headers.get('Origin')
    if supplied and supplied!=origin: return False
    if write and (supplied!=origin or not secrets.compare_digest(headers.get('X-CSRF-Token',''),token)):
        return False
    return True


def create_server(bundle: Path, store: Path, host='127.0.0.1', port=8765, ui='selection'):
    check_bind(host)
    if ui not in ('selection','legacy'): raise ValueError('UI_UNSUPPORTED')
    manifest=json.loads((bundle/'manifest.json').read_bytes())
    if 'snapshot_schema_version' in manifest or any(row['path']=='store-items.jsonl' for row in manifest['files']):
        validate_snapshot_store(bundle,store)
    else:
        for row in manifest['files']: verified_path(bundle,row)
        provenance=read_policy_provenance(bundle,manifest)
        items=read_jsonl(bundle/'items.jsonl')
        validate_item_policy(items,provenance,manifest['role'])
        c=open_store(store)
        try: import_items(c,items,manifest['role'],provenance)
        finally:c.close()
    token=secrets.token_urlsafe(32)
    assets=Path(__file__).parent/'web'
    entry='review-v2.html' if ui=='selection' else 'index.html'
    static={'/':(entry,'text/html; charset=utf-8'),'/review.js':('review.js','text/javascript; charset=utf-8'),
            '/review.css':('review.css','text/css; charset=utf-8')}
    for name in ('review-selection.js','review-draft.js','review-view.js','review-session.js','review-v2.js'):
        static['/'+name]=(name,'text/javascript; charset=utf-8')
    static['/review-v2.css']=('review-v2.css','text/css; charset=utf-8')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args): pass

        def respond(self,status,body,content_type='application/json; charset=utf-8'):
            data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status)
            for key,value in {'Content-Type':content_type,'Content-Length':str(len(data)), 'Cache-Control':'no-store',
                              'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer',
                              'Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"}.items():
                self.send_header(key,value)
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            if not authorized_request(self.headers,self.server.server_port,token):
                return self.respond(403,{'error':'HOST_OR_ORIGIN_REJECTED'})
            parsed=urlsplit(self.path); path=parsed.path
            if path in static:
                name,mime=static[path]
                try: data=(assets/name).read_bytes()
                except FileNotFoundError: return self.respond(503,{'error':'UI_NOT_READY'})
                return self.respond(200,data,mime)
            if path=='/api/session': return self.respond(200,{'csrf_token':token,'role':manifest['role'],
                                                            'capabilities':{'aliases':True,'review_workflow':True},
                                                            'alias_schema_version':'1.0',
                                                            'review_workflow_schema_version':'1.0'})
            c=open_store(store)
            try:
                if path=='/api/queue':
                    filters={k:v[0] for k,v in parse_qs(parsed.query).items() if k in ('status','source','reason','field','workflow_status')}
                    return self.respond(200,queue(c,filters))
                if path.startswith('/api/tasks/'):
                    item=get_item(c,unquote(path.removeprefix('/api/tasks/')))
                    item['review_summary']=project_review(item,decision_history(c,item['task']['task_id']))
                    item['alias_groups']=build_alias_groups(item['annotation']['occurrences'],
                                                          item['annotation'].get('alias_annotations'))
                    # Do not expose scores in the reviewer response.
                    for occurrence in item['annotation']['occurrences']: occurrence.pop('scores',None)
                    return self.respond(200,item)
                self.respond(404,{'error':'NOT_FOUND'})
            except ReviewError as exc:self.respond(404,{'error':str(exc)})
            except ValueError as exc:self.respond(400,{'error':str(exc)})
            finally:c.close()

        def do_POST(self):
            if not authorized_request(self.headers,self.server.server_port,token,write=True):
                return self.respond(403,{'error':'ORIGIN_OR_CSRF_REJECTED'})
            if self.headers.get('Content-Type')!='application/json':return self.respond(415,{'error':'JSON_REQUIRED'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1_000_000: return self.respond(413,{'error':'INVALID_BODY_SIZE'})
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('JSON_OBJECT_REQUIRED')
            except (ValueError,UnicodeError):return self.respond(400,{'error':'INVALID_JSON'})
            c=open_store(store)
            try:
                if self.path=='/api/decisions':return self.respond(200,apply_decision(c,body))
                if self.path=='/api/export':
                    destination=store.parent/'exports'/secrets.token_hex(12)
                    result=export_reference(c,destination)
                    return self.respond(200,{**result,'path':str(destination.resolve())})
                self.respond(404,{'error':'NOT_FOUND'})
            except (ValueError,KeyError,TypeError) as exc:
                self.respond(409 if str(exc)=='STALE_REVISION' else 400,{'error':str(exc)})
            finally:c.close()

    return ThreadingHTTPServer((host,port),Handler)


def serve_review(bundle: Path, store: Path, host='127.0.0.1', port=8765):
    server=create_server(bundle,store,host,port)
    print(f'Local review: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
