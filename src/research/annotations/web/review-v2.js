import {mountReviewView} from './review-view.js';
import {createReviewSession} from './review-session.js';

let csrf;
const request = async (path, body) => {
  const response = await fetch(path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json','X-CSRF-Token':csrf}, body:JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw Object.assign(new Error(result.error || 'Request failed'), {status:response.status,code:result.error});
  if (path === '/api/session') csrf = result.csrf_token;
  return result;
};
const view = mountReviewView(document.getElementById('review-root'), {
  onAction:action => session.dispatch(action), onSave:(...args) => session.save(...args),
  onNavigate:id => session.navigate(id), onFilter:value => session.setFilters(value),
  onExport:() => session.exportReference(),
});
const session = createReviewSession({request,view,reviewer:'Krushi'});
session.openTask(null);
