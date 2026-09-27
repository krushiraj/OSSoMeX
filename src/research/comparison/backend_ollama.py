"""Historical local prompts with strict JSON and exact unique span alignment."""

from copy import deepcopy
import json
import re

from ..adapters.base import PredictionInput
from ..adapters.ollama_llm import FIXED_EXAMPLES, SYSTEM_PROMPT, USER_TEMPLATE, OllamaLLMAdapter
from ..schema import PUBLIC_SCHEMA, validate_public_records
from .alignment import align_unique
from .backends import HTTPBackend, json_text, local_url, sha256, window_result


class OllamaBackend(HTTPBackend):
    def load(self, config):
        identity, raw = {}, {}
        self.loaded = None
        try:
            identity = self._prepare(config)
            native = self.arm['config']
            self.base_url = local_url(native.get('base_url', 'http://localhost:11434')).rstrip('/')
            self.model = native['model_id']
            if not isinstance(self.model, str) or not self.model:
                raise ValueError('model_id is required')
            additions = self.arm['request_options']
            if set(additions) - {'options', 'format'} or not isinstance(additions.get('options', {}), dict):
                raise ValueError('request_options accepts only named options and format additions')
            self.options = {key: native.get(key, default) for key, default in (
                ('temperature', 0), ('seed', 42), ('num_ctx', 8192), ('num_predict', 1024))}
            self.options.update(deepcopy(additions.get('options', {})))
            self.format = additions.get('format', 'json' if native.get('use_format_json', True) else None)
            self.few_shot = self.arm.get('variant') == 'L1'
            sources = {'system': SYSTEM_PROMPT, 'user_template': USER_TEMPLATE,
                       'fixed_examples': FIXED_EXAMPLES if self.few_shot else 'none',
                       'public_schema': json.dumps(PUBLIC_SCHEMA, separators=(',', ':'))}
            identity.update({'model_id': self.model, 'verified': False,
                             'prompt_sources': sources,
                             'prompt_sha256': sha256(json_text(sources).encode('utf-8')),
                             'prompt_source_sha256': {k: sha256(v.encode('utf-8')) for k, v in sources.items()},
                             'effective_request_options': {'options': deepcopy(self.options), 'format': self.format},
                             'policy_differences': ['Historical L1 labels the Python example [] despite current language scope.'] if self.few_shot else []})
            response = self._request('GET', self.base_url + '/api/tags', raw, timeout=native.get('health_timeout_seconds', 10))
            identity['health'] = raw
            if response.status_code != 200:
                raise ValueError(f'model inventory HTTP {response.status_code}')
            data = json.loads(response.text)
            if not isinstance(data, dict) or not isinstance(data.get('models'), list):
                raise ValueError('model inventory requires models array')
            matches = [entry for entry in data['models'] if isinstance(entry, dict) and entry.get('name') == self.model]
            if len(matches) != 1 or not isinstance(matches[0].get('digest'), str) or not matches[0]['digest']:
                raise ValueError('requested model not uniquely available with digest')
            identity.update({'model_digest': matches[0]['digest'], 'verified': True})
            return self._loaded('ready', None, identity)
        except Exception as exc:
            identity['health'] = raw
            return self._loaded('unavailable', str(exc), identity)

    def predict(self, window):
        raw, unresolved = {}, []
        try:
            self._require_ready(window)
            item = PredictionInput(document_id=window['document_id'], text_revision=window['window_text_revision'],
                                   chunk_id=window['window_id'], character_offsets=(0, len(window['text'])),
                                   section_type=window.get('section_type', 'unknown'), text=window['text'])
            prompt = OllamaLLMAdapter()._build_prompt(item, self.few_shot)
            payload = {'model': self.model, 'stream': False,
                       'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
                       'options': deepcopy(self.options)}
            if self.format is not None:
                payload['format'] = deepcopy(self.format)
            raw.update({'request': deepcopy(payload), 'request_sha256': sha256(json_text(payload).encode('utf-8'))})
            response = self._request('POST', self.base_url + '/api/chat', raw, json=payload,
                                     timeout=self.arm['config'].get('timeout_seconds', 240))
            if response.status_code != 200:
                raise ValueError(f'inference HTTP {response.status_code}')
            data = json.loads(response.text)
            raw['native'] = data
            if (not isinstance(data, dict) or not isinstance(data.get('message'), dict)
                    or not isinstance(data['message'].get('content'), str)):
                raise ValueError('chat response requires message.content string')
            content = data['message']['content']
            raw['content'] = content
            if data.get('done') is False or data.get('done_reason') == 'length':
                raise ValueError('incomplete or token-limited chat response')
            candidate = content.strip()
            if candidate.startswith('```'):
                match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', candidate, flags=re.DOTALL)
                if not match:
                    raise ValueError('response requires one complete JSON code fence')
                candidate = match.group(1)
            records = json.loads(candidate)
            if not isinstance(records, list):
                raise ValueError('response must be a complete JSON array')
            valid, invalid = validate_public_records(records)
            if invalid:
                raw['invalid_records'] = invalid
                raise ValueError('schema-invalid extraction records')
            spans = []
            for rec in valid:
                for label, value in [('SOFTWARE', rec['name']), ('VERSION', rec['version'])]:
                    if value is None:
                        continue
                    aligned = align_unique(window['text'], value, rec['context_sentence'])
                    if aligned['status'] == 'unresolved':
                        unresolved.append({**aligned, 'label': label})
                    else:
                        spans.append({key: value for key, value in {**aligned, 'label': label,
                                      'score': None, 'score_kind': None}.items() if key != 'status'})
            return window_result(window, spans=spans, unresolved=unresolved, raw=raw,
                                 error='unresolved_candidates' if unresolved else None)
        except Exception as exc:
            return window_result(window, unresolved=unresolved, raw=raw, error=str(exc))
