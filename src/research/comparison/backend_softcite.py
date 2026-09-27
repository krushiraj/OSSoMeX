"""Strict Softcite text endpoint without guessed spans or offset units."""

import json

from .alignment import native_span
from .backends import HTTPBackend, local_url, score_fields, verify_evidence, window_result


def _verify_offset_evidence(evidence):
    if evidence.get('method') == 'pinned_implementation':
        if not isinstance(evidence.get('implementation_revision'), str) or not evidence['implementation_revision'].strip():
            raise ValueError('offset implementation evidence requires a pinned revision')
        return
    if evidence.get('method') != 'synthetic_probe':
        raise ValueError('offset evidence requires pinned implementation or positive synthetic probe')
    matched = []
    for unit in ('codepoint', 'utf16'):
        try:
            native_span(evidence.get('text'), {'text': evidence.get('span_text'),
                        'start': evidence.get('native_start'), 'end': evidence.get('native_end')}, unit=unit)
            matched.append(unit)
        except (ValueError, TypeError):
            pass
    expected = 'utf16' if evidence['offset_unit'] == 'utf16' else 'codepoint'
    if matched != [expected]:
        raise ValueError('synthetic offset probe must positively distinguish Unicode units')


class SoftciteBackend(HTTPBackend):
    def load(self, config):
        identity, raw = {}, {}
        self.loaded = None
        try:
            identity = self._prepare(config)
            native = self.arm['config']
            self.endpoint = local_url(native['endpoint'])
            health = local_url(native.get('health_endpoint', self.endpoint.rsplit('/', 1)[0] + '/isalive'))
            if 'text' in self.arm['request_options']:
                raise ValueError('request_options cannot override frozen text')
            self.offset_unit = None
            identity.update({'model_id': native.get('model_id'), 'verified': False, 'offset_unit': 'unverified'})
            if 'local_metadata' in native:
                metadata = verify_evidence(native['local_metadata'])
                if metadata.get('backend_id') != native.get('model_id') or not metadata.get('image_digest') or not metadata.get('weight_sha256'):
                    raise ValueError('local metadata does not identify pinned image and weights')
                identity.update({'verified': True, 'local_metadata': metadata, 'local_metadata_source': native['local_metadata']})
            if 'offset_contract' in native:
                evidence = verify_evidence(native['offset_contract'])
                if (evidence.get('backend_id') != native.get('model_id') or not evidence.get('evidence')
                        or evidence.get('offset_unit') not in ('codepoint', 'unicode_codepoint_half_open', 'utf16')):
                    raise ValueError('invalid pinned offset contract evidence')
                _verify_offset_evidence(evidence)
                self.offset_unit = evidence['offset_unit']
                identity.update({'offset_unit': self.offset_unit, 'offset_contract': evidence,
                                 'offset_contract_source': native['offset_contract']})
            self.empty_204 = False
            if 'empty_204_contract' in native:
                evidence = verify_evidence(native['empty_204_contract'])
                if (evidence.get('backend_id') != native.get('model_id') or not evidence.get('evidence')
                        or evidence.get('http_204_means_no_mentions') is not True):
                    raise ValueError('invalid pinned HTTP 204 contract evidence')
                self.empty_204 = True
                identity['empty_204_contract'] = evidence
            response = self._request('GET', health, raw, timeout=native.get('health_timeout_seconds', 10))
            identity['health'] = raw
            if response.status_code != 200:
                raise ValueError(f'health HTTP {response.status_code}')
            return self._loaded('ready', None, identity)
        except Exception as exc:
            identity['health'] = raw
            return self._loaded('unavailable', str(exc), identity)

    def predict(self, window):
        raw = {}
        try:
            self._require_ready(window)
            response = self._request('POST', self.endpoint, raw,
                                     data={'text': window['text'], **self.arm['request_options']},
                                     timeout=self.arm['config'].get('timeout_seconds', 180))
            if response.status_code == 204 and self.empty_204:
                return window_result(window, raw=raw)
            if response.status_code != 200:
                raise ValueError(f'inference HTTP {response.status_code}; unverified empty response is failure')
            payload = json.loads(response.text)
            raw['native'] = payload
            if not isinstance(payload, dict) or not isinstance(payload.get('software'), list):
                raise ValueError('Softcite response requires software array')
            spans = []
            for mention in payload['software']:
                if not isinstance(mention, dict) or not isinstance(mention.get('software-name'), dict):
                    raise ValueError('software entry requires software-name object')
                fields = [('SOFTWARE', mention['software-name'])]
                if 'version' in mention and mention['version'] is not None:
                    fields.append(('VERSION', mention['version']))
                for label, item in fields:
                    if not isinstance(item, dict) or not isinstance(item.get('rawForm'), str) or not item['rawForm']:
                        raise ValueError('native span requires nonempty rawForm and offsets')
                    if self.offset_unit is None and not window['text'].isascii():
                        raise ValueError('unverified Unicode offset contract')
                    spans.append(native_span(window['text'], {
                        'label': label, 'text': item['rawForm'], 'start': item.get('offsetStart'),
                        'end': item.get('offsetEnd'),
                        **score_fields(item.get('confidence'), 'softcite_extraction_confidence')},
                        unit=self.offset_unit or 'codepoint'))
            return window_result(window, spans=spans, raw=raw)
        except Exception as exc:
            return window_result(window, raw=raw, error=str(exc))
