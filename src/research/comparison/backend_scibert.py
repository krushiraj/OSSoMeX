"""One resident local Detector with strict window-local result validation."""

from copy import deepcopy

from .alignment import native_span
from .backends import ComparisonBackend, REPO_ROOT, capabilities, score_fields, window_result
from .contracts import OFFSET_UNIT


def create_detector(checkpoint, device):
    from ..training.predict import Detector
    return Detector(checkpoint, device=device)


class SciBertBackend(ComparisonBackend):
    def __init__(self, detector_factory=None):
        super().__init__()
        self.detector_factory = detector_factory or create_detector
        self.detector = None

    def load(self, config):
        self.close()
        identity = {}
        try:
            identity = self._prepare(config)
            native = self.arm['config']
            checkpoint = str((REPO_ROOT / native['checkpoint']).resolve())
            self.detector = self.detector_factory(checkpoint, device=native.get('device', 'auto'))
            if self.detector.manifest['capabilities'] != capabilities():
                raise ValueError('checkpoint capabilities do not match span-only comparison')
            identity.update({'checkpoint': checkpoint, 'checkpoint_sha256': self.detector.identity,
                             'checkpoint_manifest': deepcopy(self.detector.manifest), 'verified': True,
                             'scores_calibrated': False})
            return self._loaded('ready', None, identity)
        except Exception as exc:
            self.detector = None
            return self._loaded('unavailable', str(exc), identity)

    def predict(self, window):
        raw = {}
        try:
            self._require_ready(window)
            document = {'document_id': window['window_id'], 'text': window['text'],
                        'text_revision': window['window_text_revision']}
            native = self.detector.predict(document)
            raw['native'] = deepcopy(native)
            if (not isinstance(native, dict) or not isinstance(native.get('chunks'), list)
                    or len(native['chunks']) != 1 or native['chunks'][0].get('status') != 'success'):
                raise ValueError('expected exactly one successful internal detector chunk')
            if (native.get('document_id') != document['document_id']
                    or native.get('text_revision') != document['text_revision']
                    or native.get('checkpoint_sha256') != self.detector.identity
                    or native.get('capabilities') != self.loaded['capabilities']
                    or native.get('offset_unit') != OFFSET_UNIT):
                raise ValueError('native detector identity/revision/contract mismatch')
            if native.get('status') not in ('success', 'no_mentions') or not isinstance(native.get('spans'), list):
                raise ValueError('native detector prediction failed or malformed')
            spans = []
            for item in native['spans']:
                if not isinstance(item, dict) or item.get('label') not in ('SOFTWARE', 'VERSION') or 'score' not in item:
                    raise ValueError('invalid native detector span')
                spans.append(native_span(window['text'], {
                    **item, **score_fields(item['score'], 'token_probability_geometric_mean_uncalibrated')}, unit=OFFSET_UNIT))
            if (native['status'] == 'success') != bool(spans):
                raise ValueError('native detector status/span mismatch')
            return window_result(window, spans=spans, raw=raw)
        except Exception as exc:
            return window_result(window, raw=raw, error=str(exc))

    def close(self):
        self.detector = None
        super().close()
