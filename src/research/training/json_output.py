"""JSON rendering for human and machine prediction output."""

import json


def render_prediction(result: dict, output_format: str = 'pretty') -> str:
    if output_format not in ('pretty', 'jsonl'):
        raise ValueError('unknown output format')
    return json.dumps(result, ensure_ascii=False, indent=2 if output_format == 'pretty' else None,
                      allow_nan=False)
