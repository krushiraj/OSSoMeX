"""Human display formats and unchanged machine-readable JSON output."""

import json
import os
import shutil


def terminal_colors(stream):
    return bool(stream.isatty() and not os.environ.get('NO_COLOR') and os.environ.get('TERM') != 'dumb')


def render_prediction(result: dict, output_format: str = 'pretty', *, color=False, width=None) -> str:
    if output_format == 'table':
        from .table_output import render_table
        return render_table(result, color=color, width=width or shutil.get_terminal_size((120, 24)).columns)
    if output_format not in ('pretty', 'jsonl'):
        raise ValueError('unknown output format')
    return json.dumps(result, ensure_ascii=False, indent=2 if output_format == 'pretty' else None,
                      allow_nan=False)
