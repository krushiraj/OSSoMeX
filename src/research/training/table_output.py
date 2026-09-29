"""Compact occurrence-level output without changing model predictions."""

import unicodedata


HEADERS = ('Software', 'Intent', 'Version', 'Aliases', 'Sentiment')
COLORS = {'used': '30;42', 'created': '30;46', 'shared': '97;44', 'mentioned': '97;100',
          'positive': '30;42', 'negative': '97;41', 'mixed': '30;43', 'not expressed': '97;100',
          'unknown': '30;43'}


def _safe(value):
    parts = []
    for char in str(value):
        if unicodedata.category(char) in ('Cc', 'Cf', 'Cs', 'Zl', 'Zp'):
            parts.append(('\\x%02x' if ord(char) <= 255 else '\\u%04x') % ord(char))
        else:
            parts.append(char)
    return ''.join(parts)


def _width(value):
    return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in 'WF' else 1
               for char in value)


def _wrap(value, width):
    lines, line, size = [], '', 0
    for char in value:
        step = _width(char)
        if size + step > width:
            lines.append(line)
            line, size = '', 0
        line += char
        size += step
    return [*lines, line]


def _paint(value, style, color):
    return f'\x1b[{style}m{value}\x1b[0m' if color and style else value


def _field(row, key):
    field = row.get(key, {})
    if field.get('status') != 'success':
        return 'unknown'
    value = field['value']
    if key == 'versions':
        return ', '.join(edge['text'] for edge in value) or '—'
    if key == 'intents':
        return ', '.join(value)
    return value.replace('_', ' ')


def render_table(result, *, color=False, width=120):
    if result.get('schema_version') != 'full-label-prediction-1':
        raise ValueError('table format requires a full-label prediction')
    fields = result['field_predictions']
    output = [f"{_safe(result['document_id'])}: {_safe(result['status'])} · {len(fields)} software mentions"]
    groups = result['alias_predictions']['groups']
    names = {row['mention_id']: row['name'] for row in fields}
    alias_unknown = result['stage_status']['alias']['status'] not in ('success', 'not_applicable')
    uncertain_versions = {(span['start'], span['end'])
                          for span in (result.get('detector_diagnostics') or {}).get('spans', [])
                          if span['label'] == 'VERSION' and span['score'] < .8}
    rows, styles, review = [], [], False
    for row in fields:
        memberships = [group for group in groups if row['mention_id'] in group['member_mention_ids']]
        aliases = list(dict.fromkeys(names[identity] for group in memberships
            for identity in group['member_mention_ids'] if identity != row['mention_id'] and identity in names))
        if alias_unknown:
            aliases.append('unknown')
        alias = ', '.join(aliases) or '—'
        values = [row['name'], _field(row, 'intents'), _field(row, 'versions'), alias, _field(row, 'sentiment')]
        uncertain = (alias_unknown or 'unknown' in values or
            row.get('software', {}).get('scores', {}).get('software', 1) < .8 or
            any((edge['span']['start'], edge['span']['end']) in uncertain_versions
                for edge in (row.get('versions', {}).get('value') or [])) or
            any(reason.startswith('low_confidence:') for key in ('intents', 'sentiment')
                for reason in row.get(key, {}).get('reasons', [])) or
            any(group.get('negative_chord_pair_ids') for group in memberships) or
            any(reason.startswith('low_confidence:') for pair in [*result['pair_predictions']['linker'],
                    *result['alias_predictions']['pairs']]
                if row['mention_id'] in (pair['first_id'], pair['second_id'])
                for reason in pair.get('reasons', [])))
        review |= uncertain
        if uncertain:
            values[0] += ' !'
        rows.append([_safe(value) for value in values])
        styles.append(['30;43' if uncertain else '97;44', COLORS.get(values[1].split(', ')[0]),
                       COLORS.get(values[2]), '30;43' if alias_unknown else '97;45' if aliases else None,
                       COLORS.get(values[4])])
    if rows:
        widths = [_width(header) for header in HEADERS]
        desired = [max(_width(row[i]) for row in [HEADERS, *rows]) for i in range(len(HEADERS))]
        budget = max(sum(widths), width - 3 * (len(HEADERS) - 1))
        while sum(widths) < budget and widths != desired:
            index = max(range(len(widths)), key=lambda i: desired[i] - widths[i])
            widths[index] += 1
        output.append(' | '.join(header + ' ' * (size - _width(header)) for header, size in zip(HEADERS, widths)))
        output.append('-+-'.join('-' * size for size in widths))
        for row, row_styles in zip(rows, styles):
            wrapped = [_wrap(cell, size) for cell, size in zip(row, widths)]
            for line in range(max(map(len, wrapped))):
                cells = [cell[line] if line < len(cell) else '' for cell in wrapped]
                output.append(' | '.join(_paint(cell + ' ' * (size - _width(cell)), style, color)
                    for cell, size, style in zip(cells, widths, row_styles)))
    elif result['status'] == 'no_mentions':
        output.append('No software mentions detected.')
    else:
        output.append('Prediction failed or incomplete; no resolved software rows. Use --format pretty for diagnostics.')
    incomplete = [f'{stage}: {entry["status"]}' for stage, entry in result['stage_status'].items()
                  if entry['status'] not in ('success', 'not_applicable')]
    if incomplete:
        output.append(_paint('Incomplete: ' + '; '.join(incomplete), '33', color))
    if review:
        output.append(_paint('! Review: low confidence, incomplete fields, or conflicting alias links.', '33', color))
    output.append('Experimental predictions · uncalibrated scores · — none linked; unknown = not resolved')
    return '\n'.join(output)
