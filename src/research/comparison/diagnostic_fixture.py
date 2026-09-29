"""Freeze explicitly annotated, exposed examples without promoting them to gold."""

import argparse
import json
from pathlib import Path
import re

from ..contracts import text_revision
from ..data.artifacts import atomic_write_new
from .backends import sha256
from .links import score_links
from .runner import json_bytes, jsonl_bytes, publish


def parse_annotated(annotated):
    parts, spans, cursor, length = [], {}, 0, 0
    for match in re.finditer(r'\[\[([^|\]]+)\|([^\]]+)\]\]', annotated):
        prefix = annotated[cursor:match.start()]
        parts.append(prefix)
        length += len(prefix)
        key, value = match.groups()
        if not re.fullmatch(r'[sv][1-9][0-9]*', key) or key in spans:
            raise ValueError('invalid or duplicate annotation endpoint ID')
        spans[key] = {'start': length, 'end': length + len(value), 'text': value,
                      'label': 'SOFTWARE' if key.startswith('s') else 'VERSION'}
        parts.append(value)
        length += len(value)
        cursor = match.end()
    parts.append(annotated[cursor:])
    text = ''.join(parts)
    if '[[' in text or ']]' in text:
        raise ValueError('malformed annotation marker')
    return text, spans


def freeze_fixture(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_bytes = source.read_bytes()
    spec = json.loads(source_bytes)
    if spec.get('schema_version') != 'annotated-diagnostic-1' or spec.get('review_kind') != 'agent_provisional':
        raise ValueError('only explicit agent-provisional diagnostics are supported')
    docs, refs, review = [], [], ['# CLI diagnostic reference review', '', spec['purpose'], '', spec['normalization'], '']
    for item in spec['documents']:
        text, spans = parse_annotated(item['annotated'])
        doc = {'document_id': item['document_id'], 'text': text, 'text_revision': text_revision(text),
               'work_group_id': item['work_group_id'], 'role': 'diagnostic', 'development_exposed': True,
               'metadata': {'source_kind': item['source_kind'], 'source_ids_unresolved': True}}
        ref = {key: doc[key] for key in ('document_id', 'text_revision')}
        ref.update(spans=list(spans.values()), coverage=[{'start': 0, 'end': len(text),
            'label': label, 'complete': True, 'review_kind': 'agent_provisional'} for label in ('SOFTWARE', 'VERSION')],
            link_coverage=[{'start': 0, 'end': len(text), 'complete': True, 'review_kind': 'agent_provisional'}],
            version_links=[{'software': spans[a], 'version': spans[b]} for a, b in item['links']],
            ignored_versions=[spans[key] for key in item.get('ignored_versions', [])],
            ignored_software=[spans[key] for key in item.get('ignored_software', [])],
            provenance={'review_kind': 'agent_provisional', 'human_reviewed': False, 'role': 'diagnostic',
                        'prediction_exposed': True, 'source_kind': item['source_kind'],
                        'annotation_spec_sha256': sha256(source_bytes), 'review_notes': item['review_notes']},
            attribute_proposals=item['attribute_proposals'])
        docs.append(doc)
        refs.append(ref)
        review += [f"## {doc['document_id']}", '', text, '',
                   '| ID | Label | Text | Start:end |', '| --- | --- | --- | --- |']
        review += [f"| {key} | {span['label']} | {span['text']} | {span['start']}:{span['end']} |"
                   for key, span in spans.items()]
        review += ['', 'Version links: ' + (', '.join(f'{a} → {b}' for a, b in item['links']) or 'none'),
                   'Masked versions: ' + ', '.join(item.get('ignored_versions', [])),
                   'Masked names for ownership: ' + ', '.join(item.get('ignored_software', [])),
                   'Attribute proposals (not scored): ' + json.dumps(item['attribute_proposals']),
                   '', item['review_notes'], '']
    score_links(docs, refs, [], review_kind='agent_provisional')
    sources = []
    for key in ('source_transcript', 'source_pdf'):
        path = Path(spec[key])
        sources.append({'kind': key, 'path': str(path),
                        'sha256': sha256(path.read_bytes()) if path.is_file() else None})
    output.mkdir(parents=True, exist_ok=False)
    files = []
    publish(output, files, 'annotation-spec.json', source_bytes)
    publish(output, files, 'inputs.jsonl', jsonl_bytes(docs))
    publish(output, files, 'references.jsonl', jsonl_bytes(refs))
    publish(output, files, 'review.md', ('\n'.join(review) + '\n').encode())
    manifest = {'schema_version': 'cli-diagnostic-fixture-1', 'documents': len(docs),
                'heldout_quality_evaluated': False, 'training_allowed': False,
                'review_kind': 'agent_provisional', 'sources': sources, 'files': files}
    atomic_write_new(output / 'manifest.json', json_bytes(manifest))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_fixture(args.source, args.output), indent=2))
