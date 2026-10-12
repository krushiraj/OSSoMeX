"""Resumable pulls of public registry metadata. Raw responses are cached with SHA-256 sidecars."""

import json
import time
from pathlib import Path

from ..data.acquire import fetch_public
from ..data.manifest import digest, json_bytes, write_once
from .sources import (LICENCES, biotools_page_url, ecosystems_page_url, qid, wikidata_detail_url,
                      wikidata_layer_list_url, wikidata_list_url)

MAX_PAGES = 2000


def _get(url, destination, policy, delay):
    cached = destination.exists()
    record = fetch_public(url, destination, policy)
    if not cached and delay:
        time.sleep(delay)
    return json.loads(destination.read_bytes()), {'path': None, 'url': url, 'sha256': record['sha256']}


def _entry(run, destination, row, role='page'):
    return {**row, 'path': str(destination.relative_to(run)), 'role': role}


def _pages(run, source, url_for, done, policy, delay, limit):
    files = []
    for page in range(1, min(limit or MAX_PAGES, MAX_PAGES) + 1):
        destination = run / 'raw' / source / f'page-{page:04d}.json'
        data, row = _get(url_for(page), destination, policy, delay)
        files.append(_entry(run, destination, row))
        if done(data):
            return files
    if limit:
        return files
    raise ValueError(f'PAGE_LIMIT_EXCEEDED: {source}')


def pull(config: dict, output: Path) -> dict:
    run = Path(output)
    policy = {'format': 'json', 'user_agent': config['user_agent']}
    delay, limit = config.get('delay_seconds', 0.2), config.get('max_pages')
    batch = config.get('wikidata_batch', 250)
    if type(batch) is not int or not 1 <= batch <= 400:
        raise ValueError('INVALID_WIKIDATA_BATCH')
    selected = config.get('sources', ['ecosystems', 'biotools', 'wikidata'])
    sources = {}
    if 'ecosystems' in selected:
        sources['ecosystems'] = _pages(run, 'ecosystems', ecosystems_page_url, lambda d: not d, policy, delay, limit)
    if 'biotools' in selected:
        sources['biotools'] = _pages(run, 'biotools', biotools_page_url, lambda d: not d.get('next'), policy, delay, limit)
    if 'wikidata' in selected:
        destination = run / 'raw/wikidata/list.json'
        data, row = _get(wikidata_list_url(), destination, policy, delay)
        files = [_entry(run, destination, row, 'list')]
        qids = sorted({qid(b['s']['value']) for b in data['results']['bindings']}, key=lambda q: int(q[1:]))
        batches = [qids[i:i + batch] for i in range(0, len(qids), batch)][:limit or None]
        for number, chunk in enumerate(batches, 1):
            destination = run / 'raw' / 'wikidata' / f'page-{number:04d}.json'
            _, row = _get(wikidata_detail_url(chunk), destination, policy, delay)
            files.append(_entry(run, destination, row))
        seen = set(qids)
        # Optional layers append batches for new items only, so earlier pages stay cached byte for byte.
        for layer in config.get('wikidata_layers', []):
            destination = run / 'raw' / 'wikidata' / f"layer-{layer['name']}-list.json"
            data, row = _get(wikidata_layer_list_url(layer), destination, policy, delay)
            files.append(_entry(run, destination, row, 'list'))
            extra = sorted({qid(b['s']['value']) for b in data['results']['bindings']} - seen, key=lambda q: int(q[1:]))
            seen |= set(extra)
            for number, start in enumerate(range(0, len(extra), batch), 1):
                destination = run / 'raw' / 'wikidata' / f"layer-{layer['name']}-{number:04d}.json"
                _, row = _get(wikidata_detail_url(extra[start:start + batch]), destination, policy, delay)
                files.append(_entry(run, destination, row))
        sources['wikidata'] = files
    manifest = {'status': 'success', 'config': config,
                'sources': {s: {'pages': len(f), 'licence': LICENCES[s], 'files': f} for s, f in sources.items()}}
    manifest['sha256'] = digest(json_bytes(manifest))
    write_once(run / 'pull-manifest.json', json_bytes(manifest))
    return manifest
