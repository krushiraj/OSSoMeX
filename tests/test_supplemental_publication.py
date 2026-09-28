import importlib
import os
from pathlib import Path
import tempfile
from urllib.parse import urlencode

import pytest

from research.data.manifest import digest
from test_detector_data import exposure_spec, preparation_sources
from test_supplemental_acquisition import FakeFetch, QUERY, setup
from test_supplemental_annotation import acquired


REQUEST = 'requests/' + digest(('https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({
    'query': QUERY, 'format': 'json', 'resultType': 'lite', 'pageSize': 200,
    'cursorMark': '*', 'synonym': 'false'})).encode())
TARGETS = [
    ('collection', name) for name in ('config.json', 'frames/metadata.json', REQUEST,
        REQUEST + '.json', 'regions.jsonl', 'bundle/documents.jsonl', 'bundle/manifest.json', 'manifest.json')
] + [
    ('tasks', name) for name in ('tasks.jsonl', 'policy.md', 'prompts/annotate-aliases.md', 'manifest.json')
] + [
    ('readiness', name) for name in ('report.json', 'manifest.json')
] + [
    ('training', name) for name in ('sources/exposure_bundles-0/identities.jsonl',
                                  'config.json', 'items.jsonl', 'manifest.json')
]


def producer(monkeypatch, tmp_path, kind):
    if kind == 'collection':
        module, config, output = setup(monkeypatch, tmp_path, FakeFetch([{'doi': '10.1000/p1'}]))
        return lambda: module.collect_supplemental(config, output), output
    if kind in ('tasks', 'readiness'):
        bundle = acquired(monkeypatch, tmp_path)
        output = tmp_path / kind
        if kind == 'tasks':
            module = importlib.import_module('research.annotations.supplemental')
            return lambda: module.prepare_supplemental_tasks(bundle, output), output
        module = importlib.import_module('research.data.supplemental_readiness')
        return lambda: module.assess_readiness(bundle, None, output), output
    module = importlib.import_module('research.training.data')
    config, documents = preparation_sources(tmp_path)
    config['exposure_bundles'] = [exposure_spec(tmp_path, documents, 'train_reserved')]
    output = tmp_path / 'training'
    return lambda: module.prepare_data(config, output), output


def fail_publication(monkeypatch, target, failure):
    original_open, original_temporary, original_fsync = Path.open, tempfile.NamedTemporaryFile, os.fsync
    selected_fds = set()

    class PartialStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, payload):
            self.stream.write(payload[:max(1, len(payload) // 2)])
            self.stream.flush()
            raise OSError('injected partial publication')

    def opening(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if Path(path) == target and mode == 'xb' and failure == 'partial':
            return PartialStream(stream)
        return stream

    def temporary(*args, **kwargs):
        stream = original_temporary(*args, **kwargs)
        if Path(kwargs.get('dir', '.')) == target.parent and kwargs.get('prefix') == f'.{target.name}-':
            if failure == 'partial':
                return PartialStream(stream)
            selected_fds.add(stream.fileno())
        return stream

    def fsync(fd):
        if fd in selected_fds:
            raise OSError('injected fsync failure')
        return original_fsync(fd)

    monkeypatch.setattr(Path, 'open', opening)
    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', temporary)
    monkeypatch.setattr(os, 'fsync', fsync)


@pytest.mark.parametrize(('kind', 'name'), TARGETS)
@pytest.mark.parametrize('failure', ['partial', 'fsync'])
def test_new_producers_never_publish_incomplete_companions_or_manifest(monkeypatch, tmp_path, kind, name, failure):
    run, output = producer(monkeypatch, tmp_path, kind)
    target = output / name
    fail_publication(monkeypatch, target, failure)
    with pytest.raises(OSError, match='injected'):
        run()
    assert not target.exists()
    assert not (output / 'manifest.json').exists()
    assert not list(output.rglob('.*.tmp'))


@pytest.mark.parametrize(('kind', 'name'), TARGETS)
def test_new_producers_preserve_concurrent_publication_collision(monkeypatch, tmp_path, kind, name):
    run, output = producer(monkeypatch, tmp_path, kind)
    target = output / name
    original_link = os.link

    def collision(source, destination):
        if Path(destination) == target:
            target.write_bytes(b'concurrent publisher')
        return original_link(source, destination)

    monkeypatch.setattr(os, 'link', collision)
    with pytest.raises(FileExistsError):
        run()
    assert target.read_bytes() == b'concurrent publisher'
    if name != 'manifest.json':
        assert not (output / 'manifest.json').exists()
    assert not list(output.rglob('.*.tmp'))


@pytest.mark.parametrize('entry', ['empty', 'partial', 'complete', 'file', 'symlink', 'dangling'])
@pytest.mark.parametrize('command', ['collect_supplemental', 'freeze_candidate_frames'])
def test_supplemental_rejects_every_existing_output_before_fetch(monkeypatch, tmp_path, entry, command):
    fake = FakeFetch()
    module, config, output = setup(monkeypatch, tmp_path, fake)
    if entry == 'complete':
        module.collect_supplemental(config, output)
    elif entry == 'file':
        output.write_bytes(b'existing file')
    elif entry in ('symlink', 'dangling'):
        destination = tmp_path / 'old-research'
        if entry == 'symlink':
            destination.mkdir()
            (destination / 'sentinel').write_bytes(b'existing research')
        output.symlink_to(destination, target_is_directory=True)
    else:
        output.mkdir()
        if entry == 'partial':
            (output / 'sentinel').write_bytes(b'existing research')
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    entries = set(tmp_path.rglob('*'))
    fake.calls.clear()
    with pytest.raises(FileExistsError):
        getattr(module, command)(config, output)
    assert fake.calls == []
    assert set(tmp_path.rglob('*')) == entries
    assert {path: path.read_bytes() for path in before} == before


def test_collection_claims_root_exclusively_before_fetch(monkeypatch, tmp_path):
    fake = FakeFetch()
    module, config, output = setup(monkeypatch, tmp_path, fake)
    original_mkdir = Path.mkdir

    def concurrent_root(path, *args, **kwargs):
        if path == output and not path.exists():
            original_mkdir(path)
            (path / 'sentinel').write_bytes(b'concurrent research')
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', concurrent_root)
    with pytest.raises(FileExistsError):
        module.collect_supplemental(config, output)
    assert fake.calls == []
    assert list(output.iterdir()) == [output / 'sentinel']
    assert (output / 'sentinel').read_bytes() == b'concurrent research'
