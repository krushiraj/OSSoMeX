from concurrent.futures import ThreadPoolExecutor
import errno
import os
from threading import Barrier

import pytest


def artifacts():
    from research.data import artifacts as module
    return module


def test_atomic_write_publishes_complete_bytes_and_creates_parents(tmp_path):
    path = tmp_path / "new" / "manifest.json"
    payload = b'{"status":"complete"}\n'
    artifacts().atomic_write_new(path, payload)
    assert path.read_bytes() == payload
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", [OSError(errno.ENOSPC, "disk full"), KeyboardInterrupt()],
                         ids=["disk_full", "interruption"])
def test_fsync_failure_leaves_no_final_file_or_owned_temp(tmp_path, monkeypatch, failure):
    module = artifacts()
    other_temporary = tmp_path / ".another-publisher.tmp"
    other_temporary.write_bytes(b"other publisher")

    def fail_fsync(_):
        raise failure

    monkeypatch.setattr(module.os, "fsync", fail_fsync)
    output = tmp_path / "manifest.json"
    with pytest.raises(type(failure)):
        module.atomic_write_new(output, b"complete only after fsync")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [other_temporary]
    assert other_temporary.read_bytes() == b"other publisher"


def test_existing_identical_content_still_raises(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        artifacts().atomic_write_new(output, b"immutable")
    assert output.read_bytes() == b"immutable"
    assert list(tmp_path.iterdir()) == [output]


def test_existing_different_content_is_not_replaced(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        artifacts().atomic_write_new(output, b"replacement")
    assert output.read_bytes() == b"original"


def test_dangling_destination_symlink_is_not_replaced(tmp_path):
    output = tmp_path / "manifest.json"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        artifacts().atomic_write_new(output, b"replacement")
    assert output.is_symlink()
    assert not (tmp_path / "missing").exists()


def test_two_publishers_cannot_replace_each_other(tmp_path, monkeypatch):
    module = artifacts()
    output = tmp_path / "manifest.json"
    barrier = Barrier(2, timeout=5)
    original_link = os.link

    def publish_together(source, destination):
        barrier.wait()
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", publish_together)

    def publish(payload):
        try:
            module.atomic_write_new(output, payload)
            return ("published", payload)
        except FileExistsError:
            return ("collision", payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, [b"first" * 10000, b"second" * 10000]))
    assert sorted(status for status, _ in results) == ["collision", "published"]
    assert output.read_bytes() == next(payload for status, payload in results if status == "published")
    assert list(tmp_path.iterdir()) == [output]


def test_cleanup_does_not_touch_another_publishers_file(tmp_path, monkeypatch):
    module = artifacts()
    output = tmp_path / "manifest.json"
    other_temporary = tmp_path / ".manifest.json-other.tmp"
    other_temporary.write_bytes(b"in progress")

    def collision(source, destination):
        destination.write_bytes(b"winner")
        raise FileExistsError(destination)

    monkeypatch.setattr(module.os, "link", collision)
    with pytest.raises(FileExistsError):
        module.atomic_write_new(output, b"loser")
    assert output.read_bytes() == b"winner"
    assert other_temporary.read_bytes() == b"in progress"
    assert set(tmp_path.iterdir()) == {output, other_temporary}
