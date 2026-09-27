"""Publish complete immutable artifacts without replacing another publisher."""

import os
from pathlib import Path
import tempfile


def atomic_write_new(path: Path, payload: bytes) -> None:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking complete bytes on the same filesystem cannot replace a concurrent publisher.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
