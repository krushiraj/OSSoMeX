"""JSONL is separated by LF, not by Unicode characters inside JSON strings."""
import json
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]
