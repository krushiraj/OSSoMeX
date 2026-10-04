from copy import deepcopy
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def contract_pack():
    path = Path(__file__).resolve().parents[1] / "examples/scibert-contract-examples.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fixture_case(contract_pack):
    return lambda case_id: deepcopy(next(
        case for case in contract_pack["extraction_cases"] if case["case_id"] == case_id
    ))
