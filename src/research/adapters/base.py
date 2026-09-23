"""Adapter interface shared by all four arms (protocol section 09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUSES = {
    "success",
    "no_mentions",
    "partial",
    "input_failed",
    "model_failed",
    "timeout",
    "rate_limited",
    "invalid_json",
    "schema_failed",
    "alignment_failed",
    "refused",
    "out_of_memory",
    "unsupported",
}


@dataclass
class PredictionInput:
    document_id: str
    text_revision: str
    chunk_id: str
    character_offsets: tuple[int, int]
    section_type: str
    text: str
    ownership_region: tuple[int, int] | None = None
    page_map: Any = None


@dataclass
class ArmResult:
    status: str
    public_records: list[dict] = field(default_factory=list)
    evidence_records: list[dict] = field(default_factory=list)
    raw_response: Any = None
    warnings: list[str] = field(default_factory=list)
    feature_support: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown adapter status: {self.status}")


class Adapter:
    """Base class. Subclasses implement load/predict_one/close."""

    name = "base"
    feature_support: dict[str, bool] = {}

    def load(self, config: dict) -> dict:
        raise NotImplementedError

    def predict_one(self, item: PredictionInput, run_context: dict) -> ArmResult:
        raise NotImplementedError

    def predict_batch(self, items: list[PredictionInput], run_context: dict) -> list[ArmResult]:
        return [self.predict_one(item, run_context) for item in items]

    def close(self) -> None:
        return None
