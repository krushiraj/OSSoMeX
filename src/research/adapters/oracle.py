"""Deterministic oracle adapters used by the mandatory harness fixtures.

These implement the protocol section 09 "Perfect predictions / all missing /
extra hallucinations" fixtures and the no-software negative fixtures. They do
not model any real software and must never be reported as an arm's quality.
"""

from __future__ import annotations

from .base import Adapter, ArmResult, PredictionInput


class OracleAdapter(Adapter):
    name = "oracle"
    feature_support = {
        "name": True,
        "version": True,
        "intents": True,
        "sentiment": True,
        "auto_resolution": False,
    }

    def __init__(self, records: list[dict] | None = None, mode: str = "perfect"):
        self._records = records or []
        self._mode = mode

    def load(self, config: dict) -> dict:
        self._mode = config.get("mode", self._mode)
        self._records = config.get("records", self._records)
        return {"mode": self._mode, "n_records": len(self._records)}

    def predict_one(self, item: PredictionInput, run_context: dict) -> ArmResult:
        if self._mode == "all_missing":
            return ArmResult(status="no_mentions")
        if self._mode == "hallucinate":
            records = [
                {
                    "name": "GhostTool",
                    "version": None,
                    "context_sentence": item.text[:80] or "invalid",
                    "intents": ["mentioned"],
                    "sentiment": "not_expressed",
                }
            ]
            return ArmResult(status="success", public_records=records)
        records = list(self._records)
        return ArmResult(status="success" if records else "no_mentions", public_records=records)
