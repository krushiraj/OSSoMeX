"""Arms B and M: SciBERT / ModernBERT encoders with task heads.

A base checkpoint is NOT a finished software NER system; task heads must be
trained on labeled data. This adapter therefore refuses to run unless a trained
checkpoint (with the shared heads) is present. Running a freshly initialized
token head would be a plumbing exercise, not an extractor, and is never reported
as quality (protocol section 11, blocked-training fallback).

On Apple Silicon this uses MPS when available.
"""

from __future__ import annotations

from pathlib import Path

from .base import Adapter, ArmResult, PredictionInput


class EncoderAdapter(Adapter):
    name = "encoder"

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id
        self._config: dict = {}
        self._available = False

    def load(self, config: dict) -> dict:
        import torch  # noqa: PLC0415

        self._config = config
        self._model_id = config.get("model_id", self._model_id)
        checkpoint = config.get("checkpoint") or config.get("model_path")
        self._available = bool(checkpoint) and Path(checkpoint).exists()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        return {
            "model_id": self._model_id,
            "checkpoint_configured": checkpoint,
            "checkpoint_available": self._available,
            "device": device,
            "feature_support": self.feature_support(),
        }

    def feature_support(self) -> dict:
        return {
            "name": self._available,
            "version": self._available,
            "intents": self._available and bool(self._config.get("intents_head")),
            "sentiment": self._available and bool(self._config.get("sentiment_head")),
            "auto_resolution": False,
        }

    def predict_one(self, item: PredictionInput, run_context: dict) -> ArmResult:
        if not self._available:
            return ArmResult(
                status="unsupported",
                error=(
                    "No trained checkpoint available. Encoder arms need task training "
                    "on labeled software-mention data before inference. See reports/BLOCKERS.md."
                ),
                feature_support=self.feature_support(),
            )
        from transformers import pipeline  # noqa: PLC0415

        ner = pipeline("token-classification", model=self._config["checkpoint"])
        results = ner(item.text, aggregation_strategy="simple")
        records = []
        for hit in results:
            records.append(
                {
                    "name": hit.get("word"),
                    "version": None,
                    "context_sentence": item.text[:200],
                    "intents": ["mentioned"],
                    "sentiment": "not_expressed",
                }
            )
        return ArmResult(
            status="success" if records else "no_mentions",
            public_records=records,
            evidence_records=[],
            raw_response=results,
        )