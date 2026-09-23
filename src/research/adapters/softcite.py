"""Arm S: Softcite software-mention service (protocol section 10).

The adapter talks to a pinned Softcite text endpoint. If the service cannot be
reached or does not expose the expected contract, the failure is reported
honestly as ``model_failed``/``unsupported`` rather than an empty extraction.
Native sentiment is treated as unsupported unless the pinned service proves
otherwise.
"""

from __future__ import annotations

import hashlib

import requests

from .base import Adapter, ArmResult, PredictionInput


def mention_id(document_id: str, text_revision: str, start: int, end: int) -> str:
    raw = f"{document_id}|{text_revision}|{start}|{end}".encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:24]


class SoftciteAdapter(Adapter):
    name = "softcite"
    feature_support = {
        "name": True,
        "version": True,
        "intents": True,
        "sentiment": False,
        "auto_resolution": False,
    }

    def __init__(self) -> None:
        self._config: dict = {}
        self._session = requests.Session()

    def load(self, config: dict) -> dict:
        self._config = config
        endpoint = config.get("endpoint")
        health = config.get("health_endpoint", (endpoint or "").rsplit("/", 1)[0] + "/isalive" if endpoint else None)
        ready = False
        detail = "endpoint not configured"
        if health:
            try:
                resp = self._session.get(health, timeout=config.get("health_timeout_seconds", 10))
                ready = resp.status_code == 200
                detail = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:
                detail = str(exc)
        return {
            "endpoint": endpoint,
            "ready": ready,
            "health_detail": detail,
            "feature_support": self.feature_support,
        }

    def predict_one(self, item: PredictionInput, run_context: dict) -> ArmResult:
        endpoint = self._config.get("endpoint")
        if not endpoint:
            return ArmResult(
                status="unsupported",
                error="Softcite endpoint not configured; see reports/BLOCKERS.md",
            )
        timeout = self._config.get("timeout_seconds", 120)
        try:
            resp = self._session.post(endpoint, data={"text": item.text}, timeout=timeout)
        except requests.Timeout:
            return ArmResult(status="timeout")
        except requests.RequestException as exc:
            return ArmResult(status="model_failed", error=str(exc))
        if resp.status_code == 204:
            return ArmResult(status="no_mentions")
        if resp.status_code != 200:
            return ArmResult(status="model_failed", error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            return ArmResult(status="invalid_json", raw_response=resp.text[:2000], error=str(exc))
        records, evidence, warnings = self._map(item, payload)
        status = "success" if records else "no_mentions"
        return ArmResult(
            status=status,
            public_records=records,
            evidence_records=evidence,
            raw_response=payload,
            warnings=warnings,
        )

    def _map(self, item: PredictionInput, payload: dict) -> tuple[list[dict], list[str], list[str]]:
        warnings: list[str] = []
        records: list[dict] = []
        evidence: list[dict] = []
        mentions = payload.get("software") or payload.get("mentions") or []
        chunk_start = item.character_offsets[0]
        for mention in mentions:
            name_info = mention.get("software-name") or {}
            name = name_info.get("rawForm") or name_info.get("normalizedForm")
            if not name:
                continue
            version_info = mention.get("version")
            if isinstance(version_info, dict):
                version = version_info.get("rawForm") or version_info.get("normalizedForm")
            else:
                version = version_info or None
            context_attr = mention.get("mentionContextAttributes") or {}
            intents = [
                label for label in ("created", "used", "shared")
                if (context_attr.get(label, {}).get("value") if isinstance(context_attr.get(label), dict) else context_attr.get(label))
            ]
            if not intents:
                intents = ["mentioned"]
            context = mention.get("context") or item.text[:200]
            if isinstance(context, dict):
                context = context.get("sentence") or item.text[:200]
            name_span = None
            off_start = name_info.get("offsetStart")
            off_end = name_info.get("offsetEnd")
            if isinstance(off_start, int) and isinstance(off_end, int):
                name_span = (chunk_start + off_start, chunk_start + off_end)
            else:
                local = item.text.find(name)
                if local >= 0:
                    name_span = (chunk_start + local, chunk_start + local + len(name))
            records.append(
                {
                    "name": name,
                    "version": version,
                    "context_sentence": context,
                    "intents": intents,
                    "sentiment": "not_expressed",
                }
            )
            evidence.append(
                {
                    "document_id": item.document_id,
                    "text_revision": item.text_revision,
                    "mention_id": mention_id(
                        item.document_id, item.text_revision,
                        name_span[0] if name_span else chunk_start,
                        name_span[1] if name_span else chunk_start,
                    ),
                    "name_span": {"start": name_span[0], "end": name_span[1]} if name_span else None,
                    "sentence_span": None,
                    "section_type": item.section_type,
                    "version_links": [],
                    "alignment_status": "exact" if name_span else "unaligned",
                    "chunk_ids": [item.chunk_id],
                    "note": "sentiment unsupported by native Softcite; set not_expressed placeholder for schema only",
                }
            )
        if mentions:
            warnings.append("sentiment feature unsupported by native Softcite; not scored as correct")
        return records, evidence, warnings
