"""Adapters for the four arms."""

from __future__ import annotations

from .base import Adapter, ArmResult, PredictionInput

_LOADERS = {
    "oracle": ("research.adapters.oracle", "OracleAdapter"),
    "llm": ("research.adapters.ollama_llm", "OllamaLLMAdapter"),
    "softcite": ("research.adapters.softcite", "SoftciteAdapter"),
    "scibert": ("research.adapters.encoder", "EncoderAdapter"),
    "modernbert": ("research.adapters.encoder", "EncoderAdapter"),
}


def get_adapter(name: str) -> Adapter:
    if name not in _LOADERS:
        raise KeyError(f"unknown arm: {name}")
    module_name, class_name = _LOADERS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


__all__ = ["Adapter", "ArmResult", "PredictionInput", "get_adapter"]
