"""Evaluation module for running standardized benchmarks on AI models."""

from __future__ import annotations

__all__ = ["HarnessWrapper", "EvaluationResult", "EvaluationHistory", "register_aios_model"]

from .harness_wrapper import HarnessWrapper, EvaluationResult
from .history import EvaluationHistory

# Import adapter to ensure it's registered (optional import)
try:
    from .aios_lm_eval_adapter import register_aios_model
except ImportError:
    # lm_eval not installed, adapter unavailable
    def register_aios_model():
        pass
