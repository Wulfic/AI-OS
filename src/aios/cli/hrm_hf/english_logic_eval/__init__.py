"""
English Logic Compliance Evaluation Module

This module provides functions to evaluate the quality of English text generation
from the HRM model, focusing on:
- Readability (Flesch-Kincaid, Gunning Fog, etc.)
- Grammar and syntax (basic structural checks)
- Logical coherence (repetition, vocabulary diversity)
- Semantic quality (contradictions, nonsensical patterns)

All metrics are designed to be lightweight and deterministic for fast evaluation.
"""

from __future__ import annotations

# Re-export all public functions to maintain original API
from .readability import (
    count_syllables,
    flesch_reading_ease,
    flesch_kincaid_grade,
    gunning_fog_index,
)
from .text_analysis import (
    calculate_sentence_stats,
    check_grammar_basics,
    calculate_vocabulary_diversity,
    check_logical_coherence,
)
from .evaluator import evaluate_english_logic
from .model_evaluation import (
    generate_samples_from_model,
    evaluate_generated_samples,
)

__all__ = [
    # Readability metrics
    "count_syllables",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    # Text analysis
    "calculate_sentence_stats",
    "check_grammar_basics",
    "calculate_vocabulary_diversity",
    "check_logical_coherence",
    # Main evaluator
    "evaluate_english_logic",
    # Model evaluation
    "generate_samples_from_model",
    "evaluate_generated_samples",
]
