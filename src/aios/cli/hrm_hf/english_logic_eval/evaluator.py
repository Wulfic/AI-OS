"""
English Logic Evaluator Module

Main evaluation orchestrator that combines all metrics
and computes overall English quality scores.
"""

from __future__ import annotations

import re
from typing import Dict, Any

from .readability import flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index
from .text_analysis import (
    calculate_sentence_stats,
    check_grammar_basics,
    calculate_vocabulary_diversity,
    check_logical_coherence,
)


def evaluate_english_logic(text: str) -> Dict[str, Any]:
    """
    Main evaluation function that combines all English logic compliance checks.
    
    Returns a comprehensive dict with all metrics.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "error": "empty_text",
            "text_length": 0,
        }
    
    text = text.strip()
    
    result: Dict[str, Any] = {
        "text_length": len(text),
        "word_count": len(re.findall(r'\b\w+\b', text)),
    }
    
    # Readability metrics
    try:
        result["flesch_reading_ease"] = flesch_reading_ease(text)
        result["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
        result["gunning_fog_index"] = gunning_fog_index(text)
    except Exception as e:
        result["readability_error"] = str(e)
    
    # Sentence stats
    try:
        result.update(calculate_sentence_stats(text))
    except Exception as e:
        result["sentence_stats_error"] = str(e)
    
    # Grammar checks
    try:
        grammar = check_grammar_basics(text)
        result["grammar_issues"] = grammar
        result["grammar_issue_count"] = sum(grammar.values())
    except Exception as e:
        result["grammar_check_error"] = str(e)
    
    # Vocabulary diversity
    try:
        result.update(calculate_vocabulary_diversity(text))
    except Exception as e:
        result["vocabulary_error"] = str(e)
    
    # Logical coherence
    try:
        coherence = check_logical_coherence(text)
        result["coherence"] = coherence
    except Exception as e:
        result["coherence_error"] = str(e)
    
    # Calculate overall quality score (0-1, higher is better)
    try:
        quality_score = 0.0
        weight_sum = 0.0
        
        # Readability contribution (target: 60-70 Flesch)
        if "flesch_reading_ease" in result:
            flesch = result["flesch_reading_ease"]
            # Optimal range is 60-70, penalize deviations
            if 60 <= flesch <= 70:
                quality_score += 0.3
            elif 50 <= flesch <= 80:
                quality_score += 0.2
            elif 40 <= flesch <= 90:
                quality_score += 0.1
            weight_sum += 0.3
        
        # Grammar contribution (fewer issues = better)
        if "grammar_issue_count" in result:
            # Normalize by text length (issues per 100 words)
            issues_per_100 = (result["grammar_issue_count"] / max(1, result["word_count"])) * 100
            grammar_score = max(0.0, 1.0 - (issues_per_100 / 10.0))  # 10+ issues per 100 words = 0
            quality_score += grammar_score * 0.25
            weight_sum += 0.25
        
        # Vocabulary diversity contribution
        if "lexical_diversity" in result:
            # Target: 0.5-0.8 is good
            lex_div = result["lexical_diversity"]
            if 0.5 <= lex_div <= 0.8:
                quality_score += 0.25
            elif 0.3 <= lex_div <= 0.9:
                quality_score += 0.15
            weight_sum += 0.25
        
        # Coherence contribution
        if "coherence" in result:
            coherence_score = 1.0
            if result["coherence"].get("excessive_repetition", False):
                coherence_score -= 0.3
            if result["coherence"].get("contradictory_patterns", 0) > 0:
                coherence_score -= 0.2
            coherence_score -= result["coherence"].get("nonsense_score", 0.0) * 0.5
            quality_score += max(0.0, coherence_score) * 0.2
            weight_sum += 0.2
        
        result["english_quality_score"] = round(quality_score / max(0.01, weight_sum), 4)
    except Exception as e:
        result["quality_score_error"] = str(e)
        result["english_quality_score"] = 0.0
    
    return result
