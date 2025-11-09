"""
Text Analysis Module

Provides text structure and quality analysis:
- Sentence statistics
- Grammar checking (basic)
- Vocabulary diversity
- Logical coherence

All checks are deterministic and lightweight.
"""

from __future__ import annotations

import re
from typing import Dict, Any
from collections import Counter

from .readability import count_syllables


def calculate_sentence_stats(text: str) -> Dict[str, float]:
    """
    Calculate basic sentence statistics.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return {
            "avg_sentence_length": 0.0,
            "sentence_count": 0,
            "min_sentence_length": 0,
            "max_sentence_length": 0,
        }
    
    sentence_lengths = []
    for sent in sentences:
        words = re.findall(r'\b\w+\b', sent)
        sentence_lengths.append(len(words))
    
    return {
        "avg_sentence_length": round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0.0,
        "sentence_count": len(sentences),
        "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
    }


def check_grammar_basics(text: str) -> Dict[str, Any]:
    """
    Perform basic grammar checks (deterministic, lightweight).
    Returns dict with pass/fail flags and counts.
    """
    issues = {
        "missing_capitalization": 0,
        "missing_punctuation": 0,
        "double_spaces": 0,
        "incomplete_sentences": 0,
        "repeated_words": 0,
    }
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    for sent in sentences:
        if sent and sent[0].islower():
            issues["missing_capitalization"] += 1
    
    # Check if text ends with punctuation
    if text.strip() and text.strip()[-1] not in '.!?':
        issues["missing_punctuation"] += 1
    
    # Check for double spaces
    issues["double_spaces"] = len(re.findall(r'  +', text))
    
    # Check for very short "sentences" (likely incomplete)
    for sent in sentences:
        words = re.findall(r'\b\w+\b', sent)
        if len(words) < 3:
            issues["incomplete_sentences"] += 1
    
    # Check for repeated consecutive words
    words = re.findall(r'\b\w+\b', text.lower())
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            issues["repeated_words"] += 1
    
    return issues


def calculate_vocabulary_diversity(text: str) -> Dict[str, float]:
    """
    Calculate vocabulary diversity metrics.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if not words:
        return {
            "unique_words": 0,
            "total_words": 0,
            "lexical_diversity": 0.0,
        }
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Type-Token Ratio (TTR)
    ttr = unique_words / total_words if total_words > 0 else 0.0
    
    return {
        "unique_words": unique_words,
        "total_words": total_words,
        "lexical_diversity": round(ttr, 4),
    }


def check_logical_coherence(text: str) -> Dict[str, Any]:
    """
    Check for logical coherence issues (deterministic heuristics).
    """
    issues = {
        "excessive_repetition": False,
        "repetition_ratio": 0.0,
        "contradictory_patterns": 0,
        "nonsense_score": 0.0,
    }
    
    # Check for excessive phrase repetition
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    
    if len(sentences) > 1:
        sentence_counts = Counter(sentences)
        max_repeat = max(sentence_counts.values())
        issues["repetition_ratio"] = round(max_repeat / len(sentences), 4)
        issues["excessive_repetition"] = max_repeat > len(sentences) * 0.3
    
    # Check for obvious contradictory patterns (simple heuristic)
    # Look for "yes ... no" or "is ... is not" patterns
    text_lower = text.lower()
    contradiction_patterns = [
        (r'\byes\b.*\bno\b', r'\bno\b.*\byes\b'),
        (r'\bis\b.*\bis not\b', r'\bis not\b.*\bis\b'),
        (r'\btrue\b.*\bfalse\b', r'\bfalse\b.*\btrue\b'),
    ]
    
    for pattern1, pattern2 in contradiction_patterns:
        if re.search(pattern1, text_lower) or re.search(pattern2, text_lower):
            issues["contradictory_patterns"] += 1
    
    # Nonsense detection: check for extremely low lexical diversity or very long repeated character sequences
    vocab_div = calculate_vocabulary_diversity(text)
    if vocab_div["total_words"] > 10 and vocab_div["lexical_diversity"] < 0.2:
        issues["nonsense_score"] += 0.5
    
    # Check for repeated character sequences (e.g., "aaaaa", "hahahaha")
    repeated_chars = re.findall(r'(\w)\1{4,}', text)
    if repeated_chars:
        issues["nonsense_score"] += 0.3
    
    issues["nonsense_score"] = round(min(1.0, issues["nonsense_score"]), 4)
    
    return issues
