"""
Readability Metrics Module

Provides standard readability metrics for English text:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index

All metrics are lightweight and deterministic.
"""

from __future__ import annotations

import re
from typing import Dict


def count_syllables(word: str) -> int:
    """
    Estimate syllable count for a word using simple heuristics.
    Not perfect but good enough for readability metrics.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    
    # Remove common endings that don't add syllables
    word = re.sub(r'(es|ed)$', '', word)
    
    # Count vowel groups
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    # Adjust for silent e
    if word.endswith('e'):
        syllable_count -= 1
    
    # Ensure at least 1 syllable
    return max(1, syllable_count)


def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Higher scores indicate easier readability.
    Range: 0-100 (though can go negative for very complex text)
    
    90-100: Very easy (5th grade)
    60-70: Standard (8th-9th grade)
    0-30: Very difficult (college graduate)
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    
    total_syllables = sum(count_syllables(word) for word in words)
    num_words = len(words)
    num_sentences = len(sentences)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
    
    # Flesch Reading Ease formula
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
    return round(score, 2)


def flesch_kincaid_grade(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.
    Returns the US school grade level needed to understand the text.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    
    total_syllables = sum(count_syllables(word) for word in words)
    num_words = len(words)
    num_sentences = len(sentences)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
    
    # Flesch-Kincaid Grade Level formula
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
    return round(max(0.0, grade), 2)


def gunning_fog_index(text: str) -> float:
    """
    Calculate Gunning Fog Index.
    Estimates years of formal education needed to understand text.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    
    # Count "complex" words (3+ syllables)
    complex_words = sum(1 for word in words if count_syllables(word) >= 3)
    num_words = len(words)
    num_sentences = len(sentences)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
    
    # Gunning Fog formula
    fog = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))
    return round(fog, 2)
