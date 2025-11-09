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

import re
from typing import Dict, List, Optional, Any
from collections import Counter
import math


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


def generate_samples_from_model(
    model,
    tokenizer,
    device,
    num_samples: int = 5,
    max_length: int = 100,
    temperature: float = 0.8,
    prompts: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate text samples from the model for evaluation.
    
    Args:
        model: The student model to evaluate
        tokenizer: HuggingFace tokenizer
        device: torch device
        num_samples: Number of samples to generate
        max_length: Maximum token length per sample
        temperature: Sampling temperature
        prompts: Optional list of prompts to condition on
    
    Returns:
        List of generated text strings
    """
    import torch
    
    if prompts is None:
        # Default prompts for evaluation
        prompts = [
            "The purpose of",
            "In order to",
            "This system",
            "The main reason",
            "One important",
        ]
    
    generated_texts = []
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(prompts))):
            prompt = prompts[i]
            try:
                # Encode prompt
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                
                # Generate
                # Unwrap DDP model if needed
                model_unwrapped = model.module if hasattr(model, 'module') else model
                
                # Simple greedy generation with temperature
                generated = input_ids.clone()
                
                for _ in range(max_length):
                    # Prepare batch dict (HRM-style)
                    batch = {
                        "inputs": generated,
                        "targets": generated,  # Not used for generation
                        "puzzle_identifiers": torch.zeros((1,), dtype=torch.int64, device=device),
                    }
                    
                    # Get initial carry and run forward
                    carry = model_unwrapped.initial_carry(batch)
                    _, output = model_unwrapped(carry, batch)
                    logits = output["logits"]
                    
                    # Get next token logits
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to sequence
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop if EOS or sequence too long
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Decode
                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                generated_texts.append(text)
                
            except Exception as e:
                generated_texts.append(f"[Generation error: {str(e)}]")
    
    model.train()
    return generated_texts


def evaluate_generated_samples(
    model,
    tokenizer,
    device,
    num_samples: int = 5,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate samples and evaluate their English logic compliance.
    
    Returns aggregated metrics across all samples.
    """
    samples = generate_samples_from_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=num_samples,
        prompts=prompts,
    )
    
    if not samples:
        return {"error": "no_samples_generated"}
    
    # Evaluate each sample
    evaluations = []
    for sample in samples:
        eval_result = evaluate_english_logic(sample)
        evaluations.append(eval_result)
    
    # Aggregate metrics
    aggregated = {
        "num_samples": len(samples),
        "samples": samples[:3],  # Include first 3 samples for inspection
    }
    
    # Average numerical metrics
    numeric_keys = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog_index",
        "avg_sentence_length",
        "lexical_diversity",
        "english_quality_score",
        "grammar_issue_count",
    ]
    
    for key in numeric_keys:
        values = [e.get(key, 0) for e in evaluations if key in e and isinstance(e.get(key), (int, float))]
        if values:
            aggregated[f"avg_{key}"] = round(sum(values) / len(values), 4)
    
    # Count coherence issues
    total_excessive_repetition = sum(1 for e in evaluations if e.get("coherence", {}).get("excessive_repetition", False))
    total_contradictions = sum(e.get("coherence", {}).get("contradictory_patterns", 0) for e in evaluations)
    
    aggregated["samples_with_excessive_repetition"] = total_excessive_repetition
    aggregated["total_contradiction_patterns"] = total_contradictions
    
    return aggregated
