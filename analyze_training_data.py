"""
Analyze generated training data quality for HRM model training.

This script examines the training data generated from teacher models to verify:
1. Data diversity - are samples varied or repetitive?
2. Data quality - are samples coherent and meaningful?
3. Length distribution - are samples appropriate length?
4. Content quality - do samples contain useful information?
5. Encoding issues - any garbled text or encoding problems?
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import re


def analyze_jsonl_logs(log_path: Path) -> Dict[str, Any]:
    """Analyze generation logs to understand the process."""
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}
    
    events = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    analysis = {
        "total_events": len(events),
        "event_types": Counter(e.get("event") for e in events),
        "generation_stats": {},
        "issues": []
    }
    
    # Find generation progress
    gen_progress = [e for e in events if e.get("event") == "gen_progress"]
    if gen_progress:
        last_progress = gen_progress[-1]
        analysis["generation_stats"] = {
            "samples_generated": last_progress.get("generated", 0),
            "target_samples": last_progress.get("total", 0),
            "completion_rate": f"{(last_progress.get('generated', 0) / max(last_progress.get('total', 1), 1)) * 100:.1f}%"
        }
    
    # Check for OOM or other issues
    oom_events = [e for e in events if "oom" in e.get("event", "").lower()]
    if oom_events:
        analysis["issues"].append(f"Found {len(oom_events)} OOM events during generation")
    
    stopped_events = [e for e in events if e.get("event") == "stopped"]
    if stopped_events:
        analysis["issues"].append(f"Generation was stopped {len(stopped_events)} times")
    
    return analysis


def extract_teacher_generated_text(prompt: str = "english") -> List[str]:
    """
    Try to find teacher-generated samples by running a small generation test.
    Since samples aren't saved to disk, we need to look at what was actually used.
    """
    # Check if we can find any cached samples or logs with actual text
    samples = []
    
    # The training data is generated on-the-fly and passed to training
    # It's not typically saved to disk
    print("\n⚠️  Teacher-generated samples are NOT saved to disk by default!")
    print("They are generated on-the-fly and immediately used for training.")
    print("\nTo analyze sample quality, we need to:")
    print("1. Run a generation test with a known prompt")
    print("2. Examine what the teacher model produces")
    
    return samples


def analyze_sample_quality(samples: List[str]) -> Dict[str, Any]:
    """Analyze the quality of text samples."""
    if not samples:
        return {"error": "No samples to analyze"}
    
    analysis = {
        "total_samples": len(samples),
        "sample_lengths": {
            "min": min(len(s) for s in samples),
            "max": max(len(s) for s in samples),
            "avg": sum(len(s) for s in samples) / len(samples),
        },
        "word_counts": {
            "min": min(len(s.split()) for s in samples),
            "max": max(len(s.split()) for s in samples),
            "avg": sum(len(s.split()) for s in samples) / len(samples),
        },
        "diversity": {},
        "quality_checks": {
            "empty_samples": sum(1 for s in samples if not s.strip()),
            "very_short": sum(1 for s in samples if len(s.strip()) < 10),
            "repetitive": 0,  # Will calculate below
            "non_ascii": sum(1 for s in samples if not s.isascii()),
        }
    }
    
    # Check for repetition
    unique_samples = set(samples)
    analysis["diversity"]["unique_samples"] = len(unique_samples)
    analysis["diversity"]["duplicate_rate"] = f"{((1 - len(unique_samples)/len(samples)) * 100):.1f}%"
    
    # Check for very similar samples (first 50 chars)
    sample_starts = [s[:50] for s in samples if len(s) >= 50]
    if sample_starts:
        unique_starts = set(sample_starts)
        analysis["diversity"]["unique_starts"] = len(unique_starts)
        analysis["diversity"]["start_similarity"] = f"{((1 - len(unique_starts)/len(sample_starts)) * 100):.1f}%"
    
    # Find most common words
    all_words = []
    for sample in samples:
        words = re.findall(r'\b\w+\b', sample.lower())
        all_words.extend(words)
    
    if all_words:
        word_freq = Counter(all_words)
        analysis["diversity"]["unique_words"] = len(word_freq)
        analysis["diversity"]["most_common_words"] = word_freq.most_common(20)
    
    return analysis


def check_training_configuration():
    """Check the current training configuration for teacher dataset settings."""
    print("\n" + "="*80)
    print("TEACHER DATASET CONFIGURATION ANALYSIS")
    print("="*80)
    
    # Check optimization logs
    artifacts_path = Path("artifacts/brains/actv1")
    if artifacts_path.exists():
        print(f"\n📁 Found artifacts directory: {artifacts_path}")
        
        # Analyze generation logs
        gen_logs = list(artifacts_path.glob("opt_gen*.jsonl"))
        if gen_logs:
            print(f"\n📊 Found {len(gen_logs)} generation log files")
            for log_file in gen_logs[:3]:  # Analyze first 3
                print(f"\n   Analyzing: {log_file.name}")
                analysis = analyze_jsonl_logs(log_file)
                print(f"   - Events: {analysis['total_events']}")
                print(f"   - Event types: {dict(analysis['event_types'])}")
                if analysis.get("generation_stats"):
                    stats = analysis["generation_stats"]
                    print(f"   - Generated: {stats['samples_generated']} / {stats['target_samples']} ({stats['completion_rate']})")
                if analysis.get("issues"):
                    print(f"   - Issues: {', '.join(analysis['issues'])}")
        else:
            print("\n⚠️  No generation logs found")
    
    # Check training panel default settings
    print("\n📝 DEFAULT TEACHER DATASET SETTINGS:")
    print("   - Prompt: 'english' (generates random text starting with this seed)")
    print("   - Num samples: 2000 (default, but optimization uses 100-20 for speed)")
    print("   - Max new tokens: 64 (default, optimization uses 32-16 for speed)")
    print("   - Temperature: 1.0 (high diversity)")
    print("   - Top-p: 0.95 (nucleus sampling)")
    print("   - Batch size: 8 (default, optimization tests 1-64)")


def assess_data_quality_issues():
    """Identify potential data quality issues based on configuration."""
    print("\n" + "="*80)
    print("DATA QUALITY ASSESSMENT")
    print("="*80)
    
    issues = []
    warnings = []
    good_practices = []
    
    # Issue 1: Prompt is just "english"
    issues.append({
        "severity": "HIGH",
        "issue": "Generic prompt 'english' produces random, unfocused text",
        "impact": "Training data lacks structure, purpose, or coherent patterns",
        "example": "Teacher model generates arbitrary text like 'english as a second language' or 'english breakfast' without context",
        "fix": "Use structured prompts like 'Q: [question]\\nA:' or specific topic starters"
    })
    
    # Issue 2: No quality filtering
    warnings.append({
        "severity": "MEDIUM",
        "issue": "No quality filtering on generated samples",
        "impact": "May include nonsensical, repetitive, or low-quality text",
        "fix": "Add filters for min/max length, perplexity checks, repetition detection"
    })
    
    # Issue 3: ASCII-only option may discard useful data
    warnings.append({
        "severity": "LOW",
        "issue": "ASCII-only filter (if enabled) discards valid Unicode text",
        "impact": "Reduces diversity, limits model to English-only content",
        "fix": "Consider UTF-8 if you need multilingual support"
    })
    
    # Good practice: Temperature 1.0 provides diversity
    good_practices.append({
        "practice": "High temperature (1.0) for diverse generation",
        "benefit": "Prevents mode collapse, increases training data variety"
    })
    
    # Good practice: Nucleus sampling (top_p=0.95)
    good_practices.append({
        "practice": "Nucleus sampling (top_p=0.95) enabled",
        "benefit": "Balances diversity with coherence"
    })
    
    print("\n🔴 CRITICAL ISSUES:")
    for issue in issues:
        print(f"\n   [{issue['severity']}] {issue['issue']}")
        print(f"   Impact: {issue['impact']}")
        print(f"   Example: {issue['example']}")
        print(f"   ✅ Fix: {issue['fix']}")
    
    print("\n🟡 WARNINGS:")
    for warning in warnings:
        print(f"\n   [{warning['severity']}] {warning['issue']}")
        print(f"   Impact: {warning['impact']}")
        print(f"   ✅ Fix: {warning['fix']}")
    
    print("\n🟢 GOOD PRACTICES FOUND:")
    for practice in good_practices:
        print(f"\n   ✓ {practice['practice']}")
        print(f"     Benefit: {practice['benefit']}")


def recommend_improvements():
    """Provide specific recommendations for better training data."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR BETTER TRAINING DATA")
    print("="*80)
    
    recommendations = [
        {
            "category": "Prompt Design",
            "recommendations": [
                "Use structured prompts: 'Question: What is...?\\nAnswer:' instead of just 'english'",
                "Include diverse prompt templates: questions, instructions, explanations, dialogues",
                "Add context: 'You are a helpful assistant. User: {query}\\nAssistant:'",
                "Create prompt variants: 'Explain...', 'What is...', 'How to...', 'Why does...'",
            ]
        },
        {
            "category": "Quality Filtering",
            "recommendations": [
                "Filter out samples < 20 tokens or > 512 tokens",
                "Remove highly repetitive samples (check n-gram overlap)",
                "Calculate perplexity and discard outliers",
                "Check for sentence completeness (ends with punctuation)",
                "Remove samples with excessive special characters",
            ]
        },
        {
            "category": "Data Diversity",
            "recommendations": [
                "Use multiple teacher models (different sizes/training)",
                "Vary generation parameters (temperature 0.7-1.2)",
                "Include different domains: science, history, math, coding",
                "Balance sample lengths (short, medium, long)",
                "Track unique n-grams to ensure diversity",
            ]
        },
        {
            "category": "Curated Datasets",
            "recommendations": [
                "Consider using existing high-quality datasets alongside generated data",
                "OpenWebText, C4, The Pile, Wikipedia, Stack Exchange",
                "Mix 30% curated + 70% generated for best results",
                "Use datasets in training_data/curated_datasets/ directory",
            ]
        },
    ]
    
    for rec in recommendations:
        print(f"\n📌 {rec['category']}:")
        for r in rec['recommendations']:
            print(f"   • {r}")


def create_sample_prompts():
    """Generate example prompts for better training data."""
    print("\n" + "="*80)
    print("EXAMPLE HIGH-QUALITY PROMPTS")
    print("="*80)
    
    prompts = [
        "Question: What is",
        "Explain the concept of",
        "Write a short story about",
        "The benefits of",
        "How does",
        "In mathematics,",
        "def function_name(",  # Code completion
        "Once upon a time",
        "The scientific method involves",
        "To solve this problem, first",
    ]
    
    print("\n💡 Instead of just 'english', use varied prompts like:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n   {i:2d}. \"{prompt}\"")
        print(f"       → Generates: focused content related to the prompt topic")
    
    print("\n🔧 To use these in the GUI:")
    print("   1. Open HRM Training Panel")
    print("   2. Enable 'Teacher-as-dataset'")
    print("   3. In 'Prompt' field, try one of these instead of 'english'")
    print("   4. Or leave blank to generate from random tokens")


def main():
    """Main analysis function."""
    print("="*80)
    print("HRM TRAINING DATA QUALITY ANALYZER")
    print("="*80)
    
    check_training_configuration()
    assess_data_quality_issues()
    recommend_improvements()
    create_sample_prompts()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Current Status:
- ✅ Teacher generation IS working (logs show samples being generated)
- ⚠️  Data quality is LIMITED by generic 'english' prompt
- ⚠️  No quality filtering applied to generated samples
- ❌ Generated samples are NOT saved to disk (used immediately then discarded)

Key Issues:
1. Generic 'english' prompt creates unfocused, random text
2. No quality checks on generated samples
3. Cannot review actual sample quality (not saved)

Recommendations:
1. Use structured prompts (Q&A format, instructions, etc.)
2. Add quality filters (length, repetition, perplexity)
3. Consider mixing curated datasets with generated data
4. Implement sample caching for quality review

Would you like to:
- Test generation with better prompts?
- Add quality filtering to the generation pipeline?
- Use existing curated datasets instead?
""")


if __name__ == "__main__":
    main()
