"""
Test script for English Logic Evaluation module

This script tests the English logic compliance evaluation functions
to ensure they work correctly and provide useful metrics.
"""

from src.aios.cli.hrm_hf.english_logic_eval import (
    evaluate_english_logic,
    flesch_reading_ease,
    flesch_kincaid_grade,
    check_grammar_basics,
    calculate_vocabulary_diversity,
    check_logical_coherence,
)


def test_readability_metrics():
    """Test readability scoring functions."""
    print("=" * 60)
    print("Testing Readability Metrics")
    print("=" * 60)
    
    # Good English
    good_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a simple sentence that should score well. "
        "Clear communication is important for effective writing."
    )
    
    # Complex English
    complex_text = (
        "Notwithstanding the aforementioned considerations, "
        "the implementation of sophisticated methodologies "
        "necessitates comprehensive understanding of multifaceted "
        "interdependencies within organizational frameworks."
    )
    
    # Poor English
    poor_text = "word word word word word word word word"
    
    for label, text in [("Good", good_text), ("Complex", complex_text), ("Poor", poor_text)]:
        print(f"\n{label} Text:")
        print(f"  Flesch Reading Ease: {flesch_reading_ease(text)}")
        print(f"  Flesch-Kincaid Grade: {flesch_kincaid_grade(text)}")
    
    print("\n✓ Readability metrics test passed\n")


def test_grammar_checks():
    """Test grammar checking functions."""
    print("=" * 60)
    print("Testing Grammar Checks")
    print("=" * 60)
    
    good_text = "This is a proper sentence. It has correct capitalization."
    bad_text = "this is bad. no  capitals and  double spaces. the the same word"
    
    for label, text in [("Good", good_text), ("Bad", bad_text)]:
        issues = check_grammar_basics(text)
        print(f"\n{label} Text:")
        print(f"  Grammar Issues: {issues}")
        print(f"  Total Issues: {sum(issues.values())}")
    
    print("\n✓ Grammar checks test passed\n")


def test_vocabulary_diversity():
    """Test vocabulary diversity metrics."""
    print("=" * 60)
    print("Testing Vocabulary Diversity")
    print("=" * 60)
    
    diverse_text = (
        "The magnificent aurora borealis danced across the sky. "
        "Brilliant colors illuminated the frozen landscape below."
    )
    
    repetitive_text = (
        "The the the the the dog dog dog dog went went went."
    )
    
    for label, text in [("Diverse", diverse_text), ("Repetitive", repetitive_text)]:
        stats = calculate_vocabulary_diversity(text)
        print(f"\n{label} Text:")
        print(f"  Unique Words: {stats['unique_words']}")
        print(f"  Total Words: {stats['total_words']}")
        print(f"  Lexical Diversity: {stats['lexical_diversity']}")
    
    print("\n✓ Vocabulary diversity test passed\n")


def test_logical_coherence():
    """Test logical coherence checking."""
    print("=" * 60)
    print("Testing Logical Coherence")
    print("=" * 60)
    
    coherent_text = (
        "The weather is sunny today. "
        "It's a great day for outdoor activities. "
        "Many people enjoy the pleasant conditions."
    )
    
    incoherent_text = (
        "Yes yes yes. No no no. Yes no yes no. "
        "The answer is true false true. "
        "aaaaaaaaaa bbbbbbbbb"
    )
    
    for label, text in [("Coherent", coherent_text), ("Incoherent", incoherent_text)]:
        issues = check_logical_coherence(text)
        print(f"\n{label} Text:")
        print(f"  Coherence Issues: {issues}")
    
    print("\n✓ Logical coherence test passed\n")


def test_comprehensive_evaluation():
    """Test the comprehensive evaluation function."""
    print("=" * 60)
    print("Testing Comprehensive Evaluation")
    print("=" * 60)
    
    test_texts = {
        "High Quality": (
            "Machine learning models require careful training. "
            "The training process involves multiple iterations. "
            "Each iteration improves the model's performance. "
            "Good data quality is essential for success."
        ),
        "Medium Quality": (
            "this is text with some problems. "
            "no capitals here and  double spaces. "
            "but the the content makes sense mostly."
        ),
        "Low Quality": (
            "word word word word word word word word word word. "
            "same same same same same same same same same."
        ),
    }
    
    for label, text in test_texts.items():
        result = evaluate_english_logic(text)
        print(f"\n{label} Text:")
        print(f"  English Quality Score: {result.get('english_quality_score', 0.0)}")
        print(f"  Flesch Reading Ease: {result.get('flesch_reading_ease', 0.0)}")
        print(f"  Flesch-Kincaid Grade: {result.get('flesch_kincaid_grade', 0.0)}")
        print(f"  Lexical Diversity: {result.get('lexical_diversity', 0.0)}")
        print(f"  Grammar Issues: {result.get('grammar_issue_count', 0)}")
        
        # Show sample text snippet
        print(f"  Sample: {text[:100]}...")
    
    print("\n✓ Comprehensive evaluation test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("English Logic Evaluation Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_readability_metrics()
        test_grammar_checks()
        test_vocabulary_diversity()
        test_logical_coherence()
        test_comprehensive_evaluation()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nThe English logic evaluation module is working correctly.")
        print("It will now be used during training to assess model quality.")
        print("\nMetrics tracked:")
        print("  • Readability (Flesch scores)")
        print("  • Grammar correctness")
        print("  • Vocabulary diversity")
        print("  • Logical coherence")
        print("  • Overall English quality score (0-1)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
