"""Benchmark definitions and preset configurations for evaluation panel."""

from typing import Dict, List, Tuple

# Available benchmarks by category
BENCHMARKS: Dict[str, List[Tuple[str, str]]] = {
    "Language": [
        ("mmlu", "MMLU - Massive Multitask Language Understanding (57 tasks)"),
        ("hellaswag", "HellaSwag - Commonsense reasoning"),
        ("truthfulqa_mc1", "TruthfulQA - Truthfulness evaluation"),
        ("winogrande", "WinoGrande - Pronoun resolution"),
        ("drop", "DROP - Discrete reasoning over paragraphs"),
        ("boolq", "BoolQ - Boolean question answering"),
    ],
    "Coding": [
        ("humaneval", "HumanEval - Python programming (164 problems)"),
        ("mbpp", "MBPP - Mostly Basic Python Problems"),
    ],
    "Math": [
        ("gsm8k", "GSM8K - Grade school math word problems"),
        ("minerva_math", "MATH - High school competition mathematics"),
    ],
    "Science": [
        ("arc_challenge", "ARC-Challenge - Science questions (hard)"),
        ("arc_easy", "ARC-Easy - Science questions (easy)"),
        ("sciq", "SciQ - Science question answering"),
    ],
    "Reasoning": [
        ("bbh", "BigBench-Hard - Challenging reasoning tasks"),
    ]
}

# Preset configurations
PRESET_CATEGORIES = ["Language", "Coding", "Math", "Science", "All", "Custom"]
