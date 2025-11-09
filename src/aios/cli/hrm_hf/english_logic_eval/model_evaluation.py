"""
Model Evaluation Module

Provides functions for generating text samples from HRM models
and evaluating their English logic compliance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

from .evaluator import evaluate_english_logic


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
