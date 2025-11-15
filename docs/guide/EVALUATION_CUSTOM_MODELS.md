# Evaluating AI-OS Native Brains

This guide explains how AI-OS native brains (actv1, etc.) can be evaluated using standard benchmarks through the lm-evaluation-harness framework.

## Overview

AI-OS includes a custom adapter that allows native brains to be evaluated using the same benchmarks as HuggingFace models. The adapter wraps AI-OS brains and provides the interface expected by lm_eval.

## Supported Brain Types

Currently supported:
- **actv1**: Custom hierarchical reasoning model with H-layers and L-layers

Future support planned for additional brain architectures.

## How It Works

### 1. Custom Model Adapter

The `AIOSBrainModel` class in `aios.core.evaluation.aios_lm_eval_adapter` implements the lm_eval API:

- **Model Loading**: Loads AI-OS brains using the standard brain loading mechanism
- **Tokenization**: Uses the brain's configured tokenizer
- **Inference**: Delegates to the brain's `run()` method for generation tasks
- **Log Likelihood**: Computes probabilities for multiple-choice tasks using the model's forward pass

### 2. Automatic Registration

When you select a brain in the Evaluation tab, the system:

1. Detects the brain type by reading `brain.json`
2. Registers the custom adapter with lm_eval
3. Configures the evaluation to use `--model aios --model_args brain_path=/path/to/brain`

### 3. Benchmark Compatibility

The adapter supports all standard lm_eval benchmark types:

- **Multiple Choice** (MMLU, ARC, etc.): Uses `loglikelihood()` to score options
- **Generation** (HumanEval, GSM8K, etc.): Uses `generate_until()` for text generation
- **Language Modeling** (WikiText, etc.): Uses `loglikelihood_rolling()` for perplexity

## Using the GUI

### Evaluating a Brain

1. Open the **Evaluation** tab
2. Under **Model Selection**, select **AI-OS Brain** as the source
3. Choose your brain from the dropdown (e.g., "English-v1")
4. Select benchmarks to run
5. Click **Start Evaluation**

The system will automatically:
- Detect it's a native AI-OS brain
- Load the custom adapter
- Run the evaluation
- Display results

### Device Selection & Multi-GPU

- The evaluation tab now honours the **Resources** panel selection when choosing inference devices.
- On Linux, selecting multiple GPUs fans out lm-eval shards across the requested devices. Progress updates show per-device activity.
- On Windows, multi-GPU selections automatically fall back to the first GPU; the log panel surfaces a clear warning so you know the run stayed single-GPU.
- When no CUDA device is available the evaluation falls back to CPU execution and records the reason in both the log panel and analytics metadata.
- Environment overrides (`CUDA_VISIBLE_DEVICES`, `AIOS_VISIBLE_DEVICES`, `AIOS_INFERENCE_PRIMARY_DEVICE`) are applied only for the spawned lm-eval processes so subsequent operations are unaffected.

### Supported Benchmarks

All standard benchmarks work with native brains:

**Quick & General**:
- hellaswag (common sense reasoning)
- arc_easy, arc_challenge (science questions)
- winogrande (pronoun resolution)
- boolq (yes/no questions)

**Academic**:
- mmlu (57 academic subjects)
- truthfulqa_mc1, truthfulqa_mc2 (truthfulness)

**Coding & Math**:
- humaneval (code generation)
- mbpp (Python programming)
- gsm8k (grade school math)

**Language Modeling**:
- wikitext (perplexity on Wikipedia)
- lambada (next-word prediction)

## Using the CLI

You can also evaluate brains from the command line:

```bash
# Evaluate using the custom adapter
lm_eval \
  --model aios \
  --model_args brain_path=artifacts/brains/actv1/English-v1 \
  --tasks hellaswag,arc_easy \
  --device cuda:0 \
  --batch_size 1
```

Or use the AI-OS CLI wrapper:

```bash
# This automatically detects brain type and uses the adapter
aios eval run artifacts/brains/actv1/English-v1 \
  --tasks hellaswag,arc_easy \
  --device cuda:0
```

## Implementation Details

### Model Interface

The adapter implements these key methods:

```python
class AIOSBrainModel(LM):
    def loglikelihood(self, requests):
        """Score multiple choice options"""
        # Uses model.forward() to get logits
        # Computes log probabilities for each token
        
    def generate_until(self, requests):
        """Generate text with stopping criteria"""
        # Uses brain.run() for generation
        # Handles stop sequences
        
    def loglikelihood_rolling(self, requests):
        """Compute perplexity"""
        # Scores each token given context
```

### Brain Loading

Brains are loaded through the standard AI-OS mechanism:

1. Read `brain.json` to get configuration
2. Load checkpoint (`actv1_student.safetensors`)
3. Load tokenizer (configured in brain.json)
4. Move to specified device (GPU/CPU)

### Batch Processing

Currently, the adapter uses batch_size=1 for simplicity. This means:
- Each sample is processed individually
- More reliable for custom architectures
- Slightly slower than batched processing

Future versions may support batching for better performance.

## Limitations & Considerations

### Current Limitations

1. **Batch Size**: Fixed at 1 for now
2. **Speed**: Native brains may be slower than optimized HF models
3. **Memory**: Full model loaded in memory during evaluation
4. **Accuracy**: Some benchmarks may not align perfectly with brain's training

### Performance Tips

**For Faster Evaluation**:
- Use `--limit 100` to test on subset first
- Start with smaller benchmarks (hellaswag, arc_easy)
- Use GPU for significant speedup

**For Memory Constraints**:
- Evaluate one benchmark at a time
- Use CPU if GPU memory is limited
- Close other applications

### Result Interpretation

**Comparing Scores**:
- Native brains may score differently than HF models
- Architecture differences affect benchmark performance
- Focus on trends across multiple benchmarks

**Benchmark Selection**:
- Choose benchmarks aligned with brain's training
- Text-heavy brains: language modeling, QA
- Code-trained brains: HumanEval, MBPP
- Math-trained brains: GSM8K, MATH

## Troubleshooting

### "Model Not Found" Error

```
OSError: English-v1 is not a local folder
```

**Cause**: Trying to use brain name as HF model  
**Solution**: Make sure "AI-OS Brain" is selected as model source

### "Brain Type Not Supported"

```
ValueError: Unsupported brain type: xyz
```

**Cause**: Brain type not yet supported by adapter  
**Solution**: Currently only actv1 brains are supported. Check brain.json type field.

### Poor Performance

If brain scores unexpectedly low:
- Check if brain has been trained
- Verify tokenizer matches training
- Try benchmarks aligned with training data
- Compare with baseline model of similar size

### Memory Errors

```
CUDA out of memory
```

**Solutions**:
- Use `--device cpu` to run on CPU
- Close other GPU applications
- Reduce `--limit` to evaluate fewer samples
- Evaluate benchmarks one at a time

## Examples

### Quick Test

```bash
# Fast test on 100 samples
aios gui
# Select brain, choose "Quick Test" preset, set limit to 100
```

### Full MMLU Suite

```bash
# Comprehensive academic benchmark
lm_eval \
  --model aios \
  --model_args brain_path=artifacts/brains/actv1/English-v1 \
  --tasks mmlu \
  --device cuda:0 \
  --output_path artifacts/evaluation/mmlu_full
```

### Compare Multiple Brains

```bash
# Evaluate multiple brains on same benchmark
for brain in English-v1 Math-v1 Code-v1; do
  aios eval run artifacts/brains/actv1/$brain \
    --tasks hellaswag,arc_easy,gsm8k \
    --output artifacts/evaluation/$brain
done

# Compare results
aios eval compare --ids 1,2,3
```

## Advanced Usage

### Custom Benchmarks

You can create custom tasks for lm_eval:

```python
# my_custom_task.py
from lm_eval.api.task import Task

class MyTask(Task):
    def __init__(self):
        super().__init__()
        # Define your task
```

Then use it:

```bash
lm_eval \
  --model aios \
  --model_args brain_path=artifacts/brains/actv1/English-v1 \
  --tasks my_custom_task \
  --include_path ./
```

### Programmatic Evaluation

```python
from aios.core.evaluation import HarnessWrapper, register_aios_model

# Register adapter
register_aios_model()

# Create wrapper
harness = HarnessWrapper()

# Run evaluation
result = harness.run_evaluation(
    model_name="brain_path=artifacts/brains/actv1/English-v1",
    tasks=["hellaswag", "arc_easy"],
    model_type="aios",
    device="cuda:0",
)

print(f"Overall Score: {result.overall_score:.2%}")
```

## Future Enhancements

Planned improvements:

1. **Batch Support**: Enable batch_size > 1 for faster evaluation
2. **More Brain Types**: Support for additional architectures
3. **Custom Metrics**: Brain-specific evaluation metrics
4. **Fine-grained Control**: Per-layer analysis, attention visualization
5. **Comparative Analysis**: Built-in brain-to-brain comparison

## Related Documentation

- [Model Evaluation Guide](EVALUATION.md) - General evaluation guide
- [Benchmark Selection](BENCHMARK_SELECTION.md) - Choosing appropriate benchmarks
- [ACTv1 Architecture](../research/ACTV1_ARCHITECTURE.md) - Brain architecture details

---

**Last Updated**: November 8, 2025
