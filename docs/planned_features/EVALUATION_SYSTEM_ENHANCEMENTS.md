# Evaluation System Enhancements

**Status:** 📋 Planned  
**Priority:** Medium  
**Category:** Model Evaluation & Benchmarking  
**Created:** October 19, 2025  
**Based on:** Evaluation system testing results (Oct 19-20, 2025)

---

## Overview

Enhance the AI-OS evaluation system with extended benchmarking capabilities, advanced metrics, and comparison tools based on systematic testing of the current evaluation functionality.

**Current State:** Basic corpus analysis and checkpoint evaluation working via `aios english-eval`  
**Goal:** Comprehensive evaluation suite with industry-standard benchmarks and automated comparison

---

## Motivation

Recent systematic testing (Oct 2025) confirmed that:
- ✅ Current evaluation system works correctly for corpus analysis
- ✅ Multiple checkpoint formats supported (.pt, .safetensors)
- ✅ Artifact storage and retrieval functional
- ⚠️ Limited to readability metrics (Flesch scores, word counts)
- ❌ No perplexity or loss-based quality metrics
- ❌ No industry-standard benchmark support (hellaswag, arc, etc.)
- ❌ No automated comparison between checkpoints

---

## Planned Enhancements

### 1. LM-Evaluation-Harness Integration

**Objective:** Add industry-standard benchmark evaluation capabilities

#### Tasks:
- [ ] Install `lm-eval` dependency
  ```bash
  pip install lm-eval
  ```
- [ ] Integrate with existing `aios eval` commands
- [ ] Enable standard benchmarks:
  - [ ] HellaSwag (commonsense reasoning)
  - [ ] ARC (science questions)
  - [ ] MMLU (multitask understanding)
  - [ ] TruthfulQA (truthfulness)
  - [ ] GSM8K (math reasoning)
  - [ ] HumanEval (code generation)
- [ ] Add custom task configuration support
- [ ] Store benchmark results in artifact system

#### Implementation Notes:
```python
# Example integration
from lm_eval import evaluator, tasks

def run_lm_eval_benchmark(model_path: str, tasks: list[str]):
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=tasks,
        num_fewshot=0,
        batch_size=8
    )
    return results
```

#### Benefits:
- Compare against published baselines
- Validate model capabilities across diverse tasks
- Standard metrics for model comparison
- Community-recognized benchmarks

---

### 2. Perplexity & Quality Metrics

**Objective:** Add model-specific quality metrics to checkpoint evaluations

#### Tasks:
- [ ] Implement perplexity calculation on test datasets
- [ ] Add cross-entropy loss metrics
- [ ] Calculate bits-per-character/byte
- [ ] Track token-level accuracy
- [ ] Add BLEU/ROUGE scores for generation tasks
- [ ] Implement diversity metrics (distinct-n)
- [ ] Add coherence scoring

#### Metrics to Add:
```yaml
quality_metrics:
  - perplexity: "Lower is better - measures prediction confidence"
  - cross_entropy: "Average loss on test set"
  - bits_per_byte: "Compression efficiency metric"
  - token_accuracy: "Exact match rate for next token"
  - distinct_1/distinct_2: "Vocabulary diversity in generations"
  - coherence_score: "Semantic consistency measure"
```

#### Implementation Approach:
```python
def calculate_checkpoint_metrics(model, dataset):
    metrics = {
        'perplexity': calculate_perplexity(model, dataset),
        'cross_entropy': calculate_loss(model, dataset),
        'bits_per_byte': calculate_bpb(model, dataset),
        'token_accuracy': calculate_accuracy(model, dataset),
        'generation_quality': evaluate_generations(model, dataset)
    }
    return metrics
```

#### Integration Points:
- Extend `aios english-eval` to include these metrics when checkpoint provided
- Store in artifact data structure
- Display in `aios artifacts-show` output

---

### 3. Automated Comparison Tools

**Objective:** Enable side-by-side comparison of evaluation results

#### Tasks:
- [ ] Implement `aios eval compare` command
- [ ] Support multi-checkpoint comparison (2+ models)
- [ ] Generate comparison tables (markdown/HTML)
- [ ] Add visualization support:
  - [ ] Performance radar charts
  - [ ] Metric progression over training
  - [ ] Task-specific comparison graphs
- [ ] Statistical significance testing
- [ ] Automated regression detection

#### CLI Interface:
```bash
# Compare two checkpoints
aios eval compare --checkpoints checkpoint1.pt checkpoint2.pt --dataset eval.txt

# Compare multiple evaluations by artifact ID
aios eval compare --artifact-ids 2 3 4 5

# Compare with baseline
aios eval compare --checkpoint my_model.pt --baseline gpt2

# Generate report
aios eval compare --checkpoints model1.pt model2.pt --output comparison_report.html
```

#### Comparison Report Features:
- **Metric Deltas:** Show improvement/regression percentages
- **Statistical Tests:** P-values for significance
- **Ranking:** Best-to-worst across metrics
- **Recommendations:** Identify which checkpoint to use for what purpose
- **Regression Alerts:** Flag significant performance drops

#### Data Structure:
```python
@dataclass
class ComparisonResult:
    checkpoints: list[str]
    metrics: dict[str, list[float]]
    deltas: dict[str, list[float]]  # Percentage changes
    statistical_significance: dict[str, float]  # p-values
    rankings: dict[str, list[int]]
    recommendations: str
    regression_alerts: list[str]
```

---

## Implementation Plan

### Phase 1: LM-Eval Integration (Week 1-2)
1. Install and test lm-eval library
2. Create wrapper functions for common benchmarks
3. Integrate with existing CLI commands
4. Test on ActV1 models
5. Document usage and available tasks

### Phase 2: Perplexity Metrics (Week 2-3)
1. Implement perplexity calculation
2. Add to english-eval output
3. Store in artifact system
4. Add generation quality metrics
5. Test across different checkpoints

### Phase 3: Comparison Tools (Week 3-4)
1. Design comparison data structures
2. Implement `aios eval compare` command
3. Add table/visualization generation
4. Implement statistical testing
5. Create automated reports
6. Add regression detection

### Phase 4: Documentation & Testing (Week 4)
1. Comprehensive user documentation
2. Example workflows and tutorials
3. Unit tests for all new functions
4. Integration tests with real checkpoints
5. Performance benchmarking

---

## Technical Requirements

### Dependencies:
```toml
[dependencies]
lm-eval = "^0.4.0"  # LM Evaluation Harness
scipy = "^1.11.0"   # Statistical tests
matplotlib = "^3.8.0"  # Visualizations
seaborn = "^0.13.0"  # Enhanced plots
jinja2 = "^3.1.0"   # HTML report templates
```

### Compatibility:
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- Works with existing .pt and .safetensors checkpoints

---

## File Structure

```
src/aios/
├── evaluation/
│   ├── __init__.py
│   ├── lm_eval_wrapper.py      # LM-eval integration
│   ├── metrics.py               # Perplexity, quality metrics
│   ├── comparison.py            # Comparison tools
│   ├── reports.py               # Report generation
│   └── visualization.py         # Plotting functions
├── cli/
│   └── eval_commands.py         # Extended CLI commands
└── templates/
    ├── comparison_report.html   # HTML template
    └── comparison_table.md      # Markdown template
```

---

## Usage Examples

### Example 1: Standard Benchmark
```bash
# Run hellaswag benchmark
aios eval run --checkpoint artifacts/brains/actv1/final_model.pt \
              --tasks hellaswag \
              --label "actv1-hellaswag"

# View results
aios artifacts-show-latest
```

### Example 2: Comprehensive Evaluation
```bash
# Run multiple benchmarks with quality metrics
aios eval run --checkpoint my_model.pt \
              --tasks hellaswag,arc_easy,arc_challenge \
              --dataset eval_dataset.txt \
              --include-perplexity \
              --include-generation-metrics \
              --label "comprehensive-eval"
```

### Example 3: Compare Checkpoints
```bash
# Compare training progression
aios eval compare \
    --checkpoints artifacts/brains/actv1/English-v1/actv1_student.safetensors \
                  artifacts/brains/actv1/English-v2/actv1_student.safetensors \
                  artifacts/brains/actv1/English-v3/actv1_student.safetensors \
                  artifacts/brains/actv1/English-v4/actv1_student.safetensors \
    --dataset training_data/eval_test_dataset.txt \
    --output training_progression.html \
    --show-deltas
```

### Example 4: Automated Testing
```bash
# Compare new checkpoint against baseline
aios eval compare \
    --checkpoint new_checkpoint.pt \
    --baseline artifacts/brains/actv1/final_model.pt \
    --dataset validation_set.txt \
    --fail-on-regression \
    --threshold 5.0  # Fail if >5% regression on any metric
```

---

## Success Metrics

### Quantitative:
- [ ] 10+ standard benchmarks supported
- [ ] 5+ quality metrics per evaluation
- [ ] Comparison reports generated in <30 seconds
- [ ] 100% compatibility with existing checkpoints
- [ ] <1 minute evaluation time for standard datasets

### Qualitative:
- [ ] Users can easily compare model versions
- [ ] Clear identification of best checkpoint for tasks
- [ ] Automated CI/CD integration possible
- [ ] Reports are readable and actionable

---

## Testing Strategy

### Unit Tests:
- Metric calculation accuracy
- Statistical test correctness
- Report generation validity

### Integration Tests:
- End-to-end benchmark runs
- Multi-checkpoint comparisons
- Artifact storage/retrieval

### Validation Tests:
- Compare against known baselines
- Verify statistical significance calculations
- Cross-check with manual evaluations

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| lm-eval dependency conflicts | High | Pin compatible versions, test thoroughly |
| Slow benchmark evaluation | Medium | Add batching, caching, parallel execution |
| Large artifact storage | Medium | Implement result compression, selective storage |
| API changes in lm-eval | Medium | Pin version, abstract wrapper layer |
| Comparison complexity | Low | Start simple, iterate based on feedback |

---

## Future Enhancements

### Post-V1:
- [ ] Multi-GPU distributed evaluation
- [ ] Cloud-based benchmark execution
- [ ] Continuous evaluation dashboard
- [ ] A/B testing framework
- [ ] Automatic hyperparameter tuning based on eval results
- [ ] Custom benchmark creation wizard
- [ ] Integration with experiment tracking (MLflow, W&B)

### Advanced Features:
- [ ] Model capability mapping (what tasks is model good at?)
- [ ] Automatic prompt optimization based on eval results
- [ ] Cross-model ensemble recommendations
- [ ] Failure analysis and debugging tools

---

## References

- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Evaluate Library](https://huggingface.co/docs/evaluate)
- [OpenAI Evals Framework](https://github.com/openai/evals)
- Current evaluation test results: `artifacts/evaluation/evaluation_test_results.md`

---

## Related Issues

- Extends existing `aios english-eval` functionality
- Complements training metrics in `artifacts/brains/actv1/metrics.jsonl`
- Supports model selection for production deployment

---

## Changelog

- **2025-10-19:** Initial plan created based on systematic evaluation testing
- **Next:** Prioritize and schedule implementation

---

**Note:** This plan is based on successful validation of the current evaluation system. All proposed enhancements build on working infrastructure and verified checkpoint compatibility.
