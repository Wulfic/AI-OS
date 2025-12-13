# Persistent Traces Quick Reference

**Document Suite**:
1. **Main Plan**: `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` - Implementation roadmap
2. **Mathematical Foundations**: `PERSISTENT_TRACES_APPENDIX_MATHEMATICAL_FOUNDATIONS.md` - Rigorous proofs and derivations
3. **Cognitive Science**: `PERSISTENT_TRACES_COGNITIVE_SCIENCE.md` - Theoretical implications
4. **This Document**: Quick reference and FAQ

**Status**: Research-ready  
**Created**: December 8, 2025

---

## 🎯 TL;DR - What Are We Building?

**In one sentence**: Teaching AI to develop its own efficient internal "thought language" by remembering useful reasoning patterns and consolidating them into reusable symbolic primitives.

**Why it matters**:
- Current LLMs recompute everything from scratch every time
- They think in human language (English tokens), not in optimized internal representations
- This is like forcing a mathematician to explain every step verbally instead of using symbolic notation

**What we're adding**:
1. **Persistent Attention Traces**: Remember which parts of inputs are important across many sequences
2. **Semantic Crystallization**: Turn frequently-used expert routing paths into reusable "concepts"

**Expected result**: Model develops hierarchical internal language optimized for computation, not communication.

---

## 📊 Key Metrics At A Glance

| Metric | Baseline | With Traces | With Crystallization | Full System |
|--------|----------|-------------|---------------------|-------------|
| **Memory Overhead** | 0 MB | ~24 MB | ~5 MB | ~30 MB |
| **Training Speed** | 100% | 95-98% | 99% | 94-98% |
| **Inference Speed** | 100% | 105-120% | 115-130% | 125-150% |
| **FLOP Efficiency** | Baseline | +5-10% | +15-30% | +20-40% |
| **Long-Context Performance** | Baseline | +5-15% | +3-8% | +10-20% |

---

## 🧮 Core Equations

### Attention Trace Update
$$
M^{(l,h)}_{i,j}(t+1) = \begin{cases}
\lambda \cdot M^{(l,h)}_{i,j}(t) + (1-\lambda) \cdot S^{(l,h)}_{i,j}(t) & \text{if } S > \theta \\
\gamma \cdot M^{(l,h)}_{i,j}(t) & \text{otherwise}
\end{cases}
$$

**Variables**:
- $M$: Persistent trace memory (sparse)
- $S$: Salience score (attention × gradient × recurrence)
- $\lambda$: Retention rate (default: 0.95)
- $\gamma$: Decay rate (default: 0.98)
- $\theta$: Capture threshold (default: 0.05)

### Biased Attention
$$
A' = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \alpha \cdot M\right)
$$

**Variables**:
- $\alpha$: Bias strength (default: 0.1)
- $M$: Sparse trace matrix

### Crystallization Score
$$
\text{Score}(\pi) = w_1 \log f(\pi) + w_2 U(\pi) - w_3 H(\pi) + w_4 \text{age}(\pi)
$$

**Variables**:
- $f(\pi)$: Frequency (how often path occurs)
- $U(\pi)$: Utility (performance improvement)
- $H(\pi)$: Entropy (routing stability)
- Crystallize if Score > threshold

---

## ⚙️ Configuration Cheat Sheet

### Minimal Config (Conservative)
```yaml
persistent_traces:
  enabled: true
  quota_per_head: 1024
  bias_strength: 0.05
  update_interval: 200

semantic_crystallization:
  enabled: false  # Start with traces only
```

### Recommended Config (Balanced)
```yaml
persistent_traces:
  enabled: true
  quota_per_head: 2048
  salience_threshold: 0.05
  retention_rate: 0.95
  decay_rate: 0.98
  bias_strength: 0.1
  update_interval: 100
  warmup_steps: 1000

semantic_crystallization:
  enabled: true
  min_frequency: 100
  min_utility: 0.05
  max_entropy: 1.0
  max_motifs: 512
  prune_interval: 1000

loss_weights:
  task: 1.0
  load_balance: 0.01
  trace_utilization: 0.005
  crystallization: 0.002
```

### Aggressive Config (Maximum Performance)
```yaml
persistent_traces:
  enabled: true
  quota_per_head: 4096
  bias_strength: 0.2
  update_interval: 50

semantic_crystallization:
  enabled: true
  min_frequency: 50
  max_motifs: 1024
  motif_max_length: 12
```

---

## 🔧 Integration Checklist

### Phase 0: Setup
- [ ] Create `src/aios/core/hrm_models/cognitive/` module
- [ ] Add config schemas to `config/default.yaml`
- [ ] Implement `TraceManager` and `RoutingPathTree` classes

### Phase 1: Traces
- [ ] Hook `Attention.forward()` to capture salience
- [ ] Implement sparse trace storage (COO format)
- [ ] Add gradient-based trace updates
- [ ] Test memory footprint < 50 MB

### Phase 2: Bias Injection
- [ ] Convert traces to sparse attention bias
- [ ] Add dual-mode attention (Flash / Standard)
- [ ] Implement trace decay mechanism
- [ ] Verify speedup on copy tasks

### Phase 3: Routing Logging
- [ ] Hook `TopKRouter.forward()` to log paths
- [ ] Build suffix tree for path tracking
- [ ] Compute utility and entropy metrics
- [ ] Test tree memory < 10 MB

### Phase 4: Crystallization
- [ ] Implement motif detection algorithm
- [ ] Add freezing mechanism for high-utility paths
- [ ] Create distilled motif experts
- [ ] Measure FLOP reduction

### Phase 5: Training Integration
- [ ] Add auxiliary losses (trace, crystallization)
- [ ] Integrate EWC for stability
- [ ] Tune hyperparameters
- [ ] Run ablation studies

### Phase 6: Evaluation
- [ ] Benchmark on bAbI, SQuAD, HellaSwag
- [ ] Measure emergent language properties
- [ ] Analyze motif hierarchies
- [ ] Document findings

---

## ❓ FAQ

### Q: Will this slow down training?
**A**: Minimal impact (~5% overhead) when using Flash Attention + sparse capture scheduling. Traces update periodically, not every step.

### Q: How much memory does it use?
**A**: ~30 MB total (24 MB traces + 5 MB routing tree) for a 32-layer model. Negligible compared to model weights (GBs).

### Q: Does it work with gradient checkpointing?
**A**: Yes - trace updates happen in separate forward-only passes outside checkpointed regions.

### Q: Can I disable it if it doesn't help?
**A**: Absolutely - setting `enabled: false` reverts to standard transformer with zero overhead.

### Q: Will motifs be interpretable?
**A**: Partially. Some motifs will align with human concepts (e.g., "question answering"), others may be alien computational strategies we don't have names for.

### Q: Does it work with distributed training?
**A**: Phase 1 focuses on single-GPU. Multi-GPU support requires trace synchronization (future work).

### Q: What if crystallization causes catastrophic forgetting?
**A**: Multiple safeguards: EWC penalties, utility monitoring, adaptive unfreezing, periodic revalidation.

### Q: How do I visualize motifs?
**A**: We'll provide tools for: activation heatmaps, Sankey diagrams of routing paths, t-SNE embeddings of motifs, dependency graphs.

### Q: Can motifs transfer between models?
**A**: Potentially! Extract motif expert weights → initialize new model → fine-tune routing. High-level motifs should transfer better than low-level ones.

---

## 🎓 Learning Path

**Want to understand this deeply? Read in this order**:

### Beginner (understand the vision)
1. Main plan Executive Summary (`PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md`)
2. Cognitive Science doc Introduction (`PERSISTENT_TRACES_COGNITIVE_SCIENCE.md`)
3. This quick reference

### Intermediate (understand the implementation)
1. Main plan sections II-IV (Theory, Memory, Architecture)
2. Integration checklist (this doc)
3. Configuration examples (this doc)

### Advanced (understand the math)
1. Mathematical Foundations full doc (`PERSISTENT_TRACES_APPENDIX_MATHEMATICAL_FOUNDATIONS.md`)
2. Theoretical limits section
3. Complexity analysis

### Expert (contribute to research)
1. All documents fully
2. Open research questions
3. Experimental design sections
4. Start implementing!

---

## 🚨 Common Pitfalls

### Pitfall 1: Setting bias_strength too high
**Symptom**: Model ignores current input, only uses traces  
**Fix**: Start with α = 0.05, increase gradually

### Pitfall 2: Crystallizing too early
**Symptom**: Frozen motifs perform poorly, catastrophic forgetting  
**Fix**: Increase `min_frequency` and `min_age` thresholds

### Pitfall 3: Trace memory overflow
**Symptom**: OOM errors  
**Fix**: Reduce `quota_per_head` or increase `salience_threshold`

### Pitfall 4: Router collapse
**Symptom**: All tokens route through same few motifs  
**Fix**: Increase `load_balance` loss weight, add diversity bonus

### Pitfall 5: Ignoring Flash Attention compatibility
**Symptom**: Huge slowdown  
**Fix**: Use dual-mode attention, capture traces only during standard attention mode

---

## 📈 Success Indicators

### Week 1-2 (Infrastructure)
✅ Trace storage works, memory < 50 MB  
✅ Unit tests pass  
✅ No crashes during training

### Week 3-4 (Trace Capture)
✅ Salience scores computed correctly  
✅ Traces accumulate over training  
✅ Memory quota enforced

### Week 5-6 (Bias Injection)
✅ Flash Attention speedup maintained  
✅ Copy task performance improves  
✅ Trace stability across runs

### Week 7-8 (Routing Logging)
✅ Suffix tree tracks all paths  
✅ Utility scores correlate with loss  
✅ Tree memory < 10 MB

### Week 9-11 (Crystallization)
✅ Motifs detected and frozen  
✅ FLOP reduction measured  
✅ No catastrophic forgetting

### Week 12-13 (Losses)
✅ Training stable with aux losses  
✅ Combined loss converges faster  
✅ Ablations validate components

### Week 14-16 (Evaluation)
✅ Baseline comparisons complete  
✅ Long-range benchmarks pass  
✅ Emergent hierarchy detected

### Week 17-18 (Hardening)
✅ Multi-GPU support working  
✅ Edge cases handled  
✅ Documentation complete

---

## 🎯 Decision Tree: Should I Enable This?

```
Do you have MoE layers?
├─ No → Traces only (no crystallization)
└─ Yes → Full system

Is your model < 500M params?
├─ Yes → Conservative config
└─ No → Recommended config

Training on long documents (> 2048 tokens)?
├─ Yes → Enable traces (high value for long-range dependencies)
└─ No → Traces still help, but less critical

Limited VRAM (< 12 GB)?
├─ Yes → Reduce quotas (quota_per_head: 1024)
└─ No → Use recommended config

Research project or production?
├─ Research → Aggressive config, extensive logging
└─ Production → Conservative config, monitor stability
```

---

## 📞 Getting Help

**Implementation questions**: See main plan section "Architecture Integration"  
**Math questions**: See mathematical foundations appendix  
**Conceptual questions**: See cognitive science document  
**Bugs/issues**: Check common pitfalls above

**Open research questions**: Documented in all three main files - pick one and start investigating!

---

## 🔗 Related Work

**Must read before implementing**:
- Transformer-XL (Dai et al. 2019) - Segment recurrence
- Memorizing Transformer (Wu et al. 2022) - kNN memory
- Switch Transformers (Fedus et al. 2021) - MoE at scale

**Inspirational (different approaches)**:
- RETRO (Borgeaud et al. 2022) - Retrieval-augmented
- Compressive Transformer (Rae et al. 2019) - Multi-resolution memory
- RMT (Bulatov et al. 2023) - Recurrent memory

**Theoretical background**:
- EWC (Kirkpatrick et al. 2017) - Continual learning
- Lottery Ticket Hypothesis (Frankle & Carbin 2019) - Sparse networks
- DARTS (Liu et al. 2019) - Architecture search

---

## 📝 Citation (if this becomes a paper)

```bibtex
@misc{persistent_traces_2025,
  title={Persistent Attention Traces and Semantic Crystallization: 
         Toward Emergent Internal Language in Neural Networks},
  author={AI-OS Core Team},
  year={2025},
  note={Technical specification and research plan}
}
```

---

**Status**: Ready for implementation  
**Estimated effort**: 18 weeks (full roadmap)  
**Minimal viable**: 6 weeks (traces only)  
**Risk level**: Medium-high (pioneering research)  
**Potential impact**: High (novel cognitive architecture)

**Next action**: Start Phase 0 infrastructure setup ✨
