# Persistent Traces & Semantic Crystallization: Mathematical Foundations

**Companion Document to**: `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md`  
**Status**: Theoretical Analysis  
**Created**: December 8, 2025

---

## 📐 Detailed Mathematical Derivations

### 1. Trace Memory Update Algorithm

#### 1.1 Formal Problem Statement

Given:
- Attention weights $A^{(l,h)}_{i,j}(t) \in [0,1]$ at timestep $t$
- Loss gradients $\nabla_A \mathcal{L}$
- Persistence history $p_{i,j}(t)$ tracking edge recurrence
- Memory budget $B$ (max traces per head)

Find:
- Sparse trace memory $M^{(l,h)}$ that maximizes expected future utility
- Update rule that balances plasticity and stability

#### 1.2 Salience Score Derivation

**Intuition**: Attention edges that are both strong AND important for the loss should persist.

**Components**:
1. **Attention magnitude**: $A^{(l,h)}_{i,j}$ - raw attention weight
2. **Gradient importance**: $\left|\frac{\partial \mathcal{L}}{\partial A^{(l,h)}_{i,j}}\right|$ - how much loss depends on this edge
3. **Temporal consistency**: $p_{i,j}(t) = \sum_{\tau=t-w}^{t} \mathbb{1}[A^{(l,h)}_{i,j}(\tau) > \epsilon]$ - edge frequency

**Combined salience**:
$$
S^{(l,h)}_{i,j}(t) = \underbrace{A^{(l,h)}_{i,j}(t)}_{\text{strength}} \cdot \underbrace{\left|\frac{\partial \mathcal{L}}{\partial A^{(l,h)}_{i,j}}\right|}_{\text{importance}} \cdot \underbrace{\log(1 + p_{i,j}(t))}_{\text{recurrence bonus}}
$$

The $\log(1 + p)$ term provides diminishing returns - edges don't need to recur infinitely to be valuable.

#### 1.3 Exponential Moving Average (EMA) Update

**Standard EMA**:
$$
M^{(l,h)}_{i,j}(t+1) = \lambda \cdot M^{(l,h)}_{i,j}(t) + (1-\lambda) \cdot S^{(l,h)}_{i,j}(t)
$$

**Problem**: Doesn't handle sparsity - memory grows unbounded.

**Solution**: Conditional update with competitive eviction.

**Sparse EMA with Decay**:
$$
M^{(l,h)}_{i,j}(t+1) = \begin{cases}
\lambda \cdot M^{(l,h)}_{i,j}(t) + (1-\lambda) \cdot S^{(l,h)}_{i,j}(t) & \text{if } S^{(l,h)}_{i,j}(t) > \theta_{\text{sal}} \\
\gamma \cdot M^{(l,h)}_{i,j}(t) & \text{otherwise} \\
0 & \text{if } M^{(l,h)}_{i,j}(t+1) < \epsilon_{\text{prune}}
\end{cases}
$$

**Competitive Eviction** (when $|M^{(l,h)}| > B$):
$$
\text{evict} = \arg\min_{(i,j) \in M^{(l,h)}} M^{(l,h)}_{i,j}(t)
$$

Remove lowest-salience trace to make room for new high-salience trace.

#### 1.4 Steady-State Analysis

**Question**: Does the trace memory converge to a stable distribution?

**Assumptions**:
- Attention distribution is stationary: $A^{(l,h)}_{i,j}(t) \sim \mathcal{D}_A$
- Salience distribution is stationary: $S^{(l,h)}_{i,j}(t) \sim \mathcal{D}_S$

**Steady state**: $\mathbb{E}[M_{i,j}(t+1)] = \mathbb{E}[M_{i,j}(t)] = M^*_{i,j}$

For edges receiving reinforcement:
$$
M^*_{i,j} = \frac{(1-\lambda) \mathbb{E}[S_{i,j}]}{1 - \lambda}
$$

For edges not receiving reinforcement:
$$
M^*_{i,j} = 0 \quad \text{(geometric decay drives to zero)}
$$

**Convergence rate**: 
- Reinforced edges converge in $O(\frac{1}{1-\lambda})$ steps
- Decaying edges vanish in $O(\frac{1}{1-\gamma})$ steps

**Memory occupancy**:
$$
|M^{(l,h)}| \approx \min\left(B, \text{Pr}[S > \theta_{\text{sal}}] \cdot T^2\right)
$$

If threshold is calibrated such that $\text{Pr}[S > \theta_{\text{sal}}] = \frac{B}{T^2}$, memory stays at quota.

### 2. Attention Bias Injection

#### 2.1 Modified Attention Mechanism

**Standard attention**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Trace-biased attention**:
$$
\text{Attention}_{\text{biased}}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \alpha \cdot M\right) V
$$

#### 2.2 Bias Strength Analysis

**Question**: How does $\alpha$ affect attention distribution?

**Softmax sensitivity**: For logits $z$ and perturbation $\delta$:
$$
\frac{\partial}{\partial \delta_i} \text{softmax}(z + \delta)_i = \text{softmax}(z)_i (1 - \text{softmax}(z)_i)
$$

**Effective bias**:
$$
\Delta A_{i,j} \approx \alpha \cdot M_{i,j} \cdot A_{i,j} (1 - A_{i,j})
$$

Maximum effect when $A_{i,j} \approx 0.5$ (uncertain attention).

**Recommended $\alpha$ calibration**:
$$
\alpha = \frac{1}{\max_{i,j} M_{i,j}} \cdot \beta
$$

where $\beta \in [0.1, 0.3]$ controls overall bias strength. This normalizes bias relative to trace strength.

#### 2.3 Information-Theoretic Analysis

**Question**: Does trace bias reduce attention entropy?

**Attention entropy** (before bias):
$$
H(A) = -\sum_{j} A_{i,j} \log A_{i,j}
$$

**Attention entropy** (after bias):
$$
H(A') = -\sum_{j} A'_{i,j} \log A'_{i,j}
$$

**Theorem**: If $M_{i,j} > 0$ is concentrated on subset $\mathcal{S} \subset \{1, \ldots, T\}$, then $H(A') < H(A)$.

**Proof**: Bias increases probability mass on $\mathcal{S}$, reducing uncertainty.

**Entropy reduction**:
$$
\Delta H = H(A) - H(A') \approx \alpha \sum_{j \in \mathcal{S}} M_{i,j} \log\left(\frac{|\mathcal{S}|}{T}\right)
$$

Stronger bias and higher concentration → greater entropy reduction → faster convergence.

### 3. Routing Path Crystallization

#### 3.1 Suffix Tree Construction

**Input**: Sequence of routing paths $\{\pi_1, \pi_2, \ldots, \pi_N\}$ where $\pi_i = [e_1^{(1)}, e_2^{(2)}, \ldots, e_L^{(L)}]$

**Output**: Suffix tree $\mathcal{T}$ with nodes representing path prefixes.

**Construction Algorithm**:
```
function BuildRoutingTree(paths):
    tree = Node(root)
    for each path π in paths:
        node = tree.root
        for each layer l in 1..L:
            expert_id = π[l]
            if expert_id not in node.children:
                node.children[expert_id] = Node(expert_id, layer=l)
            node = node.children[expert_id]
            node.count += 1
            node.total_reward += reward(path)
    return tree
```

**Complexity**:
- Time: $O(N \cdot L)$ for $N$ paths of length $L$
- Space: $O(|V|)$ where $|V|$ is number of unique path prefixes
- Worst case: $O(K^L)$ for $K$ experts per layer (all paths unique)
- Best case: $O(L)$ (all paths identical)
- Typical case: $O(K \cdot L)$ (bounded branching)

#### 3.2 Utility Estimation

**Conditional utility**:
$$
U(\pi) = \mathbb{E}[r \mid \pi] - \mathbb{E}[r]
$$

**Sample-based estimator**:
$$
\hat{U}(\pi) = \frac{1}{f(\pi)} \sum_{i: \pi(x_i) = \pi} r_i - \frac{1}{N} \sum_{i=1}^{N} r_i
$$

**Variance**:
$$
\text{Var}[\hat{U}(\pi)] = \frac{\sigma_r^2}{f(\pi)} + \frac{\sigma_r^2}{N}
$$

**Minimum frequency requirement**: To ensure $\text{Var}[\hat{U}] < \epsilon^2$:
$$
f_{\min} > \frac{\sigma_r^2}{\epsilon^2}
$$

For $\sigma_r = 1$ and $\epsilon = 0.1$ (10% precision), need $f_{\min} > 100$.

#### 3.3 Crystallization Decision

**Multi-criteria optimization**: Find motifs that maximize:
$$
\text{Score}(\pi) = w_1 \cdot \log f(\pi) + w_2 \cdot U(\pi) - w_3 \cdot H(\pi) + w_4 \cdot \text{age}(\pi)
$$

**Thresholding**:
$$
\text{Crystallize}(\pi) \Leftrightarrow \text{Score}(\pi) > \theta_{\text{crystal}}
$$

**Pareto frontier**: Alternatively, find motifs that are Pareto-optimal across $(f, U, -H)$.

#### 3.4 Frozen Motif Implementation

**Conceptual**: Crystallized motif $\pi = [e_1, e_2, \ldots, e_k]$ becomes single expert $E_{\pi}$.

**Forward pass**:
$$
E_{\pi}(h_0) = E_{e_k} \circ E_{e_{k-1}} \circ \cdots \circ E_{e_1}(h_0)
$$

**Computational savings**:
- Standard MoE: $k$ router calls + $k \cdot \text{top\_k}$ expert evaluations
- Crystallized: 0 router calls + 1 frozen expert evaluation (deterministic path)

**FLOP reduction**:
$$
\text{Savings} = k \cdot (\text{router\_FLOPs} + (\text{top\_k} - 1) \cdot \text{expert\_FLOPs})
$$

For $k=4$ layers, $\text{top\_k}=2$, router = 10% expert cost:
$$
\text{Savings} \approx 4 \cdot (0.1 + 1) = 4.4 \text{ expert-equivalents}
$$

**Speedup**: ~4.4× faster than routing for this motif!

### 4. Computational Complexity Analysis

#### 4.1 Trace Management Overhead

**Per-forward-pass costs**:

| Operation | Standard | With Traces | Overhead |
|-----------|----------|-------------|----------|
| Attention (Flash) | $O(T \cdot d)$ | $O(T \cdot d)$ | 0% (no trace) |
| Attention (Standard) | $O(T^2 \cdot d)$ | $O(T^2 \cdot d + B \cdot \log B)$ | +$O(B \log B)$ sparse add |
| Trace update | - | $O(T^2 + B \log B)$ | Once per 100 steps |

**Sparse matrix addition** (bias injection):
- Dense attention logits: $T^2$ elements
- Sparse trace matrix: $B$ elements
- Addition: $O(B)$ (iterate sparse entries)
- Negligible compared to $O(T^2 \cdot d)$ attention computation

**Trace consolidation** (periodic):
- Sort by salience: $O(B \log B)$
- Decay all traces: $O(B)$
- Total: $O(B \log B)$ once per 100 steps → amortized $O(B \log B / 100)$

**Verdict**: < 1% overhead when using Flash Attention + sparse capture scheduling.

#### 4.2 Crystallization Overhead

**Per-forward-pass costs**:

| Operation | Cost |
|-----------|------|
| Log routing path | $O(L)$ |
| Update suffix tree | $O(L \log K)$ |
| Per-batch total | $O(B_{\text{batch}} \cdot L \log K)$ |

For batch size 32, 32 layers, 8 experts:
$$
\text{Cost} = 32 \cdot 32 \cdot \log(8) = 3072 \text{ ops} \approx 0.0001\% \text{ of forward pass}
$$

**Motif discovery** (periodic):
- Traverse suffix tree: $O(|V|)$ where $|V| \approx K \cdot L$
- Compute utilities: $O(|V|)$
- Sort by score: $O(|V| \log |V|)$
- Total: $O(K \cdot L \log (K \cdot L))$ once per 1000 steps

**Verdict**: Negligible overhead (< 0.01%).

#### 4.3 Memory Access Patterns

**Trace memory** (random access):
- Structure: Hash map `(layer, head, i, j) → salience`
- Access pattern: Random (depends on attention sparsity)
- Cache efficiency: Poor (each trace access likely cache miss)
- **Optimization**: Batch trace updates to improve locality

**Routing tree** (sequential access):
- Structure: Tree with pointer-based children
- Access pattern: Depth-first traversal (sequential)
- Cache efficiency: Good (children stored contiguously)

### 5. Convergence Guarantees

#### 5.1 Trace Memory Convergence

**Theorem 1**: Under stationary salience distribution, trace memory converges to steady state.

**Proof sketch**:
- EMA update is a contraction mapping for $\lambda \in (0,1)$
- Fixed point: $M^* = (1-\lambda)^{-1} \mathbb{E}[S]$ for reinforced edges
- Lyapunov function: $V(M) = \sum_{i,j} (M_{i,j} - M^*_{i,j})^2$
- $\mathbb{E}[V(M(t+1))] = \lambda^2 V(M(t))$ → exponential convergence

**Convergence rate**: $O(\lambda^t)$

#### 5.2 Crystallization Stability

**Theorem 2**: Crystallized motifs remain stable if underlying routing distribution is stationary.

**Proof sketch**:
- Motif frequency $f(\pi)$ is sample mean of Bernoulli trials
- By LLN: $f(\pi) \to p(\pi)$ as $N \to \infty$
- Utility $U(\pi)$ is sample mean of rewards
- By LLN: $U(\pi) \to \mathbb{E}[r | \pi]$ as $f(\pi) \to \infty$
- If $p(\pi) > 0$ and $\mathbb{E}[r | \pi] > \mathbb{E}[r]$, motif will eventually crystallize

**Instability risk**: If routing distribution is non-stationary (distribution shift), motifs may become stale.

**Mitigation**: Periodic motif revalidation.

---

## 🧮 Experimental Design for Emergent Language Analysis

### Hypothesis Testing

**H1: Hierarchical Structure**  
Crystallized motifs form hierarchical dependencies (motifs call other motifs).

**Measurement**:
- Build call graph of motif activations
- Compute graph depth: $d_{\max} = \max_{\pi} \text{depth}(\pi)$
- Measure branching factor: $b = \frac{|\text{edges}|}{|\text{nodes}|}$

**Test**: Compare to random graph baseline. Expect $d_{\max} > d_{\text{random}}$ and structured branching.

---

**H2: Semantic Coherence**  
Motifs cluster by task type (e.g., arithmetic motifs vs. language motifs).

**Measurement**:
- Embed motifs in feature space: $\phi(\pi) = \text{avg}(\text{activations when } \pi \text{ fires})$
- Cluster using k-means or spectral clustering
- Compute cluster purity w.r.t. task labels

**Test**: Silhouette score > 0.5 indicates meaningful clustering.

---

**H3: Compositionality**  
Motifs combine to form higher-order concepts.

**Measurement**:
- Track co-activation patterns: $P(\pi_i, \pi_j)$
- Mutual information: $MI(\pi_i, \pi_j) = \sum P(\pi_i, \pi_j) \log \frac{P(\pi_i, \pi_j)}{P(\pi_i)P(\pi_j)}$
- Find high-MI pairs → compositional primitives

**Test**: High-MI pairs should correspond to semantic composition (e.g., "question" + "retrieval" = "QA").

---

**H4: Efficiency**  
Internal language is more efficient than token-level processing.

**Measurement**:
- Bits per concept: $\log_2(|\text{motif vocabulary}|)$
- Compare to bits per token: $\log_2(|\text{token vocabulary}|)$
- Compression ratio: $\frac{\text{avg tokens per concept}}{\text{bits per motif} / \text{bits per token}}$

**Test**: Motifs should encode multi-token concepts in fewer "symbols".

---

### Visualization Techniques

**1. Motif Activation Heatmap**
- Rows: Tasks
- Columns: Motifs
- Color: Activation frequency
- Reveals task-motif specialization

**2. Routing Path Sankey Diagram**
- Nodes: Experts per layer
- Edges: Routing transitions
- Width: Frequency
- Shows dominant pathways

**3. Motif Dependency Graph**
- Nodes: Crystallized motifs
- Edges: Co-activation (high MI)
- Layout: Hierarchical (depth by layer)
- Clusters: Semantic groups

**4. t-SNE/UMAP of Motif Embeddings**
- Each motif embedded by average activation pattern
- Dimensionality reduction to 2D
- Color by task performance
- Reveals semantic topology

---

## 📊 Theoretical Limits

### Information-Theoretic Bounds

**Question**: What is the maximum compression achievable through crystallization?

**Token-level model**:
- Vocabulary size: $|V_{\text{token}}| \approx 50,000$
- Bits per token: $\log_2(50,000) \approx 15.6$ bits

**Motif-level model**:
- Motif vocabulary: $|V_{\text{motif}}| \approx 500$
- Bits per motif: $\log_2(500) \approx 8.97$ bits
- Average tokens per motif: $\approx 5$ tokens

**Compression ratio**:
$$
C = \frac{5 \cdot 15.6}{8.97} \approx 8.7\times
$$

**Theoretical limit** (Shannon entropy):
$$
H_{\max} = \log_2(K^L)
$$

where $K$ experts, $L$ layers. For $K=8$, $L=8$:
$$
H_{\max} = 8 \cdot 3 = 24 \text{ bits per motif}
$$

**Practical limit** (accounting for motif frequency distribution):
$$
H_{\text{eff}} = -\sum_{\pi} p(\pi) \log_2 p(\pi)
$$

With Zipfian distribution over motifs, expect $H_{\text{eff}} \approx 10$ bits.

### Scaling Laws

**Question**: How does performance scale with model size, trace budget, and motif count?

**Hypothesized scaling laws**:

1. **Trace effectiveness**:
$$
\text{Perplexity improvement} \propto \log(B / B_{\min})
$$
Diminishing returns - doubling trace budget yields smaller gains.

2. **Motif utility**:
$$
\text{FLOP reduction} \propto \sqrt{M}
$$
where $M$ is number of crystallized motifs. Sublinear due to motif overlap.

3. **Training time**:
$$
\text{Convergence steps} \propto \frac{1}{(1 - \lambda)(1 - \gamma)}
$$
Faster decay → faster convergence but less stability.

**Empirical validation needed** - these are testable predictions!

---

## 🔬 Advanced Topics

### 1. Multi-Task Crystallization

**Challenge**: Different tasks may benefit from different motifs.

**Solution**: Task-conditional crystallization
- Maintain separate motif registries per task type
- Router learns to select appropriate registry based on input
- Motifs specialize to task domains

**Extension**: Meta-learning over motif selection (learning to route to motifs).

### 2. Cross-Model Motif Transfer

**Question**: Can crystallized motifs transfer between models?

**Approach**:
1. Train model A, crystallize motifs
2. Extract motif expert weights
3. Initialize model B with transferred motifs
4. Fine-tune routing to use transferred motifs

**Hypothesis**: High-level motifs (reasoning patterns) should transfer better than low-level (token-specific).

### 3. Adaptive Crystallization Depth

**Observation**: Some concepts need short motifs (2-3 layers), others need deep motifs (6-8 layers).

**Solution**: Variable-length crystallization
- Terminate motif when utility plateaus
- Allow motifs to "call" other motifs (recursive composition)
- Discover optimal depth per concept automatically

### 4. Continuous Crystallization

**Current**: Discrete crystallization (freeze or don't freeze)

**Alternative**: Soft crystallization with learned "solidification" parameter
$$
\text{Expert output} = \sigma(\pi) \cdot E_{\text{frozen}}(h) + (1 - \sigma(\pi)) \cdot E_{\text{routed}}(h)
$$

where $\sigma(\pi) \in [0,1]$ is learned crystallization strength.

Allows gradual transition and adaptive unfreezing.

---

**Status**: Theoretical foundation complete  
**Next**: Implement and validate experimentally  
**Questions**: Document open research questions for community
