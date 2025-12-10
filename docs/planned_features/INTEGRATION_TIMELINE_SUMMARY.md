# Integration Timeline Summary
## Vector Stores (PF-005) + Persistent Traces/Crystallization

**Created**: December 8, 2025  
**Status**: Plans Now Compatible ✅  
**Purpose**: Coordination guide for parallel development

---

## Timeline Overview

```
                Vector Stores (PF-005)              Persistent Traces (Cognitive)
                ═══════════════════════             ═════════════════════════════

Week 1-2:       Core Infrastructure                 Phase 0: Infrastructure
                - VectorStoreClient protocol        - TraceManager, RoutingPathTree
                - Qdrant/LanceDB drivers            - Config schemas
                - Unit tests                        - Unit tests
                
Week 3-4:       Dataset Backend Integration         Phase 1: Trace Capture
                - HF/WebDataset backends            - Attention hooks
                - CLI/GUI wiring                    - Salience computation
                - Integration tests                 - Memory profiling

Week 5-6:       Production Hardening                Phase 2: Bias Injection
                - Error handling                    - Sparse → dense conversion
                - Documentation                     - Dual-mode attention
                - PowerShell examples               - Trace visualization

              ┌───────────────────────────────────────────────────────┐
Week 6.5-7.5: │  INTEGRATION PHASE (Both teams collaborate)          │
              │  Prerequisites: Both foundations complete             │
              │                                                       │
              │  - TraceVectorStore wrapper                          │
              │  - MotifVectorStore wrapper                          │
              │  - TraceEmbedder implementation                      │
              │  - MotifEmbedder implementation                      │
              │  - Sync/load protocols                               │
              │  - Integration tests                                 │
              └───────────────────────────────────────────────────────┘

Week 7-8:                                           Phase 3: Routing Path Logging
                                                    - TopKRouter hooks
                                                    - Suffix tree building
                                                    - Path visualization

Week 8:         Deployment & Monitoring             (continues independently)
                - Production deployment
                - Monitoring dashboard
                - Backup/restore scripts

Week 9-11:                                          Phase 4: Crystallization
                                                    - Motif freezing
                                                    - Distillation
                                                    - Pruning

Week 12-13:                                         Phase 5: Auxiliary Losses
                                                    - Trace utilization loss
                                                    - Crystallization entropy
                                                    - Hyperparameter tuning

Week 14-16:                                         Phase 6: Evaluation
                                                    - Benchmarks
                                                    - FLOP measurements
                                                    - Language analysis

Week 17-18:                                         Phase 7: Production Hardening
                                                    - Distributed training
                                                    - OOM safeguards
                                                    - User documentation
```

---

## Critical Dependencies

### Week 2 Milestone
**Deliverable**: `VectorStoreClient` interface finalized  
**Consumers**:
- TraceVectorStore (Week 6.5)
- MotifVectorStore (Week 6.5)

**Interface contract**:
```python
class VectorStoreClient:
    def upsert(self, ids: Sequence[str], 
               vectors: Sequence[Sequence[float]], 
               metadata: Optional[Sequence[Dict[str, Any]]]) -> None
    
    def query(self, vector: Sequence[float], 
              top_k: int, 
              filter: Optional[Dict[str, Any]]) -> List[Tuple[str, float, Dict]]
    
    def delete(self, ids: Sequence[str]) -> None
    def close(self) -> None
```

### Week 4 Milestone
**Deliverable**: Qdrant or LanceDB deployable  
**Requirement**: At least one backend fully functional for integration testing

### Week 6 Checkpoint
**Vector Stores Team**: All core features complete, ready for cognitive integration  
**Persistent Traces Team**: Phases 0-2 complete, TraceManager ready for vector persistence

### Week 6.5 Integration Kickoff
**Joint Deliverable**: TraceVectorStore and MotifVectorStore working with both backends

---

## Configuration Compatibility

Both plans now share unified `memory:` namespace in `config/default.yaml`:

```yaml
memory:
  dataset:          # PF-005 dataset backends
  vector_store:     # PF-005 storage backend
  persistent_traces:  # Cognitive memory (with vector_store integration flags)
  semantic_crystallization:  # Cognitive memory (with vector_store integration flags)
```

**No conflicts**: Each subsystem has dedicated namespace under `memory:`.

---

## Module Dependencies

```
src/aios/memory/
├── vector_store.py              ← PF-005 (Week 1-2)
└── vector_stores/
    ├── qdrant.py                ← PF-005 (Week 1-2)
    └── lancedb.py               ← PF-005 (Week 1-2)

src/aios/core/hrm_models/cognitive/
├── trace_manager.py             ← Cognitive (Week 3-4)
├── routing_tree.py              ← Cognitive (Week 1-2)
├── embedders.py                 ← Integration (Week 6.5-7.5)
└── vector_wrappers.py           ← Integration (Week 6.5-7.5)
    ├── TraceVectorStore         (depends on memory.vector_store)
    └── MotifVectorStore         (depends on memory.vector_store)

src/aios/cli/hrm_hf/data_backends/
├── base.py                      ← PF-005 (Week 3-4)
├── custom.py                    ← PF-005 (Week 3-4)
├── hf.py                        ← PF-005 (Week 3-4)
└── webdataset.py                ← PF-005 (Week 3-4)
```

**Import flow**:
- `cognitive/vector_wrappers.py` imports `memory/vector_store.py` ✅
- `cognitive/trace_manager.py` imports `cognitive/vector_wrappers.py` (conditionally) ✅
- No circular dependencies ✅

---

## Integration Testing Strategy

### Week 6.5: Smoke Tests
1. **Trace persistence cycle**:
   - Train 1000 steps with traces enabled
   - TraceManager syncs to Qdrant
   - Restart training, load traces from Qdrant
   - Verify salience values within 1% error

2. **Motif storage test**:
   - Crystallize 10 motifs during training
   - Auto-save to vector store
   - Query similar motifs by task tag
   - Verify retrieval accuracy

### Week 7: Cross-Backend Tests
- Same tests with LanceDB backend
- Verify both Qdrant and LanceDB produce identical results

### Week 7.5: Stress Tests
- Persist 100K traces, measure sync latency
- Query 10K motifs, measure retrieval speed
- Verify memory overhead < 40 MB (30 MB traces + 5 MB embedders + 5 MB overhead)

---

## Success Criteria

### PF-005 Standalone Success
- ✅ HF streaming trains 10 steps on wikitext
- ✅ WebDataset trains 10 steps from tar shards
- ✅ Qdrant upserts 1000 vectors, queries return correct top-5
- ✅ LanceDB passes same tests as Qdrant

### Persistent Traces Standalone Success
- ✅ TraceManager captures high-salience attention edges
- ✅ Bias injection improves convergence on copy tasks
- ✅ Memory overhead < 30 MB
- ✅ Training slowdown < 10%

### Integration Success
- ✅ TraceVectorStore persists 10K traces with < 1% information loss
- ✅ MotifVectorStore retrieves similar motifs with > 0.8 cosine similarity
- ✅ Works with both Qdrant and LanceDB
- ✅ Disabling vector_store gracefully falls back to RAM-only mode
- ✅ Configuration validation prevents invalid states

---

## Risk Mitigation

### Risk 1: Timeline Slippage
**Scenario**: PF-005 Week 1-2 delayed, pushes integration to Week 7.5+  
**Mitigation**: 
- Persistent Traces continues independently through Phase 3
- Integration phase can slide to Week 8 with minimal impact
- Core features work without integration

### Risk 2: Interface Changes
**Scenario**: `VectorStoreClient` API changes after Week 2  
**Mitigation**:
- Freeze interface by Week 2 (strict contract)
- Any changes require approval from both teams
- Wrapper classes (`TraceVectorStore`) insulate from minor changes

### Risk 3: Backend Incompatibility
**Scenario**: Qdrant works but LanceDB has issues  
**Mitigation**:
- Integration phase targets Qdrant first
- LanceDB support can be delayed to Week 8
- Document Qdrant as recommended backend

---

## Communication Protocol

### Weekly Sync (Weeks 1-6)
**Purpose**: Coordinate interface design, share progress  
**Attendees**: PF-005 lead + Cognitive lead  
**Agenda**: 
- Interface changes
- Timeline status
- Blockers

### Integration Sprint (Week 6.5-7.5)
**Purpose**: Joint implementation  
**Attendees**: Both teams  
**Deliverables**:
- TraceVectorStore, MotifVectorStore
- Integration tests
- Documentation

### Handoff (Week 8)
**Purpose**: Transition to maintenance  
**Deliverables**:
- Integration documentation
- Troubleshooting guide
- Performance benchmarks

---

## Document Cross-References

| Document | Section | Content |
|----------|---------|---------|
| `data-backends-vector-stores.md` | § Cognitive Memory Integration | TraceVectorStore, MotifVectorStore specs |
| `data-backends-vector-stores.md` | § Implementation Roadmap | Week 1-8 timeline |
| `data-backends-vector-stores.md` | § Unified Configuration Schema | Full `memory:` config |
| `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` | § Vector Store Integration | Embedding specs, sync protocols |
| `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` | § Phase 2.5 | Integration phase deliverables |
| `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` | Configuration section | Vector store integration flags |

---

## Conclusion

**Both plans are now fully compatible** ✅

**Key achievements**:
1. ✅ Unified `memory:` configuration namespace prevents conflicts
2. ✅ Clear module separation with explicit integration points
3. ✅ Coordinated timeline with joint integration phase (Week 6.5-7.5)
4. ✅ Optional integration - systems work standalone or together
5. ✅ Cross-references ensure both teams stay aligned
6. ✅ Shared schema, parallel development, clean handoff

**Implementation paths**:
- **Path A** (PF-005 only): 6 weeks → Dataset backends + vector stores
- **Path B** (Persistent Traces only): 18 weeks → Cognitive memory (RAM-only)
- **Path C** (Full integration): 8 weeks → Both systems + integration → Then continue Persistent Traces Phases 3-7 (10 more weeks)

**Recommendation**: 
Start both plans in parallel (Weeks 1-6), then evaluate integration ROI at Week 6 checkpoint. If cognitive memory shows promise, proceed with Week 6.5-7.5 integration. If not, each system remains valuable independently.

---

**Status**: Ready for implementation ✅  
**Last Updated**: December 8, 2025  
**Owners**: PF-005 Team + Cognitive Architecture Team
