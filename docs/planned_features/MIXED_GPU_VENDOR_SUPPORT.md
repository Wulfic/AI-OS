# Mixed GPU Vendor Support for Parallel Training

**Status:** 📋 Planned  
> Note: Any references to `docs/user_guide/*` are placeholders for future user-facing docs. For current information, use `docs/INDEX.md` and guides under `docs/guide/`.
**Priority:** Medium  
**Complexity:** Medium (2-4 hours implementation)  
**Target Version:** Future Release  
**Created:** 2025-10-18

## Overview

Enable parallel independent training to work with mixed GPU vendors in a single system. This allows users to utilize all available GPUs regardless of manufacturer for training, maximizing hardware utilization.

## Use Cases

### Primary Use Cases
1. **Developers with mixed setups**: NVIDIA + Intel Arc development machines
2. **Budget builds**: Using older NVIDIA + newer AMD GPUs together
3. **Testing environments**: Multi-vendor CI/CD systems
4. **Workstation upgrades**: Adding new GPU without removing old one

### Example Configurations
- NVIDIA RTX 3090 + Intel Arc A770
- AMD RX 7900 XTX + NVIDIA RTX 2080 Ti
- Intel Arc A380 + AMD RX 6600
- NVIDIA RTX 4090 + AMD RX 7800 XT + Intel Arc A750 (triple vendor)

## Current Limitations

### What Works Now ✅
- **Same vendor, different models**: NVIDIA RTX 3090 + RTX 2080 Ti
- **Automatic detection**: `--cuda-ids 0,1` works for NVIDIA-only

### What Doesn't Work ❌
- **Mixed vendors**: Different GPU types not detected
- **Backend selection**: Hardcoded to CUDA only
- **Device enumeration**: Only scans NVIDIA GPUs
- **VRAM checking**: Only checks CUDA devices

## Technical Requirements

### PyTorch Backend Support

#### NVIDIA GPUs (CUDA)
```python
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    count = torch.cuda.device_count()
```

#### AMD GPUs (ROCm)
```python
import torch
# Requires: pip install torch+rocm5.7
if torch.cuda.is_available():  # ROCm pretends to be CUDA
    device = torch.device('cuda:0')  # Still uses 'cuda' namespace
```

#### Intel Arc/Xe GPUs (XPU)
```python
import intel_extension_for_pytorch as ipex
if torch.xpu.is_available():
    device = torch.device('xpu:0')
    count = torch.xpu.device_count()
```

### Dependencies

**Current:**
- `torch>=2.0.0` with CUDA support

**Needed for full support:**
- `torch+rocm` (AMD support) - separate wheel
- `intel-extension-for-pytorch` (Intel Arc support)
- Optional: Auto-detect and install based on hardware

## Implementation Plan

### Phase 1: Detection & Enumeration (High Priority)

**Goal:** Detect all available GPUs across vendors

```python
def detect_all_gpus() -> list[dict]:
    """Detect GPUs from all supported vendors.
    
    Returns:
        List of GPU info dicts:
        {
            'id': 0,
            'backend': 'cuda' | 'xpu' | 'hip',
            'device_id': 0,  # Backend-specific ID
            'name': 'NVIDIA GeForce RTX 3090',
            'vendor': 'NVIDIA' | 'AMD' | 'Intel',
            'vram_total': 24 * 1024**3,  # bytes
            'vram_available': 23.5 * 1024**3,
            'compute_capability': '8.6',  # NVIDIA only
        }
    """
    gpus = []
    global_id = 0
    
    # NVIDIA (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'id': global_id,
                    'backend': 'cuda',
                    'device_id': i,
                    'name': props.name,
                    'vendor': 'NVIDIA',
                    'vram_total': props.total_memory,
                    'vram_available': torch.cuda.mem_get_info(i)[0],
                    'compute_capability': f"{props.major}.{props.minor}",
                })
                global_id += 1
    except Exception as e:
        print(f"CUDA detection failed: {e}")
    
    # Intel Arc (XPU)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                gpus.append({
                    'id': global_id,
                    'backend': 'xpu',
                    'device_id': i,
                    'name': props.name,
                    'vendor': 'Intel',
                    'vram_total': props.total_memory,
                    'vram_available': torch.xpu.mem_get_info(i)[0],
                    'compute_capability': None,
                })
                global_id += 1
    except ImportError:
        pass  # Intel extension not installed
    except Exception as e:
        print(f"XPU detection failed: {e}")
    
    # AMD (ROCm/HIP) - tricky because it uses CUDA namespace
    # Need to check GPU names to distinguish from NVIDIA
    try:
        import torch
        if hasattr(torch, 'hip') and torch.hip.is_available():
            for i in range(torch.hip.device_count()):
                props = torch.hip.get_device_properties(i)
                gpus.append({
                    'id': global_id,
                    'backend': 'hip',
                    'device_id': i,
                    'name': props.name,
                    'vendor': 'AMD',
                    'vram_total': props.total_memory,
                    'vram_available': torch.hip.mem_get_info(i)[0],
                    'compute_capability': None,
                })
                global_id += 1
    except Exception as e:
        print(f"HIP detection failed: {e}")
    
    return gpus
```

**Files to modify:**
- `src/aios/cli/hrm_hf/parallel_independent_training.py`
- New: `src/aios/cli/hrm_hf/gpu_detection.py`

### Phase 2: Device Mapping (High Priority)

**Goal:** Map user-specified GPU IDs to backend-specific devices

```python
class GPUDevice:
    """Abstraction for different GPU backends."""
    
    def __init__(self, backend: str, device_id: int, info: dict):
        self.backend = backend
        self.device_id = device_id
        self.info = info
        self._device = None
    
    @property
    def device(self) -> torch.device:
        """Get PyTorch device object."""
        if self._device is None:
            if self.backend == 'cuda':
                self._device = torch.device(f'cuda:{self.device_id}')
            elif self.backend == 'xpu':
                self._device = torch.device(f'xpu:{self.device_id}')
            elif self.backend == 'hip':
                self._device = torch.device(f'cuda:{self.device_id}')  # HIP uses cuda namespace
        return self._device
    
    def set_device(self):
        """Set this as the active device."""
        if self.backend == 'cuda':
            torch.cuda.set_device(self.device)
        elif self.backend == 'xpu':
            torch.xpu.set_device(self.device)
        elif self.backend == 'hip':
            torch.cuda.set_device(self.device)
    
    def synchronize(self):
        """Synchronize device."""
        if self.backend == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.backend == 'xpu':
            torch.xpu.synchronize(self.device)
        elif self.backend == 'hip':
            torch.cuda.synchronize(self.device)
    
    def create_stream(self):
        """Create backend-specific stream."""
        if self.backend == 'cuda':
            return torch.cuda.Stream(device=self.device)
        elif self.backend == 'xpu':
            return torch.xpu.Stream(device=self.device)
        elif self.backend == 'hip':
            return torch.cuda.Stream(device=self.device)
    
    def get_memory_info(self) -> tuple[int, int]:
        """Get (free, total) memory in bytes."""
        if self.backend == 'cuda':
            return torch.cuda.mem_get_info(self.device_id)
        elif self.backend == 'xpu':
            return torch.xpu.mem_get_info(self.device_id)
        elif self.backend == 'hip':
            return torch.cuda.mem_get_info(self.device_id)
        return (0, 0)
```

### Phase 3: AMP Backend Support (Medium Priority)

**Goal:** Handle vendor-specific AMP implementations

```python
def create_scaler(backend: str, enabled: bool):
    """Create AMP scaler for specific backend."""
    if backend == 'cuda':
        return torch.amp.GradScaler('cuda', enabled=enabled)
    elif backend == 'xpu':
        # Intel XPU uses different AMP API
        return ipex.optimize(enable_auto_mixed_precision=enabled)
    elif backend == 'hip':
        return torch.amp.GradScaler('cuda', enabled=enabled)  # HIP uses CUDA namespace
    return None

def autocast_context(backend: str, enabled: bool):
    """Get appropriate autocast context for backend."""
    if backend == 'cuda':
        return torch.amp.autocast('cuda', enabled=enabled)
    elif backend == 'xpu':
        return torch.xpu.amp.autocast(enabled=enabled)
    elif backend == 'hip':
        return torch.amp.autocast('cuda', enabled=enabled)
    return nullcontext()
```

### Phase 4: CLI Integration (Medium Priority)

**New CLI interface:**

```bash
# Current (CUDA-only):
aios hrm-hf train-actv1 --cuda-ids 0,1

# New (auto-detect all vendors):
aios hrm-hf train-actv1 --gpu-ids 0,1,2
# Where: 0=NVIDIA, 1=Intel Arc, 2=AMD

# Explicit vendor selection:
aios hrm-hf train-actv1 --gpu-ids cuda:0,xpu:0,hip:0

# List available GPUs:
aios hrm-hf list-gpus
# Output:
# ID  Vendor   Model                      VRAM    Backend
# 0   NVIDIA   GeForce RTX 3090          24 GB   cuda
# 1   Intel    Arc A770                  16 GB   xpu
# 2   AMD      Radeon RX 7900 XTX        24 GB   hip
```

**Files to modify:**
- `src/aios/cli/hrm_hf_cli.py` - Add `--gpu-ids` parameter
- `src/aios/cli/hrm_hf_cli.py` - Add `list-gpus` command

### Phase 5: Load Balancing (Low Priority - Future)

**Goal:** Assign work proportionally to GPU speed

```python
def calculate_gpu_weights(gpus: list[GPUDevice]) -> list[float]:
    """Calculate relative performance weights for GPUs.
    
    Uses heuristics based on:
    - VRAM size
    - Vendor (NVIDIA > AMD > Intel for ML)
    - Known performance tiers
    """
    weights = []
    for gpu in gpus:
        # Base weight from VRAM
        vram_gb = gpu.info['vram_total'] / (1024**3)
        weight = vram_gb / 8  # Normalize to 8GB = 1.0
        
        # Vendor multiplier (rough performance hierarchy)
        if gpu.vendor == 'NVIDIA':
            weight *= 1.0
        elif gpu.vendor == 'AMD':
            weight *= 0.8  # ~20% slower on ML workloads
        elif gpu.vendor == 'Intel':
            weight *= 0.5  # ~50% slower (Arc is newer to ML)
        
        weights.append(weight)
    
    # Normalize to sum = 1.0
    total = sum(weights)
    return [w / total for w in weights]

# Usage:
weights = calculate_gpu_weights(gpus)
# Assign data: GPU 0 gets 50%, GPU 1 gets 30%, GPU 2 gets 20%
```

## Testing Strategy

### Unit Tests
```python
def test_gpu_detection():
    """Test GPU detection works for available hardware."""
    gpus = detect_all_gpus()
    assert len(gpus) > 0
    assert all('backend' in gpu for gpu in gpus)

def test_device_abstraction():
    """Test GPUDevice works across backends."""
    gpus = detect_all_gpus()
    for gpu_info in gpus:
        device = GPUDevice(gpu_info['backend'], gpu_info['device_id'], gpu_info)
        assert device.device is not None
        device.set_device()
        device.synchronize()

def test_mixed_training():
    """Test training works with mixed GPUs."""
    # Only runs if multiple vendor GPUs available
    gpus = detect_all_gpus()
    vendors = set(gpu['vendor'] for gpu in gpus)
    if len(vendors) < 2:
        pytest.skip("Mixed GPU hardware not available")
    
    # Run short training
    run_training(gpu_ids=[0, 1], steps=10)
```

### Manual Testing Checklist
- [ ] NVIDIA + Intel Arc training completes
- [ ] NVIDIA + AMD training completes
- [ ] Intel Arc + AMD training completes
- [ ] All three vendors simultaneously
- [ ] VRAM checking works per vendor
- [ ] AMP works on each vendor
- [ ] Gradient checkpointing works on each vendor
- [ ] Checkpoint merging produces valid model
- [ ] Performance is reasonable (not worse than sequential)

## Performance Considerations

### Bottleneck Analysis

**Scenario 1: Mixed High-End GPUs**
- RTX 4090 (165 TFLOPS) + RX 7900 XTX (61 TFLOPS)
- Bottleneck: AMD ~2.7x slower
- Solution: Assign 73% data to RTX 4090, 27% to RX 7900 XTX
- Expected speedup: ~1.6x vs single RTX 4090

**Scenario 2: High-End + Low-End**
- RTX 4090 (165 TFLOPS) + Arc A380 (8 TFLOPS)
- Bottleneck: Arc ~20x slower
- Solution: Don't use Arc, or give it <10% of data
- Expected speedup: Minimal, possibly negative

**Recommendation:**
- Start without load balancing (equal distribution)
- Add load balancing in Phase 5 if needed
- Document performance expectations

## Backwards Compatibility

### Breaking Changes: None ✅
- `--cuda-ids` continues to work for NVIDIA-only setups
- New `--gpu-ids` parameter is optional
- Auto-detection falls back to CUDA if no other backends

### Migration Path
```python
# Old code (still works):
config.cuda_ids = "0,1"

# New code (recommended):
config.gpu_ids = "0,1"  # Auto-detects vendor
config.gpu_ids = "cuda:0,xpu:1"  # Explicit
```

## Documentation Needed

### User Documentation
- Update `QUICK_START.md` with multi-vendor examples
- Add (placeholder) `docs/user_guide/MIXED_GPU_TRAINING.md` (to be authored later; see docs/INDEX.md for current guidance)
- Update `README.md` with mixed GPU capabilities

### Developer Documentation
- Add `docs/development/GPU_BACKEND_ARCHITECTURE.md`
- Document `GPUDevice` abstraction layer
- Add troubleshooting guide for backend issues

## Known Limitations

### Phase 1-4 Limitations
1. **No automatic load balancing**: Slow GPU limits speed
2. **No cross-GPU communication**: Can't implement DDP across vendors
3. **Backend-specific quirks**: Some features may not work on all vendors
4. **Installation complexity**: Users need correct PyTorch builds

### Permanent Limitations
1. **Performance**: Limited by slowest GPU
2. **Memory**: Each GPU needs full model copy
3. **Synchronization**: Barrier waits for all GPUs

## Success Metrics

### Must Have ✅
- [ ] Detection works for NVIDIA + Intel Arc
- [ ] Detection works for NVIDIA + AMD
- [ ] Training completes without errors
- [ ] Checkpoints merge correctly
- [ ] All existing NVIDIA-only functionality preserved

### Nice to Have 🎯
- [ ] Performance within 10% of theoretical maximum
- [ ] Automatic load balancing implemented
- [ ] User documentation complete
- [ ] Zero user-facing configuration needed

### Stretch Goals 🚀
- [ ] Intel Arc + AMD tested and working
- [ ] Three-vendor training working
- [ ] Automatic backend installation
- [ ] GUI support for mixed GPUs

## Future Enhancements

### Post-Implementation
1. **Dynamic scaling**: Adjust work distribution based on actual throughput
2. **Health monitoring**: Detect slow/stuck GPUs and redistribute work
3. **Power management**: Respect TDP limits per GPU
4. **Cloud support**: Work with mixed instance types (A100 + V100)

### Research Opportunities
1. **Cross-vendor communication**: Investigate vendor-agnostic collective ops
2. **Unified memory**: Explore cross-GPU memory pooling
3. **Heterogeneous parallelism**: Different model parts on different vendors

## References

### PyTorch Backend Documentation
- CUDA: https://pytorch.org/docs/stable/cuda.html
- Intel XPU: https://intel.github.io/intel-extension-for-pytorch/
- AMD ROCm: https://pytorch.org/docs/stable/notes/hip.html

### Similar Implementations
- TensorFlow multi-backend: https://www.tensorflow.org/guide/gpu
- JAX multi-backend: https://jax.readthedocs.io/en/latest/jax.devices.html

## Timeline Estimate

**Aggressive (1 week):**
- Day 1-2: Phase 1 (Detection)
- Day 3-4: Phase 2 (Mapping)
- Day 5: Phase 3 (AMP)
- Day 6-7: Phase 4 (CLI) + Testing

**Realistic (2 weeks):**
- Week 1: Phases 1-3 + Initial testing
- Week 2: Phase 4 + Documentation + Comprehensive testing

**Conservative (1 month):**
- Week 1-2: Implementation
- Week 3: Testing on real mixed hardware
- Week 4: Bug fixes + Documentation + Load balancing

## Open Questions

1. **AMD ROCm detection**: How to reliably distinguish from NVIDIA CUDA?
2. **Error handling**: What if one GPU fails mid-training?
3. **Checkpoint format**: Should we store which GPU/vendor trained each checkpoint?
4. **GUI integration**: How to represent mixed GPUs in the GUI?
5. **Package management**: Should we bundle all backends or make them optional?

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-10-18 | Use abstraction layer (GPUDevice) | Cleaner than if/else everywhere |
| 2025-10-18 | Skip load balancing in Phase 1 | Get basic functionality first |
| 2025-10-18 | Keep `--cuda-ids` for backwards compat | Don't break existing usage |
| TBD | Bundle backends or optional? | Pending: install size vs user friction |

## Approval Status

- [ ] Technical Lead Review
- [ ] Architecture Review
- [ ] Product Manager Approval
- [ ] Implementation Started
- [ ] Testing Complete
- [ ] Documentation Complete
- [ ] Released

---

**Last Updated:** 2025-10-18  
**Next Review:** After Phase 1 implementation  
**Owner:** @AI-OS-Team
