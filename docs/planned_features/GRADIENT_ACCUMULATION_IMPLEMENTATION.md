# Gradient Accumulation

**Status**: Planned  
**Priority**: High  
**Objective**: Fix loss instability from small batch sizes by implementing gradient accumulation  
**Created**: 2025-10-23

---

## 📋 Executive Summary

### The Problem
Current training exhibits severe loss instability due to small batch sizes:
```
Step 448: Loss = 7.0073
Step 449: Loss = 8.1779  ← Wild 17% jump
Step 450: Loss = 7.3505
Step 451: Loss = 8.1017
Step 452: Loss = 8.7440  ← Peak instability
Step 453: Loss = 7.5572
```

**Root Cause**: Batch size too small (2-8 samples) → noisy gradient estimates → unstable training

### The Solution
**Gradient Accumulation**: Accumulate gradients over N batches before updating weights
- **Effective batch size** = physical_batch_size × gradient_accumulation_steps
- **VRAM usage** = same as physical_batch_size (no increase!)
- **Training stability** = equivalent to large batch training

### Expected Results
```
# With batch=8, gradient_accumulation_steps=4
Step 112: Loss = 7.0073  (effective_batch=32)
Step 113: Loss = 6.8234  ← Smooth 2.6% decline
Step 114: Loss = 6.6512
Step 115: Loss = 6.4891
Step 116: Loss = 6.3201  ← Stable convergence
```

---

## 🎓 Technical Background

### How Gradient Accumulation Works

**Standard Training** (current):
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()      # Compute gradients
    optimizer.step()     # Update weights immediately
    optimizer.zero_grad()
```

**With Gradient Accumulation**:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # ⚠️ CRITICAL: Scale loss!
    loss.backward()      # Gradients accumulate in model.parameters()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()     # Update weights every N batches
        optimizer.zero_grad()
```

### Why Loss Scaling is Critical
```python
# ❌ WRONG - Gradients will sum instead of average
loss.backward()

# ✅ CORRECT - Scale loss so gradients average
loss = loss / gradient_accumulation_steps
loss.backward()
```

### Memory Usage Comparison
```
Model: 1.3B parameters

┌─────────────┬──────────┬────────────┬─────────────┐
│ Batch Size  │ Accum    │ Effective  │ VRAM Usage  │
├─────────────┼──────────┼────────────┼─────────────┤
│ 2           │ 1        │ 2          │ ~6 GB       │
│ 2           │ 16       │ 32         │ ~6 GB       │ ← Same!
│ 8           │ 1        │ 8          │ ~10 GB      │
│ 8           │ 4        │ 32         │ ~10 GB      │ ← Same!
│ 32          │ 1        │ 32         │ ~24 GB      │
└─────────────┴──────────┴────────────┴─────────────┘

Conclusion: Accumulation doesn't increase VRAM!
```

---

## 🏗️ Implementation Architecture

### Files to Modify

```
src/aios/core/hrm_training/training_config/
├── optimization_fields.py          [ADD] gradient_accumulation_steps field
├── config_main.py                  [MODIFY] to_cli_args() method

src/aios/cli/
├── hrm_hf_cli.py                   [ADD] --gradient-accumulation-steps option

src/aios/cli/hrm_hf/training_logic/
├── train_epoch.py                  [MODIFY] backward/step logic (MAIN CHANGE)

src/aios/gui/components/hrm_training_panel/
├── ui_optimizations.py             [ADD] UI controls for gradient accumulation
├── variable_setup.py               [ADD] gradient_accumulation_var
├── config_builder.py               [MODIFY] include in config build
├── state_management.py             [MODIFY] save/load state
```

---

## 📝 Implementation Steps

### ✅ Phase 1: Configuration Layer

#### Step 1.1: Add Configuration Field
**File**: `src/aios/core/hrm_training/training_config/optimization_fields.py`

**Location**: After line 118 (after `load_in_4bit` field)

**Code to Add**:
```python
    # ============================================================================
    # Gradient Accumulation
    # ============================================================================
    gradient_accumulation_steps: int = 1
    """Number of batches to accumulate gradients before updating weights.
    
    Enables training with larger effective batch sizes without increasing VRAM.
    The effective batch size is: physical_batch_size × gradient_accumulation_steps
    
    Benefits:
    - Fixes loss instability from small batch sizes
    - No VRAM increase (memory usage stays at physical batch size)
    - Smoother training dynamics
    - Better gradient estimates
    
    Example:
    - batch_size=8, gradient_accumulation_steps=4 → effective_batch_size=32
    - VRAM usage: ~10GB (for batch=8)
    - Training stability: equivalent to batch=32
    
    Recommended values:
    - 1: No accumulation (default, update every batch)
    - 2-4: Mild accumulation for slightly smoother training
    - 4-8: Moderate accumulation (recommended for most cases)
    - 8-16: High accumulation for very small batch sizes
    - 16+: Extreme accumulation (use when batch=1-2 required)
    
    How to choose:
    1. Start with current batch_size and desired effective_batch_size
    2. Calculate: gradient_accumulation_steps = effective_batch_size / batch_size
    3. Test and adjust based on loss stability
    
    Memory impact:
    - Gradients: +1× model size (same as normal training)
    - Activations: Only for physical batch size
    - Total overhead: Negligible (<5% of total memory)
    
    Performance impact:
    - Slightly slower due to more forward passes
    - ~5-15% overhead depending on accumulation_steps
    - Worth it for stability improvement
    
    Compatibility:
    - Works with: AMP, gradient checkpointing, DeepSpeed ZeRO, PEFT/LoRA
    - Works across: DDP, parallel independent, single-GPU modes
    - Scheduler: Automatically adjusted to step with weight updates
    """
```

---

#### Step 1.2: Update CLI Args Conversion
**File**: `src/aios/core/hrm_training/training_config/config_main.py`

**Location**: In `to_cli_args()` method, after batch_size (around line 109)

**Find**:
```python
        args.extend(["--batch-size", str(self.batch_size)])
        args.extend(["--steps", str(self.steps)])
```

**Replace with**:
```python
        args.extend(["--batch-size", str(self.batch_size)])
        args.extend(["--steps", str(self.steps)])
        
        # Gradient accumulation
        if self.gradient_accumulation_steps > 1:
            args.extend(["--gradient-accumulation-steps", str(self.gradient_accumulation_steps)])
```

---

### ✅ Phase 2: CLI Integration

#### Step 2.1: Add CLI Option
**File**: `src/aios/cli/hrm_hf_cli.py`

**Location**: In `train_actv1` function, after `batch_size` parameter (around line 137)

**Find**:
```python
    batch_size: int = typer.Option(8, "--batch-size"),
    steps: int = typer.Option(200, "--steps"),
```

**Replace with**:
```python
    batch_size: int = typer.Option(8, "--batch-size"),
    gradient_accumulation_steps: int = typer.Option(
        1, 
        "--gradient-accumulation-steps",
        help="Accumulate gradients over N batches before updating weights. "
             "Effective batch size = batch_size × gradient_accumulation_steps. "
             "Use to train with larger effective batches without increasing VRAM. "
             "Example: batch=8, accum=4 → effective_batch=32"
    ),
    steps: int = typer.Option(200, "--steps"),
```

---

### ✅ Phase 3: Training Loop Modification (CRITICAL)

#### Step 3.1: Modify Training Logic
**File**: `src/aios/cli/hrm_hf/training_logic/train_epoch.py`

**Location**: Around line 350-380 (backward/step section)

**Find** (the entire backward pass section):
```python
                # Backward pass
                if deepspeed_engine is not None:
                    deepspeed_engine.backward(loss)
                    deepspeed_engine.step()
                elif use_amp and scaler is not None and dev == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        opt.zero_grad(set_to_none=True)
                        scaler.update()
                    else:
                        scaler.step(opt)
                        scaler.update()
                else:
                    loss.backward()
                    average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        opt.zero_grad(set_to_none=True)
                    else:
                        opt.step()
                    
                steps_done += 1
```

**Replace with**:
```python
                # Get gradient accumulation config
                gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
                accumulation_steps = max(1, int(gradient_accumulation_steps))
                
                # Scale loss for gradient accumulation
                # CRITICAL: This ensures gradients average instead of sum
                scaled_loss = loss / accumulation_steps
                
                # Backward pass (gradients accumulate automatically)
                if deepspeed_engine is not None:
                    # DeepSpeed handles accumulation internally
                    deepspeed_engine.backward(scaled_loss)
                    
                    # Only step optimizer every N batches
                    if (batch_idx + 1) % accumulation_steps == 0:
                        deepspeed_engine.step()
                        
                elif use_amp and scaler is not None and dev == "cuda":
                    # AMP with gradient accumulation
                    scaler.scale(scaled_loss).backward()
                    
                    # Only update weights every N batches
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(opt)
                        average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            opt.zero_grad(set_to_none=True)
                            scaler.update()
                        else:
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad(set_to_none=True)
                else:
                    # Standard mode with gradient accumulation
                    scaled_loss.backward()
                    
                    # Only update weights every N batches
                    if (batch_idx + 1) % accumulation_steps == 0:
                        average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            opt.zero_grad(set_to_none=True)
                        else:
                            opt.step()
                            opt.zero_grad(set_to_none=True)
                
                # Increment step counter (count actual weight updates, not batches)
                if (batch_idx + 1) % accumulation_steps == 0:
                    steps_done += 1
```

---

#### Step 3.2: Update Logging
**File**: `src/aios/cli/hrm_hf/training_logic/train_epoch.py`

**Location**: Around line 400 (logging section)

**Find**:
```python
                if steps_done % 5 == 0 or steps_done == 1:
                    try:
                        import torch
                        if dev == "cuda" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            allocated_gb = torch.cuda.memory_allocated(device_obj) / 1024**3
                            reserved_gb = torch.cuda.memory_reserved(device_obj) / 1024**3
                            max_allocated_gb = torch.cuda.max_memory_allocated(device_obj) / 1024**3
                            total_gb = torch.cuda.get_device_properties(device_obj).total_memory / 1024**3
                            
                            write_jsonl({
                                "step": steps_done,
                                "memory_gb": round(allocated_gb, 3),
```

**Replace with**:
```python
                if steps_done % 5 == 0 or steps_done == 1:
                    try:
                        import torch
                        if dev == "cuda" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            allocated_gb = torch.cuda.memory_allocated(device_obj) / 1024**3
                            reserved_gb = torch.cuda.memory_reserved(device_obj) / 1024**3
                            max_allocated_gb = torch.cuda.max_memory_allocated(device_obj) / 1024**3
                            total_gb = torch.cuda.get_device_properties(device_obj).total_memory / 1024**3
                            
                            # Get accumulation info
                            gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
                            accumulation_steps = max(1, int(gradient_accumulation_steps))
                            
                            write_jsonl({
                                "step": steps_done,
                                "batch_idx": batch_idx + 1,
                                "loss": float(loss.item()),  # Unscaled loss for logging
                                "gradient_accumulation_steps": accumulation_steps,
                                "effective_batch_size": batch_size * accumulation_steps,
                                "physical_batch_size": batch_size,
                                "memory_gb": round(allocated_gb, 3),
```

---

### ✅ Phase 4: GUI Integration

#### Step 4.1: Add Variable
**File**: `src/aios/gui/components/hrm_training_panel/variable_setup.py`

**Location**: In `setup_variables()` function, after existing optimization variables (around line 60)

**Find**:
```python
    panel.use_8bit_optimizer_var = tk.BooleanVar(value=False)
    panel.use_cpu_offload_var = tk.BooleanVar(value=False)
```

**Add after**:
```python
    panel.use_8bit_optimizer_var = tk.BooleanVar(value=False)
    panel.use_cpu_offload_var = tk.BooleanVar(value=False)
    
    # Gradient accumulation
    panel.gradient_accumulation_var = tk.StringVar(value="1")
```

---

#### Step 4.2: Add UI Controls
**File**: `src/aios/gui/components/hrm_training_panel/ui_optimizations.py`

**Location**: After Row 1 (Memory Optimizations), around line 38

**Find**:
```python
    cpu_offload_btn.pack(side="left", padx=(8, 0))
    
    # Row 2: PEFT (Parameter-Efficient Fine-Tuning)
```

**Add between**:
```python
    cpu_offload_btn.pack(side="left", padx=(8, 0))
    
    # Row 1.5: Gradient Accumulation
    grad_accum_row = ttk.Frame(opt_frame)
    grad_accum_row.pack(fill="x", pady=2)
    ttk.Label(grad_accum_row, text="Batch Scaling:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    ttk.Label(grad_accum_row, text="Accum Steps:").pack(side="left", padx=(0, 2))
    accum_combo = ttk.Combobox(grad_accum_row, textvariable=panel.gradient_accumulation_var, width=8, state="readonly")
    accum_combo['values'] = ('1', '2', '4', '8', '16', '32')
    accum_combo.pack(side="left")
    ttk.Label(grad_accum_row, text="→").pack(side="left", padx=4)
    panel.effective_batch_lbl = ttk.Label(grad_accum_row, text="Effective Batch: 8", foreground="blue")
    panel.effective_batch_lbl.pack(side="left")
    
    # Row 2: PEFT (Parameter-Efficient Fine-Tuning)
```

---

#### Step 4.3: Add Tooltip and Update Callback
**File**: `src/aios/gui/components/hrm_training_panel/ui_optimizations.py`

**Location**: In tooltip section, around line 100

**Find**:
```python
        add_tooltip(cpu_offload_btn, "CPU Offload: Moves optimizer states to system RAM\nSaves VRAM • ~30% slower training")
        add_tooltip(peft_enable_btn, "Enable PEFT: Use Low-Rank Adaptation (LoRA) for efficient fine-tuning\n↓95-99% trainable parameters (87M → 500K-2M)")
```

**Add after**:
```python
        add_tooltip(cpu_offload_btn, "CPU Offload: Moves optimizer states to system RAM\nSaves VRAM • ~30% slower training")
        add_tooltip(accum_combo, 
            "Gradient Accumulation: Accumulate gradients over N batches\n"
            "before updating weights.\n\n"
            "Benefits:\n"
            "• Fixes loss instability from small batches\n"
            "• No VRAM increase\n"
            "• Smoother training dynamics\n\n"
            "Effective Batch = Physical Batch × Accum Steps\n"
            "Example: batch=8, accum=4 → effective=32\n\n"
            "Recommended:\n"
            "• 1: No accumulation (default)\n"
            "• 4: Balanced (recommended)\n"
            "• 8-16: High stability (small batches)\n"
            "• 32: Maximum stability (batch=1-2)")
        add_tooltip(peft_enable_btn, "Enable PEFT: Use Low-Rank Adaptation (LoRA) for efficient fine-tuning\n↓95-99% trainable parameters (87M → 500K-2M)")
```

---

#### Step 4.4: Add Update Callback Function
**File**: `src/aios/gui/components/hrm_training_panel/ui_optimizations.py`

**Location**: After tooltips section, before the end of `build_optimizations_section()` function

**Add**:
```python
    # Setup callback to update effective batch label
    def update_effective_batch_label(*args):
        try:
            batch = int(panel.batch_var.get() or 8)
            accum = int(panel.gradient_accumulation_var.get() or 1)
            effective = batch * accum
            panel.effective_batch_lbl.config(text=f"Effective Batch: {effective}")
        except Exception:
            pass
    
    panel.gradient_accumulation_var.trace_add("write", update_effective_batch_label)
    panel.batch_var.trace_add("write", update_effective_batch_label)
    
    # Initial update
    update_effective_batch_label()
```

---

#### Step 4.5: Update Config Builder
**File**: `src/aios/gui/components/hrm_training_panel/config_builder.py`

**Location**: In `build_training_config()` function, after batch_size (around line 25)

**Find**:
```python
        batch_size=int(panel.batch_var.get() or 8),
        steps=int(panel.steps_var.get() or 200),
```

**Replace with**:
```python
        batch_size=int(panel.batch_var.get() or 8),
        gradient_accumulation_steps=int(panel.gradient_accumulation_var.get() or 1),
        steps=int(panel.steps_var.get() or 200),
```

---

#### Step 4.6: Update State Management
**File**: `src/aios/gui/components/hrm_training_panel/state_management.py`

**Location**: In `get_state()` function, around line 30

**Find**:
```python
            "batch_size": panel.batch_var.get(),
            "steps": panel.steps_var.get(),
```

**Add after**:
```python
            "batch_size": panel.batch_var.get(),
            "gradient_accumulation_steps": panel.gradient_accumulation_var.get(),
            "steps": panel.steps_var.get(),
```

**Location**: In `set_state()` function, around line 100

**Find**:
```python
    if "batch_size" in state:
        panel.batch_var.set(str(state["batch_size"]))
    if "steps" in state:
        panel.steps_var.set(str(state["steps"]))
```

**Add after**:
```python
    if "batch_size" in state:
        panel.batch_var.set(str(state["batch_size"]))
    if "gradient_accumulation_steps" in state:
        panel.gradient_accumulation_var.set(str(state["gradient_accumulation_steps"]))
    if "steps" in state:
        panel.steps_var.set(str(state["steps"]))
```

---

## 🧪 Testing Plan

### Test 1: Basic Functionality
**Objective**: Verify gradient accumulation works correctly

```bash
# Terminal test
cd /path/to/AI-OS  # Replace with your AI-OS directory
.\.venv\Scripts\Activate.ps1

aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --steps 20 `
  --log-file artifacts/test_grad_accum.jsonl

# Check log file
cat artifacts/test_grad_accum.jsonl | Select-String "gradient_accumulation"
```

**Expected Output**:
```json
{"gradient_accumulation_steps": 8, "effective_batch_size": 32, "physical_batch_size": 4}
```

---

### Test 2: Loss Stability Comparison
**Objective**: Demonstrate loss stability improvement

**Test 2a: Without accumulation (baseline)**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 2 `
  --gradient-accumulation-steps 1 `
  --steps 100 `
  --log-file artifacts/test_without_accum.jsonl
```

**Test 2b: With accumulation**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 2 `
  --gradient-accumulation-steps 16 `
  --steps 100 `
  --log-file artifacts/test_with_accum.jsonl
```

**Analysis Script**:
```python
import json
import matplotlib.pyplot as plt

# Load logs
with open('artifacts/test_without_accum.jsonl') as f:
    without = [json.loads(line) for line in f if 'loss' in line]

with open('artifacts/test_with_accum.jsonl') as f:
    with_accum = [json.loads(line) for line in f if 'loss' in line]

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot([x['step'] for x in without], [x['loss'] for x in without], 
         label='Without accumulation (batch=2)', alpha=0.7)
plt.plot([x['step'] for x in with_accum], [x['loss'] for x in with_accum], 
         label='With accumulation (batch=2, accum=16, effective=32)', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Stability: With vs Without Gradient Accumulation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('artifacts/gradient_accumulation_comparison.png')
print("Chart saved to artifacts/gradient_accumulation_comparison.png")
```

---

### Test 3: VRAM Usage Verification
**Objective**: Confirm VRAM usage doesn't increase with accumulation

```bash
# Test A: batch=2, accum=1
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 2 `
  --gradient-accumulation-steps 1 `
  --steps 10 `
  --log-file artifacts/vram_test_no_accum.jsonl

# Test B: batch=2, accum=16 (should use same VRAM!)
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 2 `
  --gradient-accumulation-steps 16 `
  --steps 10 `
  --log-file artifacts/vram_test_with_accum.jsonl
```

**Verification**:
```powershell
# Compare peak VRAM usage
$noAccum = Get-Content artifacts/vram_test_no_accum.jsonl | ConvertFrom-Json | Where-Object {$_.peak_gb} | Select-Object -Last 1
$withAccum = Get-Content artifacts/vram_test_with_accum.jsonl | ConvertFrom-Json | Where-Object {$_.peak_gb} | Select-Object -Last 1

Write-Host "No accumulation VRAM: $($noAccum.peak_gb) GB"
Write-Host "With accumulation VRAM: $($withAccum.peak_gb) GB"
Write-Host "Difference: $(($withAccum.peak_gb - $noAccum.peak_gb)) GB (should be ~0)"
```

---

### Test 4: GUI Integration
**Objective**: Verify GUI controls work correctly

**Steps**:
1. Launch GUI: `aios gui`
2. Navigate to HRM Training panel
3. Verify new controls appear:
   - "Batch Scaling:" label
   - "Accum Steps:" dropdown
   - "→ Effective Batch: X" label
4. Test interactions:
   - Set Batch Size: 8
   - Set Accum Steps: 4
   - Verify label shows "Effective Batch: 32"
   - Change Batch Size to 16
   - Verify label updates to "Effective Batch: 64"
5. Start training and verify:
   - Training runs without errors
   - Log shows correct accumulation settings
   - Loss curve is smoother than without accumulation

---

### Test 5: Compatibility Tests

**Test 5a: With AMP**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --amp `
  --steps 20
```

**Test 5b: With Gradient Checkpointing**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --gradient-checkpointing `
  --steps 20
```

**Test 5c: With DeepSpeed ZeRO-2**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --zero-stage zero2 `
  --steps 20
```

**Test 5d: With PEFT/LoRA**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --use-peft `
  --lora-r 16 `
  --steps 20
```

**Test 5e: Parallel Independent Mode**
```bash
aios hrm-hf train-actv1 `
  --model artifacts/hf_implant/gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --batch-size 4 `
  --gradient-accumulation-steps 8 `
  --parallel-independent `
  --cuda-ids 0,1 `
  --steps 20
```

---

## ✅ Implementation Checklist

### Phase 1: Configuration Layer
- [ ] Add `gradient_accumulation_steps` field to `optimization_fields.py`
- [ ] Update `to_cli_args()` in `config_main.py`
- [ ] Test: Import TrainingConfig and verify new field exists

### Phase 2: CLI Integration
- [ ] Add `--gradient-accumulation-steps` option to `hrm_hf_cli.py`
- [ ] Test: `aios hrm-hf train-actv1 --help` shows new option

### Phase 3: Training Loop
- [ ] Modify backward/step logic in `train_epoch.py`
- [ ] Add loss scaling
- [ ] Add conditional optimizer step
- [ ] Update logging to show accumulation metrics
- [ ] Test: Run basic training with accumulation

### Phase 4: GUI Integration
- [ ] Add `gradient_accumulation_var` in `variable_setup.py`
- [ ] Add UI controls in `ui_optimizations.py`
- [ ] Add tooltip
- [ ] Add update callback for effective batch label
- [ ] Update `config_builder.py`
- [ ] Update `state_management.py` (get_state)
- [ ] Update `state_management.py` (set_state)
- [ ] Test: Launch GUI and verify controls appear

### Phase 5: Testing
- [ ] Test 1: Basic functionality
- [ ] Test 2: Loss stability comparison
- [ ] Test 3: VRAM usage verification
- [ ] Test 4: GUI integration
- [ ] Test 5a: AMP compatibility
- [ ] Test 5b: Gradient checkpointing compatibility
- [ ] Test 5c: DeepSpeed ZeRO compatibility
- [ ] Test 5d: PEFT/LoRA compatibility
- [ ] Test 5e: Parallel independent mode compatibility

### Phase 6: Documentation
- [ ] Update CLI help text (done via option definition)
- [ ] Add tooltip to GUI (done in Phase 4)
- [ ] Update `docs/guide/features/TRAINING_OPTIMIZATIONS.md` (if exists)
- [ ] Add note to CHANGELOG.md

---

## 📊 Success Criteria

### Functional Requirements
✅ Gradient accumulation configurable via CLI and GUI  
✅ Loss stability improves with accumulation enabled  
✅ VRAM usage remains constant regardless of accumulation steps  
✅ Compatible with all existing optimizations (AMP, checkpointing, ZeRO, PEFT)  
✅ Works in all training modes (single-GPU, DDP, parallel independent)  

### Performance Requirements
✅ Training slowdown < 15% for accumulation_steps ≤ 16  
✅ Loss variance reduced by ≥ 50% compared to small batch baseline  
✅ No memory leaks during extended training  

### User Experience Requirements
✅ GUI controls intuitive and self-documenting  
✅ CLI help text clear and complete  
✅ Logging shows accumulation status and effective batch size  
✅ State persistence works (save/load GUI state)  

---

## 🎯 Quick Start (After Implementation)

### Via CLI
```bash
# Basic usage
aios hrm-hf train-actv1 \
  --model artifacts/hf_implant/gpt2 \
  --dataset-file training_data/curated_datasets/my_dataset.txt \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --steps 1000

# High stability (small batch)
aios hrm-hf train-actv1 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --steps 1000
```

### Via GUI
1. Open HRM Training panel
2. Set **Batch Size**: `8`
3. Set **Accum Steps**: `4`
4. Verify: "Effective Batch: 32"
5. Click "Start Training"

---

## 📚 References

### Research
- PyTorch gradient accumulation patterns (from pytorch/examples)
- Standard deep learning practice for memory-constrained training
- User's current loss instability pattern: 7.0→8.7→7.3 (analyzed)

### Codebase Components
- Main training loop: `src/aios/cli/hrm_hf/training_logic/train_epoch.py`
- Config system: `src/aios/core/hrm_training/training_config/`
- GUI panel: `src/aios/gui/components/hrm_training_panel/`
- CLI: `src/aios/cli/hrm_hf_cli.py`

### Related Features
- Automatic Mixed Precision (AMP)
- Gradient Checkpointing
- DeepSpeed ZeRO
- PEFT/LoRA
- Parallel Independent Training

---

## 🔄 Future Enhancements

### Potential Improvements
1. **Auto-calculate accumulation**: GUI button to automatically determine optimal accumulation steps based on target effective batch size
2. **Adaptive accumulation**: Dynamically adjust accumulation based on VRAM usage
3. **Loss-based tuning**: Automatically increase accumulation if loss variance exceeds threshold
4. **Multi-GPU load balancing**: Different accumulation steps per GPU based on VRAM capacity

### Not in Scope (For Now)
- Dynamic gradient accumulation (changing during training)
- Per-layer gradient accumulation
- Gradient accumulation with different batch sizes per step

---

## 📞 Support & Troubleshooting

### Common Issues

**Q: Loss not improving with accumulation**  
A: Verify loss scaling is applied (`loss / accumulation_steps`). Check logs for `scaled_loss` in backward pass.

**Q: VRAM usage increased**  
A: Check that `batch_size` in config matches physical batch, not effective batch. Accumulation shouldn't increase VRAM.

**Q: Training slower than expected**  
A: Normal with high accumulation steps. Trade-off between speed and stability. Try reducing accumulation_steps.

**Q: Gradients not accumulating**  
A: Ensure `optimizer.zero_grad()` only called after weight update, not every batch.

**Q: GUI not showing new controls**  
A: Restart GUI. Check that all Phase 4 files were modified correctly.

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-23  
**Author**: AI-OS Development Team  
**Status**: Ready for Implementation
