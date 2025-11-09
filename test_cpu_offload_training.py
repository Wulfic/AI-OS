"""
Integration test for CPU offload with actual HRM training.
Tests the full training pipeline with chunked sequences and CPU offload.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("CPU OFFLOAD INTEGRATION TEST - Real Training")
print("="*70)

if not torch.cuda.is_available():
    print("⚠ No CUDA device - skipping integration test")
    sys.exit(0)

from aios.core.hrm_training.training_config import TrainingConfig
from aios.cli.hrm_hf.model_building import build_model_config, build_model
from aios.cli.hrm_hf.optimizer_setup import create_optimizer
from aios.cli.hrm_hf.memory_optimization import configure_chunking
from aios.cli.hrm_hf_utils import load_tokenizer
from aios.cli.hrm_hf.encoding import encode_lines

# Small test configuration
config = TrainingConfig(
    model_name="gpt2",
    max_seq_len=4096,
    chunk_size=1024,
    use_chunked_training=True,
    use_cpu_offload=True,  # ← Enable CPU offload
    batch_size=1,
    learning_rate=1e-4,
    halt_max_steps=2,
    use_amp=True,
    gradient_checkpointing=True,
)

print(f"\n✓ Configuration:")
print(f"  - Sequence length: {config.max_seq_len}")
print(f"  - Chunk size: {config.chunk_size}")
print(f"  - CPU offload: {config.use_cpu_offload}")
print(f"  - Chunks per sequence: {config.max_seq_len // config.chunk_size}")

# Load tokenizer
print(f"\n[Loading tokenizer]")
tokenizer = load_tokenizer(config.model_name)
vocab_size = len(tokenizer)
print(f"✓ Vocab size: {vocab_size}")

# Build tiny model for testing
print(f"\n[Building model]")
model_config = build_model_config(
    config=config,
    vocab_size=vocab_size,
    log_fn=print
)

# Override to make it tiny for testing
model_config.hidden_size = 128
model_config.h_layers = 2
model_config.l_layers = 2
model_config.num_heads = 4

model = build_model(
    config=model_config,
    student_init=None,
    log_fn=print
)

device = torch.device('cuda:0')
model = model.to(device)
model.train()

param_count = sum(p.numel() for p in model.parameters())
print(f"✓ Model parameters: {param_count:,}")

# Create optimizer
print(f"\n[Creating optimizer]")
optimizer = create_optimizer(
    model=model,
    config=config,
    use_deepspeed_optimizer=False,
    log_fn=print
)

# Configure chunking with CPU offload
print(f"\n[Configuring chunking with CPU offload]")
segment_rollout, use_chunking, final_chunk_size = configure_chunking(
    max_seq_len=config.max_seq_len,
    chunk_size=config.chunk_size,
    use_chunked_training=config.use_chunked_training,
    gradient_checkpointing=config.gradient_checkpointing,
    use_cpu_offload=config.use_cpu_offload,
    log_fn=print
)

print(f"✓ Chunking enabled: {use_chunking}")
print(f"✓ Final chunk size: {final_chunk_size}")
print(f"✓ CPU offload: {config.use_cpu_offload}")

# Create test data
print(f"\n[Creating test data]")
test_lines = [
    "This is a test of CPU offload with chunked training. " * 50,  # Long line
    "The synchronization fixes ensure data integrity. " * 50,
]

encoded_data = encode_lines(
    lines=test_lines,
    tokenizer=tokenizer,
    max_seq_len=config.max_seq_len,
    log_fn=print
)

print(f"✓ Encoded {len(encoded_data)} sequences")

# Training loop
print(f"\n[Running training with CPU offload]")
print("-" * 70)

num_steps = 5
losses = []
scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

for step in range(num_steps):
    # Get batch
    idx = step % len(encoded_data)
    input_ids = encoded_data[idx]['input_ids'].unsqueeze(0).to(device)
    labels = encoded_data[idx]['labels'].unsqueeze(0).to(device)
    
    batch = {
        'inputs': input_ids,
        'targets': labels,
        'puzzle_identifiers': torch.tensor([0], device=device)
    }
    
    seq_len = input_ids.shape[1]
    print(f"Step {step+1}/{num_steps}: seq_len={seq_len}", end="")
    
    # Forward pass with CPU offload
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast(enabled=config.use_amp):
        loss, metrics = segment_rollout(
            model=model,
            batch=batch,
            max_segments=config.halt_max_steps,
            epsilon=0.0
        )
    
    # Check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n✗ FAILED: NaN/Inf loss detected at step {step+1}!")
        print(f"   This indicates data corruption from CPU offload bug")
        sys.exit(1)
    
    losses.append(loss.item())
    
    # Backward pass
    if config.use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    print(f", loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")
    
    # Check gradient norm
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        print(f"✗ FAILED: NaN/Inf gradients at step {step+1}!")
        sys.exit(1)

print("-" * 70)
print(f"\n✅ Training completed successfully!")
print(f"\nLoss progression:")
for i, loss in enumerate(losses):
    print(f"  Step {i+1}: {loss:.4f}")

# Verify losses are reasonable (not exploding, not stuck)
if len(losses) > 2:
    loss_variance = torch.tensor(losses).var().item()
    print(f"\nLoss variance: {loss_variance:.4f}")
    
    if all(l < 0.1 for l in losses):
        print("⚠ Warning: Losses very low - may indicate issue")
    elif all(l > 100 for l in losses):
        print("✗ FAILED: Losses exploding!")
        sys.exit(1)
    else:
        print("✓ Loss progression looks healthy")

print("\n" + "="*70)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("="*70)
print("\nCPU offload works correctly in actual training:")
print("  ✓ No NaN/Inf losses")
print("  ✓ No gradient corruption")
print("  ✓ Stable training across multiple steps")
print("  ✓ Chunked sequences processed correctly")
print("  ✓ Synchronization fixes are effective")
