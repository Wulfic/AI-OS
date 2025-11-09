"""Test script to verify resume checkpoint detection with legacy path."""
from pathlib import Path
import json

# Simulate the resume check logic
bundle_dir = "artifacts/brains"
brain_name = "actv1"

brain_path = Path(bundle_dir) / brain_name
checkpoint_path = brain_path / "actv1_student.safetensors"
legacy_checkpoint_path = brain_path / "final_model.safetensors"
chunk_tracker_state_path = brain_path / "chunk_tracker_state.json"

print("="*60)
print("RESUME CHECKPOINT DETECTION TEST")
print("="*60)

print(f"\nBrain: {brain_name}")
print(f"Path: {brain_path}")

print(f"\nChecking for checkpoint files:")
print(f"  actv1_student.safetensors: {checkpoint_path.exists()}")
print(f"  final_model.safetensors: {legacy_checkpoint_path.exists()}")
print(f"  chunk_tracker_state.json: {chunk_tracker_state_path.exists()}")

# Apply fallback logic (same as in start_training.py)
if not checkpoint_path.exists() and legacy_checkpoint_path.exists():
    checkpoint_path = legacy_checkpoint_path
    print(f"\n[OK] Using legacy checkpoint: {legacy_checkpoint_path}")

print(f"\nFinal checkpoint path: {checkpoint_path}")
print(f"Checkpoint exists: {checkpoint_path.exists()}")

# Check chunk tracker state
if chunk_tracker_state_path.exists():
    print(f"\n[OK] chunk_tracker_state.json found")
    try:
        with chunk_tracker_state_path.open("r", encoding="utf-8") as f:
            tracker_state = json.load(f)
        
        completed_chunks = tracker_state.get("completed_chunks", [])
        total_samples = tracker_state.get("total_samples_trained", 0)
        current_epoch = tracker_state.get("current_epoch", 0)
        
        print(f"  Completed chunks: {len(completed_chunks)}")
        print(f"  Total samples trained: {total_samples:,}")
        print(f"  Current epoch: {current_epoch}")
        
        if len(completed_chunks) > 0 or total_samples > 0:
            if checkpoint_path.exists():
                print(f"\n✓ RESUME DIALOG SHOULD BE SHOWN")
                print(f"  Reason: ChunkTracker has progress AND checkpoint exists")
            else:
                print(f"\n✗ RESUME DIALOG WILL NOT BE SHOWN")
                print(f"  Reason: ChunkTracker has progress BUT checkpoint missing")
        else:
            print(f"\n✗ RESUME DIALOG WILL NOT BE SHOWN")
            print(f"  Reason: No training progress yet")
    except Exception as e:
        print(f"\n✗ Error reading chunk tracker state: {e}")
else:
    print(f"\n✗ chunk_tracker_state.json not found")

print("\n" + "="*60)
