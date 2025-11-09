"""Test script to verify resume checkpoint paths match."""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test configuration
bundle_dir = "artifacts/brains"
brain_name = "test-brain-001"

# Simulate setup_output_directory behavior
out_dir_path = Path(bundle_dir) / brain_name
save_dir = str(out_dir_path)

# Simulate parallel_training_v3 save paths
training_checkpoint_path = Path(save_dir) / "actv1_student.safetensors"
training_state_path = Path(save_dir) / "chunk_tracker_state.json"

# Simulate start_training.py resume check paths
brain_path = Path(bundle_dir) / brain_name
resume_checkpoint_path = brain_path / "actv1_student.safetensors"
resume_state_path = brain_path / "chunk_tracker_state.json"

print("="*60)
print("PATH VERIFICATION")
print("="*60)
print(f"\nConfiguration:")
print(f"  bundle_dir: {bundle_dir}")
print(f"  brain_name: {brain_name}")

print(f"\nTraining saves to:")
print(f"  save_dir: {save_dir}")
print(f"  checkpoint: {training_checkpoint_path}")
print(f"  state: {training_state_path}")

print(f"\nResume looks for:")
print(f"  brain_path: {brain_path}")
print(f"  checkpoint: {resume_checkpoint_path}")
print(f"  state: {resume_state_path}")

print(f"\nPath matching:")
print(f"  save_dir == brain_path: {Path(save_dir) == brain_path}")
print(f"  checkpoint paths match: {training_checkpoint_path == resume_checkpoint_path}")
print(f"  state paths match: {training_state_path == resume_state_path}")

print(f"\nFile existence check:")
print(f"  checkpoint exists: {training_checkpoint_path.exists()}")
print(f"  state exists: {training_state_path.exists()}")
print(f"  brain_path exists: {brain_path.exists()}")

if brain_path.exists():
    print(f"\nContents of {brain_path}:")
    for item in brain_path.iterdir():
        print(f"    {item.name}")

print("\n" + "="*60)
