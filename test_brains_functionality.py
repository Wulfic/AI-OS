"""Test script to verify brains panel functionality."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aios.core.brains import BrainRegistry

# Test brain registry
store_dir = "artifacts/brains"
print(f"Testing brain registry in: {store_dir}")
print("=" * 60)

reg = BrainRegistry(store_dir=store_dir)

# Check if masters.json and pinned.json exist
masters_path = os.path.join(store_dir, "masters.json")
pinned_path = os.path.join(store_dir, "pinned.json")

print(f"\nMasters file: {masters_path}")
print(f"Exists: {os.path.exists(masters_path)}")
if os.path.exists(masters_path):
    import json
    with open(masters_path) as f:
        print(f"Content: {json.load(f)}")

print(f"\nPinned file: {pinned_path}")
print(f"Exists: {os.path.exists(pinned_path)}")
if os.path.exists(pinned_path):
    import json
    with open(pinned_path) as f:
        print(f"Content: {json.load(f)}")

# Test loading
print("\n" + "=" * 60)
print("Loading pinned and masters...")
loaded_pinned = reg.load_pinned()
loaded_masters = reg.load_masters()
print(f"Loaded pinned: {loaded_pinned}, Set: {reg.pinned}")
print(f"Loaded masters: {loaded_masters}, Set: {reg.masters}")

# Test listing brains
print("\n" + "=" * 60)
print("Listing all brains in store...")
actv1_dir = os.path.join(store_dir, "actv1")
if os.path.exists(actv1_dir):
    files = [f for f in os.listdir(actv1_dir) if f.endswith('.npz')]
    print(f"Found {len(files)} brain files:")
    for f in files[:5]:  # Show first 5
        name = f[:-4]  # Remove .npz
        is_pinned = name in reg.pinned
        is_master = name in reg.masters
        print(f"  - {name}: pinned={is_pinned}, master={is_master}")
else:
    print(f"Directory not found: {actv1_dir}")

# Test pin/unpin
print("\n" + "=" * 60)
print("Testing pin/unpin functionality...")
test_name = "test_brain_123"
print(f"Pinning '{test_name}'...")
reg.pin(test_name)
print(f"Pinned set after pin: {reg.pinned}")
print(f"Masters set: {reg.masters}")

# Check if file was written
if os.path.exists(pinned_path):
    import json
    with open(pinned_path) as f:
        saved = json.load(f)
        print(f"Saved to file: {saved}")
        print(f"Test brain in saved list: {test_name in saved}")

# Test mark_master
print("\n" + "=" * 60)
print(f"Testing mark_master functionality...")
print(f"Marking '{test_name}' as master...")
reg.mark_master(test_name)
print(f"Masters set after mark: {reg.masters}")
print(f"Pinned set after mark: {reg.pinned}")

# Check if file was written
if os.path.exists(masters_path):
    import json
    with open(masters_path) as f:
        saved = json.load(f)
        print(f"Saved to masters file: {saved}")
        print(f"Test brain in masters list: {test_name in saved}")

# Test unmark_master
print("\n" + "=" * 60)
print("Testing unmark_master functionality...")
if hasattr(reg, 'unmark_master'):
    reg.unmark_master(test_name)
    print(f"Masters set after unmark: {reg.masters}")
    if os.path.exists(masters_path):
        import json
        with open(masters_path) as f:
            print(f"Saved to masters file: {json.load(f)}")
else:
    print("unmark_master method not found!")

print("\n" + "=" * 60)
print("Test complete!")
