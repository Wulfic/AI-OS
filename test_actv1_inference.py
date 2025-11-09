"""Test script for ACTv1 brain inference."""

from aios.core.brains import ACTv1Brain

# Create brain instance
print("Creating ACTv1Brain instance...")
brain = ACTv1Brain(
    name='English-v1',
    modalities=['text'],
    checkpoint_path='artifacts/brains/actv1/English-v1/actv1_student.safetensors'
)

# Test inference
print("\nTesting inference with 'Hello'...")
try:
    result = brain.run({'payload': 'Hello'})
    print(f"\nResult: {result}")
except Exception as e:
    import traceback
    print(f"\nError: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
