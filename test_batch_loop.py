"""Test to verify batch processing doesn't duplicate data."""

def test_batch_processing():
    """Test that batching through a chunk doesn't repeat data."""
    
    # Simulate a chunk
    chunk_samples = [f"ChunkSample {i}" for i in range(4000)]
    batch_size = 2
    
    print(f"Chunk samples: {len(chunk_samples)}")
    print(f"Batch size: {batch_size}")
    print(f"Expected batches: {(len(chunk_samples) + batch_size - 1) // batch_size}")
    print()
    
    all_processed_samples = []
    batch_count = 0
    
    # This is the loop from parallel_training_v3.py
    for batch_start in range(0, len(chunk_samples), batch_size):
        batch_end = min(batch_start + batch_size, len(chunk_samples))
        batch_lines = chunk_samples[batch_start:batch_end]
        
        if not batch_lines:
            continue
        
        all_processed_samples.extend(batch_lines)
        batch_count += 1
        
        # Print first few and last few batches
        if batch_count <= 3 or batch_count >= 1998:
            print(f"Batch {batch_count}: start={batch_start}, end={batch_end}, "
                  f"samples={len(batch_lines)}, first='{batch_lines[0]}', last='{batch_lines[-1]}'")
    
    print()
    print(f"Total batches: {batch_count}")
    print(f"Total samples processed: {len(all_processed_samples)}")
    print(f"Original chunk size: {len(chunk_samples)}")
    
    if len(all_processed_samples) == len(chunk_samples):
        print("✅ All samples processed exactly once")
    else:
        print(f"❌ MISMATCH: {len(all_processed_samples)} != {len(chunk_samples)}")
    
    # Check for duplicates
    if len(all_processed_samples) != len(set(all_processed_samples)):
        print("❌ DUPLICATES FOUND in processed samples!")
    else:
        print("✅ No duplicates")
    
    # Check order
    if all_processed_samples == chunk_samples:
        print("✅ Order preserved")
    else:
        print("❌ Order NOT preserved")


def test_batch_with_step_limit():
    """Test batch processing with step limit (config.steps)."""
    
    print("\n" + "="*60)
    print("Testing with STEP LIMIT")
    print("="*60 + "\n")
    
    # Simulate a chunk
    chunk_samples = [f"ChunkSample {i}" for i in range(4000)]
    batch_size = 2
    max_steps_this_chunk = 100  # config.steps
    
    print(f"Chunk samples: {len(chunk_samples)}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps per chunk: {max_steps_this_chunk}")
    print(f"Steps needed for full chunk: {(len(chunk_samples) + batch_size - 1) // batch_size}")
    print()
    
    all_processed_samples = []
    step_in_chunk = 0
    
    # This is the loop from parallel_training_v3.py
    for batch_start in range(0, len(chunk_samples), batch_size):
        # Stop if we've reached the per-chunk step limit
        if step_in_chunk >= max_steps_this_chunk:
            print(f"Reached per-chunk step limit ({max_steps_this_chunk}), stopping")
            break
        
        batch_end = min(batch_start + batch_size, len(chunk_samples))
        batch_lines = chunk_samples[batch_start:batch_end]
        
        if not batch_lines:
            continue
        
        all_processed_samples.extend(batch_lines)
        step_in_chunk += 1
        
        # Print first few batches
        if step_in_chunk <= 3 or step_in_chunk >= 99:
            print(f"Step {step_in_chunk}: batch_start={batch_start}, batch_end={batch_end}, "
                  f"samples={len(batch_lines)}, first='{batch_lines[0]}', last='{batch_lines[-1]}'")
    
    print()
    print(f"Total steps: {step_in_chunk}")
    print(f"Total samples processed: {len(all_processed_samples)}")
    print(f"Samples NOT processed: {len(chunk_samples) - len(all_processed_samples)}")
    print()
    
    # In this case, we expect to only process the first max_steps_this_chunk * batch_size samples
    expected_samples = max_steps_this_chunk * batch_size
    if len(all_processed_samples) == expected_samples:
        print(f"✅ Processed exactly {expected_samples} samples as expected")
    else:
        print(f"⚠️  Expected {expected_samples} samples, got {len(all_processed_samples)}")
    
    # The key question: When this chunk is marked complete, will the next chunk start 
    # at the right position, or will it repeat these samples?
    print()
    print("🔍 CRITICAL QUESTION:")
    print(f"   Samples processed: 0-{len(all_processed_samples)-1}")
    print(f"   Samples NOT processed: {len(all_processed_samples)}-{len(chunk_samples)-1}")
    print()
    print("   When we move to the next chunk (chunk_id + 1), will it load:")
    print(f"   A) Samples {len(chunk_samples)}-{len(chunk_samples)*2-1} ✅ (correct)")
    print(f"   B) Samples 0-{len(chunk_samples)-1} ❌ (duplicate!)")


if __name__ == "__main__":
    test_batch_processing()
    test_batch_with_step_limit()
