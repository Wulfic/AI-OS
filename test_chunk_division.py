"""Test script to verify chunk division is working correctly."""

# Simulating the _load_chunk logic for HuggingFace datasets

def test_chunk_division():
    """Test that chunks are being divided correctly from a block."""
    
    # Simulate a cached block with 100,000 samples
    cached_block = [f"Sample {i}" for i in range(100000)]
    
    # Parameters
    chunk_size = 4000
    samples_per_block = 100000
    
    # Calculate expected chunks
    expected_chunks = (samples_per_block + chunk_size - 1) // chunk_size
    print(f"Block has {len(cached_block)} samples")
    print(f"Chunk size: {chunk_size}")
    print(f"Expected chunks: {expected_chunks}")
    print()
    
    # Test chunk extraction (simulating BlockManager._load_chunk for HF)
    all_chunks = []
    for chunk_id in range(expected_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(cached_block))
        chunk_samples = cached_block[start_idx:end_idx]
        
        all_chunks.append(chunk_samples)
        
        print(f"Chunk {chunk_id}: start_idx={start_idx}, end_idx={end_idx}, "
              f"samples={len(chunk_samples)}, first='{chunk_samples[0]}', last='{chunk_samples[-1]}'")
        
        # Verify no overlap with previous chunk
        if chunk_id > 0:
            prev_chunk = all_chunks[chunk_id - 1]
            # Check last sample of previous chunk != first sample of current chunk
            if prev_chunk[-1] == chunk_samples[0]:
                print(f"  ⚠️  OVERLAP DETECTED: Previous chunk's last sample equals this chunk's first sample!")
    
    # Verify coverage
    total_samples_in_chunks = sum(len(c) for c in all_chunks)
    print()
    print(f"Total samples in all chunks: {total_samples_in_chunks}")
    print(f"Total samples in block: {len(cached_block)}")
    
    if total_samples_in_chunks == len(cached_block):
        print("✅ All samples accounted for - no duplication or gaps")
    else:
        print(f"❌ MISMATCH: {total_samples_in_chunks} != {len(cached_block)}")
    
    # Check for any duplicate samples across chunks
    all_samples_from_chunks = []
    for chunk in all_chunks:
        all_samples_from_chunks.extend(chunk)
    
    if len(all_samples_from_chunks) != len(set(all_samples_from_chunks)):
        print("❌ DUPLICATES FOUND across chunks!")
    else:
        print("✅ No duplicates found")
    
    # Verify order is preserved
    if all_samples_from_chunks == cached_block:
        print("✅ Sample order preserved")
    else:
        print("❌ Sample order NOT preserved")


def test_local_file_chunk_division():
    """Test chunk division for local files."""
    
    print("\n" + "="*60)
    print("Testing LOCAL FILE chunk division")
    print("="*60 + "\n")
    
    # Simulate all local samples (e.g., small dataset)
    all_local_samples = [f"LocalSample {i}" for i in range(25000)]
    
    # Parameters
    chunk_size = 4000
    samples_per_block = 100000
    block_id = 0
    
    print(f"Total local samples: {len(all_local_samples)}")
    print(f"Samples per block: {samples_per_block}")
    print(f"Chunk size: {chunk_size}")
    print()
    
    # Simulating _load_chunk for local files (non-block-structured)
    # This is the PROBLEMATIC code path
    all_chunks = []
    max_chunks = (len(all_local_samples) + chunk_size - 1) // chunk_size
    
    for chunk_id in range(max_chunks):
        # THIS IS THE BUG: It uses (block_id * samples_per_block) even for small files!
        start_idx = (block_id * samples_per_block) + (chunk_id * chunk_size)
        end_idx = min(start_idx + chunk_size, len(all_local_samples))
        
        if start_idx >= len(all_local_samples):
            print(f"Chunk {chunk_id}: start_idx={start_idx} >= {len(all_local_samples)} - BEYOND END")
            break
        
        chunk_samples = all_local_samples[start_idx:end_idx]
        all_chunks.append(chunk_samples)
        
        print(f"Chunk {chunk_id}: start_idx={start_idx}, end_idx={end_idx}, "
              f"samples={len(chunk_samples)}, first='{chunk_samples[0]}', last='{chunk_samples[-1]}'")
    
    # Verify coverage
    total_samples_in_chunks = sum(len(c) for c in all_chunks)
    print()
    print(f"Total samples in all chunks: {total_samples_in_chunks}")
    print(f"Total samples available: {len(all_local_samples)}")
    
    if total_samples_in_chunks == len(all_local_samples):
        print("✅ All samples accounted for")
    else:
        print(f"⚠️  Only used {total_samples_in_chunks}/{len(all_local_samples)} samples")
        print(f"   Missing: {len(all_local_samples) - total_samples_in_chunks} samples")


def test_block_structured_local():
    """Test chunk division for block-structured local datasets."""
    
    print("\n" + "="*60)
    print("Testing BLOCK-STRUCTURED LOCAL chunk division")
    print("="*60 + "\n")
    
    # Simulate a block from a block-structured dataset
    block_samples = [f"BlockSample {i}" for i in range(100000)]
    
    # Parameters
    chunk_size = 4000
    block_id = 0
    
    print(f"Block {block_id} samples: {len(block_samples)}")
    print(f"Chunk size: {chunk_size}")
    print()
    
    # Simulating _load_chunk for block-structured local files
    all_chunks = []
    max_chunks = (len(block_samples) + chunk_size - 1) // chunk_size
    
    for chunk_id in range(max_chunks):
        # This is correct - it uses only the block's samples
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(block_samples))
        
        if start_idx >= len(block_samples):
            break
        
        chunk_samples = block_samples[start_idx:end_idx]
        all_chunks.append(chunk_samples)
        
        if chunk_id < 3 or chunk_id >= max_chunks - 2:
            print(f"Chunk {chunk_id}: start_idx={start_idx}, end_idx={end_idx}, "
                  f"samples={len(chunk_samples)}")
    
    # Verify coverage
    total_samples_in_chunks = sum(len(c) for c in all_chunks)
    print()
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total samples in all chunks: {total_samples_in_chunks}")
    print(f"Total samples in block: {len(block_samples)}")
    
    if total_samples_in_chunks == len(block_samples):
        print("✅ All samples accounted for")
    else:
        print(f"❌ MISMATCH: {total_samples_in_chunks} != {len(block_samples)}")


if __name__ == "__main__":
    print("="*60)
    print("Testing HUGGINGFACE chunk division")
    print("="*60 + "\n")
    test_chunk_division()
    
    test_local_file_chunk_division()
    test_block_structured_local()
