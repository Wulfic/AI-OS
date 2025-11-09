"""Diagnose the chunk assignment logic based on the screenshot."""

def simulate_training_cycles():
    """Simulate what should happen in training based on screenshot settings."""
    
    # Settings from screenshot
    dataset_chunk_size = 1024
    samples_per_block = 100000
    batch_size = 2
    steps_per_chunk = 1024  # User sets steps = chunk_size for full coverage
    
    # Calculate chunks per block
    chunks_per_block = (samples_per_block + dataset_chunk_size - 1) // dataset_chunk_size
    print(f"Configuration:")
    print(f"  Dataset chunk size: {dataset_chunk_size}")
    print(f"  Samples per block: {samples_per_block}")
    print(f"  Steps per chunk: {steps_per_chunk}")
    print(f"  Batch size: {batch_size}")
    print(f"  Chunks per block: {chunks_per_block}")
    print()
    
    # Simulate training state
    current_block_id = 0
    samples_this_block = 0
    cycle = 0
    
    # Simulate what happens in iterate mode
    print("Simulating iterate mode training:")
    print("=" * 60)
    
    for iteration in range(10):  # Simulate 10 iterations
        print(f"\nIteration {iteration} (Cycle {cycle}):")
        print(f"  current_block_id = {current_block_id}")
        print(f"  samples_this_block = {samples_this_block}")
        
        # BEFORE training: Calculate which chunk we're about to train
        chunk_id_in_block = (samples_this_block // dataset_chunk_size) % chunks_per_block
        print(f"  → Chunk to train: Block {current_block_id}, Chunk {chunk_id_in_block}")
        
        # Load data for this chunk
        chunk_samples = dataset_chunk_size
        print(f"  → Loading {chunk_samples} samples")
        
        # Train on the chunk
        print(f"  → Training {steps_per_chunk} steps...")
        
        # AFTER training: Update samples_this_block
        samples_this_block += chunk_samples
        print(f"  → samples_this_block after training: {samples_this_block}")
        
        # Check if block is complete
        block_complete = samples_this_block >= samples_per_block
        if block_complete:
            print(f"  ✅ Block {current_block_id} COMPLETE!")
            current_block_id += 1
            samples_this_block = 0
            print(f"  → Advancing to Block {current_block_id}")
        
        cycle += 1
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("Based on the logic above, chunks should progress correctly: 0, 1, 2, ...")
    print()
    print("From the screenshot, we see:")
    print("  GPU 0: Chunk 14, then Chunk 14 again")
    print("  GPU 1: Chunk 13, then Chunk 13 again")
    print()
    print("This suggests ONE of these issues:")
    print("  1. claim_chunk() isn't properly marking chunks as in-progress")
    print("  2. samples_this_block isn't being persisted in config")
    print("  3. The chunk tracker state isn't being loaded correctly")
    print("  4. Linear dataset mode isn't enabled")


def test_chunk_tracker_logic():
    """Test the chunk tracker claim/complete logic."""
    
    print("\n\n" + "=" * 60)
    print("Testing ChunkTracker Logic")
    print("=" * 60)
    
    class SimpleChunkTracker:
        def __init__(self):
            self.completed_chunks = {}
            self.in_progress_chunks = {}
        
        def claim_chunk(self, block_id, chunk_id, gpu_id):
            chunk_key = (block_id, chunk_id)
            # Check if already completed
            if chunk_key in self.completed_chunks:
                print(f"  ❌ Chunk ({block_id}, {chunk_id}) already completed")
                return False
            # NOTE: The actual code doesn't add to in_progress here!
            print(f"  ✅ Chunk ({block_id}, {chunk_id}) claimed by GPU {gpu_id}")
            return True
        
        def mark_chunk_complete(self, block_id, chunk_id, gpu_id, step, samples):
            chunk_key = (block_id, chunk_id)
            self.completed_chunks[chunk_key] = {"gpu_id": gpu_id, "step": step}
            print(f"  ✅ Chunk ({block_id}, {chunk_id}) marked complete")
    
    tracker = SimpleChunkTracker()
    
    # Simulate what happens in iterate mode
    print("\nSimulating 5 iterations:")
    for i in range(5):
        print(f"\nIteration {i}:")
        # This simulates the bug - samples_this_block stays at 0!
        samples_this_block = 0  # BUG: Not actually updating!
        chunk_id = (samples_this_block // 1024) % 98
        print(f"  samples_this_block = {samples_this_block}")
        print(f"  chunk_id = {chunk_id}")
        
        # Try to claim
        can_claim = tracker.claim_chunk(0, chunk_id, 0)
        if can_claim:
            print(f"  Training chunk {chunk_id}...")
            tracker.mark_chunk_complete(0, chunk_id, 0, i * 1024, 1024)
        else:
            print(f"  Skipping chunk {chunk_id} (already trained)")
    
    print("\n" + "=" * 60)
    print("Result: Chunk 0 gets trained once, then skipped 4 times")
    print("This matches the screenshot behavior!")


if __name__ == "__main__":
    simulate_training_cycles()
    test_chunk_tracker_logic()
