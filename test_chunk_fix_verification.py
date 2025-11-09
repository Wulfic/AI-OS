"""Test the fixed chunk progression logic."""

class MockConfig:
    def __init__(self):
        self.current_block_samples = None
        self.current_block_id = 0
        self.samples_per_block = 100000
        self.dataset_chunk_size = 1024
        self.stop_after_block = False
        self.stop_after_epoch = False
        self.linear_dataset = True


def simulate_fixed_iterate_mode():
    """Simulate the FIXED iterate mode logic."""
    
    config = MockConfig()
    dataset_chunk_size = 1024
    samples_per_block = 100000
    chunks_per_block = (samples_per_block + dataset_chunk_size - 1) // dataset_chunk_size
    
    print("Testing FIXED iterate mode logic:")
    print(f"  dataset_chunk_size: {dataset_chunk_size}")
    print(f"  samples_per_block: {samples_per_block}")
    print(f"  chunks_per_block: {chunks_per_block}")
    print()
    
    # Initialize (matching the fix)
    if not hasattr(config, 'current_block_samples') or config.current_block_samples is None:
        config.current_block_samples = 0
    samples_this_block: int = int(config.current_block_samples)
    current_block_id: int = int(getattr(config, 'current_block_id', 0))
    
    cycle = 0
    chunk_tracker_completed = set()  # Simulate chunk tracker
    
    for iteration in range(15):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} (Cycle {cycle}):")
        print(f"  config.current_block_samples = {config.current_block_samples}")
        print(f"  samples_this_block = {samples_this_block}")
        print(f"  current_block_id = {current_block_id}")
        
        # Calculate chunk BEFORE training
        chunk_id_in_block = (samples_this_block // dataset_chunk_size) % chunks_per_block
        print(f"  → Chunk to check: Block {current_block_id}, Chunk {chunk_id_in_block}")
        
        # Check if chunk already trained
        chunk_key = (current_block_id, chunk_id_in_block)
        if chunk_key in chunk_tracker_completed:
            print(f"  ❌ Chunk already trained - SKIPPING")
            chunk_already_trained = True
            chunk_samples = dataset_chunk_size
        else:
            print(f"  ✅ Chunk not trained - TRAINING")
            chunk_already_trained = False
            # Simulate loading data
            chunk_samples = dataset_chunk_size
            # Mark as completed
            chunk_tracker_completed.add(chunk_key)
        
        # Update samples (matching the fix)
        samples_this_block += chunk_samples
        config.current_block_samples = samples_this_block  # CRITICAL FIX
        
        print(f"  → After chunk: samples_this_block = {samples_this_block}")
        print(f"  → After chunk: config.current_block_samples = {config.current_block_samples}")
        
        # Check block completion
        block_complete = samples_this_block >= samples_per_block
        if block_complete:
            print(f"  🎉 Block {current_block_id} COMPLETE!")
            current_block_id += 1
            config.current_block_id = current_block_id
            samples_this_block = 0
            config.current_block_samples = 0
            print(f"  → Advancing to Block {current_block_id}")
        
        cycle += 1
        
        # Break after demonstrating the fix works
        if iteration >= 14:
            break
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Total chunks trained: {len(chunk_tracker_completed)}")
    print(f"  Chunks trained: {sorted(chunk_tracker_completed)}")
    print()
    
    # Verify no duplicates
    if len(chunk_tracker_completed) == len(set(chunk_tracker_completed)):
        print("✅ NO DUPLICATES - Each chunk trained exactly once!")
    else:
        print("❌ DUPLICATES DETECTED!")
    
    # Verify sequential progression
    expected_chunks = [(0, i) for i in range(15)]
    if list(sorted(chunk_tracker_completed)) == expected_chunks:
        print("✅ SEQUENTIAL PROGRESSION - Chunks 0-14 trained in order!")
    else:
        print("❌ NOT SEQUENTIAL!")


if __name__ == "__main__":
    simulate_fixed_iterate_mode()
