"""
Parallel Independent Training - Windows-Compatible Version
Multi-GPU training without DDP by training on different dataset chunks independently.
Uses simpler architecture optimized for Windows spawn multiprocessing.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import multiprocessing as mp


@dataclass
class ParallelConfig:
    """Configuration for parallel independent training"""
    cuda_ids: List[int]
    num_blocks: int
    merge_interval: int
    checkpoint_dir: str
    save_dir: str
    batch_size: int
    max_seq_len: int
    dataset_size: int
    learning_rate: float = 1e-4
    steps_per_block: Optional[int] = None


def load_block_data(block_id: int, start_idx: int, end_idx: int, config: ParallelConfig):
    """Load data for a specific block - placeholder for actual data loading"""
    # This will be replaced with actual dataset loading
    dataset_size = end_idx - start_idx
    for idx in range(dataset_size):
        # Yield dummy data - replace with actual data loader
        input_ids = torch.randint(0, 50000, (config.batch_size, config.max_seq_len))
        labels = input_ids.clone()
        yield {'input_ids': input_ids, 'labels': labels}


def worker_train_block(
    gpu_id: int,
    block_id: int,
    start_idx: int,
    end_idx: int,
    config: ParallelConfig,
    model_factory: Callable,
    optimizer_factory: Callable
):
    """Train a single block on a specific GPU"""
    
    # Set device
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    print(f"🔥 GPU {gpu_id}: Starting Block {block_id} (idx {start_idx}-{end_idx})")
    
    # Create model and optimizer
    model = model_factory()
    model = model.to(device)
    optimizer = optimizer_factory(model)
    
    # Training loop
    model.train()
    total_loss = 0.0
    step_count = 0
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_data in load_block_data(block_id, start_idx, end_idx, config):
        input_ids = batch_data['input_ids'].to(device)
        labels = batch_data['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (simple example - replace with actual model forward)
        outputs = model(input_ids.float())
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step_count += 1
        
        if step_count % 10 == 0:
            avg_loss = total_loss / step_count
            print(f"  GPU {gpu_id} Block {block_id}: Step {step_count}, Loss: {avg_loss:.4f}")
        
        # Early stopping for testing
        if config.steps_per_block and step_count >= config.steps_per_block:
            break
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"gpu{gpu_id}_block{block_id}.safetensors"
    try:
        from safetensors.torch import save_file as save_safetensors
        save_safetensors(model.state_dict(), str(checkpoint_path))
    except ImportError:
        checkpoint_path = checkpoint_dir / f"gpu{gpu_id}_block{block_id}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'block_id': block_id,
            'gpu_id': gpu_id,
            'steps_completed': step_count,
            'final_loss': total_loss / step_count
        }, checkpoint_path)
    
    print(f"✅ GPU {gpu_id}: Completed Block {block_id} ({step_count} steps, avg loss: {total_loss/step_count:.4f})")
    print(f"   Checkpoint saved: {checkpoint_path}")
    
    return str(checkpoint_path)


def merge_checkpoints(checkpoint_paths: List[str], output_path: str):
    """Merge multiple checkpoints by averaging weights"""
    print(f"\n🔄 Merging {len(checkpoint_paths)} checkpoints...")
    
    # Load all checkpoints (support both formats)
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    checkpoints = []
    for p in checkpoint_paths:
        try:
            state_dict = load_safetensors(p, device='cpu')
            checkpoints.append({'model_state_dict': state_dict})
        except Exception:
            checkpoints.append(torch.load(p, map_location='cpu'))
    
    # Extract state dicts
    state_dicts = [ckpt['model_state_dict'] for ckpt in checkpoints]
    
    # Average weights
    merged_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)
        merged_state[key] = stacked.mean(dim=0)
    
    # Save merged checkpoint
    save_safetensors(merged_state, output_path)
    
    print(f"✅ Merged checkpoint saved: {output_path}")
    return output_path


def train_parallel_independent(
    config: ParallelConfig,
    model_factory: Callable,
    optimizer_factory: Callable
):
    """
    Main function to coordinate parallel independent training.
    
    Args:
        config: Training configuration
        model_factory: Function that returns a new model instance
        optimizer_factory: Function that takes a model and returns optimizer
    """
    
    print("="*60)
    print("🚀 Starting Parallel Independent Training")
    print(f"GPUs: {config.cuda_ids}")
    print(f"Blocks: {config.num_blocks}")
    print(f"Merge Interval: {config.merge_interval} blocks")
    print("="*60)
    
    # Create block assignments
    block_size = config.dataset_size // config.num_blocks
    blocks = []
    for i in range(config.num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < config.num_blocks - 1 else config.dataset_size
        blocks.append((i, start_idx, end_idx))
    
    print(f"\nCreated {len(blocks)} blocks of ~{block_size} samples each")
    print(f"Checkpoints will be saved to: {config.checkpoint_dir}\n")
    
    # Train blocks
    completed_checkpoints = []
    num_gpus = len(config.cuda_ids)
    
    for block_start in range(0, len(blocks), num_gpus):
        # Process batch of blocks (one per GPU)
        batch_blocks = blocks[block_start:block_start + num_gpus]
        
        # Use multiprocessing to train blocks in parallel
        with mp.Pool(processes=len(batch_blocks)) as pool:
            results = []
            for idx, (block_id, start_idx, end_idx) in enumerate(batch_blocks):
                gpu_id = config.cuda_ids[idx]
                result = pool.apply_async(
                    worker_train_block,
                    args=(gpu_id, block_id, start_idx, end_idx, config, model_factory, optimizer_factory)
                )
                results.append(result)
            
            # Wait for all to complete and collect checkpoints
            for result in results:
                checkpoint_path = result.get()  # This blocks until complete
                completed_checkpoints.append(checkpoint_path)
        
        # Merge if we've hit the merge interval
        if len(completed_checkpoints) >= config.merge_interval:
            save_dir = Path(config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            merge_path = save_dir / f"merged_after_block_{block_start + len(batch_blocks) - 1}.pt"
            merge_checkpoints(completed_checkpoints, str(merge_path))
            
            # Clear completed checkpoints list
            completed_checkpoints = []
    
    # Final merge if there are remaining checkpoints
    if completed_checkpoints:
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        final_path = save_dir / "final_model.pt"
        merge_checkpoints(completed_checkpoints, str(final_path))
    else:
        # No remaining checkpoints, copy the last merge as final
        last_merge = list(Path(config.save_dir).glob("merged_after_block_*.pt"))[-1]
        final_path = Path(config.save_dir) / "final_model.pt"
        import shutil
        shutil.copy(last_merge, final_path)
        print(f"\n✅ Final model: {final_path}")
    
    print("\n" + "="*60)
    print("✅ Parallel Independent Training Complete!")
    print(f"Final model saved: {final_path}")
    print("="*60)


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    
    # Simple test model
    def model_factory():
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 100)
        )
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test configuration
    config = ParallelConfig(
        cuda_ids=[0, 1] if torch.cuda.device_count() >= 2 else [0],
        num_blocks=4,
        merge_interval=2,
        checkpoint_dir="artifacts/test_parallel_v2",
        save_dir="artifacts/test_final_v2",
        batch_size=2,
        max_seq_len=128,
        dataset_size=400,
        steps_per_block=5  # Small for testing
    )
    
    train_parallel_independent(config, model_factory, optimizer_factory)
