"""
Parallel Independent Training - Proof of Concept
Multi-GPU training without DDP by training on different dataset chunks independently.
"""

import os
import time
import threading
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn


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


@dataclass
class TrainingBlock:
    """Represents a chunk of data to be trained"""
    id: int
    start_idx: int
    end_idx: int
    status: str = "pending"  # pending, in_progress, completed
    gpu_id: Optional[int] = None
    steps_completed: int = 0
    
    def __repr__(self):
        return f"Block({self.id}, idx:{self.start_idx}-{self.end_idx}, {self.status})"


class BlockQueue:
    """Thread-safe queue for managing training blocks"""
    
    def __init__(self, blocks: List[TrainingBlock]):
        self.blocks = blocks
        self.lock = threading.Lock()
        self._completed = 0
    
    def get_next_block(self, gpu_id: int) -> Optional[TrainingBlock]:
        """Get the next available block for a GPU"""
        with self.lock:
            for block in self.blocks:
                if block.status == "pending":
                    block.status = "in_progress"
                    block.gpu_id = gpu_id
                    return block
        return None
    
    def mark_complete(self, block_id: int):
        """Mark a block as completed"""
        with self.lock:
            self.blocks[block_id].status = "completed"
            self._completed += 1
    
    def all_complete(self) -> bool:
        """Check if all blocks are completed"""
        with self.lock:
            return self._completed == len(self.blocks)
    
    def get_progress(self) -> Dict:
        """Get current progress statistics"""
        with self.lock:
            pending = sum(1 for b in self.blocks if b.status == "pending")
            in_progress = sum(1 for b in self.blocks if b.status == "in_progress")
            completed = sum(1 for b in self.blocks if b.status == "completed")
            
            return {
                "total": len(self.blocks),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
                "percent": (completed / len(self.blocks)) * 100
            }


class CheckpointMerger:
    """Merge model checkpoints from multiple GPUs"""
    
    @staticmethod
    def merge_checkpoints(checkpoint_paths: List[str]) -> Dict:
        """Average weights from multiple checkpoints"""
        if not checkpoint_paths:
            raise ValueError("No checkpoints provided")
        
        # Load all checkpoints (support both safetensors and torch)
        from safetensors.torch import load_file as load_safetensors
        checkpoints = []
        for p in checkpoint_paths:
            try:
                # Try safetensors first
                state_dict = load_safetensors(p, device='cpu')
                checkpoints.append({'model_state_dict': state_dict})
            except Exception:
                # Fallback to torch.load
                checkpoints.append(torch.load(p, map_location='cpu'))
        
        # Extract state dicts
        state_dicts = [ckpt['model_state_dict'] for ckpt in checkpoints]
        
        # Average weights
        merged_state = {}
        for key in state_dicts[0].keys():
            # Stack tensors and compute mean
            tensors = [sd[key].float() for sd in state_dicts]
            stacked = torch.stack(tensors)
            merged_state[key] = stacked.mean(dim=0)
        
        return {
            'model_state_dict': merged_state,
            'num_merged': len(checkpoints),
            'source_files': checkpoint_paths
        }
    
    @staticmethod
    def merge_with_ema(current_state: Dict, new_checkpoint: Dict, alpha: float = 0.9) -> Dict:
        """Exponential moving average merge for stability"""
        merged_state = {}
        
        for key in current_state.keys():
            merged_state[key] = (
                alpha * current_state[key] +
                (1 - alpha) * new_checkpoint[key]
            )
        
        return merged_state


class IndependentWorker:
    """Worker process that trains on assigned blocks"""
    
    def __init__(self, gpu_id: int, queue: BlockQueue, config):
        self.gpu_id = gpu_id
        self.queue = queue
        self.config = config
        self.device = torch.device(f"cuda:{gpu_id}")
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_block_data(self, block: TrainingBlock):
        """Load data for a specific block"""
        # This would load from your actual dataset
        # For now, just a placeholder
        from torch.utils.data import TensorDataset, DataLoader
        
        # Simulate loading block data
        # In reality, this would use your dataset infrastructure
        block_size = block.end_idx - block.start_idx
        dummy_data = torch.randn(block_size, self.config.max_seq_len)
        dummy_labels = torch.randint(0, 50000, (block_size,))
        
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Important: 0 for spawn mode
        )
        
        return dataloader
    
    def train_block(self, block: TrainingBlock, model: nn.Module, optimizer):
        """Train on a single block"""
        model.train()
        dataloader = self.load_block_data(block)
        
        total_loss = 0
        num_steps = 0
        
        for step, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (simplified - adapt to your model)
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_steps += 1
            
            if step % 10 == 0:
                print(f"[GPU {self.gpu_id}] Block {block.id} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, block: TrainingBlock, model: nn.Module, optimizer, loss: float):
        """Save checkpoint after completing a block"""
        checkpoint_path = self.checkpoint_dir / f"gpu{self.gpu_id}_block{block.id}.safetensors"
        
        try:
            from safetensors.torch import save_file as save_safetensors
            save_safetensors(model.state_dict(), str(checkpoint_path))
        except ImportError:
            # Fallback to torch.save if safetensors not available
            checkpoint_path = self.checkpoint_dir / f"gpu{self.gpu_id}_block{block.id}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'block_id': block.id,
                'gpu_id': self.gpu_id,
                'loss': loss,
                'start_idx': block.start_idx,
                'end_idx': block.end_idx
            }, checkpoint_path)
        
        print(f"[GPU {self.gpu_id}] Saved checkpoint: {checkpoint_path}")
    
    def run(self, model_factory, optimizer_factory):
        """Main worker loop"""
        # Create model and optimizer for this GPU
        model = model_factory().to(self.device)
        optimizer = optimizer_factory(model)
        
        print(f"[GPU {self.gpu_id}] Worker started")
        
        while block := self.queue.get_next_block(self.gpu_id):
            print(f"[GPU {self.gpu_id}] Starting Block {block.id} (idx {block.start_idx}-{block.end_idx})")
            
            try:
                # Train on this block
                loss = self.train_block(block, model, optimizer)
                
                # Save checkpoint
                self.save_checkpoint(block, model, optimizer, loss)
                
                # Mark complete
                self.queue.mark_complete(block.id)
                
                print(f"[GPU {self.gpu_id}] Completed Block {block.id} | Avg Loss: {loss:.4f}")
                
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Error in Block {block.id}: {e}")
                # Reset block status so another GPU can try
                with self.queue.lock:
                    block.status = "pending"
                    block.gpu_id = None
        
        print(f"[GPU {self.gpu_id}] Worker finished - no more blocks")


class ParallelTrainingCoordinator:
    """Coordinates parallel independent training across GPUs"""
    
    def __init__(self, config):
        self.config = config
        self.num_gpus = len(config.cuda_ids)
        
        # Create blocks
        self.blocks = self._create_blocks()
        self.queue = BlockQueue(self.blocks)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Created {len(self.blocks)} blocks for {self.num_gpus} GPUs")
        print(f"Checkpoints will be saved to: {config.checkpoint_dir}")
    
    def _create_blocks(self) -> List[TrainingBlock]:
        """Split dataset into training blocks"""
        # Determine total dataset size
        # This would come from your actual dataset
        total_samples = getattr(self.config, 'dataset_size', 10000)
        num_blocks = self.config.num_blocks
        
        block_size = total_samples // num_blocks
        blocks = []
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < num_blocks - 1 else total_samples
            blocks.append(TrainingBlock(i, start_idx, end_idx))
        
        return blocks
    
    def _worker_process(self, gpu_id: int, model_factory, optimizer_factory):
        """Worker process function"""
        try:
            worker = IndependentWorker(gpu_id, self.queue, self.config)
            worker.run(model_factory, optimizer_factory)
        except Exception as e:
            print(f"[GPU {gpu_id}] Worker crashed: {e}")
            import traceback
            traceback.print_exc()
    
    def _monitor_progress(self):
        """Monitor training progress and merge checkpoints periodically"""
        merge_interval = self.config.merge_interval  # blocks
        last_merge_count = 0
        
        while not self.queue.all_complete():
            time.sleep(5)  # Check every 5 seconds
            
            progress = self.queue.get_progress()
            print(f"\n📊 Progress: {progress['completed']}/{progress['total']} blocks "
                  f"({progress['percent']:.1f}%) | "
                  f"In Progress: {progress['in_progress']} | "
                  f"Pending: {progress['pending']}")
            
            # Check if we should merge
            if progress['completed'] >= last_merge_count + merge_interval:
                self._merge_checkpoints()
                last_merge_count = progress['completed']
    
    def _merge_checkpoints(self):
        """Merge available checkpoints"""
        checkpoint_files = sorted(Path(self.config.checkpoint_dir).glob("gpu*_block*.pt"))
        
        if len(checkpoint_files) < 2:
            return  # Need at least 2 checkpoints to merge
        
        print(f"\n🔄 Merging {len(checkpoint_files)} checkpoints...")
        
        try:
            merged = CheckpointMerger.merge_checkpoints([str(f) for f in checkpoint_files])
            
            merge_path = Path(self.config.checkpoint_dir) / f"merged_step{len(checkpoint_files)}.safetensors"
            from safetensors.torch import save_file as save_safetensors
            save_safetensors(merged['model_state_dict'], str(merge_path))
            
            print(f"✅ Merged checkpoint saved: {merge_path}\n")
        except Exception as e:
            print(f"❌ Merge failed: {e}\n")
    
    def start_training(self, model_factory, optimizer_factory):
        """Launch parallel training"""
        print("\n🚀 Starting Parallel Independent Training")
        print(f"GPUs: {self.config.cuda_ids}")
        print(f"Blocks: {len(self.blocks)}")
        print(f"Merge Interval: {self.config.merge_interval} blocks\n")
        
        # Start worker processes
        processes = []
        for gpu_id in self.config.cuda_ids:
            p = mp.Process(
                target=self._worker_process,
                args=(gpu_id, model_factory, optimizer_factory)
            )
            p.start()
            processes.append(p)
            print(f"Started worker for GPU {gpu_id}")
        
        # Monitor in main thread
        monitor_thread = threading.Thread(target=self._monitor_progress)
        monitor_thread.start()
        
        # Wait for all workers to complete
        for p in processes:
            p.join()
        
        # Wait for monitor thread
        monitor_thread.join()
        
        # Final merge
        print("\n🏁 All blocks completed - performing final merge...")
        self._merge_checkpoints()
        
        # Create final model
        self._create_final_model()
        
        print("\n✅ Parallel Independent Training Complete!")
    
    def _create_final_model(self):
        """Create final merged model from all checkpoints"""
        checkpoint_files = sorted(Path(self.config.checkpoint_dir).glob("gpu*_block*.safetensors"))
        if not checkpoint_files:
            # Fallback to .pt files
            checkpoint_files = sorted(Path(self.config.checkpoint_dir).glob("gpu*_block*.pt"))
        
        if not checkpoint_files:
            print("⚠️ No checkpoints found for final merge")
            return
        
        merged = CheckpointMerger.merge_checkpoints([str(f) for f in checkpoint_files])
        
        final_path = Path(self.config.save_dir) / "final_model.safetensors"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        from safetensors.torch import save_file as save_safetensors
        save_safetensors(merged['model_state_dict'], str(final_path))
        print(f"💾 Final model saved: {final_path}")


# Example usage
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        cuda_ids = [0, 1]
        num_blocks = 10
        merge_interval = 2
        checkpoint_dir = "artifacts/parallel_checkpoints"
        save_dir = "artifacts/final_model"
        batch_size = 4
        max_seq_len = 512
        dataset_size = 10000
    
    config = Config()
    
    # Define model and optimizer factories
    def model_factory():
        # Simple dummy model for demonstration
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50000)  # vocab size
        )
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Start training
    coordinator = ParallelTrainingCoordinator(config)
    coordinator.start_training(model_factory, optimizer_factory)
