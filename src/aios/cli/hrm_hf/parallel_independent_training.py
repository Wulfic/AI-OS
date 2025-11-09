"""
Parallel Independent Training for ACTV1
Trains different data blocks on different GPUs sequentially, then merges checkpoints.

This bypasses Windows DDP limitations by avoiding distributed training entirely.
Each GPU trains independently on its assigned data blocks, and checkpoints are merged
using weight averaging.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import warnings

# CRITICAL: Set HF_HOME before any transformers imports to suppress deprecation warning
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
if "HF_HOME" not in os.environ:
    # Use default cache location
    cache_home = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = cache_home

# Filter any remaining warnings about TRANSFORMERS_CACHE
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)

import torch
from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

# Reuse existing infrastructure
from ..hrm_hf_utils import load_tokenizer as _load_tokenizer
from .model_building import (
    calculate_vocab_size,
    build_model_config,
    build_model,
    count_model_parameters,
)
from .optimizer_setup import create_optimizer
from .checkpoint_saver import CheckpointSaver
from .data import get_training_lines
from .encoding import adjust_tokenizer_padding, encode_lines
from .memory_optimization import configure_chunking

# NOTE: This file is deprecated. Use parallel_training_v3.py instead.
# The new implementation uses BlockManager and ChunkTracker for proper
# data distribution without duplication.


def merge_checkpoints(checkpoint_paths: list[str], output_path: str) -> None:
    """Merge multiple checkpoints by averaging weights"""
    print(f"\n{'='*60}")
    print(f"[MERGE] Merging {len(checkpoint_paths)} checkpoints...")
    print(f"{'='*60}")
    
    # Load all checkpoints (support both formats)
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    checkpoints = []
    for p in checkpoint_paths:
        try:
            state_dict = load_safetensors(p, device='cpu')
            checkpoints.append({'model': state_dict})
        except Exception:
            checkpoints.append(torch.load(p, map_location='cpu'))
    
    # Extract model state dicts
    state_dicts = [ckpt['model'] for ckpt in checkpoints]
    
    # Average weights
    merged_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)
        merged_state[key] = stacked.mean(dim=0)
    
    # Save merged checkpoint
    save_safetensors(merged_state, output_path)
    print(f"[OK] Merged checkpoint saved: {output_path}\n")


def train_block_on_gpu(
    config: "TrainingConfig",
    block_id: int,
    start_step: int,
    end_step: int,
    gpu_id: int,
    checkpoint_dir: Path,
    tokenizer,
    vocab_size: int,
    all_lines: list[str],
    block_start_line: int,
    block_end_line: int,
) -> str:
    """DEPRECATED: Train a single block on a specific GPU.
    
    This worker continuously requests chunks from the current block,
    trains on them, and reports progress to ChunkTracker.
    
    Args:
        config: Training configuration
        gpu_id: GPU device ID
        checkpoint_dir: Directory to save checkpoints
        tokenizer: Tokenizer for encoding text
        vocab_size: Vocabulary size
        block_manager: Manages dataset blocks
        chunk_tracker: Tracks trained chunks
        max_steps: Maximum steps before stopping
        current_block_id: ID of the block to train on
        
    Returns:
        Path to saved checkpoint
    """
    
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    # Create dedicated CUDA stream for this GPU to ensure true parallelism
    stream = torch.cuda.Stream(device=device)
    
    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] Training Block {block_id}")
    print(f"   Steps: {start_step} -> {end_step} ({end_step - start_step} steps)")
    print(f"   Lines: {block_start_line} -> {block_end_line}")
    print(f"   CUDA Stream: {stream}")
    print(f"{'='*60}\n")
    
    # Build model
    model_config = build_model_config(
        config=config,
        vocab_size=vocab_size,
        log_fn=print
    )
    
    model = build_model(
        config=model_config,
        student_init=config.student_init,  # Use config student_init if provided
        log_fn=print
    )
    model = model.to(device)
    
    # Build optimizer
    optimizer = create_optimizer(
        model=model,
        config=config,
        use_deepspeed_optimizer=False,  # No DeepSpeed in parallel independent mode
        log_fn=print
    )
    
    # Configure chunking and gradient checkpointing
    segment_rollout, use_chunking, final_chunk_size = configure_chunking(
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
        use_chunked_training=config.use_chunked_training,
        gradient_checkpointing=config.gradient_checkpointing,
        use_cpu_offload=config.use_cpu_offload,
        log_fn=print
    )
    
    # Setup AMP scaler if enabled
    use_amp = config.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"[GPU {gpu_id}] AMP (Automatic Mixed Precision) enabled")
    
    # Get data for this block
    block_lines = all_lines[block_start_line:block_end_line]
    
    # Training loop
    model.train()
    total_loss = 0.0
    steps_completed = 0
    batch_idx = 0
    loss_history = []  # Accumulate losses to reduce .item() synchronization
    
    print(f"Starting training loop for {end_step - start_step} steps...")
    
    try:
        # Training loop
        while steps_completed < (end_step - start_step):
                # Get batch of lines
                batch_start = batch_idx * config.batch_size
                batch_end = min(batch_start + config.batch_size, len(block_lines))
                
                if batch_start >= len(block_lines):
                    # Restart from beginning if we run out
                    batch_idx = 0
                    batch_start = 0
                    batch_end = min(config.batch_size, len(block_lines))
                
                batch_lines = block_lines[batch_start:batch_end]
                if not batch_lines:
                    print(f"WARNING: Empty batch at idx {batch_idx}, skipping")
                    batch_idx += 1
                    continue
                
                batch_idx += 1
                
                # Encode batch
                input_ids, labels = encode_lines(tokenizer, batch_lines, config.max_seq_len)
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # Prepare batch for segment_rollout
                batch = {
                    'inputs': input_ids,
                    'targets': labels,
                    'puzzle_identifiers': torch.arange(len(batch_lines), device=device)
                }
                
                # Forward pass using configured segment_rollout (with gradient checkpointing if enabled)
                optimizer.zero_grad()
                
                # Use AMP context if enabled
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, metrics = segment_rollout(
                        model=model,
                        batch=batch,
                        max_segments=config.halt_max_steps,
                        epsilon=0.0
                    )
                
                # Backward pass with AMP scaling
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    # Check for NaN/Inf gradients
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"WARNING: NaN/Inf gradients detected at step {steps_completed}, skipping update")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    
                    # Check for NaN/Inf gradients
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"WARNING: NaN/Inf gradients detected at step {steps_completed}, skipping update")
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                
                # Accumulate loss tensor without synchronization
                loss_history.append(loss.detach())
                steps_completed += 1
                
                # Only synchronize and print every 5 steps to reduce overhead
                if steps_completed % 5 == 0 or steps_completed == (end_step - start_step):
                    # Synchronize stream before getting loss values
                    torch.cuda.synchronize(device)
                    # Convert accumulated losses to CPU
                    recent_losses = [l.item() for l in loss_history[-5:]]
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    total_loss = sum([l.item() for l in loss_history]) / len(loss_history)
                    print(f"  GPU {gpu_id} Block {block_id}: Step {start_step + steps_completed}/{end_step}, Loss: {avg_loss:.4f}")
    
    except Exception as e:
        print(f"\n[ERROR] Training failed for GPU {gpu_id} Block {block_id}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Ensure all operations complete before saving checkpoint
    torch.cuda.synchronize(device)
    
    # Calculate final loss
    final_avg_loss = sum([l.item() for l in loss_history]) / len(loss_history) if loss_history else 0.0
    
    # Save checkpoint
    print(f"Saving checkpoint for GPU {gpu_id} Block {block_id}...")
    checkpoint_path = checkpoint_dir / f"gpu{gpu_id}_block{block_id}_step{end_step}.pt"
    
    try:
        checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'block_id': block_id,
            'gpu_id': gpu_id,
            'steps_completed': steps_completed,
            'step': end_step,
            'final_loss': final_avg_loss,
            'config': {
                'hidden_size': config.hidden_size,
                'vocab_size': vocab_size,
                'max_seq_len': config.max_seq_len,
            }
        }
        
        # Save using safetensors format
        try:
            from safetensors.torch import save_file as save_safetensors
            checkpoint_path_st = checkpoint_dir / f"gpu{gpu_id}_block{block_id}.safetensors"
            save_safetensors(model.state_dict(), str(checkpoint_path_st))
            checkpoint_path = checkpoint_path_st
            print(f"Checkpoint saved successfully to {checkpoint_path} (safetensors)")
        except ImportError:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved successfully to {checkpoint_path} (torch)")
        
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    final_avg_loss = sum([l.item() for l in loss_history]) / len(loss_history) if loss_history else 0.0
    print(f"\n[OK] GPU {gpu_id} Block {block_id}: Complete!")
    print(f"   Steps: {steps_completed}, Avg Loss: {final_avg_loss:.4f}")
    print(f"   Checkpoint: {checkpoint_path}\n")
    
    # Free memory
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    return str(checkpoint_path)


def run_parallel_independent_training(config: "TrainingConfig") -> None:
    """
    Main orchestrator for parallel independent training.
    Trains different data blocks on different GPUs sequentially, then merges.
    """
    
    print("\n" + "="*60)
    print("PARALLEL INDEPENDENT TRAINING MODE")
    print("="*60)
    print("This mode bypasses DDP by training data blocks independently")
    print("on different GPUs, then merging checkpoints via weight averaging.")
    print("="*60 + "\n")
    
    # Setup output directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "parallel_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load brain metadata to get correct tokenizer - check save_dir even if student_init is None
    from .brain_metadata import load_brain_metadata, extract_tokenizer_from_metadata
    brain_metadata = load_brain_metadata(
        student_init=config.student_init,
        log_fn=print,
        save_dir=str(save_dir)  # Fallback to brain directory for fresh starts
    )
    tokenizer_path = extract_tokenizer_from_metadata(brain_metadata, config.model)
    
    # Load tokenizer
    print(f"[LOAD] Loading tokenizer from: {tokenizer_path}")
    tokenizer = _load_tokenizer(tokenizer_path)
    adjust_tokenizer_padding(tokenizer)
    vocab_size = calculate_vocab_size(tokenizer, print)
    print(f"[OK] Vocabulary size: {vocab_size:,}\n")
    
    # Load dataset into memory
    print(f"[DATA] Loading dataset: {config.dataset_file}")
    from aios.data.datasets import read_text_lines_sample_any
    all_lines = get_training_lines(
        dataset_file=config.dataset_file,
        ascii_only=config.ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
        dataset_chunk_size=config.dataset_chunk_size  # Use config value
    )
    all_lines = list(all_lines)  # Convert to list for indexing
    print(f"[OK] Loaded {len(all_lines):,} lines\n")
    
    # Parse cuda_ids if it's a string
    cuda_ids = config.cuda_ids
    if isinstance(cuda_ids, str):
        cuda_ids = [int(x.strip()) for x in cuda_ids.split(',')]
    else:
        cuda_ids = list(cuda_ids) if cuda_ids else [0]
    
    # Calculate blocks
    num_gpus = len(cuda_ids)
    total_steps = config.steps
    
    # Each GPU trains for the FULL number of steps on its data subset
    # This is different from DDP where steps are divided
    # Result: If you request 1000 steps with 2 GPUs, you get 1000 steps per GPU = 2000 total
    blocks = []
    for i in range(num_gpus):
        blocks.append({
            'id': i,
            'gpu_id': cuda_ids[i],
            'start_step': 0,  # Each GPU starts from step 0
            'end_step': total_steps,  # Each GPU trains for full steps
        })
    
    # Divide dataset lines evenly across blocks
    lines_per_block = len(all_lines) // len(blocks)
    for i, block in enumerate(blocks):
        block['start_line'] = i * lines_per_block
        block['end_line'] = (i + 1) * lines_per_block if i < len(blocks) - 1 else len(all_lines)
    
    print(f"[PLAN] Training Plan:")
    print(f"   Steps per GPU: {total_steps}")
    print(f"   Total effective steps: {total_steps * num_gpus} ({total_steps} × {num_gpus} GPUs)")
    print(f"   GPUs: {cuda_ids}")
    print(f"   Blocks: {len(blocks)}")
    for block in blocks:
        print(f"   Block {block['id']}: GPU {block['gpu_id']}, "
              f"Steps 0-{total_steps} ({total_steps} steps), "
              f"Lines {block['start_line']}-{block['end_line']}")
    print()
    
    # Launch each GPU's training in its own thread (CUDA supports this!)
    print(f"[PARALLEL] Launching {len(blocks)} training threads (one per GPU)...")
    threads = []
    checkpoint_paths_list: list[str | None] = [None] * len(blocks)  # Store results
    exceptions: list[Exception | None] = [None] * len(blocks)  # Store any exceptions
    
    def train_wrapper(idx, block):
        """Wrapper to catch exceptions and store results"""
        try:
            ckpt_path = train_block_on_gpu(
                config=config,
                block_id=block['id'],
                start_step=block['start_step'],
                end_step=block['end_step'],
                gpu_id=block['gpu_id'],
                checkpoint_dir=checkpoint_dir,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                all_lines=all_lines,
                block_start_line=block['start_line'],
                block_end_line=block['end_line'],
            )
            checkpoint_paths_list[idx] = ckpt_path
        except Exception as e:
            exceptions[idx] = e
            print(f"\\n[ERROR] Block {idx} on GPU {block['gpu_id']} failed: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        # Launch all training threads
        for i, block in enumerate(blocks):
            t = threading.Thread(target=train_wrapper, args=(i, block), daemon=False)
            threads.append(t)
            t.start()
            print(f"   [OK] Launched Block {i} on GPU {block['gpu_id']} (Thread: {t.name})")
        
        print(f"\\n[PARALLEL] All {len(blocks)} threads running, waiting for completion...")
        
        # Wait for all threads to complete
        for i, t in enumerate(threads):
            t.join()
            exc = exceptions[i]
            if exc is not None:
                raise exc
        
        # Collect checkpoint paths
        checkpoint_paths = checkpoint_paths_list
        
        print(f"\\n[OK] All {len(blocks)} blocks completed!")
        for i, ckpt_path in enumerate(checkpoint_paths):
            print(f"   Block {i}: {ckpt_path}")
            
    except Exception as e:
        print(f"\\n[FATAL ERROR] Parallel training failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Merge all checkpoints (filter out any None values)
    final_checkpoint_path = Path(save_dir) / "final_model.pt"
    valid_checkpoints = [str(p) for p in checkpoint_paths if p is not None]
    merge_checkpoints(valid_checkpoints, str(final_checkpoint_path))
    
    print("="*60)
    print("[OK] PARALLEL INDEPENDENT TRAINING COMPLETE!")
    print(f"   Final model: {final_checkpoint_path}")
    print(f"   Individual checkpoints: {checkpoint_dir}")
    print("="*60 + "\n")

