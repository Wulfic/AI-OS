"""
Core chunked training implementation for extreme context lengths.

This module provides the main chunked_segment_rollout function that enables
memory-efficient training for sequences that exceed available VRAM.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Callable
import gc
import os
import torch
import torch.nn.functional as F

# Import CPU offloading functions for extreme-scale contexts
from ..extreme_scale_optimizations import offload_carry_to_cpu, restore_carry_to_gpu

# Import helper functions from sibling modules
from .reward_helpers import _compute_min_halt_steps, _exact_match_reward_from_preds
from .carry_management import _compress_carry_state, _checkpoint_forward

# Enable expandable segments to reduce memory fragmentation (PyTorch recommendation)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Aggressive memory management settings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')


def chunked_segment_rollout(
    model,
    batch: Dict[str, torch.Tensor],
    max_segments: int,
    chunk_size: int,
    epsilon: float = 0.0,
    ce_loss_fn_arg: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ignore_index: int = -100,
    gradient_checkpointing: bool = False,
    use_cpu_offload: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run deep supervision segments with chunked sequence processing for extreme context lengths.
    
    This function splits long sequences into chunks and processes them sequentially,
    maintaining the recurrent carry state across chunks. This enables training on
    sequences much longer than would fit in VRAM if processed all at once.
    
    Memory savings example:
    - Without chunking: 100K tokens → ~40GB VRAM
    - With chunking (2K chunks): 100K tokens → ~2GB VRAM
    
    Args:
        model: HRM model instance with initial_carry() and forward() methods
        batch: Dict with keys:
            - inputs: [B, S] input token IDs (S can be very large)
            - targets: [B, S] target token IDs
            - puzzle_identifiers: [B] puzzle IDs
        max_segments: Maximum number of deep supervision segments
        chunk_size: Number of tokens to process at once (e.g., 2048)
        epsilon: Exploration probability for minimum halt steps
        ce_loss_fn_arg: Optional custom cross-entropy loss function
        ignore_index: Index to ignore in loss computation
        gradient_checkpointing: If True, use gradient checkpointing within chunks
        use_cpu_offload: If True, offload carry state to CPU between chunks (for 100K+ contexts)
        
    Returns:
        (total_loss, metrics) where metrics includes:
            - ce: Cross-entropy loss
            - bce_halt: Binary cross-entropy for halt predictor
            - bce_continue: Binary cross-entropy for continue predictor
            - used_segments: Number of segments used
            - num_chunks: Number of chunks processed
            - effective_seq_len: Total sequence length processed
    """
    assert model.training, "Call model.train() before chunked_segment_rollout"
    
    inputs = batch["inputs"]  # [B, S]
    targets = batch["targets"]  # [B, S]
    device = inputs.device
    B, S = inputs.shape
    
    # Default CE loss function
    if ce_loss_fn_arg is None:
        # Import the safe CE loss function with NaN protection
        from ..train_utils import _sequence_ce_loss
        ce_loss_fn = lambda lg, tg: _sequence_ce_loss(lg, tg, ignore_index=ignore_index)
    else:
        ce_loss_fn = ce_loss_fn_arg
    
    # Unwrap DDP if needed
    model_unwrapped = model.module if hasattr(model, 'module') else model
    
    # Calculate number of chunks
    num_chunks = (S + chunk_size - 1) // chunk_size
    
    # Initialize carry state with MINIMAL size (just 1 position for recurrent state)
    # This is critical: we don't need full chunk-size carry, just a starting state
    # The model will expand it as needed during forward pass, then we compress it back
    init_batch = {
        "inputs": inputs[:, :1],  # Just first position
        "targets": targets[:, :1],
        "puzzle_identifiers": batch["puzzle_identifiers"],
    }
    carry = model_unwrapped.initial_carry(init_batch)
    
    # Set carry.current_data to None initially - it will be set per chunk
    if hasattr(carry, 'current_data'):
        carry.current_data = None
    
    # Metrics accumulation - use lists to accumulate losses, then stack at the end
    ce_losses = []
    bce_halt_losses = []
    bce_continue_losses = []
    used_segments = 0
    
    # Compute minimum halt steps per sample
    mmin = _compute_min_halt_steps(B, max_segments, epsilon, device)
    
    # Deep supervision loop
    for m in range(max_segments):
        used_segments += 1
        
        # Clear CUDA cache before each segment to reduce fragmentation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Process sequence in chunks
        chunk_ce_losses = []
        all_chunk_predictions = []  # Store only argmax predictions, not full logits
        outputs: Optional[Dict[str, torch.Tensor]] = None  # Initialize to avoid unbound variable
        final_outputs: Optional[Dict[str, torch.Tensor]] = None  # Will store outputs from final chunk
        
        # Safety check: ensure we have at least one chunk
        if num_chunks == 0:
            raise ValueError(f"No chunks to process: seq_len={S}, chunk_size={chunk_size}")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, S)
            
            # CPU OFFLOADING RESTORE: If we offloaded carry to CPU after previous chunk, restore it now
            if use_cpu_offload and chunk_idx > 0:
                carry = restore_carry_to_gpu(carry, {}, device)
            
            # AGGRESSIVE MEMORY CLEANUP before processing chunk
            # This maximizes available memory for the forward pass
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Extract chunk - CRITICAL: Clone to free full tensor memory!
            # Slicing creates views that keep the full tensor in memory
            # Cloning creates new tensors so the full sequence can be garbage collected
            chunk_batch = {
                "inputs": inputs[:, start_idx:end_idx].clone(),
                "targets": targets[:, start_idx:end_idx].clone(),
                "puzzle_identifiers": batch["puzzle_identifiers"],
            }
            
            # CRITICAL FIX: Update carry.current_data to match chunk size
            # The model's forward() blends carry.current_data with chunk_batch,
            # so they must have matching sequence dimensions
            if hasattr(carry, 'current_data'):
                # Create new dict to avoid holding references to old data
                carry.current_data = {
                    "inputs": chunk_batch["inputs"],
                    "targets": chunk_batch["targets"],
                    "puzzle_identifiers": batch["puzzle_identifiers"],
                }
            
            # Forward pass through chunk
            try:
                # ULTRA-AGGRESSIVE: Move input batch to GPU only when needed
                # This helps when using CPU offloading
                if use_cpu_offload and device.type == 'cuda':
                    # Clear cache before forward pass
                    torch.cuda.empty_cache()
                    gc.collect()
                
                if gradient_checkpointing:
                    # Use gradient checkpointing for memory efficiency
                    carry, outputs = _checkpoint_forward(model_unwrapped, carry, chunk_batch)
                else:
                    # CRITICAL FIX: Always enable gradients for the forward pass
                    # The loss computation needs gradients, even if we use one-step approximation
                    # (we just won't backward through earlier segments, but the final loss needs grad_fn)
                    carry, outputs = model_unwrapped(carry, chunk_batch)
            except RuntimeError as e:
                # If OOM, try ULTRA-AGGRESSIVE cleanup and retry once
                if "out of memory" in str(e).lower():
                    if device.type == 'cuda':
                        # Emergency memory cleanup
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    
                    # Try to free any cached tensors
                    import sys
                    if hasattr(sys, 'modules'):
                        for module in sys.modules.values():
                            if hasattr(module, '__dict__'):
                                for attr_name in list(module.__dict__.keys()):
                                    if attr_name.startswith('_cache'):
                                        try:
                                            delattr(module, attr_name)
                                        except:
                                            pass
                    
                    # One retry attempt after cleanup
                    if gradient_checkpointing:
                        carry, outputs = _checkpoint_forward(model_unwrapped, carry, chunk_batch)
                    else:
                        carry, outputs = model_unwrapped(carry, chunk_batch)
                else:
                    raise
            
            # Ensure outputs was set by forward pass
            if outputs is None:
                raise RuntimeError("Model forward pass did not return outputs")
            
            chunk_logits = outputs["logits"]  # [B, chunk_len, V]
            chunk_targets = chunk_batch["targets"]
            
            # Compute CE loss for this chunk (handles padding-only chunks gracefully)
            chunk_ce = ce_loss_fn(chunk_logits, chunk_targets)
            chunk_ce_losses.append(chunk_ce)
            
            # MEMORY OPTIMIZATION: Store only predictions (argmax), not full logits
            # This reduces memory from ~100MB per chunk to ~2KB per chunk!
            # For 100k context: 20GB → 400KB savings
            # CRITICAL: We need predictions from ALL chunks to compute reward against full targets
            chunk_preds = chunk_logits.argmax(dim=-1).detach()  # [B, chunk_len]
            all_chunk_predictions.append(chunk_preds)
            
            # Free chunk_logits immediately after getting predictions
            del chunk_logits
            
            # Save outputs from last chunk for Q-value computation later
            # (We'll delete outputs from non-final chunks to save memory)
            if chunk_idx == num_chunks - 1:
                final_outputs = outputs
            else:
                # ULTRA-AGGRESSIVE CLEANUP: Free outputs dict immediately for non-final chunks
                # This prevents holding onto Q-values and other intermediate tensors
                del outputs
            
            # CRITICAL: Compress carry state to only keep last position
            # This prevents memory from accumulating across chunks
            # Reduces carry from [B, chunk_size, hidden] to [B, 1, hidden]
            # ULTRA-AGGRESSIVE: Delete old carry BEFORE creating compressed version
            old_carry = carry
            carry = _compress_carry_state(carry, device, model_unwrapped)
            del old_carry
            
            # Immediate cleanup after compression
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # CPU OFFLOADING: Move carry to CPU RAM between chunks (optional, for extreme contexts)
            # This saves GPU VRAM at the cost of CPU<->GPU transfer overhead
            # Only offload between chunks (not before the last chunk since we need it on GPU soon)
            carry_metadata = {}
            if use_cpu_offload and chunk_idx < num_chunks - 1:
                carry, carry_metadata = offload_carry_to_cpu(carry, device)
            
            # EXTREME-SCALE MEMORY CLEANUP (for 100K+ token contexts)
            # Free memory immediately and aggressively after computing loss
            # Note: Keep outputs from last chunk for Q-value computation
            del chunk_targets, chunk_batch
            
            # Force CUDA synchronization and cache clearing
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Wait for all ops to complete
                torch.cuda.empty_cache()  # Clear cache after EVERY chunk
                
                # Additional cleanup for extreme contexts (>100K tokens)
                if num_chunks > 50:  # ~100K tokens with 2K chunks
                    # Force more aggressive memory defragmentation
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Force Python garbage collection for extreme contexts
            if num_chunks > 50:
                gc.collect()
            elif chunk_idx % 10 == 0:  # Every 10 chunks for normal contexts
                gc.collect()
        
        # After loop: final_outputs contains results from the last chunk
        # Q-learning targets for halt/continue
        if final_outputs is None:
            raise RuntimeError("No outputs from chunk processing - empty sequence?")
            
        # Aggregate CE loss across chunks
        ce_loss = torch.stack(chunk_ce_losses).mean()
        ce_losses.append(ce_loss)
        
        # Compute halt/continue targets using predictions
        # (concatenate all chunks for reward computation)
        if all_chunk_predictions:
            full_predictions = torch.cat(all_chunk_predictions, dim=1)  # [B, S]
            r_halt = _exact_match_reward_from_preds(full_predictions, targets, ignore_index)
        else:
            r_halt = torch.zeros(B, device=device)
        
        # Q-values from last chunk
        qh = final_outputs["q_halt_logits"]  # [B] - from last chunk
        qc = final_outputs["q_continue_logits"]  # [B]
        
        # Continue target
        if "target_q_continue" in final_outputs:
            tgt_continue = final_outputs["target_q_continue"].detach().clamp(0.0, 1.0)
        else:
            tgt_continue = torch.sigmoid(qh.detach())
        
        # BCE loss for Q-values
        halt_mask = (m + 1 >= mmin)
        if halt_mask.any():
            bce_halt_loss = F.binary_cross_entropy_with_logits(
                qh[halt_mask], r_halt[halt_mask], reduction="mean"
            )
            bce_halt_losses.append(bce_halt_loss)
        
        bce_continue_loss = F.binary_cross_entropy_with_logits(
            qc, tgt_continue, reduction="mean"
        )
        bce_continue_losses.append(bce_continue_loss)
        
        # Early stopping if all samples halted
        if carry.halted.all():
            break
    
    # Compute total losses from accumulated lists
    # Use .sum() to get a scalar with proper grad_fn from the stack operation
    if ce_losses:
        total_ce = torch.stack(ce_losses).sum()
    else:
        # No CE losses - create a zero that requires grad for proper backward() support
        total_ce = torch.zeros(1, device=device, requires_grad=True).sum()
    
    if bce_halt_losses:
        total_bce_halt = torch.stack(bce_halt_losses).sum()
    else:
        # No BCE halt losses - create a zero that requires grad
        total_bce_halt = torch.zeros(1, device=device, requires_grad=True).sum()
    
    if bce_continue_losses:
        total_bce_continue = torch.stack(bce_continue_losses).sum()
    else:
        # No BCE continue losses - create a zero that requires grad
        total_bce_continue = torch.zeros(1, device=device, requires_grad=True).sum()
    
    # Total loss - sum of all components
    # Even if some components are zero with requires_grad=True, the total will have a grad_fn
    # This ensures backward() always works, even in edge cases where some losses are empty
    total_loss = total_ce + total_bce_halt + total_bce_continue
    
    # Metrics
    metrics = {
        "ce": total_ce.detach(),
        "bce_halt": total_bce_halt.detach(),
        "bce_continue": total_bce_continue.detach(),
        "used_segments": torch.tensor(float(used_segments), device=device),
        "num_chunks": torch.tensor(float(num_chunks), device=device),
        "effective_seq_len": torch.tensor(float(S), device=device),
    }
    
    return total_loss, metrics
