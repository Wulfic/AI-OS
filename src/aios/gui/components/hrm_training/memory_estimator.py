"""
Accurate Memory Estimator for HRM Training

This module provides highly accurate VRAM and RAM estimations that account for:
- Model precision (FP32, FP16/BF16 with AMP)
- Optimization techniques (Gradient Checkpointing, LoRA/PEFT, CPU Offload, 8-bit optimizer)
- DeepSpeed ZeRO stages (0, 1, 2, 3)
- Multi-GPU distribution (DDP)
- Chunked training for long contexts
- PyTorch memory reservation and fragmentation overhead

IMPORTANT: This estimator predicts PyTorch's RESERVED memory (not just peak usage).
This matches what you'll see as "reserved_gb" in training metrics, which is typically
20-30% higher than "peak_gb" due to caching and fragmentation.

Based on empirical measurements and PyTorch/DeepSpeed documentation.
Updated October 2025 to fix critical underestimation bugs.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import math


class MemoryEstimator:
    """Accurate memory estimation for HRM training configurations."""
    
    # Constants
    BYTES_FP32 = 4
    BYTES_FP16 = 2
    BYTES_INT32 = 4
    GB = 1024 ** 3
    
    def __init__(
        self,
        # Model architecture
        total_params: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int = 50257,
        
        # Training config
        batch_size: int = 1,
        seq_len: int = 512,
        
        # Optimizations
        use_amp: bool = True,
        use_gradient_checkpointing: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_trainable_ratio: float = 0.01,  # ~1% of params trainable with LoRA
        use_cpu_offload: bool = False,
        use_8bit_optimizer: bool = False,
        
        # DeepSpeed
        zero_stage: str = "none",  # "none", "zero1", "zero2", "zero3"
        num_gpus: int = 1,
        
        # Context
        use_chunking: bool = False,
        chunk_size: Optional[int] = None,
    ):
        self.total_params = total_params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_trainable_ratio = lora_trainable_ratio
        self.use_cpu_offload = use_cpu_offload
        self.use_8bit_optimizer = use_8bit_optimizer
        
        self.zero_stage = zero_stage
        self.num_gpus = num_gpus
        
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size or self._get_auto_chunk_size()
        
        # Calculate effective parameters (for LoRA)
        if self.use_lora:
            # LoRA only trains adapters (~1% of original params)
            self.trainable_params = int(total_params * lora_trainable_ratio)
            # LoRA adds extra parameters for adapters
            self.lora_adapter_params = 2 * lora_r * hidden_size * num_layers  # Rough estimate
        else:
            self.trainable_params = total_params
            self.lora_adapter_params = 0
    
    def _get_auto_chunk_size(self) -> int:
        """Auto-determine chunk size based on context length and model size."""
        # FIXED: Lowered threshold from 8192 to 2048
        # Old logic caused 16+ GB VRAM usage at seq_len=8192 due to O(n²) attention memory
        # Even small models need chunking for sequences > 2048 to stay under VRAM limits
        if self.seq_len <= 2048:
            return self.seq_len  # No chunking needed for short sequences
        
        # For sequences > 2048, use model-size-based chunking
        # Larger models need smaller chunks due to higher per-layer memory
        if self.total_params > 200_000_000:
            return 512
        elif self.total_params > 100_000_000:
            return 1024
        else:
            return 2048
    
    def estimate_vram(self) -> Dict[str, Any]:
        """
        Estimate VRAM usage per GPU with detailed breakdown.
        
        Returns dict with:
            - model_gb: Model weights
            - optimizer_gb: Optimizer state (AdamW)
            - gradients_gb: Gradient tensors
            - activations_gb: Forward activations
            - overhead_gb: Framework overhead
            - total_gb: Total per GPU
            - breakdown: Dict with detailed component breakdown
        """
        
        # Determine precision for different components
        model_bytes_per_param = self.BYTES_FP32  # Models usually keep FP32 master weights
        
        if self.use_amp:
            # With AMP: activations in FP16, but model/optimizer stay FP32
            activation_bytes = self.BYTES_FP16
            gradient_bytes = self.BYTES_FP16  # Gradients also in FP16
        else:
            activation_bytes = self.BYTES_FP32
            gradient_bytes = self.BYTES_FP32
        
        # ===== 1. MODEL WEIGHTS =====
        model_gb = (self.total_params * model_bytes_per_param) / self.GB
        
        # LoRA adds adapter parameters
        if self.use_lora:
            lora_adapter_gb = (self.lora_adapter_params * model_bytes_per_param) / self.GB
            model_gb += lora_adapter_gb
        
        # ZeRO-3 partitions model weights across GPUs
        # IMPORTANT: ZeRO-3 with single GPU provides NO memory savings!
        # It only adds overhead from all-gather operations during forward/backward
        # NOTE: In practice, many users have multiple GPUs but only train on 1
        if self.zero_stage == "zero3" and self.num_gpus > 1:
            model_gb_per_gpu = model_gb / self.num_gpus
        else:
            model_gb_per_gpu = model_gb
        
        # ===== 2. OPTIMIZER STATE =====
        # AdamW needs 2x parameters for momentum + variance
        if self.use_lora:
            # Only optimize trainable parameters
            optimizer_params = self.trainable_params + self.lora_adapter_params
        else:
            optimizer_params = self.total_params
        
        optimizer_gb = (optimizer_params * model_bytes_per_param * 2) / self.GB
        
        # 8-bit optimizer reduces to INT8 (75% reduction)
        if self.use_8bit_optimizer:
            optimizer_gb *= 0.25
        
        # ZeRO-1, 2, 3 partition optimizer across GPUs
        if self.zero_stage in ["zero1", "zero2", "zero3"] and self.num_gpus > 1:
            optimizer_gb_per_gpu = optimizer_gb / self.num_gpus
        else:
            optimizer_gb_per_gpu = optimizer_gb
        
        # CPU offload moves optimizer to RAM
        if self.use_cpu_offload:
            optimizer_gb_vram = 0
            optimizer_gb_ram = optimizer_gb_per_gpu
        else:
            optimizer_gb_vram = optimizer_gb_per_gpu
            optimizer_gb_ram = 0
        
        # ===== 3. GRADIENTS =====
        if self.use_lora:
            gradient_params = self.trainable_params + self.lora_adapter_params
        else:
            gradient_params = self.total_params
        
        gradients_gb = (gradient_params * gradient_bytes) / self.GB
        
        # ZeRO-2 and ZeRO-3 partition gradients
        if self.zero_stage in ["zero2", "zero3"] and self.num_gpus > 1:
            gradients_gb_per_gpu = gradients_gb / self.num_gpus
        else:
            gradients_gb_per_gpu = gradients_gb
        
        # ===== 4. ACTIVATIONS =====
        # Use effective sequence length (chunk size for chunked training)
        effective_seq = self.chunk_size if self.use_chunking else self.seq_len
        
        # Per-layer activations:
        # - Input: batch * seq * hidden
        # - Attention scores: batch * heads * seq * seq  (THIS IS THE KILLER FOR LONG SEQUENCES!)
        # - Attention output: batch * seq * hidden
        # - FFN intermediate: batch * seq * (hidden * expansion)
        num_heads = max(1, self.hidden_size // 64)  # Estimate
        expansion = 2.0  # HRM ACT-v1 uses 2.0 expansion typically
        
        # CRITICAL: Attention memory scales O(n²) with sequence length!
        # For seq_len=10000: attention_scores needs 10000² = 100M elements per head per batch
        # With 8 heads: 800M elements = 3.2 GB in FP32 (or 1.6 GB in FP16) PER LAYER!
        # 
        # IMPORTANT: With chunking, the attention matrix is ONLY computed over chunk_size!
        # The KV cache is reused across chunks, but the O(n²) attention scores are only chunk_size²
        # This is THE KEY to enabling long context training on consumer GPUs.
        if self.use_chunking and effective_seq < self.seq_len:
            # With chunking: attention matrix is chunk_size x chunk_size
            attention_seq = effective_seq  # Use chunk_size, not geometric mean!
        else:
            # Without chunking: full sequence attention
            attention_seq = self.seq_len
        
        attention_memory = (
            self.batch_size * num_heads * attention_seq * attention_seq  # Score matrix
        ) * activation_bytes
        
        per_layer_act = (
            self.batch_size * effective_seq * self.hidden_size +  # Input embedding
            attention_memory +  # Attention scores (O(n²) - dominates for long seq!)
            self.batch_size * effective_seq * self.hidden_size +  # Attention output
            self.batch_size * effective_seq * int(self.hidden_size * expansion)  # FFN intermediate
        ) * activation_bytes
        
        # Total activations for all layers
        total_activations = per_layer_act * self.num_layers
        
        # HRM ACT-v1 SPECIFIC: Add overhead for dual H/L architecture
        # - Carry states (persistent across chunks, but small: batch * hidden)
        # - Halt prediction heads (minimal: batch * 2 floats per position)
        # - Cross-level communication buffers
        # Empirically measured from actual training: ~25% additional overhead
        # (Previous 45% estimate was too conservative)
        total_activations *= 1.25
        
        # Gradient checkpointing reduces stored activations by ~25%
        # We only store checkpoints at layer boundaries and recompute during backward
        # Real-world measurements show 20-30% reduction
        if self.use_gradient_checkpointing:
            total_activations *= 0.75
        
        # ===== ADD KV CACHE AND CARRY STATES (SPAN FULL SEQUENCE) =====
        # IMPORTANT: Even with chunking, KV cache and carry states span the FULL sequence!
        # This is because they accumulate across chunks for the entire context window.
        
        if self.use_chunking and effective_seq < self.seq_len:
            # KV cache: key + value for each layer, across full sequence
            # batch * seq_len * hidden * 2 (key + value) * bytes per element
            kv_cache_per_layer = (
                self.batch_size * self.seq_len * self.hidden_size * 2 * activation_bytes
            )
            kv_cache_total = kv_cache_per_layer * self.num_layers
            
            # HRM Carry states: persistent state across full sequence
            # batch * seq_len * hidden * bytes per element
            carry_states = (
                self.batch_size * self.seq_len * self.hidden_size * activation_bytes
            )
            
            # Add to activations
            total_activations += kv_cache_total + carry_states
            
            # ===== CRITICAL: CHUNK OVERLAP DURING BACKWARD PASS =====
            # During chunked training, there's a memory spike when:
            # 1. Chunk N is in backward pass (computing gradients)
            # 2. Chunk N+1 is loaded for forward pass (double buffering)
            # 3. ADDITIONAL temporary buffers for gradient accumulation
            # 4. Optimizer momentum buffers may be materialized
            # 
            # Users report spikes from ~4GB to ~11GB during these transitions.
            # This is a 2.75x multiplier on the base memory!
            #
            # The conservative approach: Account for FULL second chunk + extra buffers
            # This includes:
            # - Second chunk activations (full duplicate)
            # - Gradient accumulation workspace
            # - Temporary attention matrices during recompute
            # - Optimizer state materialization
            #
            # Calculate full chunk activations again
            per_layer_chunk_act = (
                self.batch_size * effective_seq * self.hidden_size +  # Input
                self.batch_size * num_heads * effective_seq * effective_seq +  # Attention
                self.batch_size * effective_seq * self.hidden_size +  # Output  
                self.batch_size * effective_seq * int(self.hidden_size * expansion)  # FFN
            ) * activation_bytes
            
            # Full second chunk (all layers)
            second_chunk = per_layer_chunk_act * self.num_layers
            
            # Apply HRM overhead to second chunk
            second_chunk *= 1.25
            
            # With gradient checkpointing, second chunk also gets some reduction
            # but not as much (we're recomputing during backward)
            if self.use_gradient_checkpointing:
                second_chunk *= 0.85  # Less reduction than first chunk
            
            # Add extra workspace for gradient accumulation and temporary buffers
            # This accounts for the gap between calculated and observed memory
            # CRITICAL: User observations show 15-20 GB actual vs ~10 GB calculated
            # This suggests MASSIVE temporary buffers during chunk transitions that aren't
            # captured by the simple second_chunk calculation. These include:
            # - DeepSpeed internal buffers for gradient accumulation
            # - CUDA graph workspace allocations
            # - Optimizer state materializations (momentum/variance temporarily loaded)
            # - Communication buffers for all-reduce operations
            # - Flash attention workspace (if enabled)
            # - Temporary tensors during autograd graph construction
            # Real-world observations: 2x-3x multiplier needed for large chunks with long sequences
            workspace_multiplier = 2.5  # 150% extra - aggressive but realistic
            second_chunk *= workspace_multiplier
            
            # Add overlap memory (this is the spike!)
            total_activations += second_chunk
        
        # Chunked training reduces peak memory proportionally
        # NOTE: Chunking benefits are ALREADY accounted for by using effective_seq/chunk_size
        # in the attention calculation above. DO NOT apply additional chunking_factor here
        # or we'll double-count the benefit!
        # The hybrid attention_seq calculation already handles the trade-off between
        # chunk size and full sequence memory requirements.
        
        activations_gb = total_activations / self.GB
        
        # Activations are NOT divided by num_gpus!
        # The batch_size parameter is already the PER-GPU batch size in DDP.
        # Each GPU computes activations for its own batch slice.
        # DO NOT divide by num_gpus or we massively underestimate!
        activations_gb_per_gpu = activations_gb
        
        # ===== 5. FRAMEWORK OVERHEAD =====
        # CUDA kernels, PyTorch overhead, cuDNN workspace, etc.
        base_total = model_gb_per_gpu + optimizer_gb_vram + gradients_gb_per_gpu + activations_gb_per_gpu
        
        # Base overhead: 8-15% for PyTorch internals
        # Higher percentage for smaller models (more proportional overhead)
        if base_total < 1.0:
            base_overhead_pct = 0.15
        elif base_total < 3.0:
            base_overhead_pct = 0.12
        else:
            base_overhead_pct = 0.10
        overhead_gb = base_total * base_overhead_pct
        
        # DDP overhead: Additional buffers for gradient synchronization
        if self.num_gpus > 1:
            # DDP needs communication buffers (~3% per GPU)
            overhead_gb += base_total * 0.03
        
        # ZeRO-3 has additional communication overhead
        if self.zero_stage == "zero3":
            if self.num_gpus == 1:
                # Single GPU ZeRO-3: MAJOR overhead from all-gather reconstruction!
                # Model weights are partitioned in optimizer but must be reconstructed
                # for forward/backward, causing temporary spikes to near-full memory.
                # Add 35% overhead to account for these reconstruction spikes
                overhead_gb += base_total * 0.35
            else:
                # Multi-GPU: normal 3% communication overhead
                overhead_gb += base_total * 0.03
        
        # Long sequence overhead: cuDNN workspace and temporary buffers
        # These grow with sequence length, especially for attention operations
        if self.seq_len > 2048:
            # For chunked training: workspace still needs space for chunk processing
            # Scale based on effective sequence (chunk size if chunked, else full seq)
            effective_calc_seq = self.chunk_size if self.use_chunking else self.seq_len
            # Empirically: ~0.2 GB base + scale with sqrt of sequence (diminishing returns)
            workspace_gb = 0.2 + (effective_calc_seq / 1000) * 0.15
            # Cap at reasonable maximum
            workspace_gb = min(workspace_gb, 1.0)
            overhead_gb += workspace_gb
            
            # Additional safety margin for very long sequences (>5K) with chunking
            # These have temporary spikes during chunk transitions and gradient accumulation
            if self.seq_len > 5000 and self.use_chunking:
                # Scale with sequence length: up to +20% for extreme contexts
                chunking_safety = base_total * min(0.20, (self.seq_len - 5000) / 20000)
                overhead_gb += chunking_safety
                
                # Extra overhead for large chunk sizes (>2048)
                # Large chunks increase activation memory and temporary buffers significantly
                if self.chunk_size > 2048:
                    # Scale with chunk size: up to +15% for very large chunks (4096+)
                    large_chunk_overhead = base_total * min(0.15, (self.chunk_size - 2048) / 4096 * 0.15)
                    overhead_gb += large_chunk_overhead
        
        # ===== TOTAL VRAM PER GPU (BEFORE RESERVATION) =====
        total_vram_gb = model_gb_per_gpu + optimizer_gb_vram + gradients_gb_per_gpu + activations_gb_per_gpu + overhead_gb
        
        # ===== PYTORCH MEMORY RESERVATION =====
        # PyTorch allocates more than peak usage for caching and fragmentation.
        # The "reserved_gb" metric shows this reservation, which is typically 20-40% higher than peak.
        # With DeepSpeed and chunked training, the overhead is even higher due to:
        # - Internal buffer pools
        # - Gradient accumulation staging areas
        # - Communication buffer pre-allocation
        # - CUDA graph workspace reservation
        # 
        # Based on actual measurements from users:
        # - Tiny models (<0.5GB): Higher proportional overhead ~1.40x
        # - Small models (0.5-2GB): ~1.30x reservation
        # - Medium models (2-8GB): ~1.25x reservation  
        # - Large models (8-12GB): ~1.20x reservation
        # - Very large (>12GB): ~1.15x (less proportional overhead)
        if total_vram_gb < 0.5:
            reservation_factor = 1.40  # High proportional overhead for tiny models
        elif total_vram_gb < 2.0:
            reservation_factor = 1.30
        elif total_vram_gb < 8.0:
            reservation_factor = 1.25
        elif total_vram_gb < 12.0:
            reservation_factor = 1.20
        else:
            reservation_factor = 1.15
        
        total_vram_gb *= reservation_factor
        
        return {
            "model_gb": model_gb_per_gpu,
            "optimizer_gb": optimizer_gb_vram,
            "gradients_gb": gradients_gb_per_gpu,
            "activations_gb": activations_gb_per_gpu,
            "overhead_gb": overhead_gb,
            "total_gb": total_vram_gb,
            "breakdown": {
                "model_full": model_gb,
                "optimizer_full": optimizer_gb,
                "optimizer_ram": optimizer_gb_ram,
                "gradients_full": gradients_gb,
                "trainable_params": self.trainable_params,
                "lora_adapter_params": self.lora_adapter_params,
                "effective_seq": effective_seq,
                "num_gpus": self.num_gpus,
            }
        }
    
    def estimate_ram(self) -> Dict[str, float]:
        """
        Estimate system RAM usage.
        
        Returns dict with:
            - optimizer_gb: Offloaded optimizer state (if CPU offload)
            - model_cpu_gb: Model copy in CPU RAM (before GPU transfer)
            - tokenizer_gb: Tokenizer vocabulary and buffers
            - dataset_gb: Dataset loading/streaming overhead
            - pytorch_gb: PyTorch/Python/CUDA overhead
            - training_gb: Training-specific buffers and staging
            - gpu_overflow_gb: Shared GPU memory (RAM used as VRAM overflow)
            - total_gb: Total system RAM needed
        """
        
        # Optimizer offload (if enabled)
        vram_estimate = self.estimate_vram()
        breakdown = vram_estimate.get("breakdown", {})
        optimizer_ram_gb = breakdown.get("optimizer_ram", 0.0) if isinstance(breakdown, dict) else 0.0
        
        # ===== 1. MODEL CPU COPY =====
        # PyTorch loads model on CPU first, then transfers to GPU
        # Even after transfer, keeps CPU copy for checkpointing and optimizer
        model_bytes_per_param = self.BYTES_FP32
        model_cpu_gb = (self.total_params * model_bytes_per_param) / self.GB
        
        # Add buffer overhead (PyTorch allocates extra for safety)
        model_cpu_gb *= 1.5  # 50% buffer overhead
        
        # ===== 2. TOKENIZER + VOCABULARY =====
        # HuggingFace tokenizers load full vocabulary into RAM
        # Size varies by tokenizer:
        # - GPT-2: ~0.5 GB
        # - LLaMA/Mistral: ~1-2 GB  
        # - Very large vocabs: ~2-3 GB
        if self.vocab_size <= 50000:
            tokenizer_gb = 0.5  # Small vocab (GPT-2)
        elif self.vocab_size <= 100000:
            tokenizer_gb = 1.0  # Medium vocab
        else:
            tokenizer_gb = 2.0  # Large vocab
        
        # Add tokenizer overhead (regex patterns, special tokens, etc.)
        tokenizer_gb += 0.3
        
        # ===== 3. DATASET MEMORY =====
        # Even with streaming, need RAM for tokenized batches and prefetch buffers
        # For long sequences, this is significant!
        
        if self.seq_len > 8192:
            # Streaming mode with long context
            # Need buffer for current batch + prefetch buffer (typically 2-4 batches)
            prefetch_batches = 4
            batch_memory = (self.batch_size * self.seq_len * self.BYTES_INT32) / self.GB
            dataset_gb = batch_memory * prefetch_batches
            dataset_gb += 0.5  # Streaming dataset object overhead
        else:
            # Eager loading or short context streaming
            # Full dataset in RAM (or reasonable buffer)
            dataset_gb = (self.batch_size * self.seq_len * self.BYTES_INT32 * 10) / self.GB  # ~10 batches
            dataset_gb = max(1.0, min(dataset_gb, 4.0))  # 1-4 GB range
        
        # Long sequence overhead: tokenization and string processing uses extra RAM
        if self.seq_len > 4096:
            dataset_gb *= 1.5  # 50% extra for long sequences
        
        # ===== 4. PYTORCH/CUDA/PYTHON OVERHEAD =====
        # This is much larger than naive estimates!
        pytorch_base_gb = 0.0
        
        # Python interpreter
        pytorch_base_gb += 0.5
        
        # PyTorch library (includes all operators, autograd engine, etc.)
        pytorch_base_gb += 1.5
        
        # CUDA runtime and libraries (if using CUDA)
        # cuDNN, cuBLAS, NCCL, etc. all loaded in RAM
        if self.num_gpus > 0:
            pytorch_base_gb += 2.0  # CUDA libs
        
        # DeepSpeed overhead (if using ZeRO)
        if self.zero_stage != "none":
            pytorch_base_gb += 1.0  # DeepSpeed engine and communication buffers
        
        # Additional overhead for DDP processes (each process duplicates base overhead)
        if self.num_gpus > 1:
            pytorch_ddp_gb = pytorch_base_gb * (self.num_gpus - 1) * 0.7  # Each extra process ~70% of base
        else:
            pytorch_ddp_gb = 0
        
        pytorch_total_gb = pytorch_base_gb + pytorch_ddp_gb
        
        # ===== 5. TRAINING-SPECIFIC OVERHEAD =====
        # Gradient accumulation, mixed precision shadows, checkpoint staging
        training_gb = 0.0
        
        # Gradient accumulation staging area
        if self.use_gradient_checkpointing:
            training_gb += 0.5  # Recomputation buffers
        
        # Mixed precision CPU shadows (FP32 copies of FP16 gradients)
        if self.use_amp:
            gradients_fp32_gb = (self.total_params * self.BYTES_FP32) / self.GB
            training_gb += gradients_fp32_gb * 0.5  # Partial shadowing
        
        # Chunked training overhead (staging and buffer management)
        if self.use_chunking:
            # Chunk buffers, carry state staging
            chunk_buffer_gb = (self.batch_size * self.chunk_size * self.hidden_size * 2) / self.GB
            training_gb += chunk_buffer_gb * 2  # Double buffering
        
        # Checkpoint saving staging
        training_gb += 0.5  # Temporary checkpoint buffers
        
        # ===== 6. GPU OVERFLOW TO RAM (CRITICAL!) =====
        # When GPU VRAM is full, PyTorch and OS use system RAM as overflow
        # This shows as "Shared GPU Memory" in Task Manager
        # User's screenshot showed 4.8 GB shared GPU memory!
        
        gpu_overflow_gb = 0.0
        
        # If estimated VRAM exceeds typical GPU capacity, assume overflow
        # This is aggressive but realistic for memory-constrained training
        estimated_vram_gb = vram_estimate.get("total_gb", 0)
        
        # For consumer GPUs (8-24 GB), if we're near limit, expect significant overflow
        typical_gpu_vram = 11.0  # Assume RTX 2080 Ti or similar
        
        if estimated_vram_gb > typical_gpu_vram * 0.85:
            # Heavy memory pressure → RAM overflow
            # Estimate 20-40% of VRAM usage spills to RAM
            overflow_factor = 0.3
            gpu_overflow_gb = estimated_vram_gb * overflow_factor
        elif estimated_vram_gb > typical_gpu_vram * 0.70:
            # Moderate pressure
            overflow_factor = 0.15
            gpu_overflow_gb = estimated_vram_gb * overflow_factor
        
        # Add extra overflow for chunked training with long sequences
        # (frequent data movement between GPU and CPU)
        if self.use_chunking and self.seq_len > 8192:
            gpu_overflow_gb *= 1.5
        
        # ===== TOTAL RAM =====
        total_ram_gb = (
            optimizer_ram_gb +      # Offloaded optimizer (if enabled)
            model_cpu_gb +           # Model copy in CPU
            tokenizer_gb +           # Tokenizer vocabulary
            dataset_gb +             # Dataset buffers
            pytorch_total_gb +       # PyTorch/CUDA/Python
            training_gb +            # Training-specific overhead
            gpu_overflow_gb          # GPU overflow to RAM
        )
        
        return {
            "optimizer_gb": optimizer_ram_gb,
            "model_cpu_gb": model_cpu_gb,
            "tokenizer_gb": tokenizer_gb,
            "dataset_gb": dataset_gb,
            "pytorch_gb": pytorch_total_gb,
            "training_gb": training_gb,
            "gpu_overflow_gb": gpu_overflow_gb,
            "total_gb": total_ram_gb,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete memory summary with VRAM and RAM estimates."""
        vram = self.estimate_vram()
        ram = self.estimate_ram()
        
        return {
            "vram_per_gpu_gb": vram["total_gb"],
            "ram_total_gb": ram["total_gb"],
            "vram_breakdown": vram,
            "ram_breakdown": ram,
            "configuration": {
                "total_params": self.total_params,
                "trainable_params": self.trainable_params,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "chunk_size": self.chunk_size,
                "num_gpus": self.num_gpus,
                "optimizations": {
                    "amp": self.use_amp,
                    "gradient_checkpointing": self.use_gradient_checkpointing,
                    "lora": self.use_lora,
                    "cpu_offload": self.use_cpu_offload,
                    "8bit_optimizer": self.use_8bit_optimizer,
                    "zero_stage": self.zero_stage,
                    "chunking": self.use_chunking,
                }
            }
        }
    
    def get_recommendations(self, available_vram_gb: float) -> list[str]:
        """Get optimization recommendations if memory usage is high."""
        vram = self.estimate_vram()
        current_usage = vram["total_gb"]
        
        recommendations = []
        
        if current_usage > available_vram_gb:
            overage = current_usage - available_vram_gb
            recommendations.append(f"❌ Configuration exceeds available VRAM by {overage:.1f} GB")
            
            # Suggest optimizations in order of impact
            if not self.use_amp:
                savings = current_usage * 0.4  # ~40% savings from AMP
                recommendations.append(f"💡 Enable AMP → Save ~{savings:.1f} GB")
            
            if not self.use_gradient_checkpointing:
                savings = vram["activations_gb"] * 0.6  # 60% of activations
                recommendations.append(f"💡 Enable Gradient Checkpointing → Save ~{savings:.1f} GB")
            
            if not self.use_lora:
                optimizer_savings = vram["optimizer_gb"] * 0.99  # ~99% optimizer reduction
                gradient_savings = vram["gradients_gb"] * 0.99
                total_savings = optimizer_savings + gradient_savings
                recommendations.append(f"💡 Enable LoRA → Save ~{total_savings:.1f} GB")
            
            if self.batch_size > 1:
                new_batch = self.batch_size // 2
                savings = (current_usage - current_usage * new_batch / self.batch_size)
                recommendations.append(f"💡 Reduce batch size to {new_batch} → Save ~{savings:.1f} GB")
            
            if not self.use_cpu_offload and vram["optimizer_gb"] > 0:
                recommendations.append(f"💡 Enable CPU Offload → Move {vram['optimizer_gb']:.1f} GB optimizer to RAM")
            
            if not self.use_8bit_optimizer:
                savings = vram["optimizer_gb"] * 0.75
                recommendations.append(f"💡 Enable 8-bit Optimizer → Save ~{savings:.1f} GB")
            
            if self.num_gpus == 1:
                recommendations.append(f"💡 Use multiple GPUs with ZeRO-2 → Partition memory across GPUs")
            
            if self.seq_len > 8192 and not self.use_chunking:
                recommendations.append(f"💡 Enable chunked training → Reduce activation memory")
        
        elif current_usage > available_vram_gb * 0.85:
            utilization = (current_usage / available_vram_gb) * 100
            recommendations.append(f"⚠️  High VRAM usage ({utilization:.0f}%) - may be unstable")
            recommendations.append("💡 Consider enabling more optimizations for safety margin")
        
        else:
            utilization = (current_usage / available_vram_gb) * 100
            recommendations.append(f"✅ Configuration fits comfortably ({utilization:.0f}% VRAM usage)")
        
        return recommendations


def quick_estimate(
    total_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    **optimizations
) -> Dict[str, float]:
    """
    Quick memory estimation with sensible defaults.
    
    Args:
        total_params: Total model parameters
        batch_size: Training batch size
        seq_len: Sequence length
        hidden_size: Model hidden dimension
        num_layers: Total layers
        **optimizations: Optional optimization flags (use_amp, use_gradient_checkpointing, etc.)
    
    Returns:
        Dict with VRAM and RAM estimates
    """
    estimator = MemoryEstimator(
        total_params=total_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        **optimizations
    )
    
    summary = estimator.get_summary()
    return {
        "vram_per_gpu_gb": summary["vram_per_gpu_gb"],
        "ram_total_gb": summary["ram_total_gb"],
    }
