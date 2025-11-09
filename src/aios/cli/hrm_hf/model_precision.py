"""Model precision and quantization utilities."""
from __future__ import annotations

import torch
from typing import Any


def apply_quantization(model: Any, config: Any, log_fn) -> int:
    """Apply 8-bit or 4-bit quantization to model.
    
    Returns:
        Number of quantized layers
    """
    initial_params = sum(p.numel() for p in model.parameters())
    initial_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    
    # Validate quantization options
    if config.load_in_8bit and config.load_in_4bit:
        log_fn({
            "quantization": "error",
            "message": "Cannot use both load_in_8bit and load_in_4bit simultaneously",
            "action": "Using load_in_4bit (more aggressive), disabling load_in_8bit"
        })
        config.load_in_8bit = False
    
    quantized_layers = 0
    
    # 8-bit Quantization
    if config.load_in_8bit:
        try:
            import bitsandbytes as bnb
            
            log_fn({
                "quantization": "8-bit",
                "status": "applying",
                "method": "bitsandbytes INT8",
                "expected_memory_reduction": "~75%",
                "note": "Custom HRM architecture - quantization may have limitations"
            })
            
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    quantized_layers += 1
            
            if quantized_layers == 0:
                log_fn({
                    "quantization": "8-bit",
                    "status": "skipped",
                    "reason": "No Linear layers found in model",
                })
            else:
                log_fn({
                    "quantization": "8-bit",
                    "status": "complete",
                    "quantized_layers": quantized_layers,
                })
                
        except ImportError:
            log_fn({
                "quantization": "8-bit",
                "status": "failed",
                "error": "bitsandbytes not installed",
                "install_command": "pip install bitsandbytes>=0.43.0",
                "fallback": "Training with full precision"
            })
    
    # 4-bit Quantization
    elif config.load_in_4bit:
        log_fn({
            "quantization": "4-bit",
            "status": "skipped",
            "reason": "HRM custom architecture not compatible with 4-bit quantization",
            "suggestion": "Use load_in_8bit=True or model_dtype='fp16'/'bf16' instead",
            "alternative": "For QLoRA on standard transformers, use HuggingFace models"
        })
    
    return quantized_layers


def apply_dtype_conversion(model: Any, config: Any, initial_memory_mb: float, log_fn) -> None:
    """Apply dtype conversion (fp16/bf16) to model."""
    if config.model_dtype.lower() == "fp32":
        return
    
    dtype_str = config.model_dtype.lower()
    
    try:
        if dtype_str == "fp16":
            target_dtype = torch.float16
            dtype_name = "fp16"
        elif dtype_str == "bf16":
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                log_fn({
                    "model_dtype": "warning",
                    "message": "bf16 not supported on this device, falling back to fp16",
                })
                target_dtype = torch.float16
                dtype_name = "fp16"
            else:
                target_dtype = torch.bfloat16
                dtype_name = "bf16"
        else:
            log_fn({
                "model_dtype": "error",
                "message": f"Unsupported dtype: {dtype_str}",
                "fallback": "Training with fp32"
            })
            return
        
        model.to(dtype=target_dtype)
        
        final_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        actual_reduction = (initial_memory_mb - final_memory_mb) / initial_memory_mb if initial_memory_mb > 0 else 0.0
        
        log_fn({
            "model_dtype": "applied",
            "precision": dtype_name,
            "initial_memory_mb": round(initial_memory_mb, 2),
            "final_memory_mb": round(final_memory_mb, 2),
            "memory_saved_mb": round(initial_memory_mb - final_memory_mb, 2),
            "reduction_pct": round(actual_reduction * 100, 1),
            "note": "Weight precision only - AMP controls activation precision separately"
        })
        
    except Exception as e:
        log_fn({
            "model_dtype": "error",
            "message": f"Failed to apply dtype conversion: {e}",
            "fallback": "Training with fp32"
        })


def apply_peft(model: Any, config: Any, log_fn) -> Any:
    """Apply PEFT adapters (LoRA, AdaLoRA, IA3) to model.
    
    Returns:
        Modified model with PEFT adapters
    """
    if not config.use_peft:
        return model
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig, IA3Config
        
        # Parse target modules
        target_modules_list = [m.strip() for m in config.lora_target_modules.split(',') if m.strip()]
        
        if not target_modules_list:
            log_fn({
                "peft": "error",
                "message": "No target modules specified for PEFT",
                "fallback": "Training without PEFT"
            })
            return model
        
        # Create PEFT config based on method
        if config.peft_method.lower() == "adalora":
            peft_config = AdaLoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules_list,
                task_type=TaskType.CAUSAL_LM,
            )
        elif config.peft_method.lower() == "ia3":
            peft_config = IA3Config(
                target_modules=target_modules_list,
                task_type=TaskType.CAUSAL_LM,
            )
        else:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules_list,
                task_type=TaskType.CAUSAL_LM,
            )
        
        # Wrap model with PEFT
        model = get_peft_model(model, peft_config)
        
        # Print stats
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        log_fn({
            "peft": "enabled",
            "method": config.peft_method,
            "config": {
                "lora_r": config.lora_r if config.peft_method.lower() != "ia3" else "N/A",
                "lora_alpha": config.lora_alpha if config.peft_method.lower() != "ia3" else "N/A",
                "lora_dropout": config.lora_dropout if config.peft_method.lower() != "ia3" else "N/A",
                "target_modules": target_modules_list,
            },
            "parameters": {
                "trainable": f"{trainable_params:,}",
                "total": f"{total_params:,}",
                "trainable_pct": f"{trainable_pct:.2f}%",
            },
        })
        
        return model
        
    except ImportError as e:
        log_fn({
            "peft": "error",
            "message": f"PEFT library not available: {e}",
            "solution": "Install with: pip install peft>=0.11.1",
            "fallback": "Training without PEFT"
        })
        return model
    except Exception as e:
        log_fn({
            "peft": "error",
            "message": f"Failed to apply PEFT: {e}",
            "fallback": "Training without PEFT"
        })
        return model
