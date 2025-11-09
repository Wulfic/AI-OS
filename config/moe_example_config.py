"""
MOE CONFIGURATION EXAMPLES

This file provides reference examples for working with Mixture of Experts (MoE)
in AI-OS. It shows Python API usage, configuration presets, and CLI commands.

This is a REFERENCE FILE - you don't need to execute it directly.
Use these examples to configure MoE in your own training or inference code.
"""

# Example: ACTv1 Brain Configuration with Sparse MoE
# This configuration shows how to work with trained ACTv1 brains
# that use sparse Mixture of Experts for efficient inference

# Python example (loading a trained ACTv1 brain)
"""
from aios.core.brains import BrainRegistry

# Create registry
registry = BrainRegistry(
    total_storage_limit_mb=4096,
    store_dir="artifacts/brains"
)

# Load a trained ACTv1 brain (automatically detects from actv1/ subdirectory)
brain = registry.get("English-v1")  # Loads from artifacts/brains/actv1/English-v1/

# Or explicitly create an ACTv1 brain
brain = registry.create_actv1(
    name="my_brain",
    modalities=["text"],
    checkpoint_path="artifacts/brains/actv1/my_brain/actv1_student.safetensors",
    brain_config_path="artifacts/brains/actv1/my_brain/brain.json",
    max_seq_len=2048,  # Optional override
)
"""

# Direct brain creation example with router
"""
from aios.core.brains import BrainRegistry, Router

registry = BrainRegistry(
    total_storage_limit_mb=4096,
    store_dir="artifacts/brains"
)

# Mark a brain as master so router uses it for text modality
registry.load_masters()  # Load from masters.json
registry.mark_master("English-v1")  # Or mark programmatically

# Create router - it will use the master brain automatically
router = Router(
    registry=registry,
    default_modalities=["text"],
    brain_prefix="brain",
    create_cfg={},
)

# Use the brain (routes to master brain automatically)
response = router.handle({
    "modalities": ["text"],
    "payload": "What color is the sky?"
})

print(response.get("response", response.get("text", response)))
"""

# ============================================================================
# ACTv1 BRAIN ARCHITECTURE INFO
# ============================================================================

# ACTv1 brains support Mixture of Experts (MoE) which is configured during
# training via the HRM Training panel. The configuration is stored in brain.json:
#
# {
#   "use_moe": true,
#   "num_experts": 4,
#   "num_experts_per_tok": 2,
#   "hidden_size": 256,
#   "h_layers": 8,
#   "l_layers": 8,
#   ...
# }

# To train a brain with MoE, use the HRM Training panel and enable:
# - MoE checkbox
# - Set number of experts (4, 8, 16, etc.)
# - Set experts per token (typically 1-3)
#
# The trained brain will automatically use MoE during inference.

# ============================================================================
# MoE PRESETS FOR TRAINING
# ============================================================================

# Maximum Efficiency (lowest compute, good quality)
moe_max_efficiency = {
    "enabled": True,
    "num_experts": 4,
    "experts_per_tok": 1,
}

# Balanced (recommended default)
moe_balanced = {
    "enabled": True,
    "num_experts": 8,
    "experts_per_tok": 2,
}

# High Capacity (best quality, more compute)
moe_high_capacity = {
    "enabled": True,
    "num_experts": 12,
    "experts_per_tok": 3,
}

# Multilingual (diverse expert specialization)
moe_multilingual = {
    "enabled": True,
    "num_experts": 16,
    "experts_per_tok": 3,
}

# ============================================================================
# CLI USAGE EXAMPLES
# ============================================================================

"""
# List all available brains
aios brains list

# Load a brain and mark it as master (for chat routing)
aios brains load English-v1

# Mark a brain as master explicitly
aios brains set-master English-v1

# Unload model to free GPU memory
# (In GUI: click "Unload" button in chat panel)

# Train a new brain with MoE
aios hrm-hf train-actv1 \\
    --model artifacts/hf_implant/tokenizers/mistral-7b \\
    --dataset-file training_data/my_dataset.txt \\
    --brain-name my_brain \\
    --use-moe --num-experts 8 --num-experts-per-tok 2
"""
