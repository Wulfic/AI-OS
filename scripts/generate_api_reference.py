#!/usr/bin/env python3
"""
Generate comprehensive API reference from TrainingConfig dataclasses.

Automatically extracts all parameters with their types, defaults, and docstrings
to create an up-to-date training API reference.
"""

import sys
from pathlib import Path
from dataclasses import fields
from typing import get_type_hints
import inspect

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.core.hrm_training.training_config import TrainingConfig
from aios.core.hrm_training.training_config.base_fields import BaseFields
from aios.core.hrm_training.training_config.architecture_fields import ArchitectureFields
from aios.core.hrm_training.training_config.optimization_fields import OptimizationFields
from aios.core.hrm_training.training_config.distributed_fields import DistributedFields
from aios.core.hrm_training.training_config.io_fields import IOFields
from aios.core.hrm_training.training_config.advanced_fields import AdvancedFields


def extract_field_doc(field_class, field_name: str) -> str:
    """Extract docstring for a specific field from the class."""
    # Get class source
    source = inspect.getsource(field_class)
    lines = source.split('\n')
    
    # Find the field definition
    field_line_idx = None
    for i, line in enumerate(lines):
        if f"{field_name}:" in line or f"{field_name} =" in line:
            field_line_idx = i
            break
    
    if field_line_idx is None:
        return ""
    
    # Collect docstring (triple-quoted strings after field)
    doc_lines = []
    in_doc = False
    quote_char = None
    
    for i in range(field_line_idx + 1, len(lines)):
        line = lines[i].strip()
        
        # Check for start of docstring
        if not in_doc and (line.startswith('"""') or line.startswith("'''")):
            in_doc = True
            quote_char = line[:3]
            # Get text after opening quotes
            text = line[3:]
            if text.endswith(quote_char):
                # Single-line docstring
                return text[:-3].strip()
            doc_lines.append(text)
            continue
        
        if in_doc:
            if line.endswith(quote_char):
                # End of docstring
                doc_lines.append(line[:-3])
                break
            doc_lines.append(line)
        
        # Stop if we hit another field or empty line before docstring
        if not in_doc and (line.startswith('#') or ':' in line or '=' in line):
            break
    
    return '\n'.join(doc_lines).strip()


def format_default_value(value) -> str:
    """Format default value for display."""
    if value is None:
        return "None"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return repr(value)


def get_field_category(field_name: str, field_class) -> str:
    """Determine category based on field class."""
    class_name = field_class.__name__
    if "Base" in class_name:
        return "Core"
    elif "Architecture" in class_name:
        return "Architecture"
    elif "Optimization" in class_name:
        return "Optimization"
    elif "Distributed" in class_name:
        return "Distributed"
    elif "IO" in class_name:
        return "I/O"
    elif "Advanced" in class_name:
        return "Advanced"
    return "Other"


def generate_markdown() -> str:
    """Generate complete API reference markdown."""
    
    md = []
    
    # Header
    md.append("# Training API Reference - Complete Parameter Guide")
    md.append("")
    md.append("**Auto-generated** from `TrainingConfig` dataclasses.")
    md.append("**Last Updated**: " + __import__('datetime').datetime.now().strftime("%Y-%m-%d"))
    md.append("")
    md.append("---")
    md.append("")
    
    # Table of Contents
    md.append("## 📋 Table of Contents")
    md.append("")
    md.append("- [Overview](#overview)")
    md.append("- [Core Parameters](#core-parameters)")
    md.append("- [Architecture Parameters](#architecture-parameters)")
    md.append("- [Optimization Parameters](#optimization-parameters)")
    md.append("- [Distributed Training Parameters](#distributed-training-parameters)")
    md.append("- [I/O Parameters](#io-parameters)")
    md.append("- [Advanced Parameters](#advanced-parameters)")
    md.append("- [Usage Examples](#usage-examples)")
    md.append("")
    md.append("---")
    md.append("")
    
    # Overview
    md.append("## Overview")
    md.append("")
    md.append("AI-OS uses a unified `TrainingConfig` class that combines all training parameters.")
    md.append("This ensures feature parity across CLI, GUI, and Python API interfaces.")
    md.append("")
    md.append("**Key Features**:")
    md.append("- ✅ Single source of truth for all parameters")
    md.append("- ✅ Type-safe with Python dataclasses")
    md.append("- ✅ Validation built-in (`config.validate()`)")
    md.append("- ✅ Serialization support (JSON, dict)")
    md.append("- ✅ Auto-documented with comprehensive docstrings")
    md.append("")
    md.append("---")
    md.append("")
    
    # Group fields by category
    field_classes = [
        (BaseFields, "Core Parameters", "Model, dataset, and basic optimization settings"),
        (ArchitectureFields, "Architecture Parameters", "HRM model architecture and MoE configuration"),
        (OptimizationFields, "Optimization Parameters", "DeepSpeed, quantization, and memory optimization"),
        (DistributedFields, "Distributed Training Parameters", "Multi-GPU, device placement, and evaluation"),
        (IOFields, "I/O Parameters", "Checkpointing, logging, and file paths"),
        (AdvancedFields, "Advanced Parameters", "PEFT/LoRA, experimental features, and deprecated options"),
    ]
    
    # Generate sections for each category
    for field_class, section_title, section_desc in field_classes:
        md.append(f"## {section_title}")
        md.append("")
        md.append(f"*{section_desc}*")
        md.append("")
        
        # Get all fields from this class
        class_fields = fields(field_class)
        
        # Sort fields alphabetically for easier lookup
        sorted_fields = sorted(class_fields, key=lambda f: f.name)
        
        # Generate table
        md.append("| Parameter | Type | Default | Description |")
        md.append("|-----------|------|---------|-------------|")
        
        for field in sorted_fields:
            name = field.name
            type_str = str(field.type).replace('typing.', '').replace('<class \'', '').replace('\'>', '')
            default = format_default_value(field.default)
            
            # Extract docstring
            doc = extract_field_doc(field_class, name)
            # Take first line of doc for table
            first_line = doc.split('\n')[0] if doc else "No description"
            # Escape pipes in description
            first_line = first_line.replace('|', '\\|')
            
            md.append(f"| `{name}` | `{type_str}` | `{default}` | {first_line} |")
        
        md.append("")
        
        # Detailed descriptions
        md.append("### Detailed Descriptions")
        md.append("")
        
        for field in sorted_fields:
            name = field.name
            type_str = str(field.type).replace('typing.', '').replace('<class \'', '').replace('\'>', '')
            default = format_default_value(field.default)
            
            # Extract full docstring
            doc = extract_field_doc(field_class, name)
            
            md.append(f"#### `{name}`")
            md.append("")
            md.append(f"**Type**: `{type_str}`  ")
            md.append(f"**Default**: `{default}`")
            md.append("")
            if doc:
                md.append(doc)
            else:
                md.append("*No detailed description available.*")
            md.append("")
            md.append("---")
            md.append("")
        
        md.append("")
    
    # Usage Examples
    md.append("## Usage Examples")
    md.append("")
    
    md.append("### Basic Training")
    md.append("")
    md.append("```python")
    md.append("from aios.core.hrm_training import TrainingConfig")
    md.append("from aios.cli.hrm_hf.train_actv1_impl import train_actv1_impl")
    md.append("")
    md.append("# Create configuration")
    md.append("config = TrainingConfig(")
    md.append('    model="gpt2",')
    md.append('    dataset_file="training_data/my_data.txt",')
    md.append("    max_seq_len=512,")
    md.append("    batch_size=16,")
    md.append("    steps=1000,")
    md.append("    lr=2e-4,")
    md.append(")")
    md.append("")
    md.append("# Validate")
    md.append("config.validate()")
    md.append("")
    md.append("# Train")
    md.append("train_actv1_impl(config=config)")
    md.append("```")
    md.append("")
    
    md.append("### Advanced: Multi-GPU with DeepSpeed")
    md.append("")
    md.append("```python")
    md.append("config = TrainingConfig(")
    md.append('    model="gpt2",')
    md.append('    dataset_file="large_dataset.txt",')
    md.append("    max_seq_len=2048,")
    md.append("    batch_size=8,")
    md.append("    steps=5000,")
    md.append("    # Multi-GPU")
    md.append("    ddp=True,")
    md.append('    cuda_ids="0,1,2,3",')
    md.append("    world_size=4,")
    md.append("    # DeepSpeed ZeRO-2")
    md.append('    zero_stage="zero2",')
    md.append("    # Memory optimizations")
    md.append("    gradient_checkpointing=True,")
    md.append("    use_amp=True,")
    md.append("    use_8bit_optimizer=True,")
    md.append(")")
    md.append("```")
    md.append("")
    
    md.append("### PEFT/LoRA Fine-Tuning")
    md.append("")
    md.append("```python")
    md.append("config = TrainingConfig(")
    md.append('    model="gpt2",')
    md.append('    dataset_file="finetuning_data.txt",')
    md.append("    max_seq_len=1024,")
    md.append("    batch_size=16,")
    md.append("    steps=500,")
    md.append("    # PEFT/LoRA")
    md.append("    use_peft=True,")
    md.append('    peft_method="lora",')
    md.append("    lora_r=16,")
    md.append("    lora_alpha=32,")
    md.append('    lora_target_modules="q_proj,v_proj,k_proj,o_proj",')
    md.append("    # Optional: 8-bit quantization")
    md.append("    load_in_8bit=True,")
    md.append(")")
    md.append("```")
    md.append("")
    
    md.append("### MoE (Mixture of Experts)")
    md.append("")
    md.append("```python")
    md.append("config = TrainingConfig(")
    md.append('    model="gpt2",')
    md.append('    dataset_file="training_data.txt",')
    md.append("    max_seq_len=512,")
    md.append("    batch_size=16,")
    md.append("    steps=2000,")
    md.append("    # MoE settings")
    md.append("    use_moe=True,")
    md.append("    num_experts=8,")
    md.append("    num_experts_per_tok=2,  # 75% compute reduction")
    md.append("    moe_capacity_factor=1.25,")
    md.append("    auto_adjust_moe_lr=True,  # Prevents instability")
    md.append(")")
    md.append("```")
    md.append("")
    
    md.append("### Chunked Training (Long Context)")
    md.append("")
    md.append("```python")
    md.append("config = TrainingConfig(")
    md.append('    model="gpt2",')
    md.append('    dataset_file="long_context_data.txt",')
    md.append("    max_seq_len=32768,  # 32K context")
    md.append("    batch_size=2,")
    md.append("    steps=1000,")
    md.append("    # Chunked training (auto-enabled for >8K context)")
    md.append("    use_chunked_training=True,")
    md.append("    chunk_size=2048,")
    md.append("    # Memory optimizations")
    md.append("    gradient_checkpointing=True,")
    md.append("    use_cpu_offload=True,  # For extreme contexts")
    md.append("    window_size=512,  # Sliding window attention")
    md.append(")")
    md.append("```")
    md.append("")
    
    # Footer
    md.append("---")
    md.append("")
    md.append("## Related Documentation")
    md.append("")
    md.append("- [Training Guide](../../user_guide/TRAINING_GUIDE.md) - Complete training tutorial")
    md.append("- [Optimization Guide](../../user_guide/OPTIMIZATION_GUIDE.md) - Memory optimization strategies")
    md.append("- [MoE Guide](../../user_guide/MOE_GUIDE.md) - Mixture of Experts details")
    md.append("- [PEFT Guide](../../user_guide/PEFT_GUIDE.md) - Parameter-efficient fine-tuning")
    md.append("- [Dataset Guide](../../user_guide/DATASET_LOADING_TIMEOUT.md) - Dataset loading and streaming")
    md.append("")
    md.append("---")
    md.append("")
    md.append("**Generated**: " + __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    md.append("**Script**: `scripts/generate_api_reference.py`  ")
    md.append("**Source**: `src/aios/core/hrm_training/training_config/`")
    
    return '\n'.join(md)


def main():
    """Generate and save API reference."""
    print("Generating API reference from TrainingConfig...")
    
    markdown = generate_markdown()
    
    # Save to docs
    output_path = Path(__file__).parent.parent / "docs" / "guide" / "api" / "TRAINING_API_REFERENCE.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"✅ API reference generated: {output_path}")
    print(f"📄 Total size: {len(markdown)} bytes")
    
    # Count parameters
    total_params = 0
    for field_class in [BaseFields, ArchitectureFields, OptimizationFields, 
                        DistributedFields, IOFields, AdvancedFields]:
        total_params += len(fields(field_class))
    
    print(f"📊 Total parameters documented: {total_params}")


if __name__ == "__main__":
    main()
