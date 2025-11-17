"""Metrics and memory estimation panels UI for HRM Training Panel."""

from __future__ import annotations
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def build_epoch_tracking_panel(panel: HRMTrainingPanel, parent: any) -> None:
    """Build training progress panel for dataset and chunk progress."""
    epoch_panel = ttk.LabelFrame(parent, text="📈 Training Progress")
    epoch_panel.pack(fill="x", pady=(6, 0))
    
    epoch_frame = ttk.Frame(epoch_panel)
    epoch_frame.pack(fill="x", padx=8, pady=4)
    
    # Steps - Total step counter across all GPUs
    ttk.Label(epoch_frame, text="Steps:").pack(side="left", padx=(0, 4))
    panel.met_step = ttk.Label(epoch_frame, text="-")
    panel.met_step.pack(side="left", padx=(0, 16))
    
    # Chunk - Current chunk / Total chunks
    ttk.Label(epoch_frame, text="Chunk:").pack(side="left", padx=(0, 4))
    panel.epoch_chunk_lbl = ttk.Label(epoch_frame, text="-")
    panel.epoch_chunk_lbl.pack(side="left", padx=(0, 16))
    
    # Blocks - Current block / Total blocks (or ? if unknown)
    ttk.Label(epoch_frame, text="Blocks:").pack(side="left", padx=(0, 4))
    panel.epoch_blocks_lbl = ttk.Label(epoch_frame, text="-")
    panel.epoch_blocks_lbl.pack(side="left", padx=(0, 16))
    
    # Epoch - Current epoch number
    ttk.Label(epoch_frame, text="Epoch:").pack(side="left", padx=(0, 4))
    panel.epoch_number_lbl = ttk.Label(epoch_frame, text="-")
    panel.epoch_number_lbl.pack(side="left", padx=(0, 16))
    
    # Dataset - Dataset name
    ttk.Label(epoch_frame, text="Dataset:").pack(side="left", padx=(0, 4))
    panel.epoch_dataset_lbl = ttk.Label(epoch_frame, text="-")
    panel.epoch_dataset_lbl.pack(side="left")


def initialize_epoch_tracking_display(panel: HRMTrainingPanel) -> None:
    """Initialize Training Progress display with default values.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        # Initialize with default values
        if hasattr(panel, "epoch_number_lbl"):
            panel.epoch_number_lbl.config(text="0")
        
        if hasattr(panel, "epoch_blocks_lbl"):
            panel.epoch_blocks_lbl.config(text="0/???")
        
        if hasattr(panel, "epoch_chunk_lbl"):
            panel.epoch_chunk_lbl.config(text="0/???")
        
        # Try to extract dataset name from current dataset field
        if hasattr(panel, "epoch_dataset_lbl") and hasattr(panel, "dataset_var"):
            try:
                dataset_str = panel.dataset_var.get().strip()
                if dataset_str and dataset_str != "training_data":
                    # Extract name from path or HF dataset
                    if 'hf://' in dataset_str:
                        dataset_name = dataset_str.split('hf://')[-1].split(':')[0]
                    else:
                        import os
                        name = os.path.basename(dataset_str)
                        for ext in ['.txt', '.csv', '.json', '.jsonl']:
                            name = name.replace(ext, '')
                        dataset_name = name if name else "unknown"
                    panel.epoch_dataset_lbl.config(text=dataset_name)
                    panel._dataset_name = dataset_name
                else:
                    panel.epoch_dataset_lbl.config(text="-")
            except Exception:
                panel.epoch_dataset_lbl.config(text="-")
    except Exception:
        pass


def build_memory_panels(panel: HRMTrainingPanel, parent: any) -> None:
    """Build VRAM and RAM estimation panels side by side."""
    mem_container = ttk.Frame(parent)
    mem_container.pack(fill="x", pady=(6, 0))
    
    # VRAM estimation panel
    vram_panel = ttk.LabelFrame(mem_container, text="📊 Estimated VRAM per GPU")
    vram_panel.pack(side="left", fill="both", expand=True, padx=(0, 3))
    vram_frame = ttk.Frame(vram_panel)
    vram_frame.pack(fill="x", padx=8, pady=4)
    
    ttk.Label(vram_frame, text="Model:").pack(side="left", padx=(0, 4))
    panel.vram_model_lbl = ttk.Label(vram_frame, text="-")
    panel.vram_model_lbl.pack(side="left", padx=(0, 12))
    
    ttk.Label(vram_frame, text="Optimizer:").pack(side="left", padx=(0, 4))
    panel.vram_optimizer_lbl = ttk.Label(vram_frame, text="-")
    panel.vram_optimizer_lbl.pack(side="left", padx=(0, 12))
    
    ttk.Label(vram_frame, text="Acts+Grads:").pack(side="left", padx=(0, 4))
    panel.vram_activations_lbl = ttk.Label(vram_frame, text="-")
    panel.vram_activations_lbl.pack(side="left", padx=(0, 12))
    
    ttk.Label(vram_frame, text="Total:").pack(side="left", padx=(0, 4))
    panel.vram_total_lbl = ttk.Label(vram_frame, text="- (wip)", font=("TkDefaultFont", 10, "bold"))
    panel.vram_total_lbl.pack(side="left")
    
    # RAM estimation panel
    ram_panel = ttk.LabelFrame(mem_container, text="💾 Estimated System RAM")
    ram_panel.pack(side="left", fill="both", expand=True, padx=(3, 0))
    ram_frame = ttk.Frame(ram_panel)
    ram_frame.pack(fill="x", padx=8, pady=4)
    
    ttk.Label(ram_frame, text="Dataset:").pack(side="left", padx=(0, 4))
    panel.ram_dataset_lbl = ttk.Label(ram_frame, text="-")
    panel.ram_dataset_lbl.pack(side="left", padx=(0, 12))
    
    ttk.Label(ram_frame, text="Offloaded:").pack(side="left", padx=(0, 4))
    panel.ram_offload_lbl = ttk.Label(ram_frame, text="-")
    panel.ram_offload_lbl.pack(side="left", padx=(0, 12))
    
    ttk.Label(ram_frame, text="Total:").pack(side="left", padx=(0, 4))
    panel.ram_total_lbl = ttk.Label(ram_frame, text="-", font=("TkDefaultFont", 10, "bold"))
    panel.ram_total_lbl.pack(side="left")
