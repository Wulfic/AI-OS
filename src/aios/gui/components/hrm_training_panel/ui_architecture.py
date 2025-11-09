"""Architecture display UI for HRM Training Panel.

Shows architecture parameters, MoE configuration, tokenizer, and model stats.
"""

from __future__ import annotations
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def build_architecture_display(panel: HRMTrainingPanel, parent: any) -> None:
    """Build the architecture display rows.
    
    Args:
        panel: The HRMTrainingPanel instance
        parent: Parent widget (typically a Frame)
    """
    panel._preset_buttons = []
    
    # Architecture group - Row 1: Layers, Hidden, Heads, Expansion, Cycles, PosEnc
    arch1 = ttk.Frame(parent)
    arch1.pack(fill="x", pady=2)
    ttk.Label(arch1, text="H/L layers:", width=20, anchor="e").pack(side="left")
    panel.h_layers_entry = ttk.Entry(arch1, textvariable=panel.h_layers_var, width=4, state="readonly")
    panel.h_layers_entry.pack(side="left")
    ttk.Label(arch1, text="/").pack(side="left")
    panel.l_layers_entry = ttk.Entry(arch1, textvariable=panel.l_layers_var, width=4, state="readonly")
    panel.l_layers_entry.pack(side="left")
    ttk.Label(arch1, text=" Hidden:").pack(side="left")
    panel.hidden_size_entry = ttk.Entry(arch1, textvariable=panel.hidden_size_var, width=5, state="readonly")
    panel.hidden_size_entry.pack(side="left")
    ttk.Label(arch1, text=" Heads:").pack(side="left")
    panel.num_heads_entry = ttk.Entry(arch1, textvariable=panel.num_heads_var, width=4, state="readonly")
    panel.num_heads_entry.pack(side="left")
    ttk.Label(arch1, text=" Exp:").pack(side="left", padx=(8, 0))
    panel.expansion_entry = ttk.Entry(arch1, textvariable=panel.expansion_var, width=5, state="readonly")
    panel.expansion_entry.pack(side="left")
    ttk.Label(arch1, text=" H/L cycles:").pack(side="left", padx=(8, 0))
    panel.h_cycles_entry = ttk.Entry(arch1, textvariable=panel.h_cycles_var, width=3, state="readonly")
    panel.h_cycles_entry.pack(side="left")
    ttk.Label(arch1, text="/").pack(side="left")
    panel.l_cycles_entry = ttk.Entry(arch1, textvariable=panel.l_cycles_var, width=3, state="readonly")
    panel.l_cycles_entry.pack(side="left")
    ttk.Label(arch1, text=" PosEnc:").pack(side="left", padx=(8, 0))
    panel.pos_enc_entry = ttk.Entry(arch1, textvariable=panel.pos_enc_var, width=8, state="readonly")
    panel.pos_enc_entry.pack(side="left")
    
    # Row 2: MoE, Tokenizer, Params, Size, Steps - everything on one line
    arch2 = ttk.Frame(parent)
    arch2.pack(fill="x", pady=2)
    ttk.Label(arch2, text="MoE:", width=20, anchor="e").pack(side="left")
    panel.moe_num_experts_entry = ttk.Entry(arch2, width=3, state="readonly")
    panel.moe_num_experts_entry.pack(side="left")
    ttk.Label(arch2, text="/").pack(side="left")
    panel.moe_active_experts_entry = ttk.Entry(arch2, width=3, state="readonly")
    panel.moe_active_experts_entry.pack(side="left")
    ttk.Label(arch2, text=" Tok:").pack(side="left", padx=(8, 0))
    panel.tokenizer_entry = ttk.Entry(arch2, width=12, state="readonly")
    panel.tokenizer_entry.pack(side="left")
    ttk.Label(arch2, text=" Params:").pack(side="left", padx=(8, 0))
    panel.params_entry = ttk.Entry(arch2, width=12, state="readonly")
    panel.params_entry.pack(side="left")
    ttk.Label(arch2, text=" MB:").pack(side="left", padx=(8, 0))
    panel.size_mb_entry = ttk.Entry(arch2, width=7, state="readonly")
    panel.size_mb_entry.pack(side="left")
    ttk.Label(arch2, text=" Steps:").pack(side="left", padx=(8, 0))
    panel.trained_steps_entry = ttk.Entry(arch2, width=8, state="readonly")
    panel.trained_steps_entry.pack(side="left")
    
    # Track architecture widgets
    panel._arch_widgets = [
        panel.h_layers_entry,
        panel.l_layers_entry,
        panel.hidden_size_entry,
        panel.num_heads_entry,
        panel.expansion_entry,
        panel.h_cycles_entry,
        panel.l_cycles_entry,
        panel.pos_enc_entry,
        panel.moe_num_experts_entry,
        panel.moe_active_experts_entry,
        panel.tokenizer_entry,
        panel.params_entry,
        panel.size_mb_entry,
        panel.trained_steps_entry,
    ]
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(panel.h_layers_entry, "Number of high-level transformer layers")
        add_tooltip(panel.l_layers_entry, "Number of low-level transformer layers")
        add_tooltip(panel.hidden_size_entry, "Hidden dimension size")
        add_tooltip(panel.num_heads_entry, "Number of attention heads")
        add_tooltip(panel.expansion_entry, "Feed-forward expansion factor")
        add_tooltip(panel.h_cycles_entry, "High-level recurrence cycles")
        add_tooltip(panel.l_cycles_entry, "Low-level recurrence cycles")
        add_tooltip(panel.pos_enc_entry, "Positional encoding type (learned, rope, etc.)")
        add_tooltip(panel.moe_num_experts_entry, "Total number of MoE experts")
        add_tooltip(panel.moe_active_experts_entry, "Number of active experts per token")
        add_tooltip(panel.tokenizer_entry, "Tokenizer being used")
        add_tooltip(panel.params_entry, "Total parameter count")
        add_tooltip(panel.size_mb_entry, "Model size on disk (MB)")
        add_tooltip(panel.trained_steps_entry, "Steps trained so far")
    except Exception:
        pass


def set_arch_widgets_state(panel: HRMTrainingPanel, state: str) -> None:
    """Enable/disable architecture and preset controls.
    
    Args:
        panel: The HRMTrainingPanel instance
        state: Widget state ("normal", "disabled", or "readonly")
    """
    try:
        for w in panel._arch_widgets:
            # Treat 'disabled' requests as 'readonly' to allow programmatic updates but prevent typing
            w.config(state=("readonly" if state in {"disabled", "normal"} else state))
        for b in getattr(panel, "_preset_buttons", []) or []:
            try:
                b.config(state=state)
            except Exception:
                pass
    except Exception:
        pass
