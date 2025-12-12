"""Create New Brain dialog for HRM training panel.

This dialog allows users to create new HRM student models with:
- Architecture presets (1M, 5M, 10M, 20M, 50M, Custom)
- Tokenizer selection
- Custom architecture configuration  
- MoE (Mixture of Experts) settings
- Parameter estimation
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional

# Import safe variable wrappers
from ....utils import safe_variables

from .brain_creator import create_brain_directory, apply_preset_to_panel
from .param_estimator import estimate_parameters


def show_create_dialog(parent_dialog: tk.Toplevel, panel: Any) -> None:
    """
    Show "Create New Brain" dialog.
    
    Args:
        parent_dialog: Parent selection dialog (will be closed on create)
        panel: HRM training panel instance
    """
    try:
        w = tk.Toplevel(parent_dialog)
        w.title("Create New HRM Student")
        w.grab_set()
        inner = ttk.Frame(w)
        inner.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== PRESET SELECTION =====
        preset_label = ttk.Label(inner, text="Choose architecture preset:")
        preset_label.pack(anchor="w")
        choice = safe_variables.StringVar(value="5M")
        preset_buttons = {}
        for opt in ["1M", "5M", "10M", "20M", "50M", "Custom"]:
            rb = ttk.Radiobutton(inner, text=opt, variable=choice, value=opt)
            rb.pack(anchor="w")
            preset_buttons[opt] = rb
        
        # ===== BRAIN NAME =====
        name_row = ttk.Frame(inner)
        name_row.pack(fill="x", pady=(8,0))
        name_lbl = ttk.Label(name_row, text="Brain name:", width=18, anchor="e")
        name_lbl.pack(side="left")
        name_var = safe_variables.StringVar(value=panel.brain_name_var.get().strip() or "new_brain")
        name_entry = ttk.Entry(name_row, textvariable=name_var, width=24)
        name_entry.pack(side="left")
        
        # ===== DEFAULT GOAL =====
        goal_row = ttk.Frame(inner)
        goal_row.pack(fill="x", pady=(4,0))
        goal_lbl = ttk.Label(goal_row, text="Default goal:", width=18, anchor="e")
        goal_lbl.pack(side="left")
        goal_var = safe_variables.StringVar(value="")
        goal_entry = ttk.Entry(goal_row, textvariable=goal_var, width=60)
        goal_entry.pack(side="left", fill="x", expand=True)
        
        # ===== TOKENIZER SELECTION =====
        tokenizer_row, tokenizer_var, tokenizer_id_map = _create_tokenizer_selector(inner, panel)
        
        # ===== CUSTOM ARCHITECTURE SECTION =====
        custom_frame, custom_vars = _create_custom_architecture_section(inner)
        
        # ===== PARAMETER ESTIMATOR =====
        param_frame, param_label, breakdown_label = _create_parameter_estimator(custom_frame)
        
        # Hook up parameter estimator to update on field changes
        _setup_parameter_estimator_updates(custom_vars, param_label, breakdown_label)
        
        # ===== SHOW/HIDE CUSTOM FRAME BASED ON PRESET =====
        def _on_preset_change(*args):
            if choice.get() == "Custom":
                custom_frame.pack(fill="x", pady=(8,0))
            else:
                custom_frame.pack_forget()
        choice.trace_add("write", _on_preset_change)
        
        # ===== CONFIRM BUTTON =====
        def _confirm_create():
            try:
                sel = (choice.get() or "").strip()
                
                # Apply preset or custom values
                if sel == "Custom":
                    # Custom architecture - apply custom values
                    panel.hidden_size_var.set(custom_vars["hidden"].get())
                    panel.h_layers_var.set(custom_vars["h_layers"].get())
                    panel.l_layers_var.set(custom_vars["l_layers"].get())
                    panel.num_heads_var.set(custom_vars["heads"].get())
                    panel.expansion_var.set(custom_vars["expansion"].get())
                    panel.h_cycles_var.set(custom_vars["h_cycles"].get())
                    panel.l_cycles_var.set(custom_vars["l_cycles"].get())
                    panel.pos_enc_var.set(custom_vars["pos"].get())
                else:
                    # Preset architecture
                    apply_preset_to_panel(panel, sel)
                
                # Set default goal if provided
                goal_text = goal_var.get().strip()
                if goal_text and hasattr(panel, 'default_goal_var'):
                    panel.default_goal_var.set(goal_text)
                    panel._log(f"[hrm] Set default goal: {goal_text[:60]}...")
                
                # Create brain directory
                bname = (name_var.get().strip() or "new_brain").replace(" ", "_")
                panel.brain_name_var.set(bname)
                
                # Get tokenizer info
                tokenizer_id, tokenizer_info = _get_selected_tokenizer(tokenizer_var, tokenizer_id_map, panel)
                
                # Get architecture config
                architecture = None
                if sel == "Custom":
                    architecture = {
                        "hidden_size": int(custom_vars["hidden"].get() or 512),
                        "h_layers": int(custom_vars["h_layers"].get() or 2),
                        "l_layers": int(custom_vars["l_layers"].get() or 2),
                        "num_heads": int(custom_vars["heads"].get() or 8),
                        "expansion": float(custom_vars["expansion"].get() or 2.0),
                        "h_cycles": int(custom_vars["h_cycles"].get() or 2),
                        "l_cycles": int(custom_vars["l_cycles"].get() or 2),
                        "pos_encoding": custom_vars["pos"].get(),
                        "dtype": custom_vars["dtype"].get(),
                    }
                
                # Get MoE config
                use_moe = custom_vars["use_moe"].get() if sel == "Custom" else True  # Presets have MoE enabled
                num_experts = int(custom_vars["num_experts"].get() or 8) if sel == "Custom" else 8
                num_experts_per_tok = int(custom_vars["num_experts_per_tok"].get() or 2) if sel == "Custom" else 2
                
                # Create brain directory with metadata
                brain_dir = create_brain_directory(
                    project_root=panel._project_root,
                    brain_name=bname,
                    tokenizer_id=tokenizer_id,
                    tokenizer_model=tokenizer_info.path,
                    vocab_size=tokenizer_info.vocab_size,
                    default_goal=goal_text,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    architecture=architecture,
                )
                
                # Update panel vars
                import os
                panel.student_init_var.set(os.path.join(brain_dir, "actv1_student.safetensors"))
                panel.log_file_var.set(os.path.join(brain_dir, "metrics.jsonl"))
                panel._log(f"[hrm] New student {'custom' if sel == 'Custom' else 'preset ' + sel} architecture in bundle {bname}")
                panel._log(f"[hrm] Tokenizer: {tokenizer_info.name} ({tokenizer_info.vocab_size:,} tokens)")
                
                # Warn if tokenizer not installed
                if not _check_tokenizer_installed(tokenizer_id, panel._project_root):
                    panel._log(f"[hrm] ⚠️ WARNING: Tokenizer '{tokenizer_info.name}' is not installed!")
                    panel._log(f"[hrm] Run: python scripts/download_tokenizer.py {tokenizer_id}")
                    panel._log(f"[hrm] Training will fail until tokenizer is downloaded.")
                
                panel._set_arch_widgets_state("disabled")
            except Exception as e:
                panel._log(f"[hrm] Error creating brain: {e}")
            
            # Close dialogs
            try:
                w.destroy()
            except Exception:
                pass
            try:
                parent_dialog.destroy()
            except Exception:
                pass
        
        # ===== BUTTONS =====
        btn_row = ttk.Frame(inner)
        btn_row.pack(fill="x", pady=(8,0))
        ttk.Button(btn_row, text="Create", command=_confirm_create).pack(side="left", padx=(0,6))
        ttk.Button(btn_row, text="Cancel", command=w.destroy).pack(side="left", padx=(6,0))
        
    except Exception as e:
        try:
            parent_dialog.destroy()
        except Exception:
            pass


def _create_tokenizer_selector(parent: ttk.Frame, panel: Any) -> tuple:
    """Create tokenizer selection UI with registry integration."""
    tokenizer_row = ttk.Frame(parent)
    tokenizer_row.pack(fill="x", pady=(4,0))
    tokenizer_lbl = ttk.Label(tokenizer_row, text="Tokenizer:", width=18, anchor="e")
    tokenizer_lbl.pack(side="left")
    
    tokenizer_var = safe_variables.StringVar()
    tokenizer_id_map = {}
    
    try:
        from aios.core.tokenizers import TokenizerRegistry
        
        # Build tokenizer options with status indicators
        tokenizer_options = []
        for info in TokenizerRegistry.list_available():
            installed = TokenizerRegistry.check_installed(info.id, panel._project_root)
            status = "✓" if installed else "⚠"
            display_text = f"{status} {info.name} ({info.vocab_size:,} tokens)"
            tokenizer_options.append(display_text)
            tokenizer_id_map[display_text] = info.id
        
        # Default to legacy GPT-2 base_model
        default_info = TokenizerRegistry.get_legacy_default()
        default_display = None
        for display, tok_id in tokenizer_id_map.items():
            if tok_id == default_info.id:
                default_display = display
                break
        
        tokenizer_var.set(default_display if default_display else tokenizer_options[0] if tokenizer_options else "")
        
        tokenizer_combo = ttk.Combobox(
            tokenizer_row,
            textvariable=tokenizer_var,
            values=tokenizer_options,
            state="readonly",
            width=50
        )
        tokenizer_combo.pack(side="left")
        
        # Info label showing tokenizer details
        tokenizer_info_label = ttk.Label(
            parent,
            text="",
            foreground="gray",
            wraplength=500,
            justify="left"
        )
        tokenizer_info_label.pack(fill="x", pady=(2,0))
        
        # Update info when tokenizer changes
        def _on_tokenizer_change(*args):
            try:
                display_text = tokenizer_var.get()
                tokenizer_id = tokenizer_id_map.get(display_text)
                if tokenizer_id:
                    info = TokenizerRegistry.get(tokenizer_id)
                    if info:
                        desc = f"📊 {info.vocab_size:,} tokens | ⚡ {info.compression_ratio:.1f} chars/token | 💡 {info.description}"
                        if not TokenizerRegistry.check_installed(tokenizer_id, panel._project_root):
                            desc += f"\n⚠️  Not installed. Run: python scripts/download_tokenizer.py {tokenizer_id}"
                        tokenizer_info_label.config(text=desc)
            except Exception:
                pass
        
        tokenizer_var.trace_add("write", _on_tokenizer_change)
        _on_tokenizer_change()  # Initial update
        
    except Exception:
        # Fallback if tokenizer registry not available
        tokenizer_var.set("GPT-2 (50,257 tokens)")
        tokenizer_combo = ttk.Entry(tokenizer_row, textvariable=tokenizer_var, width=50, state="readonly")
        tokenizer_combo.pack(side="left")
        tokenizer_id_map = {"GPT-2 (50,257 tokens)": "gpt2-base-model"}
    
    return tokenizer_row, tokenizer_var, tokenizer_id_map


def _create_custom_architecture_section(parent: ttk.Frame) -> tuple:
    """Create custom architecture configuration UI (hidden by default)."""
    custom_frame = ttk.LabelFrame(parent, text="Custom Architecture")
    custom_frame.pack(fill="x", pady=(8,0))
    custom_frame.pack_forget()  # Hide initially
    
    custom_vars = {}
    
    # Two-column layout
    columns_frame = ttk.Frame(custom_frame)
    columns_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    left_col = ttk.Frame(columns_frame)
    left_col.pack(side="left", fill="both", expand=True, padx=(0, 5))
    
    right_col = ttk.Frame(columns_frame)
    right_col.pack(side="left", fill="both", expand=True, padx=(5, 0))
    
    # Helper to add custom row
    def _add_custom_row(label: str, var_name: str, default: str, width: int = 8, parent=None):
        if parent is None:
            parent = left_col
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        lbl = ttk.Label(row, text=label, width=18, anchor="e")
        lbl.pack(side="left")
        var = safe_variables.StringVar(value=default)
        custom_vars[var_name] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side="left")
        return var
    
    # Architecture fields (left column)
    _add_custom_row("Hidden size:", "hidden", "512", parent=left_col)
    _add_custom_row("H layers:", "h_layers", "2", 6, parent=left_col)
    _add_custom_row("L layers:", "l_layers", "2", 6, parent=left_col)
    _add_custom_row("Num heads:", "heads", "8", 6, parent=left_col)
    _add_custom_row("Expansion:", "expansion", "2.0", 8, parent=left_col)
    _add_custom_row("H cycles:", "h_cycles", "2", 6, parent=left_col)
    _add_custom_row("L cycles:", "l_cycles", "2", 6, parent=left_col)
    
    # Position encoding
    pos_row = ttk.Frame(left_col)
    pos_row.pack(fill="x", pady=2)
    pos_lbl = ttk.Label(pos_row, text="Position encoding:", width=18, anchor="e")
    pos_lbl.pack(side="left")
    pos_var = safe_variables.StringVar(value="rope")
    custom_vars["pos"] = pos_var
    pos_combo = ttk.Combobox(pos_row, textvariable=pos_var, width=10, state="readonly")
    pos_combo['values'] = ('rope', 'learned')
    pos_combo.pack(side="left")
    
    # Model dtype
    dtype_row = ttk.Frame(left_col)
    dtype_row.pack(fill="x", pady=2)
    dtype_lbl = ttk.Label(dtype_row, text="Model dtype:", width=18, anchor="e")
    dtype_lbl.pack(side="left")
    dtype_var = safe_variables.StringVar(value="fp32")
    custom_vars["dtype"] = dtype_var
    dtype_combo = ttk.Combobox(dtype_row, textvariable=dtype_var, width=10, state="readonly")
    dtype_combo['values'] = ('fp32', 'fp16', 'bf16')
    dtype_combo.pack(side="left")
    
    # MoE Configuration (right column)
    moe_header = ttk.Label(right_col, text="Sparse MoE", font=("TkDefaultFont", 9, "bold"))
    moe_header.pack(anchor="w", pady=(0, 5))
    
    # Use MoE checkbox
    moe_check_row = ttk.Frame(right_col)
    moe_check_row.pack(fill="x", pady=2)
    use_moe_var = safe_variables.BooleanVar(value=True)
    custom_vars["use_moe"] = use_moe_var
    moe_check = ttk.Checkbutton(moe_check_row, text="Enable sparse MoE\n(75% compute reduction)", variable=use_moe_var)
    moe_check.pack(anchor="w")
    
    # Num experts
    experts_row = ttk.Frame(right_col)
    experts_row.pack(fill="x", pady=2)
    experts_lbl = ttk.Label(experts_row, text="Num experts:", width=18, anchor="e")
    experts_lbl.pack(side="left")
    num_experts_var = safe_variables.StringVar(value="8")
    custom_vars["num_experts"] = num_experts_var
    experts_entry = ttk.Entry(experts_row, textvariable=num_experts_var, width=8)
    experts_entry.pack(side="left")
    
    # Experts per token
    experts_active_row = ttk.Frame(right_col)
    experts_active_row.pack(fill="x", pady=2)
    experts_active_lbl = ttk.Label(experts_active_row, text="Experts per token:", width=18, anchor="e")
    experts_active_lbl.pack(side="left")
    experts_per_tok_var = safe_variables.StringVar(value="2")
    custom_vars["num_experts_per_tok"] = experts_per_tok_var
    experts_per_tok_entry = ttk.Entry(experts_active_row, textvariable=experts_per_tok_var, width=8)
    experts_per_tok_entry.pack(side="left")
    
    return custom_frame, custom_vars


def _create_parameter_estimator(parent: ttk.Frame) -> tuple:
    """Create parameter estimation UI."""
    param_frame = ttk.LabelFrame(parent, text="Parameter Estimation")
    param_frame.pack(fill="x", pady=(8,0), padx=5)
    
    param_label = ttk.Label(param_frame, text="Estimating...", font=("TkDefaultFont", 9, "bold"), foreground="#0066cc")
    param_label.pack(pady=5, padx=5)
    
    breakdown_label = ttk.Label(param_frame, text="", font=("TkDefaultFont", 8), foreground="#666666")
    breakdown_label.pack(pady=(0,5), padx=5)
    
    # Note about DeepSpeed
    note_frame = ttk.Frame(parent)
    note_frame.pack(fill="x", pady=(8,2), padx=5)
    note_label = ttk.Label(note_frame, text="Note: DeepSpeed ZeRO can be selected in the main training panel", 
                          foreground="blue", font=("TkDefaultFont", 8, "italic"))
    note_label.pack(anchor="w")
    
    return param_frame, param_label, breakdown_label


def _setup_parameter_estimator_updates(custom_vars: dict, param_label: ttk.Label, breakdown_label: ttk.Label) -> None:
    """Hook up parameter estimator to update when fields change."""
    def _update_estimate(*args):
        try:
            # Get current values
            hidden = int(custom_vars["hidden"].get() or 512)
            h_layers = int(custom_vars["h_layers"].get() or 2)
            l_layers = int(custom_vars["l_layers"].get() or 2)
            heads = int(custom_vars["heads"].get() or 8)
            expansion = float(custom_vars["expansion"].get() or 2.0)
            use_moe = custom_vars["use_moe"].get()
            num_experts = int(custom_vars["num_experts"].get() or 8)
            num_experts_per_tok = int(custom_vars["num_experts_per_tok"].get() or 2)
            
            # Calculate parameters
            total_params, active_params, param_text, breakdown_text = estimate_parameters(
                hidden_size=hidden,
                h_layers=h_layers,
                l_layers=l_layers,
                num_heads=heads,
                expansion=expansion,
                use_moe=use_moe,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
            )
            
            param_label.config(text=param_text)
            breakdown_label.config(text=breakdown_text)
            
        except Exception as e:
            param_label.config(text="Invalid configuration")
            breakdown_label.config(text=str(e))
    
    # Trace relevant fields
    for var_name, var in custom_vars.items():
        if var_name in ["hidden", "h_layers", "l_layers", "heads", "expansion", "use_moe", "num_experts", "num_experts_per_tok"]:
            if isinstance(var, tk.BooleanVar):
                var.trace_add("write", _update_estimate)
            else:
                var.trace_add("write", _update_estimate)
    
    # Initial estimate
    _update_estimate()


def _get_selected_tokenizer(tokenizer_var: tk.StringVar, tokenizer_id_map: dict, panel: Any) -> tuple:
    """Get selected tokenizer ID and info."""
    try:
        from aios.core.tokenizers import TokenizerRegistry
        display_text = tokenizer_var.get()
        tokenizer_id = tokenizer_id_map.get(display_text, "gpt2-base-model")
        tokenizer_info = TokenizerRegistry.get(tokenizer_id)
        return tokenizer_id, tokenizer_info
    except Exception:
        # Fallback
        from collections import namedtuple
        TokenizerInfo = namedtuple("TokenizerInfo", ["id", "name", "path", "vocab_size"])
        return "gpt2-base-model", TokenizerInfo("gpt2-base-model", "GPT-2", "gpt2", 50257)


def _check_tokenizer_installed(tokenizer_id: str, project_root: str) -> bool:
    """Check if tokenizer is installed."""
    try:
        from aios.core.tokenizers import TokenizerRegistry
        return TokenizerRegistry.check_installed(tokenizer_id, project_root)
    except Exception:
        return True  # Assume installed if can't check
