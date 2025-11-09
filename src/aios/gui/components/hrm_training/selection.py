from __future__ import annotations

import os
from typing import Any


def select_student(panel: Any) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        top = tk.Toplevel(panel)
        top.title("Select HRM Student")
        top.grab_set()
        frm = ttk.Frame(top)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Existing brains:").pack(anchor="w")
        lb = tk.Listbox(frm, width=60, height=10)
        lb.pack(fill="both", expand=True)

        name_to_dir: dict[str, str] = {}
        try:
            base = os.path.join(panel._project_root, "artifacts", "brains", "actv1")
            if os.path.isdir(base):
                for entry in sorted(os.listdir(base)):
                    # Skip internal/DDP directories
                    if entry.startswith("_"):
                        continue
                    p = os.path.join(base, entry)
                    if os.path.isdir(p):
                        name_to_dir[entry] = p
                for name in sorted(name_to_dir.keys()):
                    lb.insert("end", name)
        except Exception:
            pass

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))

        def _use_selected():
            try:
                idxs = lb.curselection()
                if idxs:
                    sel_name = lb.get(idxs[0])
                    bdir = name_to_dir.get(sel_name)
                    if bdir:
                        pt_path = os.path.join(bdir, "actv1_student.safetensors")
                        panel.student_init_var.set(pt_path)
                        panel.log_file_var.set(os.path.join(bdir, "metrics.jsonl"))
                        panel.brain_name_var.set(sel_name)
                        
                        # Read brain.json to populate training steps and architecture
                        try:
                            import json
                            brain_json_path = os.path.join(bdir, "brain.json")
                            if os.path.isfile(brain_json_path):
                                with open(brain_json_path, 'r', encoding='utf-8') as f:
                                    brain_data = json.load(f)
                                
                                # Update training steps if available
                                if "training_steps" in brain_data:
                                    panel.steps_var.set(str(brain_data["training_steps"]))
                                    panel._log(f"[hrm] Loaded training steps: {brain_data['training_steps']}")
                                
                                # Trigger VRAM estimate update which will populate MoE/tokenizer
                                panel._update_vram_estimate()
                                
                                # Force update of MoE fields immediately from brain metadata
                                # (in case _update_vram_estimate doesn't trigger the MoE update path)
                                use_moe = brain_data.get("use_moe", False)
                                if use_moe:
                                    num_experts = brain_data.get("num_experts", 8)
                                    active_per_tok = brain_data.get("num_experts_per_tok", 2)
                                    panel.moe_num_experts_entry.config(state="normal")
                                    panel.moe_num_experts_entry.delete(0, "end")
                                    panel.moe_num_experts_entry.insert(0, str(num_experts))
                                    panel.moe_num_experts_entry.config(state="readonly")
                                    
                                    panel.moe_active_experts_entry.config(state="normal")
                                    panel.moe_active_experts_entry.delete(0, "end")
                                    panel.moe_active_experts_entry.insert(0, str(active_per_tok))
                                    panel.moe_active_experts_entry.config(state="readonly")
                                else:
                                    panel.moe_num_experts_entry.config(state="normal")
                                    panel.moe_num_experts_entry.delete(0, "end")
                                    panel.moe_num_experts_entry.insert(0, "N/A")
                                    panel.moe_num_experts_entry.config(state="readonly")
                                    
                                    panel.moe_active_experts_entry.config(state="normal")
                                    panel.moe_active_experts_entry.delete(0, "end")
                                    panel.moe_active_experts_entry.insert(0, "N/A")
                                    panel.moe_active_experts_entry.config(state="readonly")
                        except Exception as e:
                            panel._log(f"[hrm] Warning: Could not read brain.json: {e}")
                    
                    panel._set_arch_widgets_state("disabled")
                    panel._log(f"[hrm] Selected brain: {sel_name}")
                top.destroy()
            except Exception:
                top.destroy()

        def _browse_any():
            try:
                from tkinter import filedialog  # type: ignore
                path = filedialog.askopenfilename(initialdir=panel._project_root, filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
                if path:
                    panel.student_init_var.set(path)
                    panel._set_arch_widgets_state("disabled")
                    panel._log(f"[hrm] Selected student init: {path}")
                    try:
                        sdir = os.path.dirname(path)
                        panel.log_file_var.set(os.path.join(sdir, "metrics.jsonl"))
                        panel.brain_name_var.set(os.path.basename(sdir))
                    except Exception:
                        pass
                top.destroy()
            except Exception:
                top.destroy()

        def _create_new():
            try:
                import tkinter as tk
                from tkinter import ttk
                w = tk.Toplevel(top)
                w.title("Create New HRM Student")
                w.grab_set()
                inner = ttk.Frame(w)
                inner.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Preset selection
                preset_label = ttk.Label(inner, text="Choose architecture preset:")
                preset_label.pack(anchor="w")
                choice = tk.StringVar(value="5M")
                preset_buttons = {}
                for opt in ["1M", "5M", "10M", "20M", "50M", "Custom"]:
                    rb = ttk.Radiobutton(inner, text=opt, variable=choice, value=opt)
                    rb.pack(anchor="w")
                    preset_buttons[opt] = rb
                
                # Brain name
                name_row = ttk.Frame(inner)
                name_row.pack(fill="x", pady=(8,0))
                name_lbl = ttk.Label(name_row, text="Brain name:", width=18, anchor="e")
                name_lbl.pack(side="left")
                name_var = tk.StringVar(value=panel.brain_name_var.get().strip() or "new_brain")
                name_entry = ttk.Entry(name_row, textvariable=name_var, width=24)
                name_entry.pack(side="left")
                
                # Default goal
                goal_row = ttk.Frame(inner)
                goal_row.pack(fill="x", pady=(4,0))
                goal_lbl = ttk.Label(goal_row, text="Default goal:", width=18, anchor="e")
                goal_lbl.pack(side="left")
                goal_var = tk.StringVar(value="")
                goal_entry = ttk.Entry(goal_row, textvariable=goal_var, width=60)
                goal_entry.pack(side="left", fill="x", expand=True)
                
                # Tokenizer selection
                tokenizer_row = ttk.Frame(inner)
                tokenizer_row.pack(fill="x", pady=(4,0))
                tokenizer_lbl = ttk.Label(tokenizer_row, text="Tokenizer:", width=18, anchor="e")
                tokenizer_lbl.pack(side="left")
                
                # Import tokenizer registry
                tokenizer_var = tk.StringVar()
                tokenizer_combo = None
                tokenizer_info_label = None
                try:
                    from aios.core.tokenizers import TokenizerRegistry
                    
                    # Build tokenizer options with status indicators
                    tokenizer_options = []
                    tokenizer_id_map = {}  # Map display text to ID
                    for info in TokenizerRegistry.list_available():
                        installed = TokenizerRegistry.check_installed(info.id, panel._project_root)
                        status = "✓" if installed else "⚠"
                        display_text = f"{status} {info.name} ({info.vocab_size:,} tokens)"
                        tokenizer_options.append(display_text)
                        tokenizer_id_map[display_text] = info.id
                    
                    # Default to legacy GPT-2 base_model for backward compatibility
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
                        inner,
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
                    
                except Exception as e:
                    # Fallback if tokenizer registry not available
                    tokenizer_var.set("GPT-2 (50,257 tokens)")
                    tokenizer_combo = ttk.Entry(tokenizer_row, textvariable=tokenizer_var, width=50, state="readonly")
                    tokenizer_combo.pack(side="left")
                    tokenizer_id_map = {"GPT-2 (50,257 tokens)": "gpt2-base-model"}
                
                # Add tooltips for presets and brain name
                try:
                    from ..tooltips import add_tooltip
                    
                    add_tooltip(preset_label, 
                        "Quick architecture presets with pre-configured parameters.\n"
                        "Select Custom for full control over all parameters.")
                    
                    if "1M" in preset_buttons:
                        add_tooltip(preset_buttons["1M"], 
                            "~1M parameters: Tiny model (hidden=128, 2+2 layers, 8 experts)\n"
                            "Fast training, minimal VRAM (~0.5 GB)\n"
                            "MoE enabled: ~75% compute reduction")
                    if "5M" in preset_buttons:
                        add_tooltip(preset_buttons["5M"], 
                            "~5M parameters: Small model (hidden=256, 2+2 layers, 8 experts)\n"
                            "Good for testing and quick experiments (~1.5 GB)\n"
                            "MoE enabled: ~75% compute reduction")
                    if "10M" in preset_buttons:
                        add_tooltip(preset_buttons["10M"], 
                            "~10M parameters: Medium model (hidden=384, 2+2 layers, 8 experts)\n"
                            "Balanced size/performance (~2.5 GB)\n"
                            "MoE enabled: ~75% compute reduction")
                    if "20M" in preset_buttons:
                        add_tooltip(preset_buttons["20M"], 
                            "~20M parameters: Large model (hidden=512, 2+2 layers, 8 experts)\n"
                            "Good quality, moderate VRAM (~4 GB)\n"
                            "MoE enabled: ~75% compute reduction")
                    if "50M" in preset_buttons:
                        add_tooltip(preset_buttons["50M"], 
                            "~50M parameters: Very large (hidden=768, 2+2 layers, 12 experts)\n"
                            "High quality, needs more VRAM (~7 GB)\n"
                            "MoE enabled: ~75% compute reduction")
                    if "Custom" in preset_buttons:
                        add_tooltip(preset_buttons["Custom"], 
                            "Custom architecture: Configure all parameters manually.\n"
                            "Reveals advanced options for hidden size, layers, heads, etc.")
                    
                    add_tooltip(name_entry,
                        "Unique name for this brain/model.\n"
                        "Will be saved to: artifacts/brains/actv1/{name}/\n"
                        "Use descriptive names like: large_context_v1, fast_inference, etc.")
                    
                    add_tooltip(goal_entry,
                        "The primary training directive for this brain.\n"
                        "This goal guides what the brain learns during training.\n\n"
                        "Examples:\n"
                        "• Learn and understand English grammar and vocabulary\n"
                        "• Master Python programming and code generation\n"
                        "• Become an expert in mathematical problem-solving\n\n"
                        "Can be edited later in the Brains tab.")
                    
                    if tokenizer_combo:
                        add_tooltip(tokenizer_combo,
                            "Select the tokenizer for this brain (CANNOT be changed later!).\n\n"
                            "This determines how text is converted to tokens:\n"
                            "• GPT-2 (50K): Legacy, good for English, backward compatible\n"
                            "• Llama 3 (128K): Modern, best overall, excellent multilingual\n"
                            "• Mistral (32K): Efficient, good balance\n"
                            "• Code Llama (32K): Optimized for programming\n\n"
                            "✓ = Installed | ⚠ = Not yet downloaded\n"
                            "Different tokenizers = different vocab sizes = different model architecture\n"
                            "Choose carefully - this locks in the brain's vocabulary!")
                    
                except Exception:
                    pass  # Tooltips are optional
                
                # Custom architecture section (initially hidden)
                custom_frame = ttk.LabelFrame(inner, text="Custom Architecture")
                custom_frame.pack(fill="x", pady=(8,0))
                custom_frame.pack_forget()  # Hide initially
                
                # Custom fields
                custom_vars = {}
                custom_widgets = {}  # Track widgets for tooltips
                
                # Create two-column layout: Architecture on left, MoE on right
                columns_frame = ttk.Frame(custom_frame)
                columns_frame.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Left column: Architecture settings
                left_col = ttk.Frame(columns_frame)
                left_col.pack(side="left", fill="both", expand=True, padx=(0, 5))
                
                # Right column: MoE settings
                right_col = ttk.Frame(columns_frame)
                right_col.pack(side="left", fill="both", expand=True, padx=(5, 0))
                
                def _add_custom_row(label: str, var_name: str, default: str, width: int = 8, parent=None):
                    if parent is None:
                        parent = left_col
                    row = ttk.Frame(parent)
                    row.pack(fill="x", pady=2)
                    lbl = ttk.Label(row, text=label, width=18, anchor="e")
                    lbl.pack(side="left")
                    var = tk.StringVar(value=default)
                    custom_vars[var_name] = var
                    entry = ttk.Entry(row, textvariable=var, width=width)
                    entry.pack(side="left")
                    custom_widgets[var_name] = (lbl, entry)
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
                pos_var = tk.StringVar(value="rope")
                custom_vars["pos"] = pos_var
                pos_combo = ttk.Combobox(pos_row, textvariable=pos_var, width=10, state="readonly")
                pos_combo['values'] = ('rope', 'learned', 'sincos')
                pos_combo.pack(side="left")
                
                # Model dtype
                dtype_row = ttk.Frame(left_col)
                dtype_row.pack(fill="x", pady=2)
                dtype_lbl = ttk.Label(dtype_row, text="Model dtype:", width=18, anchor="e")
                dtype_lbl.pack(side="left")
                dtype_var = tk.StringVar(value="fp32")
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
                use_moe_var = tk.BooleanVar(value=True)  # Default: enabled
                custom_vars["use_moe"] = use_moe_var
                moe_check = ttk.Checkbutton(moe_check_row, text="Enable sparse MoE\n(75% compute reduction)", variable=use_moe_var)
                moe_check.pack(anchor="w")
                
                # Num experts
                experts_row = ttk.Frame(right_col)
                experts_row.pack(fill="x", pady=2)
                experts_lbl = ttk.Label(experts_row, text="Num experts:", width=18, anchor="e")
                experts_lbl.pack(side="left")
                num_experts_var = tk.StringVar(value="8")
                custom_vars["num_experts"] = num_experts_var
                experts_entry = ttk.Entry(experts_row, textvariable=num_experts_var, width=8)
                experts_entry.pack(side="left")
                
                # Experts per token
                experts_active_row = ttk.Frame(right_col)
                experts_active_row.pack(fill="x", pady=2)
                experts_active_lbl = ttk.Label(experts_active_row, text="Experts per token:", width=18, anchor="e")
                experts_active_lbl.pack(side="left")
                experts_per_tok_var = tk.StringVar(value="2")
                custom_vars["num_experts_per_tok"] = experts_per_tok_var
                experts_per_tok_entry = ttk.Entry(experts_active_row, textvariable=experts_per_tok_var, width=8)
                experts_per_tok_entry.pack(side="left")
                
                # Add tooltips to custom fields
                try:
                    from ..tooltips import add_tooltip
                    
                    # Hidden size
                    if "hidden" in custom_widgets:
                        lbl, entry = custom_widgets["hidden"]
                        add_tooltip(entry, 
                            "Model width / embedding dimension.\n"
                            "Larger = more expressive but more VRAM.\n"
                            "Must be divisible by num_heads.\n"
                            "Examples: 256, 512, 768, 1024, 1536, 2048")
                    
                    # H layers
                    if "h_layers" in custom_widgets:
                        lbl, entry = custom_widgets["h_layers"]
                        add_tooltip(entry,
                            "Number of Hierarchical reasoning layers.\n"
                            "Higher-level abstract processing.\n"
                            "More layers = deeper reasoning but slower.\n"
                            "Typical: 2-8 layers")
                    
                    # L layers
                    if "l_layers" in custom_widgets:
                        lbl, entry = custom_widgets["l_layers"]
                        add_tooltip(entry,
                            "Number of Local processing layers.\n"
                            "Lower-level detail processing.\n"
                            "More layers = better detail but slower.\n"
                            "Typical: 2-8 layers")
                    
                    # Num heads
                    if "heads" in custom_widgets:
                        lbl, entry = custom_widgets["heads"]
                        add_tooltip(entry,
                            "Number of attention heads per layer.\n"
                            "More heads = more parallel attention patterns.\n"
                            "Must evenly divide hidden_size.\n"
                            "Examples: 4, 8, 12, 16, 24, 32")
                    
                    # Expansion
                    if "expansion" in custom_widgets:
                        lbl, entry = custom_widgets["expansion"]
                        add_tooltip(entry,
                            "Feed-forward network expansion factor.\n"
                            "FFN size = hidden_size × expansion.\n"
                            "Higher = more capacity but more VRAM.\n"
                            "Typical: 2.0-4.0")
                    
                    # H cycles
                    if "h_cycles" in custom_widgets:
                        lbl, entry = custom_widgets["h_cycles"]
                        add_tooltip(entry,
                            "Number of processing cycles per H layer.\n"
                            "More cycles = more refinement per layer.\n"
                            "Typical: 1-3 cycles")
                    
                    # L cycles
                    if "l_cycles" in custom_widgets:
                        lbl, entry = custom_widgets["l_cycles"]
                        add_tooltip(entry,
                            "Number of processing cycles per L layer.\n"
                            "More cycles = more refinement per layer.\n"
                            "Typical: 1-3 cycles")
                    
                    # Position encoding
                    add_tooltip(pos_combo,
                        "Position encoding method:\n\n"
                        "• rope (Rotary): Best for long contexts,\n"
                        "  relative positions, no learned params.\n"
                        "  RECOMMENDED for most use cases.\n\n"
                        "• learned: Absolute positions,\n"
                        "  trained embeddings, fixed max length.\n\n"
                        "• sincos: Classic Transformer approach,\n"
                        "  no learned params, absolute positions.")
                    
                    # Model dtype
                    add_tooltip(dtype_combo,
                        "Model weight precision:\n\n"
                        "• fp32: Full precision (default)\n"
                        "  Most stable, highest memory usage\n\n"
                        "• fp16: Half precision\n"
                        "  50% less memory, faster training\n"
                        "  May have numerical instability\n\n"
                        "• bf16: BFloat16 (recommended)\n"
                        "  50% less memory, better stability than fp16\n"
                        "  Requires modern GPU (Ampere+)")
                    
                    # MoE tooltips
                    add_tooltip(moe_check,
                        "Sparse Mixture of Experts architecture:\n\n"
                        "• Increases total model capacity\n"
                        "• Reduces active computation by ~75%\n"
                        "• Each token routed to only 2 experts\n"
                        "• Better specialization, similar speed\n\n"
                        "Recommended: Enabled for better efficiency")
                    
                    add_tooltip(experts_entry,
                        "Total number of expert networks:\n\n"
                        "More experts = more specialization\n"
                        "but higher memory for weights.\n\n"
                        "Recommendations:\n"
                        "• 4-8: Good balance (default: 8)\n"
                        "• 8-12: Large models\n"
                        "• 12-16: Very diverse tasks\n\n"
                        "Memory scales linearly with experts.")
                    
                    add_tooltip(experts_per_tok_entry,
                        "Experts activated per token (sparsity):\n\n"
                        "Lower = more efficient, less capacity\n"
                        "Higher = more capacity, less efficient\n\n"
                        "Recommendations:\n"
                        "• 1: Maximum efficiency (87.5% reduction)\n"
                        "• 2: Balanced (75% reduction, default)\n"
                        "• 3-4: Best quality (less sparse)\n\n"
                        "Active compute = (experts_per_tok / num_experts)")
                    
                except Exception:
                    pass  # Tooltips are optional
                
                # Parameter estimator
                param_frame = ttk.LabelFrame(custom_frame, text="Parameter Estimation")
                param_frame.pack(fill="x", pady=(8,0), padx=5)
                
                param_label = ttk.Label(param_frame, text="Estimating...", font=("TkDefaultFont", 9, "bold"), foreground="#0066cc")
                param_label.pack(pady=5, padx=5)
                
                breakdown_label = ttk.Label(param_frame, text="", font=("TkDefaultFont", 8), foreground="#666666")
                breakdown_label.pack(pady=(0,5), padx=5)
                
                def _estimate_params(*args):
                    try:
                        # Get current values
                        hidden = int(custom_vars["hidden"].get() or 512)
                        h_layers = int(custom_vars["h_layers"].get() or 2)
                        l_layers = int(custom_vars["l_layers"].get() or 2)
                        heads = int(custom_vars["heads"].get() or 8)
                        expansion = float(custom_vars["expansion"].get() or 2.0)
                        vocab_size = 50257  # GPT-2 tokenizer size
                        
                        # Calculate base parameters
                        # Embeddings: vocab_size * hidden
                        embed_params = vocab_size * hidden
                        
                        # Per layer params (attention + ffn)
                        qkv_params = 3 * hidden * hidden  # Q, K, V projections
                        attn_out = hidden * hidden  # Output projection
                        ffn_params = 2 * hidden * int(hidden * expansion)  # Up + down projections
                        layer_params = qkv_params + attn_out + ffn_params
                        
                        # Total layer params
                        total_layer_params = layer_params * (h_layers + l_layers)
                        
                        # Base total (without MoE)
                        base_total = embed_params + total_layer_params
                        
                        # MoE adjustment
                        use_moe = custom_vars["use_moe"].get()
                        if use_moe:
                            num_experts = int(custom_vars["num_experts"].get() or 8)
                            # Each MoE layer replaces FFN with num_experts copies
                            moe_layers = h_layers + l_layers
                            base_ffn_params = 2 * hidden * int(hidden * expansion)
                            moe_ffn_params = base_ffn_params * num_experts
                            moe_overhead = (moe_ffn_params - base_ffn_params) * moe_layers
                            router_params = hidden * num_experts * moe_layers  # Router weights
                            total_params = base_total + moe_overhead + router_params
                            
                            # Calculate active params (sparse routing)
                            experts_per_tok = int(custom_vars["num_experts_per_tok"].get() or 2)
                            active_expert_params = base_ffn_params * experts_per_tok * moe_layers
                            inactive_expert_params = base_ffn_params * (num_experts - experts_per_tok) * moe_layers
                            active_total = base_total + active_expert_params - (base_ffn_params * moe_layers) + router_params
                            compute_reduction = (1 - active_total / total_params) * 100
                            
                            param_text = f"Total: {total_params/1e6:.1f}M params ({active_total/1e6:.1f}M active, ~{compute_reduction:.0f}% reduction)"
                            breakdown = (
                                f"Embeddings: {embed_params/1e6:.1f}M  •  "
                                f"Attention: {(qkv_params + attn_out) * (h_layers + l_layers)/1e6:.1f}M  •  "
                                f"MoE Experts: {moe_ffn_params * moe_layers/1e6:.1f}M ({num_experts} experts)\n"
                                f"Active per forward: {active_total/1e6:.1f}M ({experts_per_tok} experts/token)"
                            )
                        else:
                            total_params = base_total
                            param_text = f"Total: {total_params/1e6:.1f}M params (dense model, no sparsity)"
                            breakdown = (
                                f"Embeddings: {embed_params/1e6:.1f}M  •  "
                                f"Attention: {(qkv_params + attn_out) * (h_layers + l_layers)/1e6:.1f}M  •  "
                                f"FFN: {ffn_params * (h_layers + l_layers)/1e6:.1f}M"
                            )
                        
                        param_label.config(text=param_text)
                        breakdown_label.config(text=breakdown)
                        
                    except Exception as e:
                        param_label.config(text="Invalid configuration")
                        breakdown_label.config(text=str(e))
                
                # Update estimator when any field changes
                for var_name, var in custom_vars.items():
                    if var_name in ["hidden", "h_layers", "l_layers", "heads", "expansion", "use_moe", "num_experts", "num_experts_per_tok"]:
                        if isinstance(var, tk.BooleanVar):
                            var.trace_add("write", _estimate_params)
                        else:
                            var.trace_add("write", _estimate_params)
                
                # Initial estimate
                _estimate_params()
                
                # Note about DeepSpeed
                note_frame = ttk.Frame(custom_frame)
                note_frame.pack(fill="x", pady=(8,2), padx=5)
                note_label = ttk.Label(note_frame, text="Note: DeepSpeed ZeRO can be selected in the main training panel", 
                                      foreground="blue", font=("TkDefaultFont", 8, "italic"))
                note_label.pack(anchor="w")
                
                # Show/hide custom frame based on preset selection
                def _on_preset_change(*args):
                    if choice.get() == "Custom":
                        custom_frame.pack(fill="x", pady=(8,0))
                    else:
                        custom_frame.pack_forget()
                choice.trace_add("write", _on_preset_change)
                def _confirm_create():
                    try:
                        sel = (choice.get() or "").strip()
                        
                        # If Custom is selected, apply custom values directly
                        if sel == "Custom":
                            panel.hidden_size_var.set(custom_vars["hidden"].get())
                            panel.h_layers_var.set(custom_vars["h_layers"].get())
                            panel.l_layers_var.set(custom_vars["l_layers"].get())
                            panel.num_heads_var.set(custom_vars["heads"].get())
                            panel.expansion_var.set(custom_vars["expansion"].get())
                            panel.h_cycles_var.set(custom_vars["h_cycles"].get())
                            panel.l_cycles_var.set(custom_vars["l_cycles"].get())
                            panel.pos_enc_var.set(custom_vars["pos"].get())
                        else:
                            # Apply preset (1M, 5M, 10M, 20M, 50M)
                            panel._apply_preset(sel)
                        
                        # Set default goal if provided
                        goal_text = goal_var.get().strip()
                        if goal_text and hasattr(panel, 'default_goal_var'):
                            panel.default_goal_var.set(goal_text)
                            panel._log(f"[hrm] Set default goal: {goal_text[:60]}...")
                        
                        bname = (name_var.get().strip() or "new_brain").replace(" ", "_")
                        panel.brain_name_var.set(bname)
                        
                        bundle = os.path.join(panel._project_root, "artifacts", "brains", "actv1", bname)
                        os.makedirs(bundle, exist_ok=True)
                        panel.student_init_var.set(os.path.join(bundle, "actv1_student.safetensors"))
                        panel.log_file_var.set(os.path.join(bundle, "metrics.jsonl"))
                        panel._log(f"[hrm] New student {'custom' if sel == 'Custom' else 'preset ' + sel} architecture in bundle {bname}")
                        
                        # Store tokenizer selection in brain metadata
                        try:
                            from aios.core.tokenizers import TokenizerRegistry
                            import json
                            
                            # Get selected tokenizer info
                            display_text = tokenizer_var.get()
                            tokenizer_id = tokenizer_id_map.get(display_text, "gpt2-base-model")
                            tokenizer_info = TokenizerRegistry.get(tokenizer_id)
                            
                            if tokenizer_info:
                                # Create initial brain.json with tokenizer info
                                brain_json_path = os.path.join(bundle, "brain.json")
                                
                                # Get MoE parameters from custom vars
                                use_moe = custom_vars["use_moe"].get() if sel == "Custom" else False
                                num_experts = int(custom_vars["num_experts"].get() or 8) if sel == "Custom" else 8
                                num_experts_per_tok = int(custom_vars["num_experts_per_tok"].get() or 2) if sel == "Custom" else 2
                                
                                brain_metadata = {
                                    "name": bname,
                                    "type": "actv1",
                                    "tokenizer_id": tokenizer_info.id,
                                    "tokenizer_model": tokenizer_info.path,
                                    "vocab_size": tokenizer_info.vocab_size,
                                    "checkpoint_file": "actv1_student.safetensors",
                                    "log_file": "metrics.jsonl",
                                    "default_goal": goal_text if goal_text else f"Learn and improve through training.",
                                    "created_at": __import__('time').time(),
                                    # MoE configuration
                                    "use_moe": use_moe,
                                    "num_experts": num_experts,
                                    "num_experts_per_tok": num_experts_per_tok,
                                    # Architecture parameters (for reference)
                                    "hidden_size": int(custom_vars["hidden"].get() or 256) if sel == "Custom" else None,
                                    "h_layers": int(custom_vars["h_layers"].get() or 6) if sel == "Custom" else None,
                                    "l_layers": int(custom_vars["l_layers"].get() or 6) if sel == "Custom" else None,
                                    "num_heads": int(custom_vars["heads"].get() or 8) if sel == "Custom" else None,
                                    "expansion": float(custom_vars["expansion"].get() or 2.0) if sel == "Custom" else None,
                                }
                                
                                with open(brain_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(brain_metadata, f, indent=2)
                                
                                panel._log(f"[hrm] Tokenizer: {tokenizer_info.name} ({tokenizer_info.vocab_size:,} tokens)")
                                
                                # Warn if tokenizer not installed
                                if not TokenizerRegistry.check_installed(tokenizer_id, panel._project_root):
                                    panel._log(f"[hrm] ⚠️ WARNING: Tokenizer '{tokenizer_info.name}' is not installed!")
                                    panel._log(f"[hrm] Run: python scripts/download_tokenizer.py {tokenizer_id}")
                                    panel._log(f"[hrm] Training will fail until tokenizer is downloaded.")
                        except Exception as e:
                            panel._log(f"[hrm] Warning: Could not save tokenizer metadata: {e}")
                        
                        # Note: Default goal is configured in Brains tab, no need to log here
                        panel._set_arch_widgets_state("disabled")
                    except Exception:
                        pass
                    try:
                        w.destroy()
                    except Exception:
                        pass
                    try:
                        top.destroy()
                    except Exception:
                        pass
                btn_row = ttk.Frame(inner)
                btn_row.pack(fill="x", pady=(8,0))
                ttk.Button(btn_row, text="Create", command=_confirm_create).pack(side="left", padx=(0,6))
                ttk.Button(btn_row, text="Cancel", command=w.destroy).pack(side="left", padx=(6,0))
            except Exception:
                try:
                    top.destroy()
                except Exception:
                    pass

        ttk.Button(btns, text="Use Selected", command=_use_selected).pack(side="left")
        ttk.Button(btns, text="Browse…", command=_browse_any).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="Create New…", command=_create_new).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right")
    except Exception as e:
        panel._log(f"[hrm] Failed to open selection dialog: {e}")
