from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional
from .hrm_training import (
    on_start as _on_start_helper,
    on_stop as _on_stop_helper,
    stop_all as _stop_all_helper,
)
from .hrm_training import (
    poll_metrics as _poll_metrics_helper,
    show_stopped_dialog as _show_stopped_dialog_helper,
)
from .hrm_training import (
    select_student as _select_student_helper,
    open_rank_logs as _open_rank_logs_helper,
)
from .hrm_training.optimizer_progressive import optimize_from_gui_progressive

class HRMTrainingPanel(ttk.LabelFrame):  # type: ignore[misc]
    """ACTV1 HRM Training panel for configuring and launching training runs.

    This triggers the CLI com        # Handle optimization background thread
        try:
            bg_thread = getattr(self, \"_bg_thread\", None)
            if bg_thread is not None and bg_thread.is_alive():
                self._log(\"[hrm] Force stop: signaling optimization to terminate\")
                # The background thread should check _stop_requested and exit
                bg_thread.join(timeout=5.0)  # Increased timeout for proper termination
                if bg_thread.is_alive():
                    self._log(\"[hrm] Force stop: optimization thread still running, forcing termination\")
                    # Thread will terminate on next stop check
                else:
                    self._log(\"[hrm] Force stop: optimization thread terminated successfully\")
        except Exception as e:
            self._log(f\"[hrm] Force stop optimization error: {e}\")hrm-hf train-actv1` with configurable
    dataset path and a few training knobs. It's intended as a temporary tab.
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Optional[Callable[[str], None]] = None,
        save_state_fn: Optional[Callable[[], None]] = None,
        title: str = "HRM Training",
        worker_pool: Any = None,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter not available")
        super().__init__(parent, text=title)
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._save_state_fn = save_state_fn
        self._save_after_id: Optional[str] = None
        self._worker_pool = worker_pool  # Store worker pool for async operations

        def _schedule_save(delay_ms: int = 400) -> None:
            try:
                if not callable(self._save_state_fn):
                    return
                if self._save_after_id is not None:
                    try:
                        self.after_cancel(self._save_after_id)
                    except Exception:
                        pass
                self._save_after_id = self.after(delay_ms, lambda: self._save_state_fn() if callable(self._save_state_fn) else None)
            except Exception:
                pass
        self._schedule_save = _schedule_save  # type: ignore[attr-defined]

        # Inputs
        self.dataset_var = tk.StringVar(value="training_data")
        self.model_var = tk.StringVar(value="artifacts/hf_implant/base_model")
        self.max_seq_var = tk.StringVar(value="128")
        self.batch_var = tk.StringVar(value="4")
        self.steps_var = tk.StringVar(value="100")
        self._auto_steps_calculating = False  # Flag to prevent concurrent calculations
        self.lr_var = tk.StringVar(value="5e-5")
        self.auto_adjust_moe_lr_var = tk.BooleanVar(value=True)  # Default enabled for MoE stability
        self.halt_steps_var = tk.StringVar(value="1")
        self.gradient_checkpointing_var = tk.BooleanVar(value=True)  # Default enabled for better VRAM efficiency
        self.use_amp_var = tk.BooleanVar(value=True)  # Default enabled - saves ~40-50% memory
        self.use_cpu_offload_var = tk.BooleanVar(value=False)  # Default disabled - only for extreme contexts
        self.use_8bit_optimizer_var = tk.BooleanVar(value=False)  # Default disabled - for extreme contexts or large models
        self.use_flash_attn_var = tk.BooleanVar(value=False)  # Default disabled - enable for 50K+ contexts with sliding window
        self.flash_attn_window_var = tk.StringVar(value="512")  # Window size for Flash Attention 2
        self.use_chunked_training_var = tk.BooleanVar(value=False)  # Default disabled - auto-enables for 8K+ contexts
        self.chunk_size_var = tk.StringVar(value="2048")  # Default chunk size for extreme contexts
        
        # PEFT (Parameter-Efficient Fine-Tuning) options
        self.use_peft_var = tk.BooleanVar(value=False)  # Default disabled
        self.peft_method_var = tk.StringVar(value="lora")  # lora, adalora, ia3
        self.lora_r_var = tk.StringVar(value="16")  # Rank: 8 (minimal), 16 (balanced), 32 (high)
        self.lora_alpha_var = tk.StringVar(value="32")  # Scaling factor (typically 2×rank)
        self.lora_dropout_var = tk.StringVar(value="0.05")  # Dropout rate
        self.lora_target_modules_var = tk.StringVar(value="q_proj,v_proj")  # Target modules
        
        self.kl_var = tk.StringVar(value="0.0")
        self.kl_temp_var = tk.StringVar(value="1.0")
        # Strict mode is always enabled (no checkbox)
        self.strict_var = tk.BooleanVar(value=True)
        self.eval_file_var = tk.StringVar(value="")
        # eval_minutes removed - evaluation now runs only at end of training
        self.eval_batches_var = tk.StringVar(value="10")
        self.stop_file_var = tk.StringVar(value="training_data/actv1/STOP")
        self.log_file_var = tk.StringVar(value="artifacts/brains/actv1/default/metrics.jsonl")
        self.student_init_var = tk.StringVar(value="artifacts/brains/actv1/default/actv1_student.safetensors")
        self.brain_name_var = tk.StringVar(value="default")
        self.default_goal_var = tk.StringVar(value="")
        self.bundle_dir_var = tk.StringVar(value="artifacts/brains/actv1")
        # Architecture knobs
        self.h_layers_var = tk.StringVar(value="2")
        self.l_layers_var = tk.StringVar(value="2")
        self.hidden_size_var = tk.StringVar(value="512")
        self.expansion_var = tk.StringVar(value="2.0")
        self.num_heads_var = tk.StringVar(value="8")
        self.h_cycles_var = tk.StringVar(value="2")
        self.l_cycles_var = tk.StringVar(value="2")
        self.pos_enc_var = tk.StringVar(value="rope")
        # Data filtering
        self.ascii_only_var = tk.BooleanVar(value=False)
        # Dataset progression mode
        self.linear_dataset_var = tk.BooleanVar(value=True)  # True = linear/sequential (default), False = shuffled
        # DeepSpeed ZeRO optimization
        self.zero_stage_var = tk.StringVar(value="none")  # Options: none, zero1, zero2, zero3
        # DDP behavior toggles
        self.ddp_abort_on_fail_var = tk.BooleanVar(value=False)  # If True, abort run when DDP init fails

        # Auto-persist on changes (best-effort)
        try:
            _vars_to_watch = [
                self.dataset_var, self.model_var, self.max_seq_var, self.batch_var,
                self.steps_var, self.lr_var, self.halt_steps_var, self.gradient_checkpointing_var, 
                self.use_amp_var, self.use_cpu_offload_var, self.use_8bit_optimizer_var, self.use_flash_attn_var,
                self.use_chunked_training_var, self.chunk_size_var,
                self.use_peft_var, self.peft_method_var, self.lora_r_var, self.lora_alpha_var, self.lora_dropout_var, self.lora_target_modules_var,
                self.kl_var, self.kl_temp_var, self.stop_file_var, self.log_file_var, self.student_init_var,
                self.brain_name_var, self.default_goal_var, self.bundle_dir_var, self.ascii_only_var,
                self.linear_dataset_var,  # Persist dataset progression mode
                self.h_layers_var, self.l_layers_var, self.hidden_size_var, self.expansion_var,
                self.num_heads_var, self.h_cycles_var, self.l_cycles_var, self.pos_enc_var,
                self.zero_stage_var,  # Add ZeRO optimization to auto-save
                self.ddp_abort_on_fail_var,  # Persist DDP abort setting
            ]
            for v in _vars_to_watch:
                try:
                    # BooleanVar also supports trace_add
                    v.trace_add("write", lambda *args: self._schedule_save())  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Also update VRAM estimate when key parameters change
            _vram_vars = [
                self.max_seq_var, self.batch_var, self.h_layers_var, self.l_layers_var,
                self.hidden_size_var, self.expansion_var, self.num_heads_var,
                self.h_cycles_var, self.l_cycles_var, self.halt_steps_var,
                self.zero_stage_var,  # ZeRO affects VRAM usage
            ]
            for v in _vram_vars:
                try:
                    v.trace_add("write", lambda *args: self._update_vram_estimate())  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        # Layout
        g = ttk.Frame(self)
        g.pack(fill="x")

        # Core form: two-column grid for better alignment
        form = ttk.Frame(g)
        form.pack(fill="x")
        self._form_row = 0
        def _project_root() -> str:
            # Try to detect the project root (directory containing pyproject.toml), fallback to CWD
            try:
                cur = os.path.abspath(os.getcwd())
                for _ in range(8):
                    if os.path.exists(os.path.join(cur, "pyproject.toml")):
                        return cur
                    parent = os.path.dirname(cur)
                    if parent == cur:
                        break
                    cur = parent
                return os.path.abspath(os.getcwd())
            except Exception:
                return os.path.abspath(os.getcwd())

        self._project_root = _project_root()

        def _row(label: str, widget: Any, browse_for: Optional[str] = None) -> None:
            r = self._form_row
            ttk.Label(form, text=label, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 6), pady=2)
            widget.grid(row=r, column=1, sticky="we", pady=2)
            if browse_for:
                def _browse_cmd(kind: str = browse_for, target_widget: Any = widget):
                    try:
                        from tkinter import filedialog  # type: ignore
                        path = ""
                        if kind == "file":
                            path = filedialog.askopenfilename(initialdir=self._project_root)
                        elif kind == "dir":
                            path = filedialog.askdirectory(initialdir=self._project_root)
                        elif kind == "dataset":
                            # Smart dataset browser: supports both files and directories
                            path = filedialog.askdirectory(
                                initialdir=self._project_root,
                                title="Select dataset directory (or cancel to pick a file)"
                            )
                            if not path:
                                # User cancelled or wants a file instead
                                path = filedialog.askopenfilename(
                                    initialdir=self._project_root,
                                    title="Select dataset file",
                                    filetypes=[
                                        ("Text files", "*.txt"),
                                        ("CSV files", "*.csv"),
                                        ("JSON files", "*.json *.jsonl"),
                                        ("All files", "*.*")
                                    ]
                                )
                        if path:
                            try:
                                # Set the entry widget text directly
                                target_widget.delete(0, "end")
                                target_widget.insert(0, path)
                            except Exception:
                                pass
                    except Exception:
                        pass
                ttk.Button(form, text="Browse", command=_browse_cmd).grid(row=r, column=2, sticky="w", padx=(6,0))
            form.columnconfigure(1, weight=1)
            self._form_row += 1

        dataset_entry = ttk.Entry(form, textvariable=self.dataset_var, width=60)
        
        # Helper function to populate dataset quick select dropdown
        def _populate_dataset_dropdown():
            """Find available datasets in common locations and HuggingFace favorites."""
            import os
            datasets = []
            
            # Check common dataset locations
            locations = [
                ("Z:/training_datasets", "Z: drive"),
                ("training_data", "Local training_data"),
                ("artifacts/datasets", "Local artifacts"),
            ]
            
            for base_path, label in locations:
                try:
                    if os.path.isdir(base_path):
                        for entry in sorted(os.listdir(base_path)):
                            full_path = os.path.join(base_path, entry)
                            if os.path.isdir(full_path):
                                # Check if it looks like a dataset directory
                                if any(os.path.exists(os.path.join(full_path, f)) 
                                      for f in ["dataset_info.json", "data", "dataset.arrow"]):
                                    datasets.append(("local", full_path, f"{entry} ({label})"))
                except Exception:
                    pass
            
            # Add HuggingFace favorites
            try:
                from aios.gui.components.dataset_download_panel.favorites_manager import load_favorites
                favorites = load_favorites()
                for fav in favorites:
                    hf_path = fav.get("path", "")
                    if hf_path:
                        display_name = f"🤗 {fav.get('full_name', fav.get('name', 'Unknown'))} (HuggingFace)"
                        datasets.append(("huggingface", hf_path, display_name, fav))
            except Exception as e:
                self._log(f"[hrm] Could not load HuggingFace favorites: {e}")
            
            return datasets
        
        # Add quick select button for datasets
        def _show_dataset_selector():
            """Show a dialog to quickly select from available datasets."""
            try:
                import tkinter as tk
                from tkinter import ttk
                
                datasets = _populate_dataset_dropdown()
                if not datasets:
                    self._log("[hrm] No datasets found in common locations or HuggingFace favorites")
                    # Fall back to regular browse
                    _browse_cmd = _row.__code__.co_consts[1]  # Get the browse function
                    return
                
                # Create selection dialog
                dialog = tk.Toplevel(self)
                dialog.title("Select Dataset")
                dialog.grab_set()
                dialog.geometry("700x450")
                
                frame = ttk.Frame(dialog)
                frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                ttk.Label(
                    frame, 
                    text="Available datasets (local directories and HuggingFace favorites):"
                ).pack(anchor="w", pady=(0, 5))
                
                # Listbox with scrollbar
                list_frame = ttk.Frame(frame)
                list_frame.pack(fill="both", expand=True)
                
                scrollbar = ttk.Scrollbar(list_frame)
                scrollbar.pack(side="right", fill="y")
                
                listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=90, height=15)
                listbox.pack(side="left", fill="both", expand=True)
                scrollbar.config(command=listbox.yview)
                
                # Populate listbox
                dataset_info = []
                for item in datasets:
                    if item[0] == "local":
                        _, path, display_name = item
                        listbox.insert("end", display_name)
                        dataset_info.append(("local", path, None))
                    elif item[0] == "huggingface":
                        _, hf_path, display_name, fav_data = item
                        listbox.insert("end", display_name)
                        dataset_info.append(("huggingface", hf_path, fav_data))
                
                # Info label
                info_label = ttk.Label(
                    frame,
                    text="💡 Tip: HuggingFace datasets will be streamed during training",
                    font=("", 8, "italic"),
                    foreground="gray"
                )
                info_label.pack(anchor="w", pady=(5, 0))
                
                # Buttons
                btn_frame = ttk.Frame(frame)
                btn_frame.pack(fill="x", pady=(10, 0))
                
                def _select():
                    selection = listbox.curselection()
                    if selection:
                        dataset_type, dataset_path, dataset_data = dataset_info[selection[0]]
                        
                        if dataset_type == "local":
                            # Local dataset - set path directly
                            self.dataset_var.set(dataset_path)
                            self._log(f"[hrm] Selected local dataset: {dataset_path}")
                        elif dataset_type == "huggingface":
                            # HuggingFace dataset - store as special format: hf://dataset_path:config:split
                            # Default to 'default' config and 'train' split if not specified
                            hf_identifier = f"hf://{dataset_path}"
                            config = dataset_data.get("config") if dataset_data else None
                            split = dataset_data.get("split") if dataset_data else None
                            
                            # Always include config (default to 'default' if not specified)
                            if not config:
                                config = "default"
                            hf_identifier += f":{config}"
                            
                            # Always include split (default to 'train' if not specified)
                            if not split:
                                split = "train"
                            hf_identifier += f":{split}"
                            
                            self.dataset_var.set(hf_identifier)
                            self._log(f"[hrm] Selected HuggingFace dataset: {dataset_path}")
                            self._log(f"[hrm] Using config='{config}', split='{split}'")
                            self._log(f"[hrm] Dataset will be streamed from HuggingFace Hub")
                        
                    dialog.destroy()
                
                def _browse_instead():
                    dialog.destroy()
                    # Trigger the regular browse dialog
                    try:
                        from tkinter import filedialog
                        path = filedialog.askdirectory(
                            initialdir=self._project_root,
                            title="Select dataset directory"
                        )
                        if path:
                            self.dataset_var.set(path)
                    except Exception as e:
                        self._log(f"[hrm] Browse error: {e}")
                
                ttk.Button(btn_frame, text="Select", command=_select).pack(side="left", padx=(0, 5))
                ttk.Button(btn_frame, text="Browse...", command=_browse_instead).pack(side="left", padx=5)
                ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right")
                
                # Double-click to select
                listbox.bind("<Double-Button-1>", lambda e: _select())
                
            except Exception as e:
                self._log(f"[hrm] Dataset selector error: {e}")
        
        # Create a frame for dataset entry with two buttons
        dataset_frame = ttk.Frame(form)
        dataset_frame.grid(row=self._form_row, column=1, sticky="we", pady=2)
        dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_var, width=60)
        dataset_entry.pack(side="left", fill="x", expand=True)
        
        ttk.Label(form, text="Dataset file/dir:", anchor="e").grid(row=self._form_row, column=0, sticky="e", padx=(0, 6), pady=2)
        
        # Add both Select and Browse buttons
        btn_container = ttk.Frame(form)
        btn_container.grid(row=self._form_row, column=2, sticky="w", padx=(6,0))
        ttk.Button(btn_container, text="Select", command=_show_dataset_selector, width=7).pack(side="left", padx=(0, 2))
        
        def _browse_dataset():
            try:
                from tkinter import filedialog
                # Try directory first
                path = filedialog.askdirectory(
                    initialdir=self._project_root,
                    title="Select dataset directory (or cancel to pick a file)"
                )
                if not path:
                    # User cancelled or wants a file instead
                    path = filedialog.askopenfilename(
                        initialdir=self._project_root,
                        title="Select dataset file",
                        filetypes=[
                            ("Text files", "*.txt"),
                            ("CSV files", "*.csv"),
                            ("JSON files", "*.json *.jsonl"),
                            ("All files", "*.*")
                        ]
                    )
                if path:
                    self.dataset_var.set(path)
            except Exception as e:
                self._log(f"[hrm] Browse error: {e}")
        
        ttk.Button(btn_container, text="Browse", command=_browse_dataset, width=7).pack(side="left")
        
        form.columnconfigure(1, weight=1)
        self._form_row += 1
        
        # Dataset progression mode toggle (Linear vs Shuffled)
        dataset_mode_frame = ttk.Frame(form)
        linear_check = ttk.Checkbutton(
            dataset_mode_frame, 
            text="Linear progression (sequential order, enables position tracking for pause/resume)",
            variable=self.linear_dataset_var
        )
        linear_check.pack(side="left")
        ttk.Label(form, text="Dataset mode:", anchor="e").grid(row=self._form_row, column=0, sticky="e", padx=(0, 6), pady=2)
        dataset_mode_frame.grid(row=self._form_row, column=1, sticky="w", pady=2)
        self._form_row += 1
        
        try:
            from .tooltips import add_tooltip
            add_tooltip(linear_check, 
                "Linear (default): Process data sequentially [0,1,2,...]. Tracks position for pause/resume.\n"
                "Shuffled: Randomize order each epoch. Better for generalization.\n\n"
                "Use linear for: sequential data (stories), curriculum learning, precise tracking.\n"
                "Use shuffled for: general training, classification, better model generalization.")
        except Exception:
            pass

        _row("Context length:", ttk.Entry(form, textvariable=self.max_seq_var, width=8))
        # Batch size (fixed)
        batch_entry = ttk.Entry(form, textvariable=self.batch_var, width=8)
        _row("Batch size:", batch_entry)
        
        # Steps with Auto button
        r = self._form_row
        ttk.Label(form, text="Steps:", anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 6), pady=2)
        steps_frame = ttk.Frame(form)
        steps_frame.grid(row=r, column=1, sticky="we", pady=2)
        steps_entry = ttk.Entry(steps_frame, textvariable=self.steps_var, width=8)
        steps_entry.pack(side="left")
        auto_steps_btn = ttk.Button(steps_frame, text="Auto", width=6, command=self._auto_calculate_steps)
        auto_steps_btn.pack(side="left", padx=(4, 0))
        self._auto_steps_btn = auto_steps_btn
        self._form_row += 1
        # Learning rate and Halt max steps: hidden (use defaults)
        # _row("Learning rate:", ttk.Entry(form, textvariable=self.lr_var, width=10))
        # _row("Halt max steps:", ttk.Entry(form, textvariable=self.halt_steps_var, width=8))
        # Hide manual metrics log/student init inputs; they are auto-managed via bundle/selection
        # log_entry = ttk.Entry(form, textvariable=self.log_file_var, width=60)
        # _row("Metrics log file:", log_entry, browse_for="file")
        # student_entry = ttk.Entry(form, textvariable=self.student_init_var, width=60)
        # _row("Student init (.safetensors):", student_entry, browse_for="file")
        # Brain name (for bundle)
        _row("Brain name:", ttk.Entry(form, textvariable=self.brain_name_var, width=24))
        # Architecture presets row
        # Architecture presets section removed per UI simplification
        self._preset_buttons = []
        # Architecture group - Row 1: Layers, Hidden, Heads, Expansion, Cycles, PosEnc
        arch1 = ttk.Frame(g)
        arch1.pack(fill="x", pady=2)
        ttk.Label(arch1, text="H/L layers:", width=20, anchor="e").pack(side="left")
        self.h_layers_entry = ttk.Entry(arch1, textvariable=self.h_layers_var, width=4, state="readonly")
        self.h_layers_entry.pack(side="left")
        ttk.Label(arch1, text="/").pack(side="left")
        self.l_layers_entry = ttk.Entry(arch1, textvariable=self.l_layers_var, width=4, state="readonly")
        self.l_layers_entry.pack(side="left")
        ttk.Label(arch1, text=" Hidden:").pack(side="left")
        self.hidden_size_entry = ttk.Entry(arch1, textvariable=self.hidden_size_var, width=5, state="readonly")
        self.hidden_size_entry.pack(side="left")
        ttk.Label(arch1, text=" Heads:").pack(side="left")
        self.num_heads_entry = ttk.Entry(arch1, textvariable=self.num_heads_var, width=4, state="readonly")
        self.num_heads_entry.pack(side="left")
        ttk.Label(arch1, text=" Exp:").pack(side="left", padx=(8, 0))
        self.expansion_entry = ttk.Entry(arch1, textvariable=self.expansion_var, width=5, state="readonly")
        self.expansion_entry.pack(side="left")
        ttk.Label(arch1, text=" H/L cycles:").pack(side="left", padx=(8, 0))
        self.h_cycles_entry = ttk.Entry(arch1, textvariable=self.h_cycles_var, width=3, state="readonly")
        self.h_cycles_entry.pack(side="left")
        ttk.Label(arch1, text="/").pack(side="left")
        self.l_cycles_entry = ttk.Entry(arch1, textvariable=self.l_cycles_var, width=3, state="readonly")
        self.l_cycles_entry.pack(side="left")
        ttk.Label(arch1, text=" PosEnc:").pack(side="left", padx=(8, 0))
        self.pos_enc_entry = ttk.Entry(arch1, textvariable=self.pos_enc_var, width=8, state="readonly")
        self.pos_enc_entry.pack(side="left")
        
        # Row 2: MoE, Tokenizer, Params, Size, Steps - everything on one line
        arch2 = ttk.Frame(g)
        arch2.pack(fill="x", pady=2)
        ttk.Label(arch2, text="MoE:", width=20, anchor="e").pack(side="left")
        self.moe_num_experts_entry = ttk.Entry(arch2, width=3, state="readonly")
        self.moe_num_experts_entry.pack(side="left")
        ttk.Label(arch2, text="/").pack(side="left")
        self.moe_active_experts_entry = ttk.Entry(arch2, width=3, state="readonly")
        self.moe_active_experts_entry.pack(side="left")
        ttk.Label(arch2, text=" Tok:").pack(side="left", padx=(8, 0))
        self.tokenizer_entry = ttk.Entry(arch2, width=12, state="readonly")
        self.tokenizer_entry.pack(side="left")
        ttk.Label(arch2, text=" Params:").pack(side="left", padx=(8, 0))
        self.total_params_entry = ttk.Entry(arch2, width=8, state="readonly")
        self.total_params_entry.pack(side="left")
        ttk.Label(arch2, text="/").pack(side="left")
        self.current_params_entry = ttk.Entry(arch2, width=8, state="readonly")
        self.current_params_entry.pack(side="left")
        ttk.Label(arch2, text=" MB:").pack(side="left", padx=(8, 0))
        self.size_mb_entry = ttk.Entry(arch2, width=7, state="readonly")
        self.size_mb_entry.pack(side="left")
        ttk.Label(arch2, text=" Steps:").pack(side="left", padx=(8, 0))
        self.trained_steps_entry = ttk.Entry(arch2, width=8, state="readonly")
        self.trained_steps_entry.pack(side="left")
        
        # Track architecture widgets (including MoE, Tokenizer, and Stats)
        self._arch_widgets = [
            self.h_layers_entry,
            self.l_layers_entry,
            self.hidden_size_entry,
            self.num_heads_entry,
            self.expansion_entry,
            self.h_cycles_entry,
            self.l_cycles_entry,
            self.pos_enc_entry,
            self.moe_num_experts_entry,
            self.moe_active_experts_entry,
            self.tokenizer_entry,
            self.total_params_entry,
            self.current_params_entry,
            self.size_mb_entry,
            self.trained_steps_entry,
        ]
        # Tooltips (best-effort; ignore failures silently)
        try:  # pragma: no cover - UI enhancement only
            from .tooltips import add_tooltip
            add_tooltip(dataset_entry, "Dataset file or directory to feed into HRM training.")
            add_tooltip(batch_entry, "Training batch size (fixed; internal OOM backoff may adjust during runtime).")
            # Brain configuration tooltips
            brain_name_widget = form.winfo_children()[form.grid_size()[1] * 2 - 2]  # Get brain name entry
            add_tooltip(brain_name_widget, "Unique identifier for this brain. Used to organize training artifacts and goals.")
            # Inputs hidden: metrics log and student init are managed automatically via bundle/selection
        except Exception:
            pass
        # Helper to enable/disable architecture and preset controls
        def _set_arch_widgets_state(state: str) -> None:
            try:
                for w in self._arch_widgets:
                    # Treat 'disabled' requests as 'readonly' to allow programmatic updates but prevent typing
                    w.config(state=("readonly" if state in {"disabled", "normal"} else state))
                for b in getattr(self, "_preset_buttons", []) or []:
                    try:
                        b.config(state=state)
                    except Exception:
                        pass
            except Exception:
                pass
        self._set_arch_widgets_state = _set_arch_widgets_state  # type: ignore[attr-defined]

        # ============================================================================
        # OPTIMIZATIONS SECTION
        # ============================================================================
        opt_frame = ttk.LabelFrame(g, text="🚀 Optimizations", padding=10)
        opt_frame.pack(fill="x", pady=(10, 5))
        
        # Row 1: Memory Optimizations
        opt_row1 = ttk.Frame(opt_frame)
        opt_row1.pack(fill="x", pady=2)
        ttk.Label(opt_row1, text="Memory:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        grad_ckpt_btn = ttk.Checkbutton(opt_row1, text="Grad Checkpoint ✓", variable=self.gradient_checkpointing_var)
        grad_ckpt_btn.pack(side="left")
        amp_btn = ttk.Checkbutton(opt_row1, text="AMP ✓", variable=self.use_amp_var)
        amp_btn.pack(side="left", padx=(8, 0))
        opt8bit_btn = ttk.Checkbutton(opt_row1, text="8-bit Optimizer", variable=self.use_8bit_optimizer_var)
        opt8bit_btn.pack(side="left", padx=(8, 0))
        cpu_offload_btn = ttk.Checkbutton(opt_row1, text="CPU Offload", variable=self.use_cpu_offload_var)
        cpu_offload_btn.pack(side="left", padx=(8, 0))
        
        # Row 2: PEFT (Parameter-Efficient Fine-Tuning)
        peft_row = ttk.Frame(opt_frame)
        peft_row.pack(fill="x", pady=2)
        ttk.Label(peft_row, text="PEFT:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        peft_enable_btn = ttk.Checkbutton(peft_row, text="Enable LoRA", variable=self.use_peft_var)
        peft_enable_btn.pack(side="left")
        ttk.Label(peft_row, text="Method:").pack(side="left", padx=(10, 2))
        peft_method_combo = ttk.Combobox(peft_row, textvariable=self.peft_method_var, width=8, state="readonly")
        peft_method_combo['values'] = ('lora', 'adalora', 'ia3')
        peft_method_combo.pack(side="left")
        ttk.Label(peft_row, text="Rank:").pack(side="left", padx=(8, 2))
        lora_r_entry = ttk.Entry(peft_row, textvariable=self.lora_r_var, width=6)
        lora_r_entry.pack(side="left")
        ttk.Label(peft_row, text="Alpha:").pack(side="left", padx=(6, 2))
        lora_alpha_entry = ttk.Entry(peft_row, textvariable=self.lora_alpha_var, width=6)
        lora_alpha_entry.pack(side="left")
        ttk.Label(peft_row, text="Dropout:").pack(side="left", padx=(6, 2))
        lora_dropout_entry = ttk.Entry(peft_row, textvariable=self.lora_dropout_var, width=6)
        lora_dropout_entry.pack(side="left")
        ttk.Label(peft_row, text="Modules:").pack(side="left", padx=(8, 2))
        self.lora_modules_combo = ttk.Combobox(peft_row, width=10, state="readonly")
        self.lora_modules_combo['values'] = ('Minimal', 'Balanced', 'Full')
        self.lora_modules_combo.pack(side="left")
        
        # Map display names to actual module strings
        self._lora_module_map = {
            'Minimal': 'q_proj,v_proj',
            'Balanced': 'q_proj,k_proj,v_proj,o_proj',
            'Full': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
        }
        self._lora_module_reverse_map = {v: k for k, v in self._lora_module_map.items()}
        
        # Set initial display value
        initial_val = self.lora_target_modules_var.get()
        if initial_val in self._lora_module_reverse_map:
            self.lora_modules_combo.set(self._lora_module_reverse_map[initial_val])
        else:
            self.lora_modules_combo.set('Minimal')
            self.lora_target_modules_var.set('q_proj,v_proj')
        
        # Bind selection event to update actual value
        def _on_module_select(event=None):
            display_val = self.lora_modules_combo.get()
            if display_val in self._lora_module_map:
                self.lora_target_modules_var.set(self._lora_module_map[display_val])
        self.lora_modules_combo.bind('<<ComboboxSelected>>', _on_module_select)
        
        # Bind variable changes to update combo display (for config loading)
        def _on_var_change(*args):
            actual_val = self.lora_target_modules_var.get()
            if actual_val in self._lora_module_reverse_map:
                self.lora_modules_combo.set(self._lora_module_reverse_map[actual_val])
        self.lora_target_modules_var.trace_add('write', _on_var_change)
        
        # Row 3: Advanced Optimizations
        opt_row2 = ttk.Frame(opt_frame)
        opt_row2.pack(fill="x", pady=2)
        ttk.Label(opt_row2, text="Advanced:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        flash_attn_btn = ttk.Checkbutton(opt_row2, text="FlashAttn-2", variable=self.use_flash_attn_var)
        flash_attn_btn.pack(side="left")
        ttk.Label(opt_row2, text="Window:").pack(side="left", padx=(10, 2))
        flash_window_entry = ttk.Entry(opt_row2, textvariable=self.flash_attn_window_var, width=6)
        flash_window_entry.pack(side="left")
        
        # Row 4: DeepSpeed ZeRO
        zero_row = ttk.Frame(opt_frame)
        zero_row.pack(fill="x", pady=2)
        ttk.Label(zero_row, text="DeepSpeed:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        ttk.Label(zero_row, text="ZeRO Stage:").pack(side="left", padx=(0, 2))
        zero_combo = ttk.Combobox(zero_row, textvariable=self.zero_stage_var, width=8, state="readonly")
        zero_combo['values'] = ('none', 'zero1', 'zero2', 'zero3')
        zero_combo.pack(side="left")
        self.zero_savings_lbl = ttk.Label(zero_row, text="")
        self.zero_savings_lbl.pack(side="left", padx=(10, 0))
        
        # Row 5: Chunked Training
        chunk_row = ttk.Frame(opt_frame)
        chunk_row.pack(fill="x", pady=2)
        ttk.Label(chunk_row, text="Chunked:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        chunk_enable_btn = ttk.Checkbutton(chunk_row, text="Enable", variable=self.use_chunked_training_var)
        chunk_enable_btn.pack(side="left")
        ttk.Label(chunk_row, text="Chunk Size:").pack(side="left", padx=(10, 2))
        chunk_size_combo = ttk.Combobox(chunk_row, textvariable=self.chunk_size_var, width=8, state="readonly")
        chunk_size_combo['values'] = ('32', '64', '128', '256', '512', '1024', '2048', '4096', '8192')
        chunk_size_combo.pack(side="left")
        self.chunk_info_lbl = ttk.Label(chunk_row, text="")
        self.chunk_info_lbl.pack(side="left", padx=(10, 0))
        
        # Row 6: MoE Learning Rate Auto-Adjust
        moe_lr_row = ttk.Frame(opt_frame)
        moe_lr_row.pack(fill="x", pady=2)
        ttk.Label(moe_lr_row, text="MoE LR:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        moe_lr_auto_btn = ttk.Checkbutton(moe_lr_row, text="Auto-adjust for stability", variable=self.auto_adjust_moe_lr_var)
        moe_lr_auto_btn.pack(side="left")
        ttk.Label(moe_lr_row, text="Manual LR:").pack(side="left", padx=(10, 2))
        lr_entry = ttk.Entry(moe_lr_row, textvariable=self.lr_var, width=10)
        lr_entry.pack(side="left")
        self.moe_lr_info_lbl = ttk.Label(moe_lr_row, text="")
        self.moe_lr_info_lbl.pack(side="left", padx=(10, 0))
        
        # Individual tooltips for each element
        try:  # pragma: no cover
            from .tooltips import add_tooltip
            # Memory optimizations
            add_tooltip(grad_ckpt_btn, "Gradient Checkpointing: Trades computation for memory by\nrecomputing activations during backward pass\n↓30-50% VRAM usage • Enabled by default")
            add_tooltip(amp_btn, "Automatic Mixed Precision: Uses FP16/BF16 for faster training\n↓40-50% VRAM • +20% speed • Enabled by default")
            add_tooltip(opt8bit_btn, "8-bit Optimizer: Stores optimizer states in INT8\n↓75% optimizer memory • Minimal accuracy impact")
            add_tooltip(cpu_offload_btn, "CPU Offload: Moves optimizer states to system RAM\nSaves VRAM • ~30% slower training")
            
            # PEFT options
            add_tooltip(peft_enable_btn, "Enable PEFT: Use Low-Rank Adaptation (LoRA) for efficient fine-tuning\n↓95-99% trainable parameters (87M → 500K-2M)")
            add_tooltip(peft_method_combo, "PEFT Method:\n• LoRA: Low-Rank Adaptation (best balance)\n• AdaLoRA: Adaptive rank allocation\n• IA3: Fewer parameters than LoRA")
            add_tooltip(lora_r_entry, "LoRA Rank: Controls adapter capacity\n8=minimal • 16=balanced (default) • 32=high quality")
            add_tooltip(lora_alpha_entry, "LoRA Alpha: Scaling factor for adapter weights\nTypically 2× rank (default: 32 for rank 16)")
            add_tooltip(lora_dropout_entry, "LoRA Dropout: Regularization to prevent overfitting\n0.0=none • 0.05=default • 0.1-0.3=high regularization")
            add_tooltip(self.lora_modules_combo, "Target Modules: Which layers apply LoRA\nMinimal (~500K) • Balanced (~2M) • Full (~8M params)")
            
            # Advanced options
            add_tooltip(flash_attn_btn, "Flash Attention 2: Optimized attention for long contexts\nBest for 50K+ tokens • Requires compatible GPU")
            add_tooltip(flash_window_entry, "Window Size: Sliding window size for Flash Attention 2\nDefault: 512 tokens • Range: 256-8192\nLarger = more context, more memory")
            add_tooltip(zero_combo, "DeepSpeed ZeRO: Distributed memory optimization\n• none: Standard training\n• zero1: Partition optimizer states (↓25% VRAM)\n• zero2: Partition optimizer + gradients (↓50% VRAM) [RECOMMENDED]\n• zero3: Partition everything (↓75% VRAM, slower)")
            add_tooltip(chunk_enable_btn, "Chunked Training: Split long sequences into smaller chunks\nReduces memory for extreme contexts (8K+ tokens)")
            add_tooltip(chunk_size_combo, "Chunk Size: Powers of 2 from 32-8192 tokens\nSmaller = less VRAM, slower • 2048-4096 typical")
            add_tooltip(moe_lr_auto_btn, "MoE Auto-Adjust: Automatically reduces learning rate for MoE models\nUnchecked = use manual LR value • Checked = auto-reduce to 1e-6 or 2e-6")
            add_tooltip(lr_entry, "Learning Rate: Controls training step size\n5e-5 typical for dense models • 1e-6 to 2e-6 for MoE\nAuto-adjust checkbox overrides this if enabled")
            add_tooltip(moe_lr_auto_btn, "MoE Auto-Adjust: Automatically reduces learning rate for MoE models\nUnchecked = use manual LR value • Checked = auto-reduce to 1e-6 or 2e-6")
            add_tooltip(lr_entry, "Learning Rate: Controls training step size\n5e-5 typical for dense models • 1e-6 to 2e-6 for MoE\nAuto-adjust checkbox overrides this if enabled")
        except Exception:
            pass
        
        # ============================================================================
        # END OPTIMIZATIONS SECTION
        # ============================================================================
        
        # Update ZeRO savings label when selection changes
        def _update_zero_label(*args):
            try:
                stage = self.zero_stage_var.get()
                if stage == "none":
                    self.zero_savings_lbl.config(text="")
                elif stage == "zero1":
                    self.zero_savings_lbl.config(text="↓25% VRAM")
                elif stage == "zero2":
                    self.zero_savings_lbl.config(text="↓50% VRAM (recommended)")
                elif stage == "zero3":
                    self.zero_savings_lbl.config(text="↓75% VRAM")
            except Exception:
                pass
        self.zero_stage_var.trace_add("write", _update_zero_label)
        _update_zero_label()  # Initial update
        
        # Update chunk info label when checkbox or chunk size changes
        def _update_chunk_label(*args):
            try:
                if self.use_chunked_training_var.get():
                    chunk_size = self.chunk_size_var.get().strip() or "2048"
                    self.chunk_info_lbl.config(text=f"Active: {chunk_size} token chunks")
                else:
                    self.chunk_info_lbl.config(text="")
            except Exception:
                pass
        self.use_chunked_training_var.trace_add("write", _update_chunk_label)
        self.chunk_size_var.trace_add("write", _update_chunk_label)
        _update_chunk_label()  # Initial update

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(6, 0))
        self.start_btn = ttk.Button(btns, text="Start HRM Training", command=self._on_start)
        self.start_btn.pack(side="left")

        # Optimize button with experimental label
        opt_frame = ttk.Frame(btns)
        opt_frame.pack(side="left", padx=(6, 0))
        opt_btn = ttk.Button(opt_frame, text="Optimize Settings", command=self._on_optimize)
        opt_btn.pack(side="left")
        ttk.Label(opt_frame, text="(Experimental)", font=("", 7), foreground="gray").pack(side="left", padx=(4, 0))

        ttk.Button(btns, text="Select Student", command=self._on_select_student).pack(side="left", padx=(6, 0))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop_all)
        self.stop_btn.pack(side="left", padx=(6, 0))

        # Iterate checkbox (continuous training mode)
        self.iterate_var = tk.BooleanVar(value=False)
        iterate_check = ttk.Checkbutton(btns, text="Iterate Mode", variable=self.iterate_var)
        iterate_check.pack(side="left", padx=(6, 0))
        
        # Stop after epoch completion checkbox
        self.stop_after_epoch_var = tk.BooleanVar(value=False)
        stop_after_epoch_check = ttk.Checkbutton(btns, text="Stop After Epoch", variable=self.stop_after_epoch_var)
        stop_after_epoch_check.pack(side="left", padx=(6, 0))

        # DDP failure behavior toggle (applies when multi-GPU is selected)
        ddp_abort_btn = ttk.Checkbutton(btns, text="Abort on DDP init fail", variable=self.ddp_abort_on_fail_var)
        ddp_abort_btn.pack(side="left", padx=(6, 0))

        # Clear Output button (far right)
        clear_btn = ttk.Button(btns, text="Clear Output", command=self._clear_output)
        clear_btn.pack(side="right")
        
        try:  # pragma: no cover
            from .tooltips import add_tooltip
            add_tooltip(self.start_btn, "Launch training with current settings.")
            add_tooltip(self.stop_btn, "Stop all running training processes.")
            add_tooltip(iterate_check, "Iterate Mode: Continuous training cycles until manually stopped.")
            add_tooltip(stop_after_epoch_check, "Stop After Epoch: Complete the current epoch then stop gracefully.\nUseful for pausing training at natural checkpoints.")
            add_tooltip(clear_btn, "Clear the training output log.")
            add_tooltip(ddp_abort_btn, "When enabled and DDP init fails, abort the run instead of falling back to single-GPU.")
        except Exception:
            pass

        # Progress bar
        p = ttk.Frame(self)
        p.pack(fill="x", pady=(6, 0))
        ttk.Label(p, text="Progress:").pack(side="left")
        self.progress = ttk.Progressbar(p, orient="horizontal", mode="determinate", length=240, maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.progress_lbl = ttk.Label(p, text="idle")
        self.progress_lbl.pack(side="left")
        # Removed DDP backend labels and overrides; DDP is driven by Resources tab automatically

        # Log output (shorter to make room for live metrics and status bar)
        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        
        log_scrollbar = ttk.Scrollbar(log_frame)
        log_scrollbar.pack(side="right", fill="y")
        
        # Apply theme-aware colors to Text widget
        theme_colors = self._get_theme_colors()
        self.log = tk.Text(
            log_frame, 
            height=8, 
            wrap="word", 
            yscrollcommand=log_scrollbar.set,
            bg=theme_colors["bg"],
            fg=theme_colors["fg"],
            selectbackground=theme_colors["selectbg"],
            selectforeground=theme_colors["selectfg"],
            insertbackground=theme_colors["insertbg"]
        )
        self.log.pack(side="left", fill="both", expand=True)
        
        log_scrollbar.config(command=self.log.yview)
        
        try:  # pragma: no cover
            from .tooltips import add_tooltip
            add_tooltip(self.log, "Live CLI output and training logs.")
            add_tooltip(self.progress, "Relative progress (if determinable) or indeterminate spinner during startup.")
        except Exception:
            pass

        # Live metrics panel
        m = ttk.LabelFrame(self, text="Live metrics")
        m.pack(fill="x", pady=(6, 0))
        def _mlbl(lbl: str):
            r = ttk.Frame(m)
            r.pack(side="left", padx=8)
            ttk.Label(r, text=lbl+":").pack(side="left")
            v = ttk.Label(r, text="-")
            v.pack(side="left")
            return (r, v)  # type: ignore[return-value]
        _, self.met_step = _mlbl("step")
        _, self.met_loss = _mlbl("loss")
        _, self.met_ce = _mlbl("ce_token")
        _, self.met_ppl = _mlbl("ppl")
        _, self.met_tok = _mlbl("token_acc")
        _, self.met_exact = _mlbl("exact_match")
        
        # Memory estimation panels (VRAM and RAM side by side)
        mem_container = ttk.Frame(self)
        mem_container.pack(fill="x", pady=(6, 0))
        
        # VRAM estimation panel
        vram_panel = ttk.LabelFrame(mem_container, text="📊 Estimated VRAM per GPU")
        vram_panel.pack(side="left", fill="both", expand=True, padx=(0, 4))
        vram_frame = ttk.Frame(vram_panel)
        vram_frame.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(vram_frame, text="Model:").pack(side="left", padx=(0, 4))
        self.vram_model_lbl = ttk.Label(vram_frame, text="-")
        self.vram_model_lbl.pack(side="left", padx=(0, 12))
        
        ttk.Label(vram_frame, text="Optimizer:").pack(side="left", padx=(0, 4))
        self.vram_optimizer_lbl = ttk.Label(vram_frame, text="-")
        self.vram_optimizer_lbl.pack(side="left", padx=(0, 12))
        
        ttk.Label(vram_frame, text="Acts+Grads:").pack(side="left", padx=(0, 4))
        self.vram_activations_lbl = ttk.Label(vram_frame, text="-")
        self.vram_activations_lbl.pack(side="left", padx=(0, 12))
        
        ttk.Label(vram_frame, text="Total:").pack(side="left", padx=(0, 4))
        self.vram_total_lbl = ttk.Label(vram_frame, text="-", font=("TkDefaultFont", 10, "bold"))
        self.vram_total_lbl.pack(side="left")
        
        # RAM estimation panel
        ram_panel = ttk.LabelFrame(mem_container, text="💾 Estimated System RAM")
        ram_panel.pack(side="left", fill="both", expand=True, padx=(4, 0))
        ram_frame = ttk.Frame(ram_panel)
        ram_frame.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(ram_frame, text="Dataset:").pack(side="left", padx=(0, 4))
        self.ram_dataset_lbl = ttk.Label(ram_frame, text="-")
        self.ram_dataset_lbl.pack(side="left", padx=(0, 12))
        
        ttk.Label(ram_frame, text="Offloaded:").pack(side="left", padx=(0, 4))
        self.ram_offload_lbl = ttk.Label(ram_frame, text="-")
        self.ram_offload_lbl.pack(side="left", padx=(0, 12))
        
        ttk.Label(ram_frame, text="Total:").pack(side="left", padx=(0, 4))
        self.ram_total_lbl = ttk.Label(ram_frame, text="-", font=("TkDefaultFont", 10, "bold"))
        self.ram_total_lbl.pack(side="left")
        
        self._metrics_polling_active = False
        self._run_in_progress = False
        self._last_gen_total = None
        self._last_steps_total = None
        self._gen_hist = []  # list[tuple[int,float]] for ETA during generation
        self._step_hist = []  # list[tuple[int,float]] for ETA during training
        self._stopped_dialog_shown = False
        self._bg_thread = None
        self._proc = None  # active subprocess handle (torchrun or direct) if using external process
        self._stop_requested = False  # flag set when user presses Stop
        self._last_heartbeat = None  # timestamp of last training heartbeat
        self._heartbeat_timeout = float(os.environ.get("AIOS_HEARTBEAT_TIMEOUT", "30"))  # seconds
        self._stop_escalation_timeout = float(os.environ.get("AIOS_STOP_TIMEOUT", "10"))  # seconds
        self._vram_update_after_id: Optional[str] = None  # For debouncing VRAM updates
        
        # Initial VRAM estimate
        self._update_vram_estimate()
        
        # Bind real-time VRAM estimate updates to all relevant variables
        def _schedule_vram_update(*args):
            """Schedule VRAM update with small delay to avoid rapid-fire updates."""
            # Cancel any pending update
            if hasattr(self, '_vram_update_after_id') and self._vram_update_after_id is not None:
                try:
                    self.after_cancel(self._vram_update_after_id)
                except Exception:
                    pass
            # Schedule new update after 300ms delay (debouncing)
            try:
                self._vram_update_after_id = self.after(300, self._update_vram_estimate)
            except Exception:
                pass
        
        # Bind to all variables that affect memory estimation
        self.batch_var.trace_add("write", _schedule_vram_update)
        self.max_seq_var.trace_add("write", _schedule_vram_update)
        self.h_layers_var.trace_add("write", _schedule_vram_update)
        self.l_layers_var.trace_add("write", _schedule_vram_update)
        self.hidden_size_var.trace_add("write", _schedule_vram_update)
        self.expansion_var.trace_add("write", _schedule_vram_update)
        self.num_heads_var.trace_add("write", _schedule_vram_update)
        self.gradient_checkpointing_var.trace_add("write", _schedule_vram_update)
        self.use_amp_var.trace_add("write", _schedule_vram_update)
        self.use_cpu_offload_var.trace_add("write", _schedule_vram_update)
        self.use_8bit_optimizer_var.trace_add("write", _schedule_vram_update)
        self.use_chunked_training_var.trace_add("write", _schedule_vram_update)
        self.chunk_size_var.trace_add("write", _schedule_vram_update)
        self.use_peft_var.trace_add("write", _schedule_vram_update)
        self.lora_r_var.trace_add("write", _schedule_vram_update)
        self.lora_alpha_var.trace_add("write", _schedule_vram_update)
        self.lora_dropout_var.trace_add("write", _schedule_vram_update)
        self.zero_stage_var.trace_add("write", _schedule_vram_update)
        
        # Prefill from persisted last safe batches if available
        try:
            base = os.path.join(self._project_root, "artifacts", "brains", "actv1", "last_safe.json")
            if os.path.exists(base):
                import json as _json
                with open(base, "r", encoding="utf-8") as f:
                    data = _json.loads(f.read())
                if isinstance(data, dict):
                    tb = data.get("train_batch")
                    if isinstance(tb, int) and tb > 0:
                        self.batch_var.set(str(tb))
                    elif isinstance(tb, str) and tb.isdigit():
                        self.batch_var.set(tb)
        except Exception:
            pass

    def _get_theme_colors(self) -> dict[str, str]:
        """Detect current theme and return appropriate colors for Text widgets."""
        try:
            style = ttk.Style()
            bg = style.lookup(".", "background")
            if bg and bg.startswith("#"):
                # Parse RGB values
                r = int(bg[1:3], 16)
                g = int(bg[3:5], 16)
                b = int(bg[5:7], 16)
                brightness = (r + g + b) / 3
                
                # Check for specific themes
                if brightness < 50:  # Very dark (Matrix, Halloween, or Dark)
                    if g > r and g > b:  # Greenish = Matrix
                        return {
                            "bg": "#000000",
                            "fg": "#00ff41",
                            "selectbg": "#003300",
                            "selectfg": "#00ff41",
                            "insertbg": "#00ff41"
                        }
                    elif r > 50 and r > g * 2:  # Orange-ish (low brightness but high red) = Halloween
                        return {
                            "bg": "#1a0f00",
                            "fg": "#ff6600",
                            "selectbg": "#ff6600",
                            "selectfg": "#1a0f00",
                            "insertbg": "#ff6600"
                        }
                    else:  # Dark mode
                        return {
                            "bg": "#2b2b2b",
                            "fg": "#e0e0e0",
                            "selectbg": "#404040",
                            "selectfg": "#e0e0e0",
                            "insertbg": "#e0e0e0"
                        }
                elif brightness < 128:  # Dark mode
                    return {
                        "bg": "#2b2b2b",
                        "fg": "#e0e0e0",
                        "selectbg": "#404040",
                        "selectfg": "#e0e0e0",
                        "insertbg": "#e0e0e0"
                    }
                elif r > 200 and g > 150 and b > 150 and r > b:  # Barbie mode (pinkish)
                    return {
                        "bg": "#ffe4f0",
                        "fg": "#c71585",
                        "selectbg": "#ff69b4",
                        "selectfg": "#ffffff",
                        "insertbg": "#c71585"
                    }
        except Exception:
            pass
        
        # Default to light theme
        return {
            "bg": "#ffffff",
            "fg": "#000000",
            "selectbg": "#0078d7",
            "selectfg": "#ffffff",
            "insertbg": "#000000"
        }

    def update_theme(self) -> None:
        """Update Text widget colors when theme changes."""
        try:
            theme_colors = self._get_theme_colors()
            self.log.config(
                bg=theme_colors["bg"],
                fg=theme_colors["fg"],
                selectbackground=theme_colors["selectbg"],
                selectforeground=theme_colors["selectfg"],
                insertbackground=theme_colors["insertbg"]
            )
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        """Append a line of text to the panel log and external output.

        This method is safe to call from background threads; UI updates are
        marshalled onto the Tk main loop using `after(0, ...)`.
        """
        # Forward to the shared output sink (Debug tab) if provided
        try:
            if callable(getattr(self, "_append_out", None)):
                self._append_out(msg)
        except Exception:
            pass

        line = (msg if isinstance(msg, str) else str(msg))
        if not line.endswith("\n"):
            line += "\n"

        def _ui_append() -> None:
            try:
                # Only scroll to bottom if user is already viewing the bottom
                try:
                    yview = self.log.yview()
                    at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
                except Exception:
                    at_bottom = True  # Default to scrolling if can't check
                
                self.log.insert("end", line)
                
                if at_bottom:
                    self.log.see("end")
            except Exception:
                pass

        try:
            # Schedule on the Tk event loop to avoid cross-thread UI access
            self.after(0, _ui_append)
        except Exception:
            # Best-effort direct append if `after` is unavailable
            _ui_append()

    def _auto_calculate_steps(self) -> None:
        """Auto-calculate optimal steps based on dataset size and batch size."""
        if self._auto_steps_calculating:
            return
        
        try:
            self._auto_steps_calculating = True
            dataset_path = self.dataset_var.get().strip()
            
            if not dataset_path:
                self._log("[Auto Steps] Error: No dataset selected")
                return
            
            # Disable button during calculation
            if hasattr(self, '_auto_steps_btn'):
                self._auto_steps_btn.config(state="disabled", text="...")
            
            # Get batch size
            try:
                batch_size = int(self.batch_var.get())
                if batch_size < 1:
                    batch_size = 1
            except:
                batch_size = 1
            
            # Count dataset samples
            def _count_and_update():
                try:
                    from pathlib import Path
                    import os
                    
                    dataset_count = 0
                    
                    # Handle HuggingFace dataset paths (hf://dataset_name:split:subset format)
                    if dataset_path.startswith("hf://"):
                        self._log(f"[Auto Steps] Detecting HuggingFace dataset: {dataset_path}")
                        try:
                            # Parse hf://PleIAs/common_corpus:default:train format
                            hf_path = dataset_path[5:]  # Remove "hf://" prefix
                            parts = hf_path.split(":")
                            
                            if len(parts) >= 3:
                                dataset_name = parts[0]
                                config_name = parts[1] if parts[1] != "default" else None
                                split_name = parts[2]
                            elif len(parts) == 2:
                                dataset_name = parts[0]
                                config_name = None
                                split_name = parts[1]
                            else:
                                dataset_name = hf_path
                                config_name = None
                                split_name = "train"
                            
                            self._log(f"[Auto Steps] Loading HF dataset: {dataset_name}, config={config_name}, split={split_name}")
                            
                            # Try to load dataset info without downloading full data
                            try:
                                from datasets import load_dataset_builder
                                builder = load_dataset_builder(dataset_name, config_name)
                                if builder.info.splits:
                                    split_info = builder.info.splits.get(split_name)
                                    if split_info:
                                        dataset_count = split_info.num_examples
                                        self._log(f"[Auto Steps] Found {dataset_count:,} examples in HF dataset (from metadata)")
                                    else:
                                        raise ValueError(f"Split '{split_name}' not found")
                                else:
                                    raise ValueError("No split information available")
                            except Exception as e1:
                                # Fallback: try streaming mode to avoid downloading full dataset
                                self._log(f"[Auto Steps] Metadata unavailable ({e1}), trying streaming mode...")
                                from datasets import load_dataset
                                dataset = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                                # Handle both sized and iterable datasets
                                if hasattr(dataset, '__len__'):
                                    dataset_count = len(dataset)  # type: ignore[arg-type]
                                    self._log(f"[Auto Steps] Loaded {dataset_count:,} examples from HF dataset")
                                else:
                                    # For IterableDataset, we can't get length without downloading
                                    self._log(f"[Auto Steps] Error: Cannot determine size for streaming dataset without downloading")
                                    self._log(f"[Auto Steps] Solution: Download dataset first via Datasets tab, or manually set steps")
                                    self._log(f"[Auto Steps] Tip: For large datasets, use estimated steps based on training time")
                                    return
                        
                        except ImportError:
                            self._log("[Auto Steps] Error: 'datasets' library not installed. Install with: pip install datasets")
                            return
                        except Exception as e:
                            self._log(f"[Auto Steps] Error loading HuggingFace dataset: {e}")
                            return
                    
                    else:
                        # Handle local file/directory paths
                        path = Path(dataset_path)
                        
                        # Check if path exists
                        if not path.exists():
                            self._log(f"[Auto Steps] Error: Dataset path does not exist: {dataset_path}")
                            self._log(f"[Auto Steps] Tip: For HuggingFace datasets, use format: hf://dataset_name:config:split")
                            return
                        
                        # Count samples based on path type
                        if path.is_file():
                            # Single file: count lines
                            try:
                                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                    dataset_count = sum(1 for line in f if line.strip())
                            except Exception as e:
                                self._log(f"[Auto Steps] Error reading file: {e}")
                                return
                        elif path.is_dir():
                            # Directory: count lines in all .txt files
                            txt_files = list(path.rglob('*.txt'))
                            if not txt_files:
                                self._log(f"[Auto Steps] Warning: No .txt files found in {dataset_path}")
                                return
                            
                            for txt_file in txt_files:
                                try:
                                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        dataset_count += sum(1 for line in f if line.strip())
                                except Exception:
                                    continue
                        else:
                            self._log(f"[Auto Steps] Error: Invalid dataset path type")
                            return
                    
                    if dataset_count == 0:
                        self._log(f"[Auto Steps] Warning: Dataset appears to be empty")
                        return
                    
                    # Calculate optimal steps for full dataset coverage
                    optimal_steps = max(1, dataset_count // batch_size)
                    
                    # Update UI on main thread
                    def _update_ui():
                        self.steps_var.set(str(optimal_steps))
                        self._log(f"[Auto Steps] Calculated: {optimal_steps} steps for {dataset_count:,} samples (batch_size={batch_size})")
                        self._log(f"[Auto Steps] Coverage: 100% of dataset per epoch")
                        
                        # Re-enable button
                        if hasattr(self, '_auto_steps_btn'):
                            self._auto_steps_btn.config(state="normal", text="Auto")
                        
                        self._auto_steps_calculating = False
                    
                    self.after(0, _update_ui)
                    
                except Exception as e:
                    def _show_error():
                        self._log(f"[Auto Steps] Error: {e}")
                        if hasattr(self, '_auto_steps_btn'):
                            self._auto_steps_btn.config(state="normal", text="Auto")
                        self._auto_steps_calculating = False
                    self.after(0, _show_error)
            
            # Run in worker pool if available, otherwise in thread
            if self._worker_pool:
                self._worker_pool.submit(_count_and_update)
            else:
                import threading
                threading.Thread(target=_count_and_update, daemon=True).start()
        
        except Exception as e:
            self._log(f"[Auto Steps] Error: {e}")
            if hasattr(self, '_auto_steps_btn'):
                self._auto_steps_btn.config(state="normal", text="Auto")
            self._auto_steps_calculating = False

    def _apply_preset(self, name: str) -> None:
        """Apply a rough architecture preset by name.

        Presets are approximate and meant for convenience; users can still edit
        the fields afterwards if needed.
        
        NOTE: All presets now use MoE by default (8 experts, 2 active).
        Hidden sizes are adjusted to account for MoE overhead while hitting
        target parameter counts: 1M, 5M, 10M, 20M, 50M.
        """
        p = (name or "").strip().upper()
        # Defaults with MoE enabled
        spec = {
            "h_layers": "2",
            "l_layers": "2",
            "hidden": "512",
            "expansion": "2.0",
            "heads": "8",
            "h_cycles": "2",
            "l_cycles": "2",
            "pos": "rope",
        }
        # Adjusted for MoE (8 experts, ~1.6x param multiplier)
        # Target: actual params with MoE = desired total
        if p == "1M":
            spec.update({"hidden": "128", "heads": "4"})  # ~1M with MoE
        elif p == "5M":
            spec.update({"hidden": "256", "heads": "8"})  # ~5M with MoE
        elif p == "10M":
            spec.update({"hidden": "384", "heads": "8"})  # ~10M with MoE
        elif p == "20M":
            spec.update({"hidden": "512", "heads": "8"})  # ~20M with MoE
        elif p == "50M":
            spec.update({"hidden": "768", "heads": "12"})  # ~50M with MoE

        try:
            self.h_layers_var.set(spec["h_layers"])  # type: ignore[index]
            self.l_layers_var.set(spec["l_layers"])  # type: ignore[index]
            self.hidden_size_var.set(spec["hidden"])  # type: ignore[index]
            self.expansion_var.set(spec["expansion"])  # type: ignore[index]
            self.num_heads_var.set(spec["heads"])  # type: ignore[index]
            self.h_cycles_var.set(spec["h_cycles"])  # type: ignore[index]
            self.l_cycles_var.set(spec["l_cycles"])  # type: ignore[index]
            self.pos_enc_var.set(spec["pos"])  # type: ignore[index]
            self._log(f"[hrm] Applied preset: {p}")
        except Exception:
            pass

    def _clear_output(self) -> None:
        """Clear the training output log."""
        try:
            self.log.delete("1.0", "end")
            self._log("[hrm] Output cleared")
        except Exception:
            pass

    # Tiny helpers used by extracted functions
    def _mk_bool(self, val: bool) -> tk.BooleanVar:
        try:
            return tk.BooleanVar(value=val)
        except Exception:  # pragma: no cover - headless
            class _B:
                def __init__(self, v): self._v = bool(v)
                def get(self): return self._v
                def set(self, v): self._v = bool(v)
            return _B(val)  # type: ignore[return-value]

    def get_state(self) -> dict:
        """Return a dict of current UI settings for persistence."""
        try:
            return {
                # core
                "dataset": self.dataset_var.get(),
                "model": self.model_var.get(),
                "max_seq": self.max_seq_var.get(),
                "batch": self.batch_var.get(),
                "steps": self.steps_var.get(),
                "lr": self.lr_var.get(),
                "auto_adjust_moe_lr": self.auto_adjust_moe_lr_var.get(),
                "halt_steps": self.halt_steps_var.get(),
                "gradient_checkpointing": bool(self.gradient_checkpointing_var.get()),
                "use_amp": bool(getattr(self, "use_amp_var", tk.BooleanVar(value=True)).get()),
                "use_cpu_offload": bool(getattr(self, "use_cpu_offload_var", tk.BooleanVar(value=False)).get()),
                "use_8bit_optimizer": bool(getattr(self, "use_8bit_optimizer_var", tk.BooleanVar(value=False)).get()),
                "use_flash_attn": bool(getattr(self, "use_flash_attn_var", tk.BooleanVar(value=False)).get()),
                "flash_attn_window": getattr(self, "flash_attn_window_var", tk.StringVar(value="512")).get(),
                "use_chunked_training": bool(getattr(self, "use_chunked_training_var", tk.BooleanVar(value=False)).get()),
                "chunk_size": self.chunk_size_var.get(),
                "kl": self.kl_var.get(),
                "kl_temp": self.kl_temp_var.get(),
                "stop_file": self.stop_file_var.get(),
                "log_file": self.log_file_var.get(),
                "student_init": self.student_init_var.get(),
                "h_layers": self.h_layers_var.get(),
                "l_layers": self.l_layers_var.get(),
                "hidden_size": self.hidden_size_var.get(),
                "expansion": self.expansion_var.get(),
                "num_heads": self.num_heads_var.get(),
                "h_cycles": self.h_cycles_var.get(),
                "l_cycles": self.l_cycles_var.get(),
                "pos_enc": self.pos_enc_var.get(),
                "brain_name": self.brain_name_var.get(),
                "default_goal": self.default_goal_var.get(),
                "bundle_dir": self.bundle_dir_var.get(),
                "ascii_only": bool(self.ascii_only_var.get()),
                "linear_dataset": bool(self.linear_dataset_var.get()),
                "zero_stage": self.zero_stage_var.get(),
                "ddp_abort_on_fail": bool(self.ddp_abort_on_fail_var.get()),
                # flow
                "iterate": bool(getattr(self, "iterate_var", tk.BooleanVar(value=False)).get()),
                "stop_after_epoch": bool(getattr(self, "stop_after_epoch_var", tk.BooleanVar(value=False)).get()),
            }
        except Exception:
            return {}

    def set_state(self, state: dict) -> None:
        """Apply settings from a dict produced by get_state()."""
        if not isinstance(state, dict):
            return
        # helpers
        def _set_str(var, key):
            try:
                v = state.get(key)
                if isinstance(v, (str, int, float)):
                    var.set(str(v))
            except Exception:
                pass
        def _set_bool(var, key):
            try:
                v = state.get(key)
                if isinstance(v, bool):
                    var.set(v)
                elif isinstance(v, (int, float)):
                    var.set(bool(v))
            except Exception:
                pass

        # core
        _set_str(self.dataset_var, "dataset")
        _set_str(self.model_var, "model")
        _set_str(self.max_seq_var, "max_seq")
        _set_str(self.batch_var, "batch")
        _set_str(self.steps_var, "steps")
        _set_str(self.lr_var, "lr")
        _set_bool(self.auto_adjust_moe_lr_var, "auto_adjust_moe_lr")
        _set_str(self.halt_steps_var, "halt_steps")
        _set_bool(self.gradient_checkpointing_var, "gradient_checkpointing")
        if hasattr(self, "use_amp_var"):
            _set_bool(self.use_amp_var, "use_amp")
        if hasattr(self, "use_cpu_offload_var"):
            _set_bool(self.use_cpu_offload_var, "use_cpu_offload")
        if hasattr(self, "use_8bit_optimizer_var"):
            _set_bool(self.use_8bit_optimizer_var, "use_8bit_optimizer")
        if hasattr(self, "use_flash_attn_var"):
            _set_bool(self.use_flash_attn_var, "use_flash_attn")
        if hasattr(self, "flash_attn_window_var"):
            _set_str(self.flash_attn_window_var, "flash_attn_window")
        if hasattr(self, "use_chunked_training_var"):
            _set_bool(self.use_chunked_training_var, "use_chunked_training")
        _set_str(self.chunk_size_var, "chunk_size")
        
        # PEFT options
        if hasattr(self, "use_peft_var"):
            _set_bool(self.use_peft_var, "use_peft")
        if hasattr(self, "peft_method_var"):
            _set_str(self.peft_method_var, "peft_method")
        if hasattr(self, "lora_r_var"):
            _set_str(self.lora_r_var, "lora_r")
        if hasattr(self, "lora_alpha_var"):
            _set_str(self.lora_alpha_var, "lora_alpha")
        if hasattr(self, "lora_dropout_var"):
            _set_str(self.lora_dropout_var, "lora_dropout")
        if hasattr(self, "lora_target_modules_var"):
            _set_str(self.lora_target_modules_var, "lora_target_modules")
        
        _set_str(self.kl_var, "kl")
        _set_str(self.kl_temp_var, "kl_temp")
        _set_str(self.stop_file_var, "stop_file")
        _set_str(self.log_file_var, "log_file")
        _set_str(self.student_init_var, "student_init")
        _set_str(self.brain_name_var, "brain_name")
        _set_str(self.default_goal_var, "default_goal")
        # arch
        _set_str(self.h_layers_var, "h_layers")
        _set_str(self.l_layers_var, "l_layers")
        _set_str(self.hidden_size_var, "hidden_size")
        _set_str(self.expansion_var, "expansion")
        _set_str(self.num_heads_var, "num_heads")
        _set_str(self.h_cycles_var, "h_cycles")
        _set_str(self.l_cycles_var, "l_cycles")
        _set_str(self.pos_enc_var, "pos_enc")
        _set_str(self.brain_name_var, "brain_name")
        _set_str(self.bundle_dir_var, "bundle_dir")
        _set_bool(self.ascii_only_var, "ascii_only")
        _set_bool(self.linear_dataset_var, "linear_dataset")
        _set_str(self.zero_stage_var, "zero_stage")
        _set_bool(self.ddp_abort_on_fail_var, "ddp_abort_on_fail")
        # flow
        if hasattr(self, "iterate_var"):
            _set_bool(self.iterate_var, "iterate")
        if hasattr(self, "stop_after_epoch_var"):
            _set_bool(self.stop_after_epoch_var, "stop_after_epoch")

    def _get_moe_num_experts(self) -> int:
        """Get MoE num_experts from readonly display field, with fallback to default."""
        try:
            val = self.moe_num_experts_entry.get().strip()
            if val and val != "N/A":
                return int(val)
        except Exception:
            pass
        return 8  # Default

    def _get_moe_active_experts(self) -> int:
        """Get MoE active experts per token from readonly display field, with fallback to default."""
        try:
            val = self.moe_active_experts_entry.get().strip()
            if val and val != "N/A":
                return int(val)
        except Exception:
            pass
        return 2  # Default

    def build_training_config(self) -> Any:
        """Build a TrainingConfig object from current GUI state.
        
        Returns:
            TrainingConfig object ready for training
        """
        from aios.core.hrm_training import TrainingConfig
        
        # Helper to safely get int value
        def _int(var, default=0):
            try:
                return int(var.get().strip() or default)
            except Exception:
                return default
        
        # Helper to safely get float value
        def _float(var, default=0.0):
            try:
                return float(var.get().strip() or default)
            except Exception:
                return default
        
        # Helper to safely get string value
        def _str(var, default=""):
            try:
                return var.get().strip() or default
            except Exception:
                return default
        
        # Debug: Check checkbox state before building config
        auto_adjust_value = self.auto_adjust_moe_lr_var.get()
        print(f"[GUI DEBUG] Building config: auto_adjust_moe_lr checkbox state = {auto_adjust_value}")
        
        # Get device from resources panel if available
        device = "auto"
        try:
            rp = getattr(self, "_resources_panel", None)
            if rp is not None:
                rvals = rp.get_values()
                td = str(rvals.get("train_device") or "auto").lower()
                if td in {"cpu", "cuda", "xpu", "mps", "dml"}:
                    device = td
                elif isinstance(rvals.get("train_cuda_selected"), list) and len(rvals.get("train_cuda_selected", [])) > 0:
                    device = "cuda"
        except Exception:
            pass
        
        # Get CUDA IDs and DDP settings from resources panel
        cuda_ids = None
        ddp = False
        world_size = None
        try:
            rp = getattr(self, "_resources_panel", None)
            if rp is not None:
                rvals = rp.get_values()
                sel_train = rvals.get("train_cuda_selected") or []
                if isinstance(sel_train, list) and len(sel_train) > 0:
                    cuda_ids = ",".join(str(int(i)) for i in sel_train)
                    if len(sel_train) > 1:
                        ddp = True
                        world_size = len(sel_train)
        except Exception:
            pass
        
        # Get sys_mem_cap_pct from resources CPU util
        sys_mem_cap_pct = None
        try:
            rp = getattr(self, "_resources_panel", None)
            if rp is not None:
                rvals = rp.get_values()
                cap = int(rvals.get("cpu_util_pct") or 0)
                if cap > 0:
                    sys_mem_cap_pct = cap
        except Exception:
            pass
        
        # Fix old HuggingFace dataset format (hf://dataset:split -> hf://dataset:config:split)
        dataset_file = _str(self.dataset_var)
        if dataset_file and dataset_file.startswith("hf://"):
            parts = dataset_file[5:].split(":")
            if len(parts) == 2:
                # Old format: hf://dataset:split -> Fix to hf://dataset:default:split
                dataset_path, split = parts
                dataset_file = f"hf://{dataset_path}:default:{split}"
                self.dataset_var.set(dataset_file)  # Update for next time
                self._log(f"[hrm] Auto-corrected dataset format to: {dataset_file}")
        
        # Check for incompatible DDP + ZeRO-3 configuration
        zero_stage_val = _str(self.zero_stage_var, "none")
        if ddp and world_size and world_size > 1 and zero_stage_val == "zero3":
            # DDP with multiple GPUs + ZeRO-3 is not supported - fall back to single GPU
            self._log("[hrm] WARNING: Multi-GPU DDP + ZeRO-3 is not supported")
            self._log(f"[hrm] Falling back to single GPU (using GPU {cuda_ids.split(',')[0] if cuda_ids else '0'})")
            # Use only the first GPU
            if cuda_ids:
                cuda_ids = cuda_ids.split(',')[0]
            ddp = False
            world_size = None
        
        # Build config
        config = TrainingConfig(
            model=_str(self.model_var, "artifacts/hf_implant/base_model"),
            dataset_file=dataset_file,
            max_seq_len=_int(self.max_seq_var, 128),
            batch_size=_int(self.batch_var, 4),
            steps=_int(self.steps_var, 100),
            lr=_float(self.lr_var, 5e-5),
            auto_adjust_moe_lr=self.auto_adjust_moe_lr_var.get(),
            device=device,
            halt_max_steps=_int(self.halt_steps_var, 1),
            save_dir=_str(self.bundle_dir_var, "artifacts/brains/actv1"),
            teacher=None,  # Removed feature
            teacher_device="cuda",  # Removed feature
            kl=_float(self.kl_var, 0.0),
            kl_temp=_float(self.kl_temp_var, 1.0),
            ascii_only=bool(self.ascii_only_var.get()),
            linear_dataset=bool(self.linear_dataset_var.get()),
            dataset_start_offset=0,  # GUI always starts from 0; resume handled by checkpoint loading
            sys_mem_cap_pct=sys_mem_cap_pct,
            stop_file=_str(self.stop_file_var) or None,
            log_file=_str(self.log_file_var) or None,
            student_init=_str(self.student_init_var) or None,
            brain_name=_str(self.brain_name_var) or None,
            default_goal=_str(self.default_goal_var) or None,
            bundle_dir=_str(self.bundle_dir_var, "artifacts/brains/actv1"),
            h_layers=_int(self.h_layers_var, 2),
            l_layers=_int(self.l_layers_var, 2),
            hidden_size=_int(self.hidden_size_var, 512),
            expansion=_float(self.expansion_var, 2.0),
            num_heads=_int(self.num_heads_var, 8),
            h_cycles=_int(self.h_cycles_var, 2),
            l_cycles=_int(self.l_cycles_var, 2),
            pos_encodings=_str(self.pos_enc_var, "rope"),
            
            # MoE (Mixture of Experts) configuration
            # Read from readonly display fields populated from brain metadata
            # Determine if MoE is enabled based on whether num_experts > 1
            use_moe=self._get_moe_num_experts() > 1,  # Enable MoE if experts > 1
            num_experts=self._get_moe_num_experts(),
            num_experts_per_tok=self._get_moe_active_experts(),
            moe_capacity_factor=1.25,  # Use default, not currently displayed
            
            cuda_ids=cuda_ids,
            iterate=bool(getattr(self, "iterate_var", self._mk_bool(False)).get()),
            stop_after_epoch=bool(getattr(self, "stop_after_epoch_var", self._mk_bool(False)).get()),
            optimize=False,  # Never auto-optimize during training (use Optimize button instead)
            gradient_checkpointing=bool(self.gradient_checkpointing_var.get()),
            use_amp=bool(getattr(self, "use_amp_var", self._mk_bool(True)).get()),
            use_cpu_offload=bool(getattr(self, "use_cpu_offload_var", self._mk_bool(False)).get()),
            use_8bit_optimizer=bool(getattr(self, "use_8bit_optimizer_var", self._mk_bool(False)).get()),
            use_chunked_training=bool(getattr(self, "use_chunked_training_var", self._mk_bool(False)).get()),
            chunk_size=_int(self.chunk_size_var, 2048),
            window_size=int(getattr(self, "flash_attn_window_var", tk.StringVar(value="512")).get()) if bool(getattr(self, "use_flash_attn_var", self._mk_bool(False)).get()) else None,
            
            # PEFT options
            use_peft=bool(getattr(self, "use_peft_var", self._mk_bool(False)).get()),
            peft_method=_str(self.peft_method_var, "lora"),
            lora_r=_int(self.lora_r_var, 16),
            lora_alpha=_int(self.lora_alpha_var, 32),
            lora_dropout=_float(self.lora_dropout_var, 0.05),
            lora_target_modules=_str(self.lora_target_modules_var, "q_proj,v_proj"),
            
            zero_stage=_str(self.zero_stage_var, "none"),
            ddp=ddp,
            world_size=world_size,
            strict=True,  # Always strict mode in GUI
        )
        
        return config

    def _on_start(self) -> None:
        _on_start_helper(self)

    def _open_rank_logs(self) -> None:
        _open_rank_logs_helper(self)

    def _on_select_student(self) -> None:
        _select_student_helper(self)

    def _on_optimize(self) -> None:
        """Run pre-flight optimization to find good training/gen settings.

        Executes in a background thread and updates UI fields on success.
        """
        # Require a selected student/brain so we optimize around the actual brain
        try:
            si = (self.student_init_var.get() or "").strip()
            if not si:
                # Try to resolve from brain name bundle
                bname = (self.brain_name_var.get() or "").strip()
                if bname:
                    bdir = os.path.join(self._project_root, "artifacts", "brains", "actv1", bname)
                    cand = os.path.join(bdir, "actv1_student.safetensors")
                    if os.path.exists(cand) or os.path.isdir(bdir):
                        try:
                            self.student_init_var.set(cand)
                            si = cand
                        except Exception:
                            pass
            if not si:
                self._log("[opt] Please select a student/brain before optimizing → click 'Select Student'.")
                try:
                    # Open the selection dialog to help the user pick one
                    self._on_select_student()
                except Exception:
                    pass
                return
            else:
                try:
                    self._log(f"[opt] Using student for optimization: {si}")
                except Exception:
                    pass
        except Exception:
            # If anything goes wrong resolving the student, abort gracefully
            self._log("[opt] Failed to resolve selected student; select a brain first.")
            return
        # Guard against concurrent runs
        if getattr(self, "_run_in_progress", False):
            try:
                self._log("[opt] Busy: wait for current run to finish.")
            except Exception:
                pass
            return
        try:
            self._run_in_progress = True
            self.start_btn.config(state="disabled")
            self.progress_lbl.config(text="optimizing…")
            self.progress.configure(mode="indeterminate", value=0)
            self.progress.start(10)
        except Exception:
            pass
        # Set optimization state flags
        self._run_in_progress = True
        self._stop_requested = False
        
        # Start metrics polling during optimization
        if not self._metrics_polling_active:
            self._metrics_polling_active = True
            try:
                self.after(1000, self._poll_metrics)
            except Exception:
                pass
        
        import threading
        def _bg():
            try:
                optimize_from_gui_progressive(self)
            except Exception as e:
                try:
                    self._log(f"[opt] error: {e}")
                except Exception:
                    pass
            finally:
                def _done():
                    try:
                        self.start_btn.config(state="normal")
                        self.progress.stop()
                        self.progress.configure(mode="determinate", value=0)
                        self.progress_lbl.config(text="idle")
                        # Reset stop flag
                        self._stop_requested = False
                        # Save updated settings after optimization
                        if callable(getattr(self, "_save_state_fn", None)):
                            try:
                                self._save_state_fn()  # type: ignore[misc]
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._run_in_progress = False
                try:
                    self.after(0, _done)
                except Exception:
                    _done()
        self._bg_thread = threading.Thread(target=_bg, daemon=True)
        self._bg_thread.start()

    def _default_stop_file(self) -> str:
        try:
            return os.path.join(self._project_root, "training_data", "actv1", "STOP")
        except Exception:
            return "training_data/actv1/STOP"

    def _on_stop(self) -> None:
        _on_stop_helper(self)

    def _stop_all(self) -> None:
        _stop_all_helper(self)

    def _poll_metrics(self) -> None:
        _poll_metrics_helper(self)

    def _show_stopped_dialog(self) -> None:
        _show_stopped_dialog_helper(self)

    def _estimate_model_params(self) -> int:
        """Estimate the number of model parameters based on architecture settings."""
        try:
            # Parse architecture parameters
            h_layers = int(self.h_layers_var.get() or 2)
            l_layers = int(self.l_layers_var.get() or 2)
            hidden_size = int(self.hidden_size_var.get() or 512)
            expansion = float(self.expansion_var.get() or 2.0)
            num_heads = int(self.num_heads_var.get() or 8)
            
            # Vocab size (standard tokenizer)
            vocab_size = 50257
            
            # Rough parameter estimation:
            # 1. Embedding layer: vocab_size * hidden_size
            embed_params = vocab_size * hidden_size
            
            # 2. H-layers (high-level reasoning)
            # Each transformer layer has:
            # - Self-attention: 4 * hidden_size^2 (Q, K, V, O projections)
            # - FFN: 2 * hidden_size * (hidden_size * expansion)
            h_layer_params = h_layers * (
                4 * hidden_size * hidden_size +  # attention
                2 * hidden_size * int(hidden_size * expansion)  # FFN
            )
            
            # 3. L-layers (low-level processing) - similar structure
            l_layer_params = l_layers * (
                4 * hidden_size * hidden_size +
                2 * hidden_size * int(hidden_size * expansion)
            )
            
            # 4. Output head: hidden_size * vocab_size
            head_params = hidden_size * vocab_size
            
            # 5. Layer norms and other small components (~5% overhead)
            total_base = embed_params + h_layer_params + l_layer_params + head_params
            total_params = int(total_base * 1.05)
            
            return total_params
        except Exception:
            return 0

    def _update_vram_estimate(self) -> None:
        """Update memory estimates (VRAM and RAM) using accurate MemoryEstimator.
        
        Accounts for all optimizations:
        - AMP (mixed precision)
        - Gradient checkpointing
        - LoRA/PEFT (parameter-efficient fine-tuning)
        - CPU offload
        - 8-bit optimizer
        - DeepSpeed ZeRO stages (1, 2, 3)
        - Chunked training for long contexts
        """
        try:
            from .hrm_training.memory_estimator import MemoryEstimator
            
            # Get parameters
            batch_size = int(self.batch_var.get() or 4)
            seq_len = int(self.max_seq_var.get() or 128)
            total_params = self._estimate_model_params()
            hidden_size = int(self.hidden_size_var.get() or 512)
            h_layers = int(self.h_layers_var.get() or 2)
            l_layers = int(self.l_layers_var.get() or 2)
            num_layers = h_layers + l_layers
            
            if total_params == 0:
                # Invalid params - show placeholder
                self.vram_model_lbl.config(text="-")
                self.vram_optimizer_lbl.config(text="-")
                self.vram_activations_lbl.config(text="-")
                self.vram_total_lbl.config(text="-")
                self.ram_dataset_lbl.config(text="-")
                self.ram_offload_lbl.config(text="-")
                self.ram_total_lbl.config(text="-")
                return
            
            # Get optimization settings
            use_amp = bool(getattr(self, "use_amp_var", self._mk_bool(True)).get())
            use_gradient_checkpointing = bool(getattr(self, "gradient_checkpointing_var", self._mk_bool(True)).get())
            use_lora = bool(getattr(self, "use_peft_var", self._mk_bool(False)).get())
            lora_r = int(self.lora_r_var.get() or 16) if use_lora else 16
            use_cpu_offload = bool(getattr(self, "use_cpu_offload_var", self._mk_bool(False)).get())
            use_8bit_optimizer = bool(getattr(self, "use_8bit_optimizer_var", self._mk_bool(False)).get())
            zero_stage = self.zero_stage_var.get()
            use_chunking = seq_len > 8192  # Auto-chunking for long contexts
            
            # ========== CRITICAL: Auto-enable AMP for long sequences ==========
            # For sequences > 2048, FP32 memory becomes prohibitive
            # Attention memory scales O(n²): seq_len=10000 needs 100M elements PER HEAD!
            if seq_len > 2048 and not use_amp:
                # WARN USER: This configuration will likely OOM
                try:
                    self._log(f"[hrm] ⚠️  WARNING: Long sequence ({seq_len} tokens) without AMP will use massive memory!")
                    self._log(f"[hrm] 💡 STRONGLY RECOMMENDED: Enable AMP to reduce memory by ~50%")
                    self._log(f"[hrm] FP32 attention memory for seq_len={seq_len}: ~{(seq_len * seq_len * 4 * num_layers) / 1e9:.1f} GB per batch!")
                except:
                    pass
            
            # Get number of GPUs
            num_gpus = 1
            try:
                rp = getattr(self, "_resources_panel", None)
                if rp is not None:
                    rvals = rp.get_values()
                    sel_train = rvals.get("train_cuda_selected") or []
                    if isinstance(sel_train, list) and len(sel_train) > 0:
                        num_gpus = len(sel_train)
            except Exception:
                pass
            
            # ========== CREATE MEMORY ESTIMATOR ==========
            estimator = MemoryEstimator(
                total_params=total_params,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_size=batch_size,
                seq_len=seq_len,
                use_amp=use_amp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_lora=use_lora,
                lora_r=lora_r,
                use_cpu_offload=use_cpu_offload,
                use_8bit_optimizer=use_8bit_optimizer,
                zero_stage=zero_stage,
                num_gpus=num_gpus,
                use_chunking=use_chunking,
            )
            
            # Get estimates
            summary = estimator.get_summary()
            vram = summary["vram_breakdown"]
            ram = summary["ram_breakdown"]
            
            # ========== UPDATE VRAM DISPLAY ==========
            self.vram_model_lbl.config(text=f"{vram['model_gb']:.2f} GB")
            self.vram_optimizer_lbl.config(text=f"{vram['optimizer_gb']:.2f} GB")
            
            # Activations + Gradients combined
            act_grad_total = vram['activations_gb'] + vram['gradients_gb']
            self.vram_activations_lbl.config(text=f"{act_grad_total:.2f} GB")
            
            # Color-code total based on typical GPU VRAM sizes
            per_gpu_vram_gb = vram['total_gb']
            if per_gpu_vram_gb <= 8:
                color = "green"
                recommendation = "✓ Fits most GPUs"
            elif per_gpu_vram_gb <= 12:
                color = "#DAA520"  # goldenrod
                recommendation = "⚠ Needs 12GB+ GPU"
            elif per_gpu_vram_gb <= 16:
                color = "orange"
                recommendation = "⚠ Needs 16GB+ GPU"
            elif per_gpu_vram_gb <= 24:
                color = "red"
                recommendation = "⚠ Needs 24GB+ GPU"
            else:
                color = "red"
                recommendation = "❌ Needs 40GB+ GPU"
            
            # Format total with GPU count info
            gpu_text = f"{per_gpu_vram_gb:.2f} GB"
            if num_gpus > 1:
                gpu_text += f" (×{num_gpus} GPUs)"
            self.vram_total_lbl.config(text=gpu_text, foreground=color)
            
            # ========== UPDATE RAM DISPLAY ==========
            self.ram_dataset_lbl.config(text=f"{ram['dataset_gb']:.2f} GB")
            self.ram_offload_lbl.config(text=f"{ram['optimizer_gb']:.2f} GB")
            ram_total_text = f"{ram['total_gb']:.1f} GB"
            self.ram_total_lbl.config(text=ram_total_text)
            
            # ========== UPDATE MOE STATS AND TOKENIZER DISPLAY ==========
            # MoE stats and tokenizer are determined by the selected brain's architecture
            # Read from current student brain.json if available
            try:
                import json
                import os
                
                # Try to find brain.json from multiple sources
                brain_json_path = None
                brain_dir = None
                
                # Method 1: From student_init path
                student_path = self.student_init_var.get()
                if student_path and os.path.exists(student_path):
                    brain_dir = os.path.dirname(student_path)
                    brain_json_path = os.path.join(brain_dir, "brain.json")
                
                # Method 2: From brain name (fallback)
                if not brain_json_path or not os.path.exists(brain_json_path):
                    brain_name = self.brain_name_var.get()
                    if brain_name:
                        # Try artifacts/brains/actv1/<brain_name>/brain.json
                        brain_dir = os.path.join("artifacts", "brains", "actv1", brain_name)
                        brain_json_path = os.path.join(brain_dir, "brain.json")
                
                if brain_json_path and os.path.exists(brain_json_path):
                        with open(brain_json_path) as f:
                            brain_meta = json.load(f)
                        
                        # MoE configuration
                        use_moe = brain_meta.get("use_moe", False)
                        num_experts = brain_meta.get("num_experts", 8)  # Define outside if block
                        active_per_tok = brain_meta.get("num_experts_per_tok", 2)
                        
                        if use_moe:
                            self.moe_num_experts_entry.config(state="normal")
                            self.moe_num_experts_entry.delete(0, "end")
                            self.moe_num_experts_entry.insert(0, str(num_experts))
                            self.moe_num_experts_entry.config(state="readonly")
                            
                            self.moe_active_experts_entry.config(state="normal")
                            self.moe_active_experts_entry.delete(0, "end")
                            self.moe_active_experts_entry.insert(0, str(active_per_tok))
                            self.moe_active_experts_entry.config(state="readonly")
                        else:
                            self.moe_num_experts_entry.config(state="normal")
                            self.moe_num_experts_entry.delete(0, "end")
                            self.moe_num_experts_entry.insert(0, "N/A")
                            self.moe_num_experts_entry.config(state="readonly")
                            
                            self.moe_active_experts_entry.config(state="normal")
                            self.moe_active_experts_entry.delete(0, "end")
                            self.moe_active_experts_entry.insert(0, "N/A")
                            self.moe_active_experts_entry.config(state="readonly")
                        
                        # Tokenizer detection - prefer tokenizer_id, fallback to tokenizer_name
                        tokenizer_id = brain_meta.get("tokenizer_id")
                        tokenizer_display = None
                        
                        if tokenizer_id:
                            # Try to get friendly name from TokenizerRegistry
                            try:
                                from aios.core.tokenizers import TokenizerRegistry
                                tokenizer_info = TokenizerRegistry.get(tokenizer_id)
                                if tokenizer_info:
                                    tokenizer_display = tokenizer_info.name
                            except Exception:
                                pass
                            
                            # Fallback to tokenizer_id if registry lookup failed
                            if not tokenizer_display:
                                tokenizer_display = tokenizer_id
                        else:
                            # Fallback to legacy tokenizer_name field
                            tokenizer_display = brain_meta.get("tokenizer_name", "Unknown")
                        
                        # Extract short name from full path if needed
                        if "/" in tokenizer_display:
                            tokenizer_display = tokenizer_display.split("/")[-1]
                        
                        self.tokenizer_entry.config(state="normal")
                        self.tokenizer_entry.delete(0, "end")
                        self.tokenizer_entry.insert(0, tokenizer_display)
                        self.tokenizer_entry.config(state="readonly")
                        
                        # ========== POPULATE MODEL STATS ==========
                        # Calculate total params from architecture
                        arch = brain_meta.get("arch", {})
                        h_layers = arch.get("H_layers") or brain_meta.get("h_layers")
                        l_layers = arch.get("L_layers") or brain_meta.get("l_layers")
                        hidden_size = arch.get("hidden_size") or brain_meta.get("hidden_size")
                        vocab_size = arch.get("vocab_size") or brain_meta.get("vocab_size")
                        expansion = arch.get("expansion") or brain_meta.get("expansion") or 2.0
                        # Get MoE params if needed (already loaded above)
                        num_experts_for_calc = num_experts if use_moe else 8
                        
                        total_params_str = "-"
                        if h_layers and l_layers and hidden_size and vocab_size:
                            # Embedding layer
                            embed_params = vocab_size * hidden_size
                            # Attention per layer
                            attn_params_per_layer = 4 * hidden_size * hidden_size
                            # FFN per layer
                            ffn_hidden = int(hidden_size * expansion)
                            if use_moe:
                                ffn_params_per_layer = num_experts_for_calc * (hidden_size * ffn_hidden + ffn_hidden * hidden_size) + hidden_size * num_experts_for_calc
                            else:
                                ffn_params_per_layer = hidden_size * ffn_hidden + ffn_hidden * hidden_size
                            # Layer norm
                            ln_params_per_layer = 2 * hidden_size * 2
                            # Total
                            total_layers = h_layers + l_layers
                            layer_params = total_layers * (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer)
                            output_params = hidden_size * vocab_size
                            total_params = embed_params + layer_params + output_params
                            
                            if total_params >= 1_000_000_000:
                                total_params_str = f"{total_params / 1_000_000_000:.2f}B"
                            elif total_params >= 1_000_000:
                                total_params_str = f"{total_params / 1_000_000:.2f}M"
                            else:
                                total_params_str = f"{total_params:,.0f}"
                        
                        # Get file size and current params
                        size_mb_str = "-"
                        current_params_str = "-"
                        if brain_dir:
                            student_pt_path = os.path.join(brain_dir, brain_meta.get("checkpoint_file", "actv1_student.safetensors"))
                            if os.path.exists(student_pt_path):
                                size_bytes = os.path.getsize(student_pt_path)
                                size_mb = size_bytes / (1024 * 1024)
                                size_mb_str = f"{size_mb:.2f}"
                                
                                # Calculate current params (assuming fp32)
                                current_params = size_bytes / 4
                                if current_params >= 1_000_000_000:
                                    current_params_str = f"{current_params / 1_000_000_000:.2f}B"
                                elif current_params >= 1_000_000:
                                    current_params_str = f"{current_params / 1_000_000:.2f}M"
                                else:
                                    current_params_str = f"{current_params:,.0f}"
                        
                        # Get training steps - check brain.json first, then metrics.jsonl
                        training_steps = brain_meta.get("training_steps", 0)
                        
                        # If not in brain.json, try to read from metrics.jsonl
                        if training_steps == 0 and brain_dir:
                            metrics_path = os.path.join(brain_dir, brain_meta.get("log_file", "metrics.jsonl"))
                            if os.path.exists(metrics_path):
                                try:
                                    import json
                                    with open(metrics_path, 'r') as f:
                                        # Read all lines and find the highest step number
                                        max_step = 0
                                        for line in f:
                                            try:
                                                entry = json.loads(line.strip())
                                                if "step" in entry:
                                                    max_step = max(max_step, entry["step"])
                                            except:
                                                continue
                                        training_steps = max_step
                                except Exception:
                                    pass
                        
                        trained_steps_str = f"{training_steps:,}" if training_steps else "0"
                        
                        # Update the stats fields
                        self.total_params_entry.config(state="normal")
                        self.total_params_entry.delete(0, "end")
                        self.total_params_entry.insert(0, total_params_str)
                        self.total_params_entry.config(state="readonly")
                        
                        self.current_params_entry.config(state="normal")
                        self.current_params_entry.delete(0, "end")
                        self.current_params_entry.insert(0, current_params_str)
                        self.current_params_entry.config(state="readonly")
                        
                        self.size_mb_entry.config(state="normal")
                        self.size_mb_entry.delete(0, "end")
                        self.size_mb_entry.insert(0, size_mb_str)
                        self.size_mb_entry.config(state="readonly")
                        
                        self.trained_steps_entry.config(state="normal")
                        self.trained_steps_entry.delete(0, "end")
                        self.trained_steps_entry.insert(0, trained_steps_str)
                        self.trained_steps_entry.config(state="readonly")
                else:
                    # Brain JSON not found or brain not selected, show placeholder
                    self.moe_num_experts_entry.config(state="normal")
                    self.moe_num_experts_entry.delete(0, "end")
                    self.moe_num_experts_entry.insert(0, "-")
                    self.moe_num_experts_entry.config(state="readonly")
                    
                    self.moe_active_experts_entry.config(state="normal")
                    self.moe_active_experts_entry.delete(0, "end")
                    self.moe_active_experts_entry.insert(0, "-")
                    self.moe_active_experts_entry.config(state="readonly")
                    
                    self.tokenizer_entry.config(state="normal")
                    self.tokenizer_entry.delete(0, "end")
                    self.tokenizer_entry.insert(0, "-")
                    self.tokenizer_entry.config(state="readonly")
                    
                    # Clear stats fields
                    self.total_params_entry.config(state="normal")
                    self.total_params_entry.delete(0, "end")
                    self.total_params_entry.insert(0, "-")
                    self.total_params_entry.config(state="readonly")
                    
                    self.current_params_entry.config(state="normal")
                    self.current_params_entry.delete(0, "end")
                    self.current_params_entry.insert(0, "-")
                    self.current_params_entry.config(state="readonly")
                    
                    self.size_mb_entry.config(state="normal")
                    self.size_mb_entry.delete(0, "end")
                    self.size_mb_entry.insert(0, "-")
                    self.size_mb_entry.config(state="readonly")
                    
                    self.trained_steps_entry.config(state="normal")
                    self.trained_steps_entry.delete(0, "end")
                    self.trained_steps_entry.insert(0, "-")
                    self.trained_steps_entry.config(state="readonly")
            except Exception:
                self.moe_num_experts_entry.config(state="normal")
                self.moe_num_experts_entry.delete(0, "end")
                self.moe_num_experts_entry.insert(0, "-")
                self.moe_num_experts_entry.config(state="readonly")
                
                self.moe_active_experts_entry.config(state="normal")
                self.moe_active_experts_entry.delete(0, "end")
                self.moe_active_experts_entry.insert(0, "-")
                self.moe_active_experts_entry.config(state="readonly")
                
                self.tokenizer_entry.config(state="normal")
                self.tokenizer_entry.delete(0, "end")
                self.tokenizer_entry.insert(0, "-")
                self.tokenizer_entry.config(state="readonly")
                
                # Clear stats fields
                self.total_params_entry.config(state="normal")
                self.total_params_entry.delete(0, "end")
                self.total_params_entry.insert(0, "-")
                self.total_params_entry.config(state="readonly")
                
                self.current_params_entry.config(state="normal")
                self.current_params_entry.delete(0, "end")
                self.current_params_entry.insert(0, "-")
                self.current_params_entry.config(state="readonly")
                
                self.size_mb_entry.config(state="normal")
                self.size_mb_entry.delete(0, "end")
                self.size_mb_entry.insert(0, "-")
                self.size_mb_entry.config(state="readonly")
                
                self.trained_steps_entry.config(state="normal")
                self.trained_steps_entry.delete(0, "end")
                self.trained_steps_entry.insert(0, "-")
                self.trained_steps_entry.config(state="readonly")
            
            # ========== SIMPLIFIED TOOLTIP ==========
            try:
                from .tooltips import add_tooltip
                
                # Get optimizations summary
                opts = summary["configuration"]["optimizations"]
                
                # Build tooltip
                tooltip_lines = [
                    "══════ MEMORY ESTIMATE ══════",
                    "",
                    f"🎯 VRAM per GPU: {per_gpu_vram_gb:.2f} GB",
                    f"   • Model: {vram['model_gb']:.2f} GB",
                    f"   • Optimizer: {vram['optimizer_gb']:.2f} GB",
                    f"   • Gradients: {vram['gradients_gb']:.2f} GB",
                    f"   • Activations: {vram['activations_gb']:.2f} GB",
                    f"   • Overhead: {vram['overhead_gb']:.2f} GB",
                    "",
                    f"💾 System RAM: {ram['total_gb']:.1f} GB",
                    f"   • Dataset: {ram['dataset_gb']:.2f} GB",
                    f"   • CPU Offload: {ram['optimizer_gb']:.2f} GB",
                    f"   • PyTorch: {ram['pytorch_gb']:.2f} GB",
                    "",
                    "⚙️  Active Optimizations:",
                    f"   • AMP (FP16): {'✓' if opts['amp'] else '✗'}",
                    f"   • Gradient Checkpointing: {'✓' if opts['gradient_checkpointing'] else '✗'}",
                    f"   • LoRA/PEFT: {'✓' if opts['lora'] else '✗'}",
                    f"   • CPU Offload: {'✓' if opts['cpu_offload'] else '✗'}",
                    f"   • 8-bit Optimizer: {'✓' if opts['8bit_optimizer'] else '✗'}",
                    f"   • DeepSpeed ZeRO: {opts['zero_stage']}",
                    f"   • Chunked Training: {'✓' if opts['chunking'] else '✗'}",
                    "",
                    f"📊 Configuration:",
                    f"   • Total params: {total_params/1e6:.1f}M",
                    f"   • Trainable params: {vram['breakdown']['trainable_params']/1e6:.1f}M",
                    f"   • Batch size: {batch_size}",
                    f"   • Sequence length: {seq_len}",
                    f"   • Effective chunk: {vram['breakdown']['effective_seq']}",
                    f"   • GPUs: {num_gpus}",
                    "",
                    f"💡 {recommendation}",
                ]
                
                # Add recommendations if needed
                if per_gpu_vram_gb > 12:
                    tooltip_lines.extend(["", "🔧 Suggestions to reduce VRAM:"])
                    if not opts['amp']:
                        tooltip_lines.append("   • Enable AMP → Save ~40%")
                    if not opts['gradient_checkpointing']:
                        tooltip_lines.append("   • Enable Grad Checkpoint → Save ~60% activations")
                    if not opts['lora']:
                        tooltip_lines.append("   • Enable LoRA → Save ~99% optimizer/gradients")
                    if not opts['cpu_offload'] and vram['optimizer_gb'] > 0:
                        tooltip_lines.append(f"   • Enable CPU Offload → Move {vram['optimizer_gb']:.1f} GB to RAM")
                    if batch_size > 1:
                        tooltip_lines.append(f"   • Reduce batch size → Direct VRAM savings")
                
                add_tooltip(self.vram_total_lbl, "\n".join(tooltip_lines))
                
                # RAM tooltip
                ram_tooltip = [
                    "══════ SYSTEM RAM ══════",
                    "",
                    f"Total: {ram['total_gb']:.1f} GB",
                    "",
                    "Breakdown:",
                    f"  • Dataset buffer: {ram['dataset_gb']:.2f} GB",
                    f"  • CPU offloaded optimizer: {ram['optimizer_gb']:.2f} GB",
                    f"  • PyTorch/Python: {ram['pytorch_gb']:.2f} GB",
                    "",
                    "💡 Enable CPU Offload to move optimizer",
                    "   state from VRAM to RAM (slower but saves VRAM)",
                ]
                add_tooltip(self.ram_total_lbl, "\n".join(ram_tooltip))
                
            except Exception:
                pass
                
        except Exception as e:
            # Silently fail - memory estimate is informational only
            try:
                self.vram_model_lbl.config(text="-")
                self.vram_optimizer_lbl.config(text="-")
                self.vram_activations_lbl.config(text="-")
                self.vram_total_lbl.config(text="error")
                self.ram_dataset_lbl.config(text="-")
                self.ram_offload_lbl.config(text="-")
                self.ram_total_lbl.config(text="-")
            except Exception:
                pass
    
    def _get_effective_chunk_size(self, total_params: int, seq_len: int) -> int:
        """
        Calculate effective chunk size based on model size and sequence length.
        Larger models use smaller chunks to fit in VRAM.
        """
        # Auto-chunking only for long sequences
        if seq_len <= 8192:
            return seq_len
        
        # Base chunk sizes
        if total_params > 200_000_000:  # 200M+ params
            base_chunk = 512
        elif total_params > 100_000_000:  # 100M-200M params
            base_chunk = 1024
        else:  # <100M params
            base_chunk = 2048
        
        # For very long sequences, use even smaller chunks
        if seq_len >= 100_000:
            return min(base_chunk, 512)
        elif seq_len >= 50_000:
            return min(base_chunk, 1024)
        elif seq_len >= 20_000:
            return min(base_chunk, 1536)
        else:
            return base_chunk
    
    def _estimate_dataset_memory(self, batch_size: int, seq_len: int) -> float:
        """
        Estimate dataset memory overhead.
        With streaming: only current batch tokenized (~0.16 GB for batch=2, seq=10K)
        Without streaming: minimal (text strings are lightweight)
        """
        # With streaming enabled for large datasets, memory is minimal
        # Only current batch is tokenized at a time
        bytes_per_token = 4  # int32
        
        # Estimate: input_ids + labels for current batch only
        current_batch_memory = batch_size * seq_len * bytes_per_token * 2  # input_ids + labels
        
        # Streaming keeps memory low - just current batch + some overhead
        streaming_overhead_gb = (current_batch_memory * 1.2) / (1024**3)  # 20% overhead for bookkeeping
        
        # Cap at reasonable minimum (even with tiny batches, there's some overhead)
        return max(0.05, streaming_overhead_gb)  # At least 50 MB
    
    def _estimate_teacher_params(self, model_name: str) -> int:
        """Estimate teacher model parameters based on model name."""
        model_name_lower = model_name.lower()
        
        # Common model sizes (approximate)
        if "gpt2-xl" in model_name_lower or "gpt2xl" in model_name_lower:
            return 1_500_000_000  # 1.5B
        
        elif "gpt2-medium" in model_name_lower or "gpt2medium" in model_name_lower:
            return 345_000_000  # 345M
        
        elif "distilgpt2" in model_name_lower:
            return 82_000_000   # 82M
        elif "gpt-j" in model_name_lower or "gptj" in model_name_lower:
            return 6_000_000_000  # 6B
        elif "llama-7b" in model_name_lower or "7b" in model_name_lower:
            return 7_000_000_000  # 7B
        elif "llama-13b" in model_name_lower or "13b" in model_name_lower:
            return 13_000_000_000  # 13B
        elif "phi" in model_name_lower:
            if "2.7" in model_name_lower:
                return 2_700_000_000  # 2.7B
            else:
                return 1_300_000_000  # 1.3B (Phi-1.5)
        elif "mistral" in model_name_lower:
            return 7_000_000_000  # 7B
        else:
            # Default: generic base model estimate
            return 124_000_000

    # Method _force_stop removed - functionality merged into universal stop button
