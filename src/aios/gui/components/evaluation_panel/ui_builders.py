"""UI builder functions for evaluation panel."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING

from ..tooltips import add_tooltip
from .benchmark_data import PRESET_CATEGORIES
from . import tree_management
from aios.gui.utils.model_display import get_model_display_name

if TYPE_CHECKING:
    from .panel_main import EvaluationPanel


def create_model_selection(panel: "EvaluationPanel") -> None:
    """Create the model selection section.
    
    Args:
        panel: The evaluation panel instance
    """
    model_frame = ttk.LabelFrame(panel, text="Model Selection")
    model_frame.pack(fill="x", padx=6, pady=(6, 4))
    
    # Brain selection (using brains list like chat tab)
    brain_row = ttk.Frame(model_frame)
    brain_row.pack(fill="x", padx=6, pady=4)
    ttk.Label(brain_row, text="Brain:").pack(side="left")
    panel.brain_combo = ttk.Combobox(
        brain_row,
        textvariable=panel.model_name_var,
        values=[],
        width=40
    )
    add_tooltip(panel.brain_combo, 
                "Model to Evaluate:\n\n"
                "• Select a trained brain from your artifacts/brains folder\n"
                "• Or enter a HuggingFace model name (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')\n\n"
                "AI-OS brains are automatically detected and use the native adapter.\n"
                "External models are loaded through HuggingFace transformers.")
    panel.brain_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
    panel.refresh_brains_btn = ttk.Button(brain_row, text="🔄", command=panel._refresh_brains, width=3)
    add_tooltip(panel.refresh_brains_btn, "Refresh list of available brains from artifacts/brains folder")
    panel.refresh_brains_btn.pack(side="left", padx=(4, 0))
    
    # Don't load brains here - will be loaded async after panel creation
    # This avoids blocking the UI thread with a 9+ second subprocess call


def create_benchmark_selection(panel: "EvaluationPanel") -> None:
    """Create the benchmark selection section.
    
    Args:
        panel: The evaluation panel instance
    """
    bench_frame = ttk.LabelFrame(panel, text="Benchmark Selection")
    bench_frame.pack(fill="both", expand=True, padx=6, pady=4)
    
    # Preset buttons
    preset_row = ttk.Frame(bench_frame)
    preset_row.pack(fill="x", padx=6, pady=4)
    ttk.Label(preset_row, text="Presets:").pack(side="left")
    
    for preset in PRESET_CATEGORIES:
        btn = ttk.Button(
            preset_row,
            text=preset,
            command=lambda p=preset: panel._select_preset(p),
            width=10
        )
        tooltip_texts = {
            "Language": "Language understanding benchmarks (MMLU, HellaSwag, TruthfulQA, etc.)",
            "Coding": "Programming and code generation benchmarks (HumanEval, MBPP, etc.)",
            "Math": "Mathematical reasoning and problem solving (GSM8K, MATH, etc.)",
            "Science": "Scientific knowledge and reasoning benchmarks",
            "All": "Select all available benchmarks",
            "Custom": "Manually select individual benchmarks"
        }
        add_tooltip(btn, tooltip_texts.get(preset, f"Quick select {preset} benchmark group"))
        btn.pack(side="left", padx=2)
    
    # Benchmark list and info
    list_frame = ttk.Frame(bench_frame)
    list_frame.pack(fill="both", expand=True, padx=10, pady=6)
    
    # Create Treeview for benchmarks
    columns = ("Category", "Benchmark", "Description")
    panel.bench_tree = ttk.Treeview(
        list_frame,
        columns=columns,
        show="tree headings",
        selectmode="extended",
        height=8
    )
    add_tooltip(panel.bench_tree, 
                "Benchmark Selection:\n\n"
                "Click the checkbox (☐/☑) to select/deselect benchmarks.\n\n"
                "Categories:\n"
                "• Language - Reading comprehension, reasoning, Q&A\n"
                "• Coding - Programming tasks and code generation\n"
                "• Math - Mathematical problem solving\n"
                "• Science - Scientific knowledge and reasoning\n\n"
                "Common benchmarks:\n"
                "• MMLU - Massive multitask language understanding (57 tasks)\n"
                "• HellaSwag - Commonsense reasoning\n"
                "• BoolQ - Yes/no question answering\n"
                "• TruthfulQA - Truthfulness evaluation")
    panel.bench_tree.heading("#0", text="Select")
    panel.bench_tree.column("#0", width=100, stretch=False)
    panel.bench_tree.heading("Category", text="Category")
    panel.bench_tree.column("Category", width=80)
    panel.bench_tree.heading("Benchmark", text="Benchmark")
    panel.bench_tree.column("Benchmark", width=120)
    panel.bench_tree.heading("Description", text="Description")
    panel.bench_tree.column("Description", width=400)
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=panel.bench_tree.yview)
    panel.bench_tree.configure(yscrollcommand=scrollbar.set)
    
    panel.bench_tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Create empty tree - will be populated async
    panel._tree_populated = False
    panel._tree_items = {}
    
    # Show "Loading benchmarks..." placeholder
    placeholder_id = panel.bench_tree.insert("", "end", text="", values=("Loading benchmarks...", "", ""))
    panel._tree_placeholder_id = placeholder_id
    
    # Bind click to toggle selection
    panel.bench_tree.bind("<Button-1>", lambda e: tree_management.on_benchmark_click(panel, e))
    
    # Selection info
    info_row = ttk.Frame(bench_frame)
    info_row.pack(fill="x", padx=6, pady=(4, 6))
    panel.selection_label = ttk.Label(info_row, text="Selected: 0 benchmarks", foreground="gray")
    add_tooltip(panel.selection_label, "Summary of selected benchmarks for evaluation")
    panel.selection_label.pack(side="left")


def create_configuration_section(panel: "EvaluationPanel") -> None:
    """Create the configuration section.
    
    Args:
        panel: The evaluation panel instance
    """
    config_frame = ttk.LabelFrame(panel, text="Configuration")
    config_frame.pack(fill="x", padx=6, pady=4)
    
    # Config row 1
    config1 = ttk.Frame(config_frame)
    config1.pack(fill="x", padx=6, pady=4)
    
    ttk.Label(config1, text="Batch Size:").pack(side="left")
    batch_entry = ttk.Entry(config1, textvariable=panel.batch_size_var, width=8)
    add_tooltip(batch_entry, 
                "Batch Size - Controls how many samples are processed together:\n\n"
                "• 'auto' = Automatically determine based on available GPU memory (recommended)\n"
                "• Number (e.g., 1, 8, 16) = Process this many samples at once\n\n"
                "Larger batches are faster but use more memory.\n"
                "If you get out-of-memory errors, try 'auto' or a smaller number.")
    batch_entry.pack(side="left", padx=(4, 12))
    
    ttk.Label(config1, text="Limit:").pack(side="left")
    limit_entry = ttk.Entry(config1, textvariable=panel.limit_var, width=6)
    add_tooltip(limit_entry, 
                "Dataset Limit - Percentage of dataset to evaluate:\n\n"
                "• 0 = Unlimited (evaluate full dataset - most accurate)\n"
                "• 1-100 = Evaluate only this percentage of samples\n\n"
                "Examples:\n"
                "• 10 = Evaluate 10% of dataset (quick test)\n"
                "• 50 = Evaluate 50% of dataset (faster but less accurate)\n"
                "• 0 or 100 = Full evaluation (slowest but most accurate)\n\n"
                "Useful for quick testing before running full evaluation.")
    limit_entry.pack(side="left", padx=(4, 0))
    ttk.Label(config1, text="%").pack(side="left", padx=(0, 12))
    
    ttk.Label(config1, text="Few-shot:").pack(side="left")
    fewshot_entry = ttk.Entry(config1, textvariable=panel.num_fewshot_var, width=6)
    add_tooltip(fewshot_entry, 
                "Few-Shot Examples - Number of example Q&A pairs shown before each question:\n\n"
                "• 0 = Zero-shot (no examples, tests raw knowledge)\n"
                "• 1-5 = Show this many example question-answer pairs first\n"
                "• 5 = Standard benchmark setting (recommended)\n\n"
                "How it works:\n"
                "The model sees example questions with correct answers before\n"
                "being asked the test question. This helps it understand the\n"
                "task format and expected answer style.\n\n"
                "More examples usually improve accuracy but use more tokens.\n"
                "If you get context length errors, reduce this number.")
    fewshot_entry.pack(side="left", padx=(4, 12))
    
    # Config row 2
    config2 = ttk.Frame(config_frame)
    config2.pack(fill="x", padx=6, pady=4)
    
    ttk.Label(config2, text="Output:").pack(side="left")
    output_entry = ttk.Entry(config2, textvariable=panel.output_path_var, width=40)
    add_tooltip(output_entry, 
                "Output Directory - Where to save evaluation results:\n\n"
                "Results are saved as JSON files containing:\n"
                "• Accuracy scores and metrics\n"
                "• Per-task performance breakdown\n"
                "• Metadata (model, date, configuration)\n\n"
                "Default: artifacts/evaluation")
    output_entry.pack(side="left", padx=(4, 0), fill="x", expand=True)
    browse_btn = ttk.Button(config2, text="📁", command=panel._browse_output, width=3)
    add_tooltip(browse_btn, "Browse for output directory")
    browse_btn.pack(side="left", padx=(4, 0))


def create_advanced_options(panel: "EvaluationPanel") -> None:
    """Create the advanced options section.
    
    Args:
        panel: The evaluation panel instance
    """
    # Get the last created frame (config_frame from create_configuration_section)
    config_frame = panel.winfo_children()[-1]
    
    # Advanced options
    config3 = ttk.Frame(config_frame)
    config3.pack(fill="x", padx=6, pady=(4, 6))
    
    log_samples_cb = ttk.Checkbutton(config3, text="Log samples", variable=panel.log_samples_var)
    add_tooltip(log_samples_cb, 
                "Log Samples - Save detailed predictions for each sample:\n\n"
                "When enabled:\n"
                "• Saves the model's prediction for every test question\n"
                "• Includes the input prompt and expected answer\n"
                "• Useful for debugging and analyzing errors\n\n"
                "⚠ Warning: Creates large files (100s of MB for full evaluations)\n"
                "Only enable if you need to analyze individual predictions.")
    log_samples_cb.pack(side="left", padx=(0, 12))
    
    cache_cb = ttk.Checkbutton(config3, text="Cache results", variable=panel.cache_requests_var)
    add_tooltip(cache_cb, 
                "Cache Results - Reuse results from previous evaluations:\n\n"
                "When enabled:\n"
                "• Identical requests are cached and reused\n"
                "• Significantly speeds up repeated evaluations\n"
                "• Safe for comparing different models on same benchmarks\n\n"
                "Recommended: Keep enabled unless testing changes to the model\n"
                "or if you suspect cache corruption.")
    cache_cb.pack(side="left", padx=(0, 12))
    
    integrity_cb = ttk.Checkbutton(config3, text="Check integrity", variable=panel.check_integrity_var)
    add_tooltip(integrity_cb, 
                "Check Integrity - Verify dataset files before evaluation:\n\n"
                "When enabled:\n"
                "• Validates checksums of downloaded benchmark datasets\n"
                "• Ensures no corrupted or tampered data\n"
                "• Adds a few seconds to startup time\n\n"
                "Recommended: Only enable if you suspect dataset corruption\n"
                "or for critical/official benchmark runs.")
    integrity_cb.pack(side="left")


def create_control_buttons(panel: "EvaluationPanel") -> None:
    """Create the execution control buttons.
    
    Args:
        panel: The evaluation panel instance
    """
    controls_frame = ttk.Frame(panel)
    controls_frame.pack(fill="x", padx=6, pady=4)
    
    panel.start_btn = ttk.Button(
        controls_frame,
        text="▶ Start Evaluation",
        command=panel._on_start_evaluation
    )
    add_tooltip(panel.start_btn, 
                "Start Evaluation:\n\n"
                "Runs the selected benchmarks on the chosen model.\n"
                "Progress will be shown in the progress bar below.\n\n"
                "Make sure to:\n"
                "• Select a model/brain\n"
                "• Select at least one benchmark\n"
                "• Configure settings (batch size, few-shot, etc.)")
    panel.start_btn.pack(side="left", padx=(0, 6))
    
    panel.stop_btn = ttk.Button(
        controls_frame,
        text="⏹ Stop",
        command=panel._on_stop_evaluation,
        state="disabled"
    )
    add_tooltip(panel.stop_btn, 
                "Stop Evaluation:\n\n"
                "Cancels the current evaluation.\n"
                "Partial results may be saved depending on when stopped.")
    panel.stop_btn.pack(side="left", padx=(0, 6))
    
    history_btn = ttk.Button(
        controls_frame,
        text="📊 View History",
        command=panel._on_view_history
    )
    add_tooltip(history_btn, 
                "View History:\n\n"
                "Browse past evaluation results:\n"
                "• Compare scores across different models\n"
                "• View detailed metrics and performance\n"
                "• Export results to CSV or JSON\n"
                "• Delete old evaluations")
    history_btn.pack(side="left", padx=(0, 6))
    
    clear_btn = ttk.Button(
        controls_frame,
        text="Clear Output",
        command=panel._clear_output
    )
    add_tooltip(clear_btn, "Clear the output log below (does not affect saved results)")
    clear_btn.pack(side="right")


def create_progress_section(panel: "EvaluationPanel") -> None:
    """Create the progress section.
    
    Args:
        panel: The evaluation panel instance
    """
    # Progress is now integrated into the output section header
    pass


def create_output_section(panel: "EvaluationPanel") -> None:
    """Create the output log section.
    
    Args:
        panel: The evaluation panel instance
    """
    log_frame = ttk.LabelFrame(panel, text="Output Log")
    log_frame.pack(fill="both", expand=True, padx=6, pady=4)
    
    # Progress bar in header (inside the frame)
    progress_frame = ttk.Frame(log_frame)
    progress_frame.pack(fill="x", padx=4, pady=(2, 4))
    
    ttk.Label(progress_frame, text="Progress:").pack(side="left")
    panel.progress = ttk.Progressbar(
        progress_frame,
        orient="horizontal",
        mode="determinate",
        length=200
    )
    add_tooltip(panel.progress, "Evaluation progress (benchmarks completed)")
    panel.progress.pack(side="left", fill="x", expand=True, padx=(6, 6))
    panel.progress_label = ttk.Label(progress_frame, text="idle", width=25, anchor="w")
    add_tooltip(panel.progress_label, "Current evaluation status")
    panel.progress_label.pack(side="left")
    
    # Text area with scrollbar
    text_frame = ttk.Frame(log_frame)
    text_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
    
    panel.log_text = tk.Text(text_frame, height=10, wrap="word", state="disabled")
    add_tooltip(panel.log_text, "Real-time evaluation output and progress messages")
    log_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=panel.log_text.yview)
    panel.log_text.configure(yscrollcommand=log_scrollbar.set)
    
    panel.log_text.pack(side="left", fill="both", expand=True)
    log_scrollbar.pack(side="right", fill="y")


def create_results_section(panel: "EvaluationPanel") -> None:
    """Create the results summary section.
    
    Args:
        panel: The evaluation panel instance
    """
    results_frame = ttk.LabelFrame(panel, text="Results Summary")
    results_frame.pack(fill="both", expand=True, padx=6, pady=(4, 6))
    
    # Current result summary row
    summary_row = ttk.Frame(results_frame)
    summary_row.pack(fill="x", padx=6, pady=4)
    
    ttk.Label(summary_row, text="Current:").pack(side="left", padx=(0, 4))
    panel.results_label = ttk.Label(
        summary_row,
        text="No evaluation results yet",
        foreground="gray"
    )
    add_tooltip(panel.results_label, "Summary of most recent evaluation results")
    panel.results_label.pack(side="left", fill="x", expand=True)
    
    view_btn = ttk.Button(
        summary_row,
        text="View Details",
        command=panel._view_detailed_results,
        state="disabled"
    )
    add_tooltip(view_btn, "View detailed results in separate window")
    view_btn.pack(side="right", padx=2)
    
    csv_btn = ttk.Button(
        summary_row,
        text="Export CSV",
        command=lambda: panel._export_results("csv"),
        state="disabled"
    )
    add_tooltip(csv_btn, "Export results to CSV file")
    csv_btn.pack(side="right", padx=2)
    
    json_btn = ttk.Button(
        summary_row,
        text="Export JSON",
        command=lambda: panel._export_results("json"),
        state="disabled"
    )
    add_tooltip(json_btn, "Export results to JSON file")
    json_btn.pack(side="right", padx=2)
    
    # Store button references for enabling/disabling
    panel._result_buttons = [view_btn, csv_btn, json_btn]
    
    # Recent results list
    recent_frame = ttk.Frame(results_frame)
    recent_frame.pack(fill="both", expand=True, padx=6, pady=(4, 6))
    
    recent_header = ttk.Frame(recent_frame)
    recent_header.pack(fill="x", pady=(0, 2))
    ttk.Label(recent_header, text="Recent Evaluations:", font=("", 9, "bold")).pack(side="left")
    refresh_recent_btn = ttk.Button(
        recent_header,
        text="🔄",
        command=lambda: _refresh_recent_results(panel),
        width=3
    )
    add_tooltip(refresh_recent_btn, "Refresh recent results list")
    refresh_recent_btn.pack(side="right")
    
    # Create a treeview for recent results
    columns = ("Model", "Date", "Score", "Benchmarks")
    panel.recent_tree = ttk.Treeview(
        recent_frame,
        columns=columns,
        show="headings",
        selectmode="browse",
        height=4
    )
    add_tooltip(panel.recent_tree, "Recent evaluation results. Double-click to view details.")
    
    panel.recent_tree.heading("Model", text="Model")
    panel.recent_tree.column("Model", width=150)
    panel.recent_tree.heading("Date", text="Date")
    panel.recent_tree.column("Date", width=140)
    panel.recent_tree.heading("Score", text="Score")
    panel.recent_tree.column("Score", width=80)
    panel.recent_tree.heading("Benchmarks", text="Benchmarks")
    panel.recent_tree.column("Benchmarks", width=200)
    
    recent_scrollbar = ttk.Scrollbar(recent_frame, orient="vertical", command=panel.recent_tree.yview)
    panel.recent_tree.configure(yscrollcommand=recent_scrollbar.set)
    
    panel.recent_tree.pack(side="left", fill="both", expand=True)
    recent_scrollbar.pack(side="right", fill="y")
    
    # Bind double-click to view details
    panel.recent_tree.bind("<Double-Button-1>", lambda e: _on_recent_result_double_click(panel))
    
    # Load initial recent results
    _refresh_recent_results(panel)


def _refresh_recent_results(panel: "EvaluationPanel") -> None:
    """Refresh the recent results list.
    
    Args:
        panel: The evaluation panel instance
    """
    try:
        # Clear existing items
        for item in panel.recent_tree.get_children():
            panel.recent_tree.delete(item)
        
        history = getattr(panel, "_history", None)
        if history is None:
            if not getattr(panel, "_history_notice_shown", False):
                panel._log("[eval] Evaluation history is still loading; recent results will appear once ready.")
                panel._history_notice_shown = True
            return

        if hasattr(panel, "_history_notice_shown"):
            delattr(panel, "_history_notice_shown")

        # Get recent evaluations (limit to 10)
        recent = history.get_recent_evaluations(limit=10)
        
        # Populate tree
        for eval_data in recent:
            eval_id = eval_data["id"]
            model_name = eval_data.get("model_name", "Unknown")
            model_display = get_model_display_name(model_name)
            timestamp = eval_data.get("timestamp", "")
            overall_score = eval_data.get("overall_score", 0.0)
            
            # Format date
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = timestamp[:16] if timestamp else "Unknown"
            
            # Get benchmark names
            benchmarks = []
            for score in eval_data.get("scores", []):
                benchmark = score.get("benchmark_name", "")
                if benchmark and benchmark not in benchmarks:
                    benchmarks.append(benchmark)
            
            benchmarks_str = ", ".join(benchmarks[:3])
            if len(benchmarks) > 3:
                benchmarks_str += f" +{len(benchmarks) - 3} more"
            
            # Format score
            score_str = f"{overall_score:.1%}" if overall_score > 0 else "N/A"
            
            # Insert into tree
            panel.recent_tree.insert(
                "",
                "end",
                values=(model_display, date_str, score_str, benchmarks_str),
                tags=(str(eval_id),)
            )
    
    except Exception as e:
        panel._log(f"[eval] Error refreshing recent results: {e}")


def _on_recent_result_double_click(panel: "EvaluationPanel") -> None:
    """Handle double-click on recent result.
    
    Args:
        panel: The evaluation panel instance
    """
    try:
        from aios.core.evaluation import EvaluationResult
        from aios.gui.dialogs import EvaluationResultsDialog
        
        history = getattr(panel, "_history", None)
        if history is None:
            panel._log("[eval] Evaluation history is still loading; please try again in a moment.")
            messagebox.showinfo("History Loading", "Evaluation history is still loading. Please try again shortly.")
            return

        selection = panel.recent_tree.selection()
        if not selection:
            return
        
        # Get eval_id from tags
        item = selection[0]
        tags = panel.recent_tree.item(item, "tags")
        if not tags:
            return
        
        eval_id = int(tags[0])
        
        # Get evaluation data
        eval_data = history.get_evaluation(eval_id)
        if not eval_data:
            messagebox.showerror("Error", "Evaluation not found.")
            return
        
        # Reconstruct EvaluationResult
        result = EvaluationResult(
            overall_score=eval_data.get("overall_score", 0.0),
            status=eval_data.get("status", "unknown"),
            error_message=eval_data.get("error_message", ""),
            start_time=eval_data.get("start_time", 0.0),
            end_time=eval_data.get("end_time", 0.0),
            output_path=eval_data.get("output_path", ""),
            raw_results=eval_data.get("raw_results", {}),
        )
        
        # Reconstruct benchmark_scores from scores list
        benchmark_scores = {}
        for score in eval_data.get("scores", []):
            benchmark = score["benchmark_name"]
            if benchmark not in benchmark_scores:
                benchmark_scores[benchmark] = {"scores": {}, "raw": {}}
            
            metric = score["metric_name"]
            benchmark_scores[benchmark]["scores"][metric] = score["score"]
            if score.get("stderr"):
                benchmark_scores[benchmark]["raw"][f"{metric}_stderr"] = score["stderr"]
        
        result.benchmark_scores = benchmark_scores
        
        # Open results dialog
        model_name = eval_data.get("model_name", "Unknown")
        EvaluationResultsDialog(panel, result, model_name)
    
    except Exception as e:
        panel._log(f"[eval] Error viewing result: {e}")
        messagebox.showerror("Error", f"Failed to view details:\n{e}")
