"""Dialog for viewing logged evaluation samples."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_spacing_multiplier


class EvaluationSamplesDialog(tk.Toplevel):  # type: ignore[misc]
    """Dialog for viewing detailed sample logs from evaluations."""
    
    def __init__(
        self,
        parent: Any,
        samples_path: str,
        tasks: list[str],
    ) -> None:
        """Initialize the samples viewer dialog.
        
        Args:
            parent: Parent window
            samples_path: Directory containing sample files
            tasks: List of task names evaluated
        """
        super().__init__(parent)
        
        self.samples_path = Path(samples_path)
        self.tasks = tasks
        self.samples_data: dict[str, list[dict[str, Any]]] = {}
        
        # Configure window
        self.title("Evaluation Samples")
        self.geometry("1000x700")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Apply theme to this dialog
        apply_theme_to_toplevel(self)
        
        # Build UI
        self._build_ui()
        
        # Load samples
        self._load_samples()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Get spacing multiplier for current theme
        spacing = get_spacing_multiplier()
        
        # Main container
        padding = int(10 * spacing)
        main_frame = ttk.Frame(self, padding=padding)
        main_frame.pack(fill="both", expand=True)
        
        # Top: Task selector and filter
        top_frame = ttk.Frame(main_frame)
        pady_val = (0, int(10 * spacing))
        top_frame.pack(fill="x", pady=pady_val)
        
        ttk.Label(top_frame, text="Task:").pack(side="left", padx=(0, int(5 * spacing)))
        
        self.task_var = tk.StringVar()
        self.task_combo = ttk.Combobox(
            top_frame,
            textvariable=self.task_var,
            state="readonly",
            width=25,
        )
        self.task_combo.pack(side="left", padx=(0, int(15 * spacing)))
        self.task_combo.bind("<<ComboboxSelected>>", lambda e: self._on_task_changed())
        
        # Filter controls
        ttk.Label(top_frame, text="Show:").pack(side="left", padx=(0, int(5 * spacing)))
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(
            top_frame,
            textvariable=self.filter_var,
            values=["all", "correct", "incorrect"],
            state="readonly",
            width=12,
        )
        filter_combo.pack(side="left", padx=(0, int(10 * spacing)))
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())
        
        # Stats label
        self.stats_label = ttk.Label(top_frame, text="", foreground="blue")
        self.stats_label.pack(side="left", padx=(int(15 * spacing), 0))
        
        # Middle: Sample list
        list_frame = ttk.LabelFrame(main_frame, text="Samples", padding=int(5 * spacing))
        list_frame.pack(fill="both", expand=True, pady=(0, int(10 * spacing)))
        
        # Sample listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.sample_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 9),
            height=8,
        )
        self.sample_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.sample_listbox.yview)
        
        self.sample_listbox.bind("<<ListboxSelect>>", lambda e: self._on_sample_selected())
        
        # Bottom: Sample details
        details_frame = ttk.LabelFrame(main_frame, text="Sample Details", padding=int(5 * spacing))
        details_frame.pack(fill="both", expand=True, pady=(0, int(10 * spacing)))
        
        # Text widget with scrollbar
        text_container = ttk.Frame(details_frame)
        text_container.pack(fill="both", expand=True)
        
        text_scroll = ttk.Scrollbar(text_container)
        text_scroll.pack(side="right", fill="y")
        
        self.details_text = tk.Text(
            text_container,
            wrap="word",
            yscrollcommand=text_scroll.set,
            font=("Segoe UI", 10),
            height=15,
        )
        self.details_text.pack(side="left", fill="both", expand=True)
        text_scroll.config(command=self.details_text.yview)
        
        # Configure text tags for formatting
        self.details_text.tag_configure("heading", font=("Segoe UI", 11, "bold"), foreground="#2c3e50")
        self.details_text.tag_configure("label", font=("Segoe UI", 10, "bold"), foreground="#34495e")
        self.details_text.tag_configure("correct", foreground="#27ae60", font=("Segoe UI", 10, "bold"))
        self.details_text.tag_configure("incorrect", foreground="#e74c3c", font=("Segoe UI", 10, "bold"))
        self.details_text.tag_configure("question", foreground="#2980b9")
        self.details_text.tag_configure("code", font=("Consolas", 9), background="#f8f9fa")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        ttk.Button(
            button_frame, text="Export Samples", command=self._export_samples
        ).pack(side="left", padx=(0, int(5 * spacing)))
        
        ttk.Button(
            button_frame, text="Close", command=self.destroy
        ).pack(side="right")
    
    def _load_samples(self) -> None:
        """Load sample files from the samples directory."""
        if not self.samples_path.exists():
            messagebox.showerror("Error", f"Samples directory not found:\n{self.samples_path}")
            self.destroy()
            return
        
        try:
            # Find all samples_*.jsonl files
            sample_files = list(self.samples_path.glob("samples_*.jsonl"))
            
            if not sample_files:
                messagebox.showwarning(
                    "No Samples",
                    f"No sample files found in:\n{self.samples_path}\n\n"
                    "Make sure 'Log samples' was enabled during evaluation."
                )
                self.destroy()
                return
            
            # Load each sample file
            for sample_file in sample_files:
                # Extract task name from filename: samples_<task>_<timestamp>.jsonl
                task_name = sample_file.stem.replace("samples_", "").rsplit("_", 2)[0]
                
                samples = []
                with open(sample_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                
                self.samples_data[task_name] = samples
            
            # Populate task dropdown
            task_names = sorted(self.samples_data.keys())
            self.task_combo.config(values=task_names)
            if task_names:
                self.task_var.set(task_names[0])
                self._on_task_changed()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load samples:\n{e}")
            self.destroy()
    
    def _on_task_changed(self) -> None:
        """Handle task selection change."""
        self._apply_filter()
    
    def _apply_filter(self) -> None:
        """Apply filter and update sample list."""
        task = self.task_var.get()
        if not task or task not in self.samples_data:
            return
        
        samples = self.samples_data[task]
        filter_val = self.filter_var.get()
        
        # Clear listbox
        self.sample_listbox.delete(0, tk.END)
        
        # Filter samples
        correct_count = 0
        incorrect_count = 0
        
        for i, sample in enumerate(samples):
            is_correct = sample.get("acc", 0) > 0.5  # Most metrics use 1.0 for correct, 0.0 for incorrect
            
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
            
            # Apply filter
            if filter_val == "correct" and not is_correct:
                continue
            elif filter_val == "incorrect" and is_correct:
                continue
            
            # Format list item
            status = "✓" if is_correct else "✗"
            doc_id = sample.get("doc_id", i)
            
            # Get question preview
            doc = sample.get("doc", {})
            question = ""
            if isinstance(doc, dict):
                question = doc.get("question", doc.get("prompt", ""))
            
            if not question and "arguments" in sample:
                # Try to extract from arguments
                args = sample.get("arguments", {})
                if args:
                    first_arg = list(args.values())[0] if args else {}
                    if isinstance(first_arg, dict):
                        arg_0 = first_arg.get("arg_0", "")
                        if arg_0:
                            # Extract question part (usually at the end)
                            if "Question:" in arg_0:
                                question = arg_0.split("Question:")[-1].split("\n")[0].strip()
            
            # Truncate question for display
            if len(question) > 70:
                question = question[:67] + "..."
            
            list_item = f"{status} #{doc_id}: {question}"
            self.sample_listbox.insert(tk.END, list_item)
            
            # Store original index as data
            self.sample_listbox.itemconfig(tk.END, foreground="#27ae60" if is_correct else "#e74c3c")
        
        # Update stats
        total = len(samples)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        self.stats_label.config(
            text=f"Total: {total} | Correct: {correct_count} | Incorrect: {incorrect_count} | Accuracy: {accuracy:.1f}%"
        )
    
    def _on_sample_selected(self) -> None:
        """Handle sample selection."""
        selection = self.sample_listbox.curselection()
        if not selection:
            return
        
        # Get the selected item text to find the doc_id
        item_text = self.sample_listbox.get(selection[0])
        # Extract doc_id from format: "✓ #123: question..."
        try:
            doc_id_str = item_text.split("#")[1].split(":")[0].strip()
            doc_id = int(doc_id_str)
        except (IndexError, ValueError):
            return
        
        # Find the sample with this doc_id
        task = self.task_var.get()
        samples = self.samples_data.get(task, [])
        
        sample = None
        for s in samples:
            if s.get("doc_id") == doc_id:
                sample = s
                break
        
        if not sample:
            return
        
        # Display sample details
        self._display_sample(sample)
    
    def _display_sample(self, sample: dict[str, Any]) -> None:
        """Display detailed sample information."""
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        
        # Get sample data
        doc = sample.get("doc", {})
        doc_id = sample.get("doc_id", "?")
        is_correct = sample.get("acc", 0) > 0.5
        
        # Title
        self.details_text.insert(tk.END, f"Sample #{doc_id}\n", "heading")
        self.details_text.insert(tk.END, f"Result: ", "label")
        result_text = "CORRECT ✓\n" if is_correct else "INCORRECT ✗\n"
        result_tag = "correct" if is_correct else "incorrect"
        self.details_text.insert(tk.END, result_text, result_tag)
        self.details_text.insert(tk.END, "\n")
        
        # Question/Prompt
        if isinstance(doc, dict):
            question = doc.get("question", doc.get("prompt", doc.get("input", "")))
            if question:
                self.details_text.insert(tk.END, "Question:\n", "label")
                self.details_text.insert(tk.END, f"{question}\n\n", "question")
            
            # Context/Passage (if available)
            context = doc.get("passage", doc.get("context", ""))
            if context:
                self.details_text.insert(tk.END, "Context:\n", "label")
                # Truncate very long contexts
                if len(context) > 500:
                    context = context[:497] + "..."
                self.details_text.insert(tk.END, f"{context}\n\n")
        
        # Target answer
        target = sample.get("target", "?")
        self.details_text.insert(tk.END, "Correct Answer: ", "label")
        self.details_text.insert(tk.END, f"{target}\n\n")
        
        # Model responses
        resps = sample.get("resps", [])
        filtered_resps = sample.get("filtered_resps", [])
        
        if filtered_resps:
            self.details_text.insert(tk.END, "Model Responses:\n", "label")
            for i, resp in enumerate(filtered_resps):
                if isinstance(resp, list) and len(resp) >= 2:
                    score = resp[0]
                    self.details_text.insert(tk.END, f"  Option {i+1}: {resp[1]} (log-prob: {score})\n")
                else:
                    self.details_text.insert(tk.END, f"  Response {i+1}: {resp}\n")
            self.details_text.insert(tk.END, "\n")
        
        # Arguments (prompts sent to model)
        arguments = sample.get("arguments", {})
        if arguments and len(arguments) <= 3:  # Only show if not too many
            self.details_text.insert(tk.END, "Full Prompts:\n", "label")
            for key, arg_data in arguments.items():
                if isinstance(arg_data, dict):
                    prompt = arg_data.get("arg_0", "")
                    if prompt:
                        # Truncate very long prompts
                        if len(prompt) > 300:
                            prompt = prompt[:297] + "..."
                        self.details_text.insert(tk.END, f"\n{key}:\n", "label")
                        self.details_text.insert(tk.END, f"{prompt}\n", "code")
        
        self.details_text.config(state="disabled")
    
    def _export_samples(self) -> None:
        """Export samples to a file."""
        from tkinter import filedialog
        
        task = self.task_var.get()
        if not task:
            return
        
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="Export Samples",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"samples_{task}.jsonl",
        )
        
        if not filename:
            return
        
        try:
            samples = self.samples_data.get(task, [])
            filter_val = self.filter_var.get()
            
            # Filter samples
            filtered_samples = []
            for sample in samples:
                is_correct = sample.get("acc", 0) > 0.5
                
                if filter_val == "correct" and not is_correct:
                    continue
                elif filter_val == "incorrect" and is_correct:
                    continue
                
                filtered_samples.append(sample)
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                for sample in filtered_samples:
                    f.write(json.dumps(sample) + "\n")
            
            messagebox.showinfo("Success", f"Exported {len(filtered_samples)} samples to:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export samples:\n{e}")
