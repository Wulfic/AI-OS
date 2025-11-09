"""Export utilities for evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.evaluation import EvaluationResult


def export_to_csv(result: "EvaluationResult", model_name: str) -> None:
    """Export evaluation results to CSV file.
    
    Args:
        result: EvaluationResult to export
        model_name: Name of the evaluated model
    """
    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"eval_results_{model_name.replace('/', '_')}.csv",
    )
    
    if not filepath:
        return
    
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Model", model_name])
            writer.writerow(["Overall Score", f"{result.overall_score:.4f}"])
            writer.writerow(["Duration", result.duration_str])
            writer.writerow(["Status", result.status])
            writer.writerow([])
            
            # Benchmark scores
            writer.writerow(["Benchmark", "Metric", "Score", "Std Error"])
            
            for task_name, task_data in sorted(result.benchmark_scores.items()):
                scores_dict = task_data.get("scores", {})
                raw_data = task_data.get("raw", {})
                
                for metric_name, score_value in sorted(scores_dict.items()):
                    stderr = ""
                    if f"{metric_name}_stderr" in raw_data:
                        stderr = f"{raw_data[f'{metric_name}_stderr']:.6f}"
                    
                    writer.writerow([
                        task_name,
                        metric_name,
                        f"{score_value:.6f}",
                        stderr,
                    ])
        
        messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
    
    except Exception as e:
        messagebox.showerror("Export Failed", f"Failed to export CSV:\n{e}")


def export_to_json(result: "EvaluationResult", model_name: str) -> None:
    """Export evaluation results to JSON file.
    
    Args:
        result: EvaluationResult to export
        model_name: Name of the evaluated model
    """
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=f"eval_results_{model_name.replace('/', '_')}.json",
    )
    
    if not filepath:
        return
    
    try:
        export_data = {
            "model": model_name,
            "overall_score": result.overall_score,
            "duration_seconds": result.duration,
            "duration_str": result.duration_str,
            "status": result.status,
            "error_message": result.error_message,
            "benchmark_scores": result.benchmark_scores,
            "raw_results": result.raw_results,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        
        messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
    
    except Exception as e:
        messagebox.showerror("Export Failed", f"Failed to export JSON:\n{e}")


def export_to_html(result: "EvaluationResult", model_name: str) -> None:
    """Export evaluation results to HTML report.
    
    Args:
        result: EvaluationResult to export
        model_name: Name of the evaluated model
    """
    filepath = filedialog.asksaveasfilename(
        defaultextension=".html",
        filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
        initialfile=f"eval_report_{model_name.replace('/', '_')}.html",
    )
    
    if not filepath:
        return
    
    try:
        html = generate_html_report(result, model_name)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        
        messagebox.showinfo("Export Successful", f"Report exported to:\n{filepath}")
    
    except Exception as e:
        messagebox.showerror("Export Failed", f"Failed to export HTML:\n{e}")


def generate_html_report(result: "EvaluationResult", model_name: str) -> str:
    """Generate HTML report from evaluation results.
    
    Args:
        result: EvaluationResult to format
        model_name: Name of the evaluated model
        
    Returns:
        HTML report as string
    """
    # Build benchmark rows
    benchmark_rows = []
    for task_name, task_data in sorted(result.benchmark_scores.items()):
        scores_dict = task_data.get("scores", {})
        
        for metric_name, score_value in sorted(scores_dict.items()):
            score_pct = f"{score_value:.2%}" if score_value <= 1.0 else f"{score_value:.4f}"
            benchmark_rows.append(
                f"<tr><td>{task_name}</td><td>{metric_name}</td><td>{score_pct}</td></tr>"
            )
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-item {{
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .summary-item label {{
            display: block;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .summary-item value {{
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Evaluation Report</h1>
        
        <div class="summary">
            <div class="summary-item">
                <label>Model</label>
                <value>{model_name}</value>
            </div>
            <div class="summary-item">
                <label>Overall Score</label>
                <value>{result.overall_score:.2%}</value>
            </div>
            <div class="summary-item">
                <label>Status</label>
                <value>{result.status.upper()}</value>
            </div>
            <div class="summary-item">
                <label>Duration</label>
                <value>{result.duration_str}</value>
            </div>
        </div>
        
        <h2>📊 Benchmark Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                {"".join(benchmark_rows)}
            </tbody>
        </table>
        
        <div class="footer">
            Generated by AI-OS Evaluation System | {result.duration_str} evaluation time
        </div>
    </div>
</body>
</html>"""
    
    return html
