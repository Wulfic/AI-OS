"""Export utilities for evaluation results."""

from __future__ import annotations
import json
import csv
from typing import TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    from aios.core.evaluation import EvaluationResult


def export_to_json(result: "EvaluationResult", filepath: str, model_name: str) -> None:
    """Export evaluation results to JSON format.
    
    Args:
        result: The evaluation result to export
        filepath: Path to save the JSON file
        model_name: Name of the evaluated model
    """
    export_data = {
        "model": model_name,
        "overall_score": result.overall_score,
        "duration_seconds": result.duration,
        "duration_str": result.duration_str,
        "status": result.status,
        "benchmark_scores": result.benchmark_scores,
        "raw_results": result.raw_results,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)


def export_to_csv(result: "EvaluationResult", filepath: str, model_name: str) -> None:
    """Export evaluation results to CSV format.
    
    Args:
        result: The evaluation result to export
        filepath: Path to save the CSV file
        model_name: Name of the evaluated model
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header metadata
        writer.writerow(["Model", model_name])
        writer.writerow(["Overall Score", f"{result.overall_score:.4f}"])
        writer.writerow(["Duration", result.duration_str])
        writer.writerow(["Status", result.status])
        writer.writerow([])
        
        # Benchmark scores
        writer.writerow(["Benchmark", "Metric", "Score"])
        
        for task_name, task_data in sorted(result.benchmark_scores.items()):
            scores_dict = task_data.get("scores", {})
            
            for metric_name, score_value in sorted(scores_dict.items()):
                writer.writerow([
                    task_name,
                    metric_name,
                    f"{score_value:.6f}",
                ])


def export_to_markdown(result: "EvaluationResult", filepath: str, model_name: str) -> None:
    """Export evaluation results to Markdown format.
    
    Args:
        result: The evaluation result to export
        filepath: Path to save the Markdown file
        model_name: Name of the evaluated model
    """
    with open(filepath, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# Evaluation Results: {model_name}\n\n")
        f.write(f"**Overall Score:** {result.overall_score:.2%}\n\n")
        f.write(f"**Duration:** {result.duration_str}\n\n")
        f.write(f"**Status:** {result.status}\n\n")
        
        # Benchmark scores table
        f.write("## Benchmark Scores\n\n")
        f.write("| Benchmark | Metric | Score |\n")
        f.write("|-----------|--------|-------|\n")
        
        for task_name, task_data in sorted(result.benchmark_scores.items()):
            scores_dict = task_data.get("scores", {})
            
            for metric_name, score_value in sorted(scores_dict.items()):
                f.write(f"| {task_name} | {metric_name} | {score_value:.2%} |\n")
