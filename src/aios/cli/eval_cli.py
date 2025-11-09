"""CLI commands for model evaluation."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from aios.core.evaluation import EvaluationHistory, HarnessWrapper

app = typer.Typer(help="Model evaluation commands")
console = Console()


@app.command("run")
def eval_run(
    model: str = typer.Argument(..., help="Model name or path"),
    tasks: str = typer.Option(
        ...,
        "--tasks",
        "-t",
        help="Comma-separated list of tasks (e.g., 'mmlu,hellaswag,arc_challenge')",
    ),
    output: str = typer.Option(
        "artifacts/evaluation",
        "--output",
        "-o",
        help="Output directory for results",
    ),
    model_args: str = typer.Option(
        "dtype=auto",
        "--model-args",
        help="Model arguments (e.g., 'dtype=auto,trust_remote_code=True')",
    ),
    batch_size: str = typer.Option(
        "auto",
        "--batch-size",
        "-b",
        help="Batch size or 'auto'",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of samples per task (for testing)",
    ),
    num_fewshot: int = typer.Option(
        5,
        "--num-fewshot",
        "-n",
        help="Number of few-shot examples",
    ),
    device: str = typer.Option(
        "cuda:0",
        "--device",
        "-d",
        help="Device to use (cuda:0, cpu, etc.)",
    ),
    model_type: str = typer.Option(
        "hf",
        "--model-type",
        help="Model type (hf, vllm, local-completions, etc.)",
    ),
    log_samples: bool = typer.Option(
        False,
        "--log-samples",
        help="Log individual samples",
    ),
    save_to_history: bool = typer.Option(
        True,
        "--save-history/--no-save-history",
        help="Save results to evaluation history database",
    ),
) -> None:
    """Run evaluation on a model using lm-evaluation-harness.
    
    Examples:
        aios eval run gpt2 --tasks mmlu,hellaswag
        aios eval run facebook/opt-125m --tasks arc_challenge --limit 100
        aios eval run ./my_model --tasks humaneval --device cpu
    """
    # Check if lm_eval is installed
    if not HarnessWrapper.is_lm_eval_installed():
        console.print(
            "[red]Error:[/red] lm-evaluation-harness is not installed.\n"
            "Install with: [cyan]pip install lm-eval[/cyan]",
            style="bold"
        )
        raise typer.Exit(1)
    
    console.print(f"\n[bold cyan]Starting evaluation:[/bold cyan] {model}")
    console.print(f"Tasks: {tasks}")
    console.print(f"Output: {output}\n")
    
    # Parse tasks
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    
    if not task_list:
        console.print("[red]Error:[/red] No valid tasks specified", style="bold")
        raise typer.Exit(1)
    
    # Setup logging callbacks
    def log_callback(msg: str) -> None:
        # Remove [eval] prefix if present
        msg = msg.replace("[eval] ", "")
        console.print(msg)
    
    def progress_callback(pct: float, status: str) -> None:
        # Simple progress indicator
        bar_length = 40
        filled = int(bar_length * pct)
        bar = "█" * filled + "░" * (bar_length - filled)
        console.print(f"\r[{bar}] {pct*100:.0f}% - {status}", end="")
        if pct >= 1.0:
            console.print()  # New line at end
    
    # Create wrapper
    harness = HarnessWrapper(
        log_callback=log_callback,
        progress_callback=progress_callback,
    )
    
    # Run evaluation
    try:
        result = harness.run_evaluation(
            model_name=model,
            tasks=task_list,
            model_args=model_args,
            batch_size=batch_size,
            limit=limit,
            num_fewshot=num_fewshot,
            device=device,
            output_path=output,
            log_samples=log_samples,
            model_type=model_type,
        )
        
        # Display results
        console.print(f"\n[bold green]Evaluation completed![/bold green]")
        console.print(f"Status: {result.status}")
        console.print(f"Duration: {result.duration_str}")
        console.print(f"Overall Score: [bold]{result.overall_score:.2%}[/bold]\n")
        
        # Results table
        if result.benchmark_scores:
            table = Table(title="Benchmark Scores")
            table.add_column("Benchmark", style="cyan")
            table.add_column("Metric", style="magenta")
            table.add_column("Score", style="green", justify="right")
            
            for task_name, task_data in sorted(result.benchmark_scores.items()):
                scores_dict = task_data.get("scores", {})
                for metric_name, score_value in sorted(scores_dict.items()):
                    score_str = f"{score_value:.2%}" if score_value <= 1.0 else f"{score_value:.4f}"
                    table.add_row(task_name, metric_name, score_str)
            
            console.print(table)
        
        # Save to history
        if save_to_history and result.status == "completed":
            try:
                history = EvaluationHistory()
                eval_id = history.save_evaluation(
                    result=result,
                    model_name=model,
                    model_source="cli",
                    model_args=model_args,
                    tasks=task_list,
                    config={
                        "batch_size": batch_size,
                        "limit": limit,
                        "num_fewshot": num_fewshot,
                        "device": device,
                        "model_type": model_type,
                        "log_samples": log_samples,
                    },
                )
                console.print(f"\n[dim]Saved to history (ID: {eval_id})[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to save to history: {e}")
        
        console.print(f"\nResults saved to: [cyan]{result.output_path}[/cyan]\n")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation cancelled by user[/yellow]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)


@app.command("list")
def eval_list(
    filter_text: Optional[str] = typer.Argument(
        None,
        help="Optional filter text to search task names",
    ),
) -> None:
    """List available evaluation tasks.
    
    Examples:
        aios eval list
        aios eval list mmlu
        aios eval list code
    """
    # Check if lm_eval is installed
    if not HarnessWrapper.is_lm_eval_installed():
        console.print(
            "[red]Error:[/red] lm-evaluation-harness is not installed.\n"
            "Install with: [cyan]pip install lm-eval[/cyan]",
            style="bold"
        )
        raise typer.Exit(1)
    
    console.print("\n[bold cyan]Fetching available tasks...[/bold cyan]\n")
    
    try:
        tasks = HarnessWrapper.get_available_tasks()
        
        if not tasks:
            console.print("[yellow]No tasks found or failed to query lm_eval[/yellow]")
            raise typer.Exit(1)
        
        # Filter if requested
        if filter_text:
            tasks = [t for t in tasks if filter_text.lower() in t.lower()]
            console.print(f"[dim]Filtered by: {filter_text}[/dim]\n")
        
        # Display in columns
        console.print(f"[bold]Available tasks ({len(tasks)}):[/bold]\n")
        
        # Group by category (simple heuristic)
        language_tasks = [t for t in tasks if any(x in t.lower() for x in ["mmlu", "hellaswag", "winogrande", "truthfulqa", "boolq", "drop"])]
        coding_tasks = [t for t in tasks if any(x in t.lower() for x in ["human", "mbpp", "code"])]
        math_tasks = [t for t in tasks if any(x in t.lower() for x in ["gsm8k", "math", "minerva"])]
        science_tasks = [t for t in tasks if any(x in t.lower() for x in ["arc", "sciq", "gpqa"])]
        other_tasks = [t for t in tasks if t not in language_tasks + coding_tasks + math_tasks + science_tasks]
        
        if language_tasks:
            console.print("[bold cyan]Language:[/bold cyan]")
            for task in sorted(language_tasks):
                console.print(f"  • {task}")
            console.print()
        
        if coding_tasks:
            console.print("[bold green]Coding:[/bold green]")
            for task in sorted(coding_tasks):
                console.print(f"  • {task}")
            console.print()
        
        if math_tasks:
            console.print("[bold yellow]Math:[/bold yellow]")
            for task in sorted(math_tasks):
                console.print(f"  • {task}")
            console.print()
        
        if science_tasks:
            console.print("[bold magenta]Science:[/bold magenta]")
            for task in sorted(science_tasks):
                console.print(f"  • {task}")
            console.print()
        
        if other_tasks and not filter_text:
            console.print("[bold white]Other:[/bold white]")
            for task in sorted(other_tasks[:20]):  # Limit display
                console.print(f"  • {task}")
            if len(other_tasks) > 20:
                console.print(f"  ... and {len(other_tasks) - 20} more")
            console.print()
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)


@app.command("history")
def eval_history(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of results to show",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Filter by model name (partial match)",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (completed, failed, cancelled)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
) -> None:
    """Show evaluation history.
    
    Examples:
        aios eval history
        aios eval history --limit 50
        aios eval history --model gpt2
        aios eval history --status completed --verbose
    """
    try:
        history = EvaluationHistory()
        
        # Get evaluations
        evaluations = history.get_recent_evaluations(
            limit=limit,
            model_name=model,
            status=status,
        )
        
        if not evaluations:
            console.print("\n[yellow]No evaluations found[/yellow]\n")
            return
        
        # Display statistics
        stats = history.get_statistics()
        console.print(f"\n[bold cyan]Evaluation History[/bold cyan]")
        console.print(
            f"Total: {stats['total_evaluations']} | "
            f"Models: {stats['unique_models']} | "
            f"Avg Score: {stats['average_score']:.1%} | "
            f"Recent (7d): {stats['recent_count']}\n"
        )
        
        if verbose:
            # Detailed view
            for eval_data in evaluations:
                console.print(f"[bold]ID {eval_data['id']}:[/bold] {eval_data['model_name']}")
                console.print(f"  Score: {eval_data.get('overall_score', 0):.2%}")
                console.print(f"  Status: {eval_data['status']}")
                
                tasks = eval_data.get('tasks', [])
                if isinstance(tasks, list):
                    console.print(f"  Tasks: {', '.join(tasks)}")
                
                created_at = eval_data.get('created_at', 0)
                if created_at:
                    date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    console.print(f"  Date: {date_str}")
                
                duration = eval_data.get('duration', 0)
                if duration:
                    if duration < 60:
                        duration_str = f"{duration:.1f}s"
                    elif duration < 3600:
                        duration_str = f"{duration/60:.1f}m"
                    else:
                        duration_str = f"{duration/3600:.1f}h"
                    console.print(f"  Duration: {duration_str}")
                
                console.print()
        else:
            # Table view
            table = Table()
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Model", style="green")
            table.add_column("Score", style="yellow", justify="right")
            table.add_column("Status", style="magenta")
            table.add_column("Tasks", style="blue")
            table.add_column("Date", style="white")
            
            for eval_data in evaluations:
                eval_id = str(eval_data['id'])
                model_name = eval_data['model_name']
                
                score = eval_data.get('overall_score', 0)
                score_str = f"{score:.1%}" if score else "N/A"
                
                status_val = eval_data['status']
                
                tasks = eval_data.get('tasks', [])
                if isinstance(tasks, list):
                    tasks_str = ", ".join(tasks[:2])
                    if len(tasks) > 2:
                        tasks_str += f" (+{len(tasks) - 2})"
                else:
                    tasks_str = "N/A"
                
                created_at = eval_data.get('created_at', 0)
                date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
                
                table.add_row(eval_id, model_name, score_str, status_val, tasks_str, date_str)
            
            console.print(table)
        
        console.print()
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)


@app.command("compare")
def eval_compare(
    eval_ids: str = typer.Argument(
        ...,
        help="Comma-separated evaluation IDs to compare",
    ),
) -> None:
    """Compare multiple evaluations.
    
    Examples:
        aios eval compare 1,2,3
        aios eval compare 5,8
    """
    try:
        # Parse IDs
        ids = [int(x.strip()) for x in eval_ids.split(",") if x.strip()]
        
        if len(ids) < 2:
            console.print("[red]Error:[/red] Please provide at least 2 evaluation IDs", style="bold")
            raise typer.Exit(1)
        
        history = EvaluationHistory()
        comparison = history.compare_evaluations(ids)
        
        if not comparison:
            console.print("[yellow]No comparison data available[/yellow]")
            raise typer.Exit(1)
        
        # Display comparison
        console.print(f"\n[bold cyan]Comparing {len(ids)} evaluations[/bold cyan]\n")
        
        # Models header
        console.print("[bold]Models:[/bold]")
        for i, eval_info in enumerate(comparison["evaluations"], 1):
            console.print(
                f"  {i}. {eval_info['model_name']} "
                f"(Score: {eval_info['overall_score']:.2%})"
            )
        console.print()
        
        # Benchmark comparison table
        table = Table(title="Benchmark Comparison")
        table.add_column("Benchmark", style="cyan")
        for i in range(len(comparison["evaluations"])):
            table.add_column(f"Model {i+1}", style="green", justify="right")
        
        for benchmark, scores in sorted(comparison["benchmarks"].items()):
            row = [benchmark]
            for score_data in scores:
                if score_data:
                    row.append(f"{score_data['score']:.2%}")
                else:
                    row.append("N/A")
            table.add_row(*row)
        
        console.print(table)
        console.print()
    
    except ValueError:
        console.print("[red]Error:[/red] Invalid evaluation ID format", style="bold")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
