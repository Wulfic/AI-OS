"""Chart generation utilities for evaluation results."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np
    from aios.core.evaluation import EvaluationResult

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore[assignment]
    from matplotlib.figure import Figure  # type: ignore[assignment]
    import numpy as np  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore[assignment,misc]
    FigureCanvasTkAgg = None  # type: ignore[assignment,misc]
    np = None  # type: ignore[assignment]


def create_evaluation_charts(result: "EvaluationResult", container: Any) -> None:
    """Create matplotlib visualization charts for evaluation results.
    
    Args:
        result: EvaluationResult to visualize
        container: Tkinter container to embed charts in
    """
    if not MATPLOTLIB_AVAILABLE or Figure is None or np is None or FigureCanvasTkAgg is None:
        return
    
    # Create figure with subplots
    fig = Figure(figsize=(10, 8), dpi=100)
    
    # Extract data
    task_names = []
    task_scores = []
    
    for task_name, task_data in sorted(result.benchmark_scores.items()):
        scores_dict = task_data.get("scores", {})
        if scores_dict:
            # Use first score as primary metric
            primary_score = list(scores_dict.values())[0]
            task_names.append(task_name)
            task_scores.append(float(primary_score) * 100)  # Convert to percentage
    
    if not task_names:
        return
    
    # Chart 1: Bar chart (top half)
    ax1 = fig.add_subplot(2, 1, 1)
    bars = ax1.bar(range(len(task_names)), task_scores, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Benchmark', fontsize=10)
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.set_title('Benchmark Scores Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(task_names)))
    ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    
    # Chart 2: Radar chart (bottom half) - only if 3+ benchmarks
    if len(task_names) >= 3:
        ax2 = fig.add_subplot(2, 1, 2, projection='polar')
        
        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(task_names), endpoint=False).tolist()
        scores_radar = task_scores + [task_scores[0]]  # Close the circle
        angles_radar = angles + [angles[0]]
        
        # Plot
        ax2.plot(angles_radar, scores_radar, 'o-', linewidth=2, color='steelblue')
        ax2.fill(angles_radar, scores_radar, alpha=0.25, color='steelblue')
        ax2.set_xticks(angles)
        ax2.set_xticklabels(task_names, fontsize=8)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Score (%)', fontsize=10)
        ax2.set_title('Performance Radar Chart', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True)
    
    fig.tight_layout()
    
    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def is_matplotlib_available() -> bool:
    """Check if matplotlib is available for chart generation.
    
    Returns:
        True if matplotlib is installed and functional
    """
    return MATPLOTLIB_AVAILABLE
