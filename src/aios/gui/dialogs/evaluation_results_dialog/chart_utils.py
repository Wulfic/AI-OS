"""Chart generation utilities for evaluation results."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

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
        logger.debug("Matplotlib not available, skipping chart creation")
        return

    logger.debug("Creating evaluation result charts")

    fig = Figure(figsize=(10, 8), dpi=100)

    # Extract benchmark data
    score_pairs: list[tuple[str, float]] = []
    for task_name, task_data in sorted(result.benchmark_scores.items()):
        scores_dict = task_data.get("scores", {})
        if not scores_dict:
            continue
        primary_score = list(scores_dict.values())[0]
        score_pairs.append((task_name, float(primary_score) * 100))

    if not score_pairs:
        logger.debug("No task data available for charting")
        return

    logger.debug("Creating charts for %d tasks", len(score_pairs))

    score_pairs.sort(key=lambda item: item[1], reverse=True)
    total_tasks = len(score_pairs)

    # Top benchmarks bar chart -------------------------------------------------
    top_limit = 12
    top_slice = score_pairs[:top_limit]
    ax1 = fig.add_subplot(2, 1, 1)

    display_pairs = list(reversed(top_slice))
    labels = [name for name, _ in display_pairs]
    values = [score for _, score in display_pairs]
    y_positions = np.arange(len(display_pairs))

    axis_max = max(100.0, max(values) + 5.0)
    bars = ax1.barh(y_positions, values, color='steelblue', alpha=0.85)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, axis_max)
    ax1.set_xlabel('Score (%)', fontsize=10)
    ax1.set_title(
        f"Top {len(display_pairs)} Benchmarks by Score", fontsize=12, fontweight='bold'
    )
    ax1.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax1.text(
            min(width + 1.0, axis_max - 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va='center',
            ha='left',
            fontsize=9,
        )

    if total_tasks > len(display_pairs):
        ax1.text(
            0.98,
            0.02,
            f"Showing top {len(display_pairs)} of {total_tasks} benchmarks",
            transform=ax1.transAxes,
            ha='right',
            va='bottom',
            fontsize=8,
            color='dimgray',
        )

    # Score distribution histogram --------------------------------------------
    ax2 = fig.add_subplot(2, 1, 2)
    scores_array = np.array([score for _, score in score_pairs])
    bins = min(12, max(5, int(len(scores_array) / 2)))
    ax2.hist(scores_array, bins=bins, color='steelblue', alpha=0.75, edgecolor='white')
    ax2.axvline(scores_array.mean(), color='darkorange', linestyle='--', linewidth=1.5, label='Mean')
    ax2.axvline(np.median(scores_array), color='seagreen', linestyle='-.', linewidth=1.5, label='Median')
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Score (%)', fontsize=10)
    ax2.set_ylabel('Benchmarks', fontsize=10)
    ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper left', fontsize=9)

    fig.tight_layout(pad=2.0)
    
    # Embed in tkinter
    logger.debug("Embedding charts in tkinter container")
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def is_matplotlib_available() -> bool:
    """Check if matplotlib is available for chart generation.
    
    Returns:
        True if matplotlib is installed and functional
    """
    return MATPLOTLIB_AVAILABLE
