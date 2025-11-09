from __future__ import annotations

from typing import Optional, Callable


def get_training_lines(
    *,
    dataset_file: Optional[str],
    ascii_only: bool,
    read_text_lines_sample_any: Callable[..., list[str]],
    cycle: int = 0,
    dataset_chunk_size: int = 4000,
) -> list[str]:
    """Load training lines from dataset file.

    Simplified to only support dataset file loading (teacher-dataset feature removed).
    
    Args:
        dataset_file: Path to dataset file/directory
        ascii_only: Filter to ASCII-only lines
        read_text_lines_sample_any: Function to read lines
        cycle: Current training cycle (for chunk rotation)
        dataset_chunk_size: Number of samples to load per cycle
    """
    from rich import print

    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    if not dataset_file:
        print({"started": False, "error": "no dataset provided; use --dataset-file to specify a training dataset"})
        import typer
        raise typer.Exit(code=1)
        
    lines = read_text_lines_sample_any(dataset_file, max_lines=dataset_chunk_size, cycle=cycle)
    lines = [ln for ln in lines if ln and str(ln).strip()]
    
    if ascii_only:
        lines = [ln for ln in lines if _is_ascii(str(ln))]
    
    # Better error message if no lines were loaded from dataset
    if not lines:
        print({
            "started": False, 
            "error": f"no valid lines loaded from dataset: {dataset_file}",
            "hint": "Dataset file/directory may be empty or invalid. Provide a non-empty dataset file with text data."
        })
        import typer
        raise typer.Exit(code=1)
        
    return lines
