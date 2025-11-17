"""Helper utilities for brains panel.

Pure functions with no GUI dependencies for project management, parsing, and filtering.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Tuple, Dict

# Import centralized parameter calculation
from aios.cli.hrm_hf.model_building import calculate_actv1_params


# Cache training step calculations keyed by metrics file path.
_TRAINING_STEPS_CACHE: Dict[str, Tuple[float, int]] = {}


def _extract_latest_step_from_lines(lines: list[str]) -> int:
    """Return the most recent ``step`` field from JSON lines.

    Iterates from the end of ``lines`` so we stop at the first parseable value.
    """
    for line in reversed(lines):
        payload = line.strip()
        if not payload:
            continue
        try:
            entry = json.loads(payload)
        except Exception:
            continue
        step = entry.get("step")
        if isinstance(step, (int, float)):
            return int(step)
    return 0


def _read_training_steps_from_metrics(metrics_path: str) -> int:
    """Best-effort extraction of the latest training step from metrics.jsonl.

    Reads only the tail of large files to avoid multi-second blocking on big logs.
    Falls back to scanning the whole file if the tail chunk lacks a ``step`` field.
    """
    try:
        size = os.path.getsize(metrics_path)
    except Exception:
        return 0

    # Start with a reasonable tail chunk; enlarge before falling back to full scan.
    chunk_plan = [256_000, 1_048_576, 4_194_304]

    for chunk_size in chunk_plan:
        try:
            if size <= chunk_size:
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    return _extract_latest_step_from_lines(mf.readlines())

            with open(metrics_path, "rb") as mf:
                offset = max(size - chunk_size, 0)
                mf.seek(offset)
                data = mf.read().decode("utf-8", errors="ignore")

            lines = data.splitlines()
            step = _extract_latest_step_from_lines(lines)
            if step > 0:
                return step
        except Exception:
            continue

    # Unable to find a step field in the recent portion of the file; avoid
    # scanning the full log to keep the UI responsive.
    return 0


def find_project_root() -> str:
    """Find project root by searching for pyproject.toml.
    
    Searches up to 8 parent directories from current working directory.
    
    Returns:
        Absolute path to project root, or current directory if not found.
    """
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


def parse_cli_dict(text: str) -> dict[str, Any]:
    """Parse CLI output that may include headers/noise, returning the last JSON/dict payload.
    
    Strategy:
    1. Attempt direct JSON parse
    2. Fallback: find the FIRST '{' after the header and parse from there
    3. Fallback: ast.literal_eval on balanced brace section
    4. Else, return {}
    
    Args:
        text: Raw CLI output text
        
    Returns:
        Parsed dictionary, or empty dict if parsing fails
    """
    # 1) Try direct JSON
    try:
        result = json.loads(text)
        return result
    except Exception:
        pass
    
    # 2) Find FIRST '{' - this is the start of the JSON payload after the CLI header
    try:
        first_brace = text.find('{')
        if first_brace != -1:
            candidate = text[first_brace:].strip()
            try:
                result = json.loads(candidate)
                return result
            except Exception:
                pass
    except Exception:
        pass
    
    # 3) Try ast on first brace block
    try:
        first_brace = text.find('{')
        if first_brace != -1:
            tail = text[first_brace:]
            import ast
            result = ast.literal_eval(tail)
            return result
    except Exception:
        pass
    
    return {}


def is_temporary_brain(name: str) -> bool:
    """Check if brain name matches temporary brain patterns.
    
    Temporary brains include:
    - Brains starting with '_' (like _ddp used for DDP training)
    - Brains matching 'brain-<modality>-<hash>' pattern (temporary chat router brains)
    - Internal system directories like 'parallel_checkpoints'
    
    Args:
        name: Brain name to check
        
    Returns:
        True if brain is temporary and should be filtered out
    """
    if name.startswith('_'):
        return True
    
    # Check for router-generated temporary brains: brain-text-de5aae40, brain-image-abc123, etc.
    if re.match(r'^brain-[a-z]+-[0-9a-f]{8}$', name):
        return True
    
    # Filter out internal system directories
    if name in ('parallel_checkpoints', 'checkpoints', 'temp', 'tmp', '.git', 'ddp_logs'):
        return True
    
    return False


def scan_actv1_bundles(actv1_base: str) -> dict[str, dict[str, Any]]:
    """Scan for ACTV1 bundle directories and estimate their sizes.
    
    ACTV1 bundles are directories containing actv1_student.safetensors or other model files.
    
    Args:
        actv1_base: Path to artifacts/brains/actv1 directory
        
    Returns:
        Dictionary mapping brain names to metadata dicts with at least 'size_bytes' key
    """
    bundles = {}
    
    if not os.path.isdir(actv1_base):
        return bundles
    
    try:
        for entry in sorted(os.listdir(actv1_base)):
            p = os.path.join(actv1_base, entry)
            if not os.path.isdir(p):
                continue
            
            # Skip internal system directories
            if is_temporary_brain(entry):
                continue
            
            # Estimate size
            size_b = 0
            try:
                # Try safetensors file first
                pt = os.path.join(p, "actv1_student.safetensors")
                if os.path.exists(pt):
                    size_b = int(os.path.getsize(pt))
                else:
                    # Fallback: sum entire directory
                    total = 0
                    for r, _d, files in os.walk(p):
                        for f in files:
                            try:
                                total += int(os.path.getsize(os.path.join(r, f)))
                            except Exception:
                                continue
                    size_b = int(total)
            except Exception:
                size_b = 0
            
            bundles[entry] = {"size_bytes": size_b}
    except Exception:
        pass
    
    return bundles


def load_training_steps(brain_path: str, brain_metadata: dict[str, Any]) -> int:
    """Load training steps from brain.json or metrics.jsonl.
    
    Tries multiple strategies:
    1. Read from brain_metadata directly
    2. Read from brain.json at brain_path
    3. Parse metrics.jsonl to find maximum step
    
    Args:
        brain_path: Absolute path to brain directory
        brain_metadata: Brain metadata dict (may contain training_steps)
        
    Returns:
        Number of training steps, or 0 if not found
    """
    # Try metadata first
    training_steps = int(brain_metadata.get("training_steps", 0) or 0)
    if training_steps > 0:
        return training_steps

    # Track candidate metrics files (primary first) to avoid redundant lookups.
    metrics_candidates: list[str] = []
    log_file = (brain_metadata.get("log_file") or "metrics.jsonl") if brain_metadata else "metrics.jsonl"
    if log_file:
        metrics_candidates.append(os.path.join(brain_path, log_file))

    # Try loading from brain.json for additional metadata/log path hints.
    brain_json_path = os.path.join(brain_path, "brain.json")
    if os.path.exists(brain_json_path):
        try:
            with open(brain_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            training_steps = int(meta.get("training_steps", 0) or 0)
            if training_steps > 0:
                return training_steps

            alt_log = meta.get("log_file")
            if isinstance(alt_log, str) and alt_log:
                candidate_path = os.path.join(brain_path, alt_log)
                if candidate_path not in metrics_candidates:
                    metrics_candidates.insert(0, candidate_path)
        except Exception:
            pass

    # Parse metrics files (tail-first) with caching to avoid repeated large reads.
    for metrics_path in metrics_candidates:
        if not metrics_path or not os.path.exists(metrics_path):
            continue
        try:
            mtime = os.path.getmtime(metrics_path)
            cached = _TRAINING_STEPS_CACHE.get(metrics_path)
            if cached and cached[0] == mtime:
                step = cached[1]
            else:
                step = _read_training_steps_from_metrics(metrics_path)
                _TRAINING_STEPS_CACHE[metrics_path] = (mtime, step)
            if step > 0:
                return step
        except Exception:
            continue

    return 0


def get_selected_tree_value(tree: Any, column_index: int) -> Optional[str]:
    """Extract value from selected tree item at given column index.
    
    Args:
        tree: Tkinter Treeview widget
        column_index: Column index to extract (0-based)
        
    Returns:
        String value at column, or None if no selection or error
    """
    try:
        sel = tree.selection()
        if not sel:
            return None
        values = tree.item(sel[0]).get("values", [])
        if column_index < len(values):
            return values[column_index]
        return None
    except Exception:
        return None


def calculate_params_from_metadata(metadata: dict[str, Any]) -> int:
    """Calculate approximate parameter count from brain architecture metadata.
    
    For ACTV1 MOE models, calculates based on architecture parameters.
    
    Args:
        metadata: Brain metadata dict from brain.json
        
    Returns:
        Estimated parameter count, or 0 if calculation fails
    """
    try:
        # Check if it's an ACTV1 model with architecture info
        if metadata.get("type") != "actv1":
            return 0
        
        # Check if brain.json has pre-calculated total_params (preferred)
        total_params = metadata.get("total_params")
        if total_params and isinstance(total_params, int):
            return total_params
        
        # Fallback: Calculate from architecture (for legacy brains)
        # Get architecture params (check both root level and 'arch' sub-dict)
        arch = metadata.get("arch", {})
        hidden_size = metadata.get("hidden_size") or arch.get("hidden_size", 0)
        h_layers = metadata.get("h_layers") or arch.get("H_layers", 0) or arch.get("h_layers", 0)
        l_layers = metadata.get("l_layers") or arch.get("L_layers", 0) or arch.get("l_layers", 0)
        expansion = metadata.get("expansion") or arch.get("expansion", 2.0)
        vocab_size = metadata.get("vocab_size") or arch.get("vocab_size", 0)
        use_moe = metadata.get("use_moe", False)
        num_experts = metadata.get("num_experts", 1)
        
        if hidden_size == 0 or (h_layers == 0 and l_layers == 0) or vocab_size == 0:
            return 0
        
        # Use centralized parameter calculation (single source of truth)
        try:
            return calculate_actv1_params(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                h_layers=h_layers,
                l_layers=l_layers,
                expansion=expansion,
                use_moe=use_moe,
                num_experts=num_experts
            )
        except Exception:
            return 0
        
    except Exception:
        return 0
