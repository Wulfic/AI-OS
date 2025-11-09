"""
Checkpoint utilities for train_actv1.

Handles checkpoint recovery and path resolution.
"""

import os
from pathlib import Path
from typing import Optional


def recover_checkpoint_artifacts(candidate_dirs: list[Path], log_fn) -> Path | None:
    """Promote tmp checkpoints to their final filenames before training starts.
    
    Args:
        candidate_dirs: List of directories to search for checkpoints
        log_fn: Logging function for events
    
    Returns:
        Path to recovered checkpoint, or None if none found
    """
    recovered: Path | None = None
    seen: set[Path] = set()
    
    for directory in candidate_dirs:
        if directory is None:
            continue
        try:
            bundle_path = Path(directory)
        except Exception:
            continue
        if bundle_path in seen:
            continue
        seen.add(bundle_path)
        if not bundle_path.exists() or not bundle_path.is_dir():
            continue
        
        for base_name, fmt in (
            ("actv1_student.safetensors", "safetensors"),
            ("actv1_student.pt", "pt"),
        ):
            final_path = bundle_path / base_name
            tmp_path = bundle_path / f"{base_name}.tmp"
            
            if not tmp_path.exists():
                continue
            
            replace_needed = not final_path.exists()
            if not replace_needed:
                try:
                    replace_needed = (
                        tmp_path.stat().st_size != final_path.stat().st_size
                        or tmp_path.stat().st_mtime > final_path.stat().st_mtime
                    )
                except Exception:
                    replace_needed = True
            
            try:
                backup_path = bundle_path / f"{base_name}.prev"
                if replace_needed and final_path.exists():
                    try:
                        if backup_path.exists():
                            backup_path.unlink()
                    except Exception:
                        pass
                    os.replace(final_path, backup_path)
                os.replace(tmp_path, final_path)
                
                if log_fn:
                    try:
                        log_fn({
                            "checkpoint_recovered": str(final_path),
                            "source": str(tmp_path),
                            "format": fmt,
                        })
                    except Exception:
                        pass
                
                try:
                    brain_json = bundle_path / "brain.json"
                    if brain_json.exists():
                        import json as _json
                        with brain_json.open("r", encoding="utf-8") as _f:
                            data = _json.load(_f) or {}
                    else:
                        import json as _json
                        data = {}
                    
                    data["checkpoint_file"] = final_path.name
                    data["checkpoint_format"] = fmt
                    if fmt == "safetensors":
                        data.pop("student_pt", None)
                    
                    with brain_json.open("w", encoding="utf-8") as _f:
                        _json.dump(data, _f, indent=2)
                except Exception:
                    pass
                
                if fmt == "safetensors" or recovered is None:
                    recovered = final_path
                    
            except Exception as recovery_error:
                if log_fn:
                    try:
                        log_fn({
                            "checkpoint_recover": "FAILED",
                            "source": str(tmp_path),
                            "error": str(recovery_error),
                        })
                    except Exception:
                        pass
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
    
    return recovered


def resolve_student_init_path(
    student_init: Optional[str],
    recovered: Path | None,
    candidate_dirs: list[Path],
    log_fn,
) -> Optional[str]:
    """Pick a usable checkpoint path, preferring recovered safetensors files.
    
    Args:
        student_init: User-provided checkpoint path
        recovered: Recovered checkpoint path from artifacts
        candidate_dirs: List of directories to search
        log_fn: Logging function
    
    Returns:
        Resolved checkpoint path, or None
    """
    if student_init:
        init_path = Path(student_init)
        if init_path.exists() or init_path.is_dir():
            return str(init_path)
        
        alt_candidates: list[Path] = []
        if init_path.suffix.lower() == ".pt":
            alt_candidates.append(init_path.with_suffix(".safetensors"))
        elif init_path.suffix.lower() == ".safetensors":
            alt_candidates.append(init_path.with_suffix(".pt"))
        
        alt_candidates.extend((
            init_path.parent / "actv1_student.safetensors",
            init_path.parent / "actv1_student.pt"
        ))
        
        for candidate in alt_candidates:
            if candidate.exists():
                if log_fn:
                    try:
                        log_fn({
                            "student_init_redirected": str(candidate),
                            "requested_path": str(init_path),
                        })
                    except Exception:
                        pass
                return str(candidate)
        
        if recovered and recovered.exists():
            if log_fn:
                try:
                    log_fn({
                        "student_init_redirected": str(recovered),
                        "requested_path": str(init_path),
                    })
                except Exception:
                    pass
            return str(recovered)
        
        return student_init
    
    if recovered and recovered.exists():
        if log_fn:
            try:
                log_fn({"student_init_autodetected": str(recovered)})
            except Exception:
                pass
        return str(recovered)
    
    for directory in candidate_dirs:
        if directory is None:
            continue
        try:
            bundle_path = Path(directory)
        except Exception:
            continue
        if not bundle_path.exists() or not bundle_path.is_dir():
            continue
        
        for candidate_name in ("actv1_student.safetensors", "actv1_student.pt"):
            candidate_path = bundle_path / candidate_name
            if candidate_path.exists():
                if log_fn:
                    try:
                        log_fn({"student_init_autodetected": str(candidate_path)})
                    except Exception:
                        pass
                return str(candidate_path)
    
    return None
