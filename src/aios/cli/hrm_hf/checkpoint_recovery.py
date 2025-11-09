"""Checkpoint recovery and management utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable


def _validate_safetensors(path: Path) -> bool:
    """Validate that a safetensors file is not corrupted."""
    try:
        # Try to open the file and read the header
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            # Just checking if we can read the header is enough
            return True
    except Exception:
        return False


def recover_checkpoint_artifacts(candidate_dirs: list[Path], log_fn: Callable) -> Path | None:
    """Promote tmp checkpoints to their final filenames before training starts."""
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
                    final_stat = final_path.stat()
                    tmp_stat = tmp_path.stat()
                    replace_needed = tmp_stat.st_mtime > final_stat.st_mtime
                except Exception:
                    replace_needed = True
            try:
                if replace_needed:
                    # Validate checkpoint before promoting (for safetensors)
                    if fmt == "safetensors":
                        is_valid = _validate_safetensors(tmp_path)
                        if not is_valid:
                            if log_fn:
                                log_fn({
                                    "checkpoint_recovery": "corrupted_tmp",
                                    "tmp": str(tmp_path),
                                    "action": "removing",
                                })
                            tmp_path.unlink()
                            # Also remove corrupted final file if it exists
                            if final_path.exists():
                                final_path.unlink()
                                if log_fn:
                                    log_fn({
                                        "checkpoint_recovery": "removed_corrupted_final",
                                        "path": str(final_path),
                                    })
                            continue
                    
                    if log_fn:
                        log_fn({
                            "checkpoint_recovery": "promoting_tmp",
                            "tmp": str(tmp_path),
                            "final": str(final_path),
                        })
                    if final_path.exists():
                        final_path.unlink()
                    tmp_path.rename(final_path)
                    if log_fn:
                        log_fn({
                            "checkpoint_recovery": "success",
                            "path": str(final_path),
                            "size_mb": round(final_path.stat().st_size / (1024 ** 2), 2),
                        })
                    recovered = final_path
                else:
                    tmp_path.unlink()
                    if log_fn:
                        log_fn({
                            "checkpoint_recovery": "removed_stale_tmp",
                            "tmp": str(tmp_path),
                        })
            except Exception as recovery_error:
                if log_fn:
                    log_fn({
                        "checkpoint_recovery": "failed",
                        "error": str(recovery_error),
                        "tmp": str(tmp_path),
                    })
    return recovered


def resolve_student_init_path(
    student_init: Optional[str],
    recovered: Path | None,
    candidate_dirs: list[Path],
    log_fn: Callable,
) -> Optional[str]:
    """Pick a usable checkpoint path, preferring recovered safetensors files."""
    if student_init:
        init_path = Path(student_init)
        if init_path.exists() or init_path.is_dir():
            return str(init_path)
        alt_candidates: list[Path] = []
        if init_path.suffix.lower() == ".pt":
            alt_candidates.append(init_path.with_suffix(".safetensors"))
        elif init_path.suffix.lower() == ".safetensors":
            alt_candidates.append(init_path.with_suffix(".pt"))
        alt_candidates.extend(
            (init_path.parent / "actv1_student.safetensors", init_path.parent / "actv1_student.pt")
        )
        for candidate in alt_candidates:
            if candidate.exists():
                if log_fn:
                    log_fn({
                        "student_init": "using_alternative",
                        "requested": str(student_init),
                        "using": str(candidate),
                    })
                return str(candidate)
        if recovered and recovered.exists():
            if log_fn:
                log_fn({
                    "student_init": "using_recovered",
                    "requested": str(student_init),
                    "recovered": str(recovered),
                })
            return str(recovered)
        return student_init

    if recovered and recovered.exists():
        if log_fn:
            try:
                log_fn({"student_init": "using_recovered", "path": str(recovered)})
            except Exception:
                pass
        return str(recovered)

    for directory in candidate_dirs:
        if directory is None:
            continue
        bundle_path = Path(directory)
        for candidate in (bundle_path / "actv1_student.safetensors", bundle_path / "actv1_student.pt"):
            if candidate.exists():
                if log_fn:
                    log_fn({"student_init": "found_in_bundle", "path": str(candidate)})
                return str(candidate)

    return student_init
