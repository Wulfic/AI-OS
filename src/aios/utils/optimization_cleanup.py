"""Cleanup utility for optimization artifacts to prevent folder bloat.

This module manages the artifacts/optimization folder by:
1. Identifying unique optimization runs by session ID
2. Keeping only the most recent N runs
3. Removing all files associated with older runs
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


def extract_session_id(filename: str) -> str | None:
    """Extract session ID from optimization artifact filename.
    
    Patterns:
    - gpu_metrics_{session_id}.jsonl
    - gpu_metrics_gen_{session_id}.jsonl
    - gpu_metrics_train_{session_id}.jsonl
    - progressive_results_{session_id}.json
    - results_{session_id}.json
    - gen_{session_id}.jsonl
    - stop_{session_id}.flag
    - train_{session_id}.jsonl
    
    Args:
        filename: The filename to extract session ID from
        
    Returns:
        Session ID string or None if no match
    """
    # Patterns for different file types
    patterns = [
        r'^gpu_metrics_(?:gen_|train_)?([a-f0-9]{8})\.jsonl$',
        r'^progressive_results_([a-f0-9]{8})\.json$',
        r'^results_([a-f0-9]{8})\.json$',
        r'^gen_([a-f0-9]{8})\.jsonl$',
        r'^stop_([a-f0-9]{8})\.flag$',
        r'^train_([a-f0-9]{8})\.jsonl$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
    
    return None


def group_files_by_session(optimization_dir: Path) -> Dict[str, List[Path]]:
    """Group all optimization files by their session ID.
    
    Args:
        optimization_dir: Path to artifacts/optimization directory
        
    Returns:
        Dictionary mapping session_id -> list of file paths
    """
    session_files = defaultdict(list)
    
    if not optimization_dir.exists():
        return dict(session_files)
    
    for file_path in optimization_dir.iterdir():
        if not file_path.is_file():
            continue
        
        session_id = extract_session_id(file_path.name)
        if session_id:
            session_files[session_id].append(file_path)
    
    return dict(session_files)


def get_session_timestamp(files: List[Path]) -> float:
    """Get the most recent modification time for a session's files.
    
    Args:
        files: List of file paths for a session
        
    Returns:
        Most recent modification timestamp
    """
    if not files:
        return 0.0
    
    return max(f.stat().st_mtime for f in files)


def cleanup_old_optimization_runs(
    optimization_dir: Path | str,
    keep_last_n: int = 3,
    dry_run: bool = False
) -> Dict[str, any]:
    """Clean up old optimization runs, keeping only the most recent N runs.
    
    Args:
        optimization_dir: Path to artifacts/optimization directory
        keep_last_n: Number of most recent runs to keep (default: 3)
        dry_run: If True, don't delete files, just report what would be deleted
        
    Returns:
        Dictionary with cleanup statistics
    """
    optimization_dir = Path(optimization_dir)
    
    if not optimization_dir.exists():
        logger.info(f"Optimization directory does not exist: {optimization_dir}")
        return {
            "success": True,
            "sessions_found": 0,
            "sessions_kept": 0,
            "sessions_deleted": 0,
            "files_deleted": 0,
            "bytes_freed": 0
        }
    
    # Group files by session
    session_files = group_files_by_session(optimization_dir)
    
    if not session_files:
        logger.info("No optimization sessions found")
        return {
            "success": True,
            "sessions_found": 0,
            "sessions_kept": 0,
            "sessions_deleted": 0,
            "files_deleted": 0,
            "bytes_freed": 0
        }
    
    # Sort sessions by most recent timestamp
    sessions_by_time = sorted(
        session_files.items(),
        key=lambda x: get_session_timestamp(x[1]),
        reverse=True  # Most recent first
    )
    
    # Keep the N most recent sessions
    sessions_to_keep = sessions_by_time[:keep_last_n]
    sessions_to_delete = sessions_by_time[keep_last_n:]
    
    # Statistics
    stats = {
        "success": True,
        "sessions_found": len(session_files),
        "sessions_kept": len(sessions_to_keep),
        "sessions_deleted": 0,
        "files_deleted": 0,
        "bytes_freed": 0,
        "dry_run": dry_run
    }
    
    # Delete old sessions
    for session_id, files in sessions_to_delete:
        logger.info(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} session: {session_id} ({len(files)} files)")
        
        for file_path in files:
            try:
                file_size = file_path.stat().st_size
                
                if not dry_run:
                    file_path.unlink()
                    logger.debug(f"  Deleted: {file_path.name}")
                else:
                    logger.debug(f"  Would delete: {file_path.name}")
                
                stats["files_deleted"] += 1
                stats["bytes_freed"] += file_size
                
            except Exception as e:
                logger.error(f"  Failed to delete {file_path.name}: {e}")
        
        stats["sessions_deleted"] += 1
    
    # Log summary
    if stats["sessions_deleted"] > 0:
        mb_freed = stats["bytes_freed"] / (1024 * 1024)
        logger.info(
            f"{'[DRY RUN] Would clean' if dry_run else 'Cleaned'} {stats['sessions_deleted']} old session(s), "
            f"{'would free' if dry_run else 'freed'} {mb_freed:.2f} MB "
            f"(kept {stats['sessions_kept']} most recent)"
        )
    else:
        logger.info(f"No cleanup needed - only {stats['sessions_found']} session(s) found (keeping {keep_last_n})")
    
    return stats


def cleanup_on_startup(optimization_dir: Path | str = "artifacts/optimization", keep_last_n: int = 3):
    """Cleanup function to call at the start of optimization.
    
    This is a convenience wrapper for integration into the optimizer.
    
    Args:
        optimization_dir: Path to artifacts/optimization directory
        keep_last_n: Number of most recent runs to keep (default: 3)
    """
    try:
        logger.info(f"Running optimization folder cleanup (keeping last {keep_last_n} runs)...")
        stats = cleanup_old_optimization_runs(optimization_dir, keep_last_n, dry_run=False)
        
        if stats["sessions_deleted"] > 0:
            mb_freed = stats["bytes_freed"] / (1024 * 1024)
            logger.info(f"✅ Cleaned up {stats['sessions_deleted']} old run(s), freed {mb_freed:.2f} MB")
        else:
            logger.debug("No cleanup needed")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup optimization folder: {e}")
        # Don't fail optimization if cleanup fails
