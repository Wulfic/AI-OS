from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .orchestrator import Orchestrator
from ..memory.store import get_db, init_db, list_artifacts


def default_ckpt_dir() -> Path:
    return Path.home() / ".local/share/aios/checkpoints"


def latest_checkpoint_from_db(label: Optional[str] = None) -> Optional[Path]:
    """Return latest training checkpoint path from SQLite artifacts, if present."""
    conn = get_db()
    try:
        init_db(conn)
        items = list_artifacts(conn, kind="training_checkpoint", limit=200)
        if label is not None:
            items = [it for it in items if str(it.get("label") or "") == label]
        item = items[0] if items else None
        p = (item or {}).get("data", {}).get("path") if item else None
        if isinstance(p, str) and p:
            pp = Path(p)
            return pp if pp.exists() else None
        return None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def latest_checkpoint_from_fs(ckpt_dir: Optional[Path] = None) -> Optional[Path]:
    """Fallback: pick the most recently modified .npz in checkpoint dir."""
    d = ckpt_dir or default_ckpt_dir()
    if not d.exists():
        return None
    cands = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def set_active_checkpoint(ckpt: Path) -> Path:
    """Copy the given checkpoint to an 'active.npz' in checkpoint dir (and .pt sibling if present)."""
    d = default_ckpt_dir()
    d.mkdir(parents=True, exist_ok=True)
    active_npz = d / "active.npz"
    shutil.copy2(str(ckpt), str(active_npz))
    # copy optional torch file
    pt_src = Path(str(ckpt) + ".pt")
    if pt_src.exists():
        shutil.copy2(str(pt_src), str(Path(str(active_npz) + ".pt")))
    return active_npz


def detect_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from start to find a .git directory."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def upload_checkpoint_to_repo(ckpt: Path, repo_root: Optional[Path] = None, dest_subdir: str = "artifacts/checkpoints") -> Path:
    root = repo_root or detect_repo_root() or Path.cwd()
    dest_dir = (root / dest_subdir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ckpt.name
    shutil.copy2(str(ckpt), str(dest))
    # copy torch sidecar if present
    pt_src = Path(str(ckpt) + ".pt")
    if pt_src.exists():
        shutil.copy2(str(pt_src), str(Path(str(dest) + ".pt")))
    return dest


def git_commit(paths: list[Path], message: str, repo_root: Optional[Path] = None, push: bool = False) -> bool:
    root = (repo_root or detect_repo_root() or Path.cwd()).resolve()
    try:
        subprocess.run(["git", "add", *[str(p.relative_to(root)) for p in paths]], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if push:
            subprocess.run(["git", "push"], cwd=root, check=True)
        return True
    except Exception:
        return False


def is_service_environment() -> bool:
    return platform.system().lower() == "linux" and shutil.which("systemctl") is not None


def restart_agent_service() -> bool:
    if not is_service_environment():
        return False
    try:
        # user service restart
        subprocess.run(["systemctl", "--user", "restart", "aios.service"], check=True)
        return True
    except Exception:
        return False


def health_check() -> bool:
    """Basic health: attempt to construct Orchestrator and fetch status."""
    try:
        orch = Orchestrator(config={})
        _ = orch.status()
        return True
    except Exception:
        return False


def restore_last_checkpoint(prefer_label: Optional[str] = None) -> dict:
    """Try to restore from last checkpoint: set active.npz, optionally restart service.

    Returns summary with keys: found(bool), ckpt_path(str|None), active_path(str|None), service_restarted(bool)
    """
    ck = latest_checkpoint_from_db(prefer_label) or latest_checkpoint_from_fs()
    if not ck:
        return {"found": False, "ckpt_path": None, "active_path": None, "service_restarted": False}
    active = set_active_checkpoint(ck)
    restarted = restart_agent_service()
    return {"found": True, "ckpt_path": str(ck), "active_path": str(active), "service_restarted": restarted}
