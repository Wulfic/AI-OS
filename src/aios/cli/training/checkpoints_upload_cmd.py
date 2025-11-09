from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def checkpoints_upload(
    label: Optional[str] = typer.Option(None, "--label", help="Filter to latest checkpoint with this label (DB)"),
    ckpt_path: Optional[str] = typer.Option(None, "--path", help="Path to a checkpoint .npz to upload (overrides label)"),
    dest: str = typer.Option("artifacts/checkpoints", "--dest", help="Repo subdir to copy checkpoints into"),
    commit: bool = typer.Option(True, "--commit/--no-commit", help="Commit copied files with git"),
    push: bool = typer.Option(False, "--push/--no-push", help="Also push after commit"),
):
    from aios.core.watch import upload_checkpoint_to_repo, git_commit, latest_checkpoint_from_db, latest_checkpoint_from_fs

    p: Optional[Path] = None
    if ckpt_path:
        p = Path(ckpt_path)
        if not p.exists():
            print({"uploaded": False, "error": "path not found", "path": ckpt_path})
            return
    else:
        p = latest_checkpoint_from_db(label) or latest_checkpoint_from_fs()
    if not p:
        print({"uploaded": False, "error": "no checkpoint found"})
        return
    dst = upload_checkpoint_to_repo(p, dest_subdir=dest)
    sidecar = Path(str(dst) + ".pt")
    changed = [dst] + ([sidecar] if sidecar.exists() else [])
    committed = False
    if commit:
        committed = git_commit(changed, message=f"chore(ckpt): add {dst.name}", push=push)
    print({
        "uploaded": True,
        "dest": str(dst),
        "committed": committed,
        "pushed": bool(committed and push),
    })
