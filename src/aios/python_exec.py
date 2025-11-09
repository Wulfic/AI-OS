from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _resolve_venv_python(venv_dir: Path) -> Optional[Path]:
    try:
        if os.name == "nt":
            candidate = venv_dir / "Scripts" / "python.exe"
        else:
            candidate = venv_dir / "bin" / "python"
    except Exception:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


@lru_cache(maxsize=1)
def get_preferred_python_executable() -> str:
    """Return the Python interpreter we prefer for launching CLI subprocesses.

    Resolution order:
    1. Explicit AIOS_PREFERRED_PYTHON environment variable (if valid)
    2. Active virtual environment specified by VIRTUAL_ENV
    3. Local project virtual environment directories (.venv, venv, .env)
    4. The interpreter running the current process (sys.executable)
    """
    # 1. Explicit override
    override = os.environ.get("AIOS_PREFERRED_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    # 2. Active virtual environment
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        candidate = _resolve_venv_python(Path(venv_env))
        if candidate is not None:
            return str(candidate)

    # 3. Project-local virtual environments
    try:
        project_root = Path(__file__).resolve().parents[2]
    except Exception:
        project_root = None
    if project_root is not None:
        for folder in (".venv", "venv", ".env"):
            candidate = _resolve_venv_python(project_root / folder)
            if candidate is not None:
                return str(candidate)

    # 4. Fallback to current interpreter
    return sys.executable
