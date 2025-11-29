"""Shared path helpers for the HRM training panel.

These helpers keep UI defaults aligned with the centralized
aios.system.paths module without creating circular imports.
"""

from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - fallback for early init contexts
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None

__all__ = [
    "resolve_artifact",
    "get_default_artifacts_root",
    "get_default_model_base",
    "get_default_bundle_dir",
    "get_default_brain_dir",
    "get_default_metrics_file",
    "get_default_student_checkpoint",
]


def _repo_root() -> Path:
    # src/aios/gui/components/hrm_training_panel -> repo root
    return Path(__file__).resolve().parents[5]


def resolve_artifact(relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    if system_paths is not None:
        return system_paths.resolve_artifact_path(rel)
    return (get_default_artifacts_root() / rel).resolve()


def get_default_artifacts_root() -> Path:
    if system_paths is not None:
        return system_paths.get_artifacts_root()
    return (_repo_root() / "artifacts").resolve()


def get_default_bundle_dir(family: str = "actv1") -> Path:
    if system_paths is not None:
        return system_paths.get_brain_family_dir(family)
    return (get_default_artifacts_root() / "brains" / family).resolve()


def get_default_brain_dir(brain_name: str = "default", family: str = "actv1") -> Path:
    return get_default_bundle_dir(family) / brain_name


def get_default_metrics_file(brain_name: str = "default", family: str = "actv1") -> Path:
    return get_default_brain_dir(brain_name, family) / "metrics.jsonl"


def get_default_student_checkpoint(brain_name: str = "default", family: str = "actv1") -> Path:
    return get_default_brain_dir(brain_name, family) / "actv1_student.safetensors"


def get_default_model_base() -> Path:
    return resolve_artifact("hf_implant/base_model")
