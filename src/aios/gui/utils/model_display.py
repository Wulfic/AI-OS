"""Utilities for presenting model identifiers in the GUI."""

from __future__ import annotations

from pathlib import Path


def get_model_display_name(model_identifier: str | None) -> str:
    """Return a user-friendly display name for a model identifier.

    Converts filesystem paths to just their final component while leaving
    regular model identifiers unchanged.
    """
    if not model_identifier:
        return "Unknown"

    raw_value = model_identifier.strip().strip('"')
    if not raw_value:
        return "Unknown"

    raw_value = _strip_named_path_prefix(raw_value)

    if _looks_like_path(raw_value):
        trimmed = raw_value.rstrip("\\/") or raw_value
        try:
            candidate = Path(trimmed)
            name = candidate.name
            if name:
                return name
        except Exception:
            pass
        parts = trimmed.replace("\\", "/").split("/")
        return parts[-1] if parts and parts[-1] else trimmed

    return raw_value


def _looks_like_path(value: str) -> bool:
    """Heuristically determine whether *value* represents a filesystem path."""
    if not value:
        return False

    if value.startswith(("\\\\", "./", "../", "/", "~")):
        return True

    if len(value) >= 2 and value[1] == ":":
        return True

    if "\\" in value:
        return True

    return False


def _strip_named_path_prefix(value: str) -> str:
    """Handle key=value identifiers where the value is a filesystem path."""

    lowered = value.lower()
    for prefix in ("brain_path=", "model_path=", "path="):
        if lowered.startswith(prefix):
            return value.split("=", 1)[1].strip().strip('"')

    if "=" in value:
        _, rhs = value.split("=", 1)
        candidate = rhs.strip().strip('"')
        if _looks_like_path(candidate):
            return candidate

    return value
