"""Adaptive LR profile loading for the HRM Training GUI.

This supports a dropdown in the GUI that can be extended by users.

Search order (later files override earlier by profile id):
1) Built-in defaults (hard-coded)
2) Repo defaults: <repo>/config/adaptive_lr_profiles.json
3) User overrides: <user_config_dir>/adaptive_lr_profiles.json

A profile can provide either:
- "config_path": a path to a JSON/TOML/YAML adaptive LR config file
- "overrides": a dict of AdaptiveLRConfig keys (will be written to a generated JSON)

Note: the scheduler only supports mode in {balanced, conservative, aggressive, auto}.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aios.gui import config_loader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveLRProfile:
    profile_id: str
    label: str
    description: str = ""
    config_path: str | None = None
    overrides: dict[str, Any] | None = None


def _repo_root_from_here() -> Path | None:
    try:
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "config" / "adaptive_lr_profiles.json").exists():
                return parent
    except Exception:
        return None
    return None


def _load_profiles_file(path: Path) -> list[dict[str, Any]]:
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        profiles = data.get("profiles")
        if not isinstance(profiles, list):
            return []
        return [p for p in profiles if isinstance(p, dict)]
    except Exception as e:
        logger.warning("Failed to load adaptive LR profiles from %s: %s", path, e)
        return []


def load_adaptive_lr_profiles() -> list[AdaptiveLRProfile]:
    built_in: list[AdaptiveLRProfile] = [
        AdaptiveLRProfile(profile_id="off", label="Off", description="Disable adaptive LR; manual LR enabled."),
        AdaptiveLRProfile(profile_id="auto", label="Auto", description="Adaptive LR enabled; auto-switches modes.", overrides={"mode": "auto"}),
        AdaptiveLRProfile(profile_id="balanced", label="Balanced", description="Adaptive LR enabled; balanced preset.", overrides={"mode": "balanced"}),
        AdaptiveLRProfile(profile_id="conservative", label="Conservative", description="Adaptive LR enabled; conservative preset.", overrides={"mode": "conservative"}),
        AdaptiveLRProfile(profile_id="aggressive", label="Aggressive", description="Adaptive LR enabled; aggressive preset.", overrides={"mode": "aggressive"}),
    ]

    merged: dict[str, AdaptiveLRProfile] = {p.profile_id: p for p in built_in}

    # Repo profiles
    try:
        root = _repo_root_from_here()
        if root is not None:
            repo_profiles_path = root / "config" / "adaptive_lr_profiles.json"
            for raw in _load_profiles_file(repo_profiles_path):
                pid = str(raw.get("id") or "").strip().lower()
                label = str(raw.get("label") or "").strip()
                if not pid or not label:
                    continue
                if pid == "off":
                    # Keep the GUI's Off semantics stable.
                    continue
                merged[pid] = AdaptiveLRProfile(
                    profile_id=pid,
                    label=label,
                    description=str(raw.get("description") or ""),
                    config_path=str(raw.get("config_path")).strip() if raw.get("config_path") else None,
                    overrides=dict(raw.get("overrides")) if isinstance(raw.get("overrides"), dict) else None,
                )
    except Exception:
        logger.debug("Repo adaptive LR profiles load failed", exc_info=True)

    # User profiles
    try:
        user_config_dir = config_loader.get_config_path().parent
        user_profiles_path = user_config_dir / "adaptive_lr_profiles.json"
        for raw in _load_profiles_file(user_profiles_path):
            pid = str(raw.get("id") or "").strip().lower()
            label = str(raw.get("label") or "").strip()
            if not pid or not label:
                continue
            if pid == "off":
                continue
            merged[pid] = AdaptiveLRProfile(
                profile_id=pid,
                label=label,
                description=str(raw.get("description") or ""),
                config_path=str(raw.get("config_path")).strip() if raw.get("config_path") else None,
                overrides=dict(raw.get("overrides")) if isinstance(raw.get("overrides"), dict) else None,
            )
    except Exception:
        logger.debug("User adaptive LR profiles load failed", exc_info=True)

    # Stable ordering: Off, Auto, defaults, then customs alpha by label.
    order = ["off", "auto", "balanced", "conservative", "aggressive"]
    result: list[AdaptiveLRProfile] = []
    for pid in order:
        if pid in merged:
            result.append(merged[pid])

    # Remaining custom profiles
    for pid, prof in sorted(merged.items(), key=lambda kv: (kv[1].label.lower(), kv[1].profile_id)):
        if pid in order:
            continue
        result.append(prof)

    return result


def resolve_profile_config_path(profile: AdaptiveLRProfile) -> str | None:
    """Return a config path suitable for --adaptive-lr-config.

    If profile.config_path is provided, returns that path (expanded).
    Otherwise, if overrides exist, writes a generated JSON file under the user config dir.
    Off returns None.
    """
    if profile.profile_id == "off":
        return None

    if profile.config_path:
        p = os.path.expandvars(os.path.expanduser(profile.config_path))
        return str(Path(p))

    overrides = profile.overrides or {}
    if not overrides:
        return None

    user_config_dir = config_loader.get_config_path().parent
    out_dir = user_config_dir / "generated" / "adaptive_lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaptive_lr_{profile.profile_id}.json"
    out_path.write_text(json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8")
    return str(out_path)
