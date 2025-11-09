from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class GuardRules:
    allow_paths: List[str] = field(
        default_factory=list
    )  # directories/files allowed for writes
    deny_paths: List[str] = field(
        default_factory=list
    )  # explicit deny list (takes precedence)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def is_path_allowed(target: Path, rules: GuardRules) -> bool:
    t = target.resolve()
    # deny takes precedence
    for d in rules.deny_paths:
        dp = Path(d).expanduser()
        if _is_within(t, dp) or t == dp:
            return False
    # if allow list empty, deny-by-default
    if not rules.allow_paths:
        return False
    for a in rules.allow_paths:
        ap = Path(a).expanduser()
        if _is_within(t, ap) or t == ap:
            return True
    return False


def rules_from_config(cfg: dict) -> GuardRules:
    g = (cfg or {}).get("guards", {}) or {}
    allow = list(g.get("allow_paths", []) or [])
    deny = list(g.get("deny_paths", []) or [])
    return GuardRules(
        allow_paths=[str(Path(p).expanduser()) for p in allow],
        deny_paths=[str(Path(p).expanduser()) for p in deny],
    )


def enforce_write_allowed(path: Path, cfg: dict) -> None:
    # Only enforce when write_guarded is true
    if (cfg or {}).get("autonomy", {}).get("write_guarded", True):
        rules = rules_from_config(cfg)
        if not is_path_allowed(path, rules):
            raise PermissionError(f"Write not allowed by guard rules: {path}")
