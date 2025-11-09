from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


BudgetMap = Dict[str, float]


@dataclass
class SafetyBudget:
    """Minimal safety budget tracker for constrained operations (CMDP-style).

    Tracks cumulative costs per domain (e.g., file_writes, service_changes).
    A budget value of float('inf') disables that budget.
    """

    limits: BudgetMap = field(default_factory=dict)
    usage: BudgetMap = field(default_factory=dict)

    def allow(self, domain: str, cost: float = 1.0) -> bool:
        limit = float(self.limits.get(domain, float("inf")))
        used = float(self.usage.get(domain, 0.0))
        if limit == float("inf"):
            return True
        return (used + cost) <= limit

    def record(self, domain: str, cost: float = 1.0) -> bool:
        """Record cost if allowed. Returns True if accepted, False if denied."""
        if self.allow(domain, cost):
            self.usage[domain] = float(self.usage.get(domain, 0.0)) + float(cost)
            return True
        return False

    def remaining(self, domain: str) -> float:
        limit = float(self.limits.get(domain, float("inf")))
        if limit == float("inf"):
            return float("inf")
        return max(0.0, limit - float(self.usage.get(domain, 0.0)))

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k in self.limits.keys():
            out[k] = {
                "limit": float(self.limits[k]),
                "used": float(self.usage.get(k, 0.0)),
                "remaining": float(self.remaining(k)),
            }
        # include any dynamic usage keys not present in limits
        for k in self.usage.keys():
            if k not in out:
                out[k] = {
                    "limit": float("inf"),
                    "used": float(self.usage[k]),
                    "remaining": float("inf"),
                }
        return out


def defaults_for_risk_tier(tier: str) -> BudgetMap:
    """Return default per-domain budgets for a risk tier.

    Domains:
    - file_writes: number of write operations
    - service_changes: number of start/stop/restart actions
    - pkg_ops: number of apt install/remove/hold operations
    - privileged_calls: number of root-helper calls (any method)
    """
    t = (tier or "conservative").lower()
    if t == "aggressive":
        inf = float("inf")
        return {
            "file_writes": inf,
            "service_changes": inf,
            "pkg_ops": inf,
            "privileged_calls": inf,
        }
    if t == "balanced":
        return {
            "file_writes": 50.0,
            "service_changes": 20.0,
            "pkg_ops": 10.0,
            "privileged_calls": 100.0,
        }
    # conservative (default)
    return {
        "file_writes": 10.0,
        "service_changes": 5.0,
        "pkg_ops": 2.0,
        "privileged_calls": 20.0,
    }
