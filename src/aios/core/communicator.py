from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from aios.core.directives import Directive


@dataclass
class Summary:
    headline: str
    details: str
    directives: List[str]


def format_status_summary(
    status: Dict[str, Any], directives: List[Directive]
) -> Summary:
    mode = status.get("autonomy", "autonomous_on")
    headline = f"Agent is {mode.replace('_', ' ')}."

    if directives:
        dlist = ", ".join(f'"{d.text}"' for d in directives[:3])
        more = "" if len(directives) <= 3 else f" (+{len(directives)-3} more)"
        details = f"Active directives: {dlist}{more}."
    else:
        details = 'No active directives. Add one with: aios goals-add "<your focus>"'

    return Summary(
        headline=headline,
        details=details,
        directives=[d.text for d in directives],
    )
