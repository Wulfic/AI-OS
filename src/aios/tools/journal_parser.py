from __future__ import annotations

import re
from typing import Dict

_SEVERITY_ORDER = [
    "emerg",
    "alert",
    "crit",
    "err",
    "warning",
    "notice",
    "info",
    "debug",
]


def severity_counts(text: str) -> Dict[str, int]:
    """Heuristically count log severities in journal text.

    This is a lightweight classifier using common tokens.
    It scans line-by-line and assigns at most one severity per line,
    preferring higher severity when multiple tokens appear.
    """
    counts: Dict[str, int] = {k: 0 for k in _SEVERITY_ORDER}
    if not text:
        return counts

    patterns = {
        "emerg": re.compile(r"\b(emerg|emergency)\b", re.IGNORECASE),
        "alert": re.compile(r"\b(alert)\b", re.IGNORECASE),
        "crit": re.compile(r"\b(crit|critical)\b", re.IGNORECASE),
        "err": re.compile(r"\b(err|error|failed|failure|exception|traceback)\b", re.IGNORECASE),
        "warning": re.compile(r"\b(warn|warning)\b", re.IGNORECASE),
        "notice": re.compile(r"\b(notice)\b", re.IGNORECASE),
        "info": re.compile(r"\b(info|information)\b", re.IGNORECASE),
        "debug": re.compile(r"\b(debug|dbg)\b", re.IGNORECASE),
    }

    priority = _SEVERITY_ORDER

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sev in priority:
            if patterns[sev].search(line):
                counts[sev] += 1
                break
    return counts
