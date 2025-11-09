"""Journal Operators - Journal analysis and parsing operations."""

from __future__ import annotations

import sys
import re
import subprocess
from typing import Dict, Any, List


def register_journal_operators(reg):
    """Register journal analysis operators."""
    from ..api import SimpleOperator, AsyncOperator

    def _journal_triage_heuristic(ctx: Dict[str, Any]) -> bool:
        """Read-only: scan recent journal for simple error patterns (Linux-only)."""
        if not sys.platform.startswith("linux"):
            return False
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        n = max(1, min(int(ctx.get("lines", 50)), 500))
        try:
            text = subprocess.check_output(
                ["/usr/bin/journalctl", "-u", unit, "-n", str(n), "--no-pager"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            return False
        patterns = re.compile(r"error|failed|panic|traceback", re.IGNORECASE)
        _ = bool(patterns.search(text or ""))
        return True

    reg.register(SimpleOperator(name="journal_triage_heuristic", func=_journal_triage_heuristic))

    async def _journal_triage_v2(ctx: Dict[str, Any]) -> bool:
        """Read-only: windowed journal triage with severity filter and patterns."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        since = ctx.get("since")
        until = ctx.get("until")
        priority = ctx.get("priority")
        lines = int(ctx.get("lines", 200))
        try:
            from aios.tools.service_adapter import journal_read as jr
        except Exception:
            return False
        res = await jr(unit, lines=lines, since=since, until=until, priority=priority, timeout_sec=2.0)
        via = res.get("via")
        text = (res.get("text") or "").strip()
        if via not in ("root-helper", "local"):
            return False
        patterns = re.compile(r"\b(error|failed|failure|panic|traceback|segfault)\b", re.IGNORECASE)
        _ = bool(patterns.search(text))
        return bool(text)

    reg.register(AsyncOperator(name="journal_triage_v2", async_func=_journal_triage_v2))

    async def _journal_triage_summary(ctx: Dict[str, Any]) -> bool:
        """Read-only: compute severity counts from recent journal for a unit."""
        unit = str(ctx.get("service") or ctx.get("unit") or "").strip()
        if not unit:
            return False
        try:
            from aios.tools.service_adapter import journal_read as jr
            from aios.tools.journal_parser import severity_counts
        except Exception:
            return False
        res = await jr(unit, lines=int(ctx.get("lines", 200)), timeout_sec=2.0)
        via = res.get("via")
        text = (res.get("text") or "").strip()
        if via not in ("root-helper", "local"):
            return False
        counts = severity_counts(text)
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs["journal_triage_summary"] = {
                "unit": unit,
                "via": via,
                "lines": int(ctx.get("lines", 200)),
                "counts": counts,
                "has_text": bool(text),
            }
        except Exception:
            pass
        return bool(text) or any(v > 0 for v in counts.values())

    reg.register(AsyncOperator(name="journal_triage_summary", async_func=_journal_triage_summary))

    def _journal_summary_from_text(ctx: Dict[str, Any]) -> bool:
        """Compute severity counts from provided 'journal_text' in context."""
        text = str(ctx.get("journal_text") or "")
        if not text.strip():
            return False
        try:
            from aios.tools.journal_parser import severity_counts as _sc
        except Exception:
            return False
        counts = _sc(text)
        label = str(ctx.get("label") or "journal_summary").strip() or "journal_summary"
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs[label] = {"counts": counts, "has_text": True}
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="journal_summary_from_text", func=_journal_summary_from_text))

    def _journal_severity_ratio_from_text(ctx: Dict[str, Any]) -> bool:
        """Compute simple severity ratios from provided 'journal_text'."""
        text = str(ctx.get("journal_text") or "")
        if not text.strip():
            return False
        try:
            from aios.tools.journal_parser import severity_counts as _sc
        except Exception:
            return False
        counts = _sc(text)
        info = max(1, int(counts.get("info", 0)))
        warn = int(counts.get("warning", 0))
        err = int(counts.get("err", 0))
        ratios = {"err_to_info": err / info, "warn_to_info": warn / info}
        label = str(ctx.get("label") or "journal_severity_ratio").strip() or "journal_severity_ratio"
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs[label] = {"counts": counts, "ratios": ratios}
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="journal_severity_ratio_from_text", func=_journal_severity_ratio_from_text))

    def _journal_top_words_from_text(ctx: Dict[str, Any]) -> bool:
        """Compute top-N non-stopword tokens from provided 'journal_text'."""
        text = str(ctx.get("journal_text") or "")
        if not text.strip():
            return False
        top_n = int(ctx.get("top_n", 5) or 5)
        if top_n <= 0:
            top_n = 5
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return False
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at",
            "is", "are", "was", "were", "for", "with", "by", "from", "it",
            "this", "that", "as", "be", "has", "have", "had", "not", "no",
            "service", "unit", "system", "error", "warning", "info", "debug",
        }
        counts: Dict[str, int] = {}
        for t in tokens:
            if t in stop:
                continue
            counts[t] = counts.get(t, 0) + 1
        if not counts:
            return False
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        label = str(ctx.get("label") or "journal_top_words").strip() or "journal_top_words"
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs[label] = {"top": top, "counts": counts}
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="journal_top_words_from_text", func=_journal_top_words_from_text))

    def _journal_top_severities_from_text(ctx: Dict[str, Any]) -> bool:
        """Rank severities by frequency using severity_counts from provided text."""
        text = str(ctx.get("journal_text") or "")
        if not text.strip():
            return False
        try:
            from aios.tools.journal_parser import severity_counts as _sc
        except Exception:
            return False
        counts = _sc(text)
        order = ["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
        idx = {s: i for i, s in enumerate(order)}
        items = [(s, int(c)) for s, c in counts.items() if int(c) > 0]
        if not items:
            items = []
        items.sort(key=lambda kv: (-kv[1], idx.get(kv[0], 999)))
        label = str(ctx.get("label") or "journal_top_severities").strip() or "journal_top_severities"
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs[label] = {"order": items, "counts": counts}
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="journal_top_severities_from_text", func=_journal_top_severities_from_text))

    def _journal_trend_from_text(ctx: Dict[str, Any]) -> bool:
        """Compute simple severity trends by bucketing lines in provided journal_text."""
        text = str(ctx.get("journal_text") or "")
        if not text.strip():
            return False
        try:
            from aios.tools.journal_parser import severity_counts as _sc
        except Exception:
            return False
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        b = int(ctx.get("buckets", 4) or 4)
        if b < 2:
            b = 4
        if b > 12:
            b = 12
        n = len(lines)
        chunk = max(1, (n + b - 1) // b)
        buckets: List[Dict[str, Any]] = []
        err_series: List[float] = []
        warn_series: List[float] = []
        for i in range(0, n, chunk):
            seg = "\n".join(lines[i : i + chunk])
            counts = _sc(seg)
            info = max(1, int(counts.get("info", 0)))
            warn = int(counts.get("warning", 0))
            err = int(counts.get("err", 0))
            ratios = {"err_to_info": err / info, "warn_to_info": warn / info}
            buckets.append({
                "index": len(buckets),
                "total": len(seg.splitlines()),
                "counts": counts,
                "ratios": ratios,
            })
            err_series.append(ratios["err_to_info"])
            warn_series.append(ratios["warn_to_info"])
            if len(buckets) >= b:
                break
        overall_counts = _sc("\n".join(lines))

        def _delta(series: List[float]) -> float:
            if not series:
                return 0.0
            return float(series[-1] - series[0])

        def _monotonic_non_decreasing(series: List[float]) -> bool:
            if not series:
                return False
            prev = series[0]
            for x in series[1:]:
                if x < prev:
                    return False
                prev = x
            return True

        trend = {
            "err_to_info_delta": _delta(err_series),
            "warn_to_info_delta": _delta(warn_series),
            "monotonic_increase_err": _monotonic_non_decreasing(err_series),
            "monotonic_increase_warn": _monotonic_non_decreasing(warn_series),
        }
        label = str(ctx.get("label") or "journal_trend").strip() or "journal_trend"
        try:
            outputs = ctx.setdefault("outputs", {})
            outputs[label] = {
                "buckets": buckets,
                "overall": {"counts": overall_counts},
                "trend": trend,
            }
        except Exception:
            pass
        return True

    reg.register(SimpleOperator(name="journal_trend_from_text", func=_journal_trend_from_text))
