"""System Operators - Basic system-level operations."""

from __future__ import annotations

from typing import Dict, Any


def register_system_operators(reg):
    """Register basic system operators: sys_info, web_parse_html, python_syntax_check."""
    from aios.tools.os import get_system_info
    from aios.tools.crawler import parse_html_to_text
    from ..api import SimpleOperator

    def _sys_info(_ctx):
        _ = get_system_info()
        # Always succeeds (read-only)
        return True

    reg.register(SimpleOperator(name="sys_info", func=_sys_info))

    def _web_parse_html(ctx: Dict[str, Any]) -> bool:
        """Parse provided HTML using crawler parser to simulate web expertise."""
        html = str(ctx.get("html", "")).strip()
        if not html:
            return False
        title, text = parse_html_to_text(html)
        return bool((title or text))

    reg.register(SimpleOperator(name="web_parse_html", func=_web_parse_html))

    def _python_syntax_check(ctx: Dict[str, Any]) -> bool:
        """Compile a provided code snippet to validate Python syntax (read-only)."""
        code = str(ctx.get("code", "")).strip()
        if not code:
            return False
        try:
            compile(code, "<operator:python_syntax_check>", "exec")
            return True
        except Exception:
            return False

    reg.register(SimpleOperator(name="python_syntax_check", func=_python_syntax_check))
