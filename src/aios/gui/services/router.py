from __future__ import annotations

import ast
import json
from typing import Any


def chat_route(user: str) -> list[str]:
    u = (user or "").strip()
    if not u:
        return ["status", "--recent", "1"]
    if u.startswith("/"):
        return u[1:].split()
    toks = u.split()
    if not toks:
        return ["status", "--recent", "1"]
    known = {"status", "crawl", "train", "train-parallel", "artifacts-list", "budgets-show", "budgets-usage"}
    if toks[0] in known:
        return toks
    return ["chat", u]


def parse_cli_dict(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def render_chat_output(raw: str | dict, args: list[str]) -> str:
    """Render chat output, handling both string (CLI) and dict (direct router) responses."""
    # Handle direct dict response from router
    if isinstance(raw, dict):
        data = raw
    else:
        out = raw or ""
        # Strip CLI debug headers and warnings from output
        if "[cli] $" in out:
            # Remove the CLI command header line
            lines = out.split("\n")
            out = "\n".join(line for line in lines if not line.startswith("[cli] $"))
        # Remove transformer warnings
        if "FutureWarning" in out or "TRANSFORMERS_CACHE" in out:
            lines = out.split("\n")
            cleaned_lines = []
            skip_warning = False
            for line in lines:
                if "WARNING" in line or "FutureWarning" in line or "TRANSFORMERS_CACHE" in line:
                    skip_warning = True
                    continue
                if skip_warning and line.strip() and not line.startswith(" "):
                    skip_warning = False
                if not skip_warning:
                    cleaned_lines.append(line)
            out = "\n".join(cleaned_lines)
        # Try to parse JSON from string
        try:
            data = parse_cli_dict(out.strip())
        except Exception:
            data = None
    
    # Extract clean text from structured response
    if isinstance(data, dict):
        if data.get("text"):
            txt = str(data.get("text") or "").strip()
            return txt
        if data.get("summary"):
            s = data.get("summary") or {}
            headline = s.get("headline") or ""
            details = s.get("details") or ""
            echo = data.get("echo") or ""
            msg2 = (str(headline).strip() + "\n" + str(details).strip()).strip()
            if echo:
                msg2 += f"\n\nEcho: {echo}"
            return msg2
        if data.get("error"):
            return f"Error: {data.get('error')}"
    
    # Fallback to raw string
    if isinstance(raw, str):
        if not raw or not raw.strip():
            return "Model not ready yet. Train a checkpoint or try 'status' to see system state."
        return raw
    
    return "No response from model."
