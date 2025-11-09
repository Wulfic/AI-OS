from __future__ import annotations

import subprocess as _sp
import time
from typing import Any

from aios.python_exec import get_preferred_python_executable


class CliBridgeMixin:
    """Shared helpers to run CLI commands and parse dict-like output.
    
    Includes basic caching for read-only operations to reduce subprocess overhead.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        # Simple cache for CLI results: {command_key: (result, timestamp)}
        self._cli_cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = 2.0  # Cache results for 2 seconds

    def _run_cli(self, args: list[str], use_cache: bool = True) -> str:  # type: ignore[override]
        """Run a CLI command, optionally using cache for read-only operations.
        
        Args:
            args: CLI command arguments
            use_cache: If True, cache read-only operations (list, stats, etc.)
        
        Returns:
            Command output as string
        """
        if not isinstance(args, list):
            return ""
        
        # Determine if this is a cacheable read-only operation
        cacheable = use_cache and len(args) > 0 and args[0] in {
            'brains', 'datasets-stats', 'status', 'goals-list'
        } and not any(arg in {'create', 'delete', 'add', 'remove', 'train'} for arg in args)
        
        # Check cache if cacheable
        if cacheable:
            cache_key = ' '.join(args)
            if cache_key in self._cli_cache:
                result, timestamp = self._cli_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    # Cache hit - return cached result
                    return result
        
        py_exec = get_preferred_python_executable()
        cmd = [py_exec, "-u", "-m", "aios.cli.aios", *args]
        try:
            res = _sp.run(cmd, check=False, capture_output=True, text=True)
            header = f"[cli] $ {' '.join(cmd)} (rc={res.returncode})\n"
            out = header + (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            try:
                self._debug_write(out)  # type: ignore[attr-defined]
            except Exception:
                pass
            
            result = out.strip()
            
            # Cache result if cacheable
            if cacheable:
                cache_key = ' '.join(args)
                self._cli_cache[cache_key] = (result, time.time())
                # Limit cache size to 50 entries
                if len(self._cli_cache) > 50:
                    # Remove oldest entry
                    oldest_key = min(self._cli_cache.keys(), 
                                   key=lambda k: self._cli_cache[k][1])
                    del self._cli_cache[oldest_key]
            
            return result
        except Exception as e:
            msg = f"[cli] error: {e}"
            try:
                self._debug_set_error(msg)  # type: ignore[attr-defined]
            except Exception:
                pass
            return msg

    def _parse_cli_dict(self, text: str) -> dict:
        """Best-effort parser that extracts a JSON/dict payload from noisy CLI output.

        Our CLI helpers often prepend a header line with the invoked command and
        return code, so downstream GUI code shouldn't assume the whole string is
        a clean JSON blob. This routine attempts, in order:
        - direct JSON parse
        - direct Python literal eval
        - locate the trailing JSON/dict segment by scanning from the bottom
          and parse that segment via JSON, then via literal eval as fallback
        - final attempt: slice from the last '{' and try again
        """
        import json, ast
        if not text:
            return {}
        # Fast path: exact JSON
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # Fast path: Python dict literal
        try:
            obj = ast.literal_eval(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # Robust path: scan from the bottom for a block that looks like JSON/dict
        try:
            lines = [ln for ln in (text or "").splitlines() if ln.strip()]
            for i in range(len(lines) - 1, -1, -1):
                tail = "\n".join(lines[i:]).strip()
                if not tail:
                    continue
                if tail[0] in "[{" and tail[-1] in "]}":
                    # Try JSON first
                    try:
                        obj = json.loads(tail)
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        # Fallback to Python literal (some CLIs print single quotes)
                        try:
                            obj = ast.literal_eval(tail)
                            return obj if isinstance(obj, dict) else {}
                        except Exception:
                            continue
        except Exception:
            pass
        # Last-ditch: slice from last '{'
        try:
            idx = text.rfind('{')
            if idx != -1:
                tail = text[idx:]
                try:
                    obj = json.loads(tail)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    try:
                        obj = ast.literal_eval(tail)
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        pass
        except Exception:
            pass
        return {}
