from __future__ import annotations

import logging
import subprocess as _sp
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future
from typing import Any, Callable

from aios.python_exec import get_preferred_python_executable

logger = logging.getLogger(__name__)


class CliBridgeMixin:
    """Shared helpers to run CLI commands and parse dict-like output.
    
    Includes basic caching for read-only operations to reduce subprocess overhead.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        # Simple cache for CLI results: {command_key: (result, timestamp)}
        self._cli_cache: dict[str, tuple[str, float]] = {}
        self._cli_cache_lock = threading.Lock()
        self._cache_ttl = 2.0  # Cache results for 2 seconds
        main_thread = threading.main_thread()
        self._main_thread_ident = main_thread.ident if main_thread else threading.get_ident()
        if self._main_thread_ident is None:
            self._main_thread_ident = threading.get_ident()
        self._sync_cli_guard_stack: list[str] = []
        self._enforce_sync_cli_guard = True

    @contextmanager
    def allow_sync_cli(self, reason: str = ""):
        """Temporarily permit synchronous CLI usage on the UI thread."""
        self._sync_cli_guard_stack.append(reason or "unspecified")
        try:
            yield
        finally:
            if self._sync_cli_guard_stack:
                self._sync_cli_guard_stack.pop()

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

        if getattr(self, "_enforce_sync_cli_guard", False):
            main_ident = getattr(self, "_main_thread_ident", None)
            if main_ident is None:
                main_thread = threading.main_thread()
                main_ident = main_thread.ident if main_thread else None
                self._main_thread_ident = main_ident if main_ident is not None else threading.get_ident()
            if main_ident is not None and threading.get_ident() == main_ident:
                guard_stack = getattr(self, "_sync_cli_guard_stack", None)
                if not guard_stack:
                    msg = (
                        "Synchronous _run_cli invoked on UI thread without allow_sync_cli context. "
                        "Use _run_cli_async(...) or wrap with allow_sync_cli(reason)."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
        
        # Determine if this is a cacheable read-only operation
        cacheable = use_cache and len(args) > 0 and args[0] in {
            'brains', 'datasets-stats', 'status', 'goals-list'
        } and not any(arg in {'create', 'delete', 'add', 'remove', 'train'} for arg in args)
        
        # Check cache if cacheable
        cache_key = ' '.join(args) if cacheable else None
        if cacheable and cache_key:
            with self._cli_cache_lock:
                cached_entry = self._cli_cache.get(cache_key)
            if cached_entry:
                result, timestamp = cached_entry
                if time.time() - timestamp < self._cache_ttl:
                    # Cache hit - return cached result
                    return result
        
        py_exec = get_preferred_python_executable()
        cmd = [py_exec, "-u", "-m", "aios.cli.aios", *args]
        try:
            # On Windows, use CREATE_NO_WINDOW to prevent CMD popups
            import sys as _sys
            creationflags = _sp.CREATE_NO_WINDOW if _sys.platform == "win32" else 0
            res = _sp.run(cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=creationflags)
            header = f"[cli] $ {' '.join(cmd)} (rc={res.returncode})\n"
            out = header + (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            try:
                self._debug_write(out)  # type: ignore[attr-defined]
            except Exception:
                pass
            
            result = out.strip()
            
            # Cache result if cacheable
            if cacheable and cache_key:
                with self._cli_cache_lock:
                    self._cli_cache[cache_key] = (result, time.time())
                    # Limit cache size to 50 entries
                    if len(self._cli_cache) > 50:
                        oldest_key = min(
                            self._cli_cache.keys(),
                            key=lambda k: self._cli_cache[k][1],
                        )
                        self._cli_cache.pop(oldest_key, None)
            
            return result
        except FileNotFoundError as e:
            error_context = "CLI command failed: Python executable not found"
            suggestion = "Reinstall Python or verify AIOS installation. Check that Python is in PATH"
            msg = f"[cli] {error_context}: {e}\nSuggestion: {suggestion}"
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            try:
                self._debug_set_error(msg)  # type: ignore[attr-defined]
            except Exception:
                pass
            return msg
        except PermissionError as e:
            error_context = f"CLI command failed: Permission denied executing {py_exec}"
            suggestion = "Run as administrator or check executable permissions"
            msg = f"[cli] {error_context}\nSuggestion: {suggestion}"
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            try:
                self._debug_set_error(msg)  # type: ignore[attr-defined]
            except Exception:
                pass
            return msg
        except Exception as e:
            error_context = f"CLI command failed: {' '.join(args)}"
            
            # Provide contextual suggestions
            error_str = str(e).lower()
            if "timeout" in error_str:
                suggestion = "Command timed out. Operation may be too resource-intensive or system is overloaded"
            elif "memory" in error_str:
                suggestion = "Insufficient memory. Close other applications or increase available RAM"
            elif "module" in error_str or "import" in error_str:
                suggestion = "Missing dependencies. Run 'pip install -r requirements.txt' to install required packages"
            else:
                suggestion = "Check logs for details. Verify AIOS installation and dependencies"
            
            msg = f"[cli] {error_context}: {e}\nSuggestion: {suggestion}"
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
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
        except Exception as e:
            logger.debug(f"Direct JSON parse failed: {e}")
        # Fast path: Python dict literal
        try:
            obj = ast.literal_eval(text)
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            logger.debug(f"Direct Python literal eval failed: {e}")
        # Robust path: scan from the bottom for a block that looks like JSON/dict
        try:
            lines = [ln for ln in (text or "").splitlines() if ln.strip()]
            for start in range(len(lines) - 1, -1, -1):
                line = lines[start]
                if "{" not in line and "[" not in line:
                    continue
                candidate_lines: list[str] = []
                brace_balance = 0
                bracket_balance = 0
                parsed = False
                for idx in range(start, len(lines)):
                    candidate_lines.append(lines[idx])
                    brace_balance += lines[idx].count("{") - lines[idx].count("}")
                    bracket_balance += lines[idx].count("[") - lines[idx].count("]")
                    if brace_balance <= 0 and bracket_balance <= 0:
                        payload = "\n".join(candidate_lines).strip()
                        for parser in (json.loads, ast.literal_eval):
                            try:
                                obj = parser(payload)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                continue
                        parsed = True
                        break
                if parsed:
                    continue
        except Exception as e:
            logger.debug(f"Bottom-up scan parse failed: {e}")
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
        except Exception as e:
            logger.debug(f"Last-ditch parse failed: {e}")
        
        # Parsing failed - log at debug level since empty dict is returned as fallback
        logger.debug(f"Output parsing failed: unable to extract dict from CLI output")
        logger.debug(f"Unexpected output format (length: {len(text)} chars): {text[:500]}")
        return {}

    def _run_cli_async(
        self,
        args: list[str],
        *,
        use_cache: bool = True,
        worker_pool: Any | None = None,
        ui_dispatcher: Any | None = None,
        on_success: Callable[[str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_finally: Callable[[], None] | None = None,
        description: str | None = None,
    ) -> Future[str]:
        """Execute a CLI command on a background worker.

        Args:
            args: CLI command arguments.
            use_cache: Whether to enable caching for read-only commands.
            worker_pool: Optional worker pool; defaults to ``self._worker_pool``.
            ui_dispatcher: Dispatcher used to marshal callbacks to the UI thread.
            on_success: Optional callback invoked with command output when it succeeds.
            on_error: Optional callback invoked with the raised exception.
            on_finally: Optional callback invoked after success or error.
            description: Optional context string for logging/debugging.

        Returns:
            ``concurrent.futures.Future`` representing the in-flight CLI execution.
        """

        worker = worker_pool or getattr(self, "_worker_pool", None)
        dispatcher = ui_dispatcher or getattr(self, "_ui_dispatcher", None)

        def _dispatch(callback: Callable, *callback_args: Any) -> None:
            if callback is None:
                return
            if dispatcher is not None:
                try:
                    dispatcher.dispatch(callback, *callback_args)
                    return
                except Exception:
                    pass
            callback(*callback_args)

        if not isinstance(args, list):
            future: Future[str] = Future()
            exc = TypeError("CLI args must be a list of strings")
            future.set_exception(exc)
            _dispatch(on_error, exc)
            _dispatch(on_finally)
            return future

        def _task() -> str:
            if description:
                logger.debug("Starting CLI task: %s", description)
            result = self._run_cli(args, use_cache=use_cache)
            if description:
                logger.debug("Completed CLI task: %s", description)
            return result

        if worker is None:
            future = Future()
            exc = RuntimeError("Async worker pool unavailable for CLI submission")
            future.set_exception(exc)
            _dispatch(on_error, exc)
            _dispatch(on_finally)
            return future

        future = worker.submit(_task)

        def _handle_completion(fut: Future[str]) -> None:
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - error path
                _dispatch(on_error, exc)
            else:
                _dispatch(on_success, result)
            finally:
                _dispatch(on_finally)

        if on_success is not None or on_error is not None or on_finally is not None:
            future.add_done_callback(_handle_completion)

        return future
