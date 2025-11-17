"""Goal operations module for AI-OS GUI.

This module handles:
- Goal management (add, remove, list)
- Brain goal associations
- Goal tracking
"""

from __future__ import annotations
from concurrent.futures import Future
from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    pass

from ..services import LogCategory

logger = logging.getLogger(__name__)


def setup_goal_operations(app: Any) -> None:
    """
    Set up goal operation handlers.
    
    Args:
        app: AiosTkApp instance with _run_cli available
    """
    
    def _extract_goal_payload(raw: str) -> list[dict[str, Any]]:
        """Best-effort extraction of goals list from CLI output."""
        import ast
        import json

        text = raw or "[]"

        # Fast path: leverage the CLI bridge parser when available so we share
        # the same noisy-output handling logic used elsewhere in the app.
        parser_fn = getattr(app, "_parse_cli_dict", None)
        if callable(parser_fn):
            try:
                parsed = parser_fn(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed:
                for key in ("items", "goals", "data"):
                    payload_items = parsed.get(key)
                    if isinstance(payload_items, list):
                        dict_items = [item for item in payload_items if isinstance(item, dict)]
                        if dict_items:
                            return dict_items
                if {"id", "text"}.issubset(parsed.keys()):
                    return [parsed]

        lines = [ln for ln in text.splitlines() if ln.strip()]

        # Walk upward collecting candidate segments that look like JSON/dicts.
        candidates: list[str] = []
        for idx in range(len(lines) - 1, -1, -1):
            snippet = lines[idx].strip()
            if snippet.startswith("[") or snippet.startswith("{"):
                candidates.append(snippet)
                # Attempt to include following lines if the payload spans multiple lines
                brace_balance = snippet.count("{") - snippet.count("}")
                bracket_balance = snippet.count("[") - snippet.count("]")
                if brace_balance > 0 or bracket_balance > 0:
                    payload_lines = [snippet]
                    for follow in lines[idx + 1:]:
                        payload_lines.append(follow.strip())
                        brace_balance += follow.count("{") - follow.count("}")
                        bracket_balance += follow.count("[") - follow.count("]")
                        if brace_balance <= 0 and bracket_balance <= 0:
                            break
                    candidates[-1] = "\n".join(payload_lines)
        if not candidates:
            candidates.append(text)

        for payload in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(payload)
                    if isinstance(obj, list):
                        return [item for item in obj if isinstance(item, dict)]
                    if isinstance(obj, dict):
                        for key in ("items", "goals", "data"):
                            payload_items = obj.get(key)
                            if isinstance(payload_items, list):
                                dict_items = [item for item in payload_items if isinstance(item, dict)]
                                if dict_items:
                                    return dict_items
                        if {"id", "text"}.issubset(obj.keys()):
                            return [obj]
                except Exception:
                    continue
        return []

    def _on_goal_add_for_brain(brain_name: str, goal_text: str) -> Future[str]:
        """
        Add a goal for a brain.
        
        Args:
            brain_name: Brain name
            goal_text: Goal description
        """
        try:
            logger.info(f"User action: Adding goal to brain '{brain_name}': {goal_text[:100]}")
            app._log_router.log(f"Adding goal to {brain_name}: {goal_text[:50]}...", LogCategory.TRAINING, "INFO")
            
            def _on_success(result: str) -> None:
                logger.info(f"Successfully added goal to brain '{brain_name}'")
                app._log_router.log(f"Goal added: {result}", LogCategory.TRAINING, "INFO")
                if hasattr(app, "brains_panel") and app.brains_panel:
                    try:
                        app.brains_panel.refresh(force=True)
                    except Exception:
                        app.brains_panel.refresh()

            def _on_error(exc: Exception) -> None:
                logger.error(f"Failed to add goal to brain '{brain_name}': {exc}", exc_info=True)
                app._set_error(f"Failed to add goal: {exc}")

            return app._run_cli_async(
                ["goals-add", goal_text, "--expert-id", brain_name],
                use_cache=False,
                worker_pool=getattr(app, "_worker_pool", None),
                ui_dispatcher=getattr(app, "_ui_dispatcher", None),
                on_success=_on_success,
                on_error=_on_error,
                description=f"goals-add {brain_name}",
            )
        except Exception as e:
            logger.error(f"Failed to queue add goal '{brain_name}': {e}", exc_info=True)
            app._set_error(f"Failed to add goal: {e}")
            future: Future[str] = Future()
            future.set_exception(e)
            return future
    
    def _on_goals_list_for_brain(brain_name: str) -> Future[list[dict[str, Any]]]:
        """
        List goals for a brain.
        
        Args:
            brain_name: Brain name
        
        Returns:
            List of goal dictionaries
        """
        try:
            logger.debug(f"Listing goals for brain '{brain_name}'")
            future: Future[list[dict[str, Any]]] = Future()

            def _on_success(raw: str) -> None:
                items = _extract_goal_payload(raw)
                logger.debug(f"Retrieved {len(items)} total goals from system")
                if not future.done():
                    future.set_result(items)

            def _on_error(exc: Exception) -> None:
                logger.error(f"Failed to list goals for brain '{brain_name}': {exc}", exc_info=True)
                if not future.done():
                    future.set_result([])

            # Always fetch fresh directives so UI reflects recent add/remove operations.
            app._run_cli_async(
                ["goals-list"],
                use_cache=False,
                worker_pool=getattr(app, "_worker_pool", None),
                ui_dispatcher=getattr(app, "_ui_dispatcher", None),
                on_success=_on_success,
                on_error=_on_error,
                description=f"goals-list {brain_name}",
            )
            return future
        except Exception as e:
            logger.error(f"Failed to queue goals list for brain '{brain_name}': {e}", exc_info=True)
            future: Future[list[dict[str, Any]]] = Future()
            future.set_result([])
            return future
    
    def _on_goal_remove(goal_id: int) -> None:
        """
        Remove a goal by ID.
        
        Args:
            goal_id: Goal identifier (integer)
        """
        try:
            logger.info(f"User action: Removing goal with ID {goal_id}")
            app._log_router.log(f"Removing goal {goal_id}", LogCategory.TRAINING, "INFO")
            
            def _on_success(result: str) -> None:
                logger.info(f"Successfully removed goal {goal_id}")
                app._log_router.log(f"Goal removed: {result}", LogCategory.TRAINING, "INFO")
                if hasattr(app, "brains_panel") and app.brains_panel and hasattr(app.brains_panel, "_on_tree_select"):
                    try:
                        app.brains_panel._on_tree_select()
                    except Exception:
                        app.brains_panel.refresh(force=True)

            def _on_error(exc: Exception) -> None:
                logger.error(f"Failed to remove goal {goal_id}: {exc}", exc_info=True)
                app._set_error(f"Failed to remove goal: {exc}")

            return app._run_cli_async(
                ["goals-remove", str(goal_id)],
                use_cache=False,
                worker_pool=getattr(app, "_worker_pool", None),
                ui_dispatcher=getattr(app, "_ui_dispatcher", None),
                on_success=_on_success,
                on_error=_on_error,
                description=f"goals-remove {goal_id}",
            )
        except Exception as e:
            logger.error(f"Failed to queue goal removal {goal_id}: {e}", exc_info=True)
            app._set_error(f"Failed to remove goal: {e}")
            future: Future[str] = Future()
            future.set_exception(e)
            return future
    
    # Attach handlers to app
    app._on_goal_add_for_brain = _on_goal_add_for_brain
    app._on_goals_list_for_brain = _on_goals_list_for_brain
    app._on_goal_remove = _on_goal_remove
