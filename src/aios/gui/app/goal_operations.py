"""Goal operations module for AI-OS GUI.

This module handles:
- Goal management (add, remove, list)
- Brain goal associations
- Goal tracking
"""

from __future__ import annotations
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
    
    def _on_goal_add_for_brain(brain_name: str, goal_text: str) -> None:
        """
        Add a goal for a brain.
        
        Args:
            brain_name: Brain name
            goal_text: Goal description
        """
        try:
            app._log_router.log(f"Adding goal to {brain_name}: {goal_text[:50]}...", LogCategory.TRAINING)
            result = app._run_cli(["goals-add", goal_text, "--expert-id", brain_name])
            app._log_router.log(f"Goal added: {result}", LogCategory.TRAINING)
            
            # Refresh brains panel
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
        except Exception as e:
            logger.error(f"Failed to add goal: {e}")
            app._set_error(f"Failed to add goal: {e}")
    
    def _on_goals_list_for_brain(brain_name: str) -> list[dict]:
        """
        List goals for a brain.
        
        Args:
            brain_name: Brain name
        
        Returns:
            List of goal dictionaries
        """
        try:
            result = app._run_cli(["goals-list"])
            data = app._parse_cli_dict(result or "[]")
            
            if isinstance(data, list):
                return data
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to list goals: {e}")
            return []
    
    def _on_goal_remove(goal_id: int) -> None:
        """
        Remove a goal by ID.
        
        Args:
            goal_id: Goal identifier (integer)
        """
        try:
            app._log_router.log(f"Removing goal {goal_id}", LogCategory.TRAINING)
            result = app._run_cli(["goals-remove", str(goal_id)])
            app._log_router.log(f"Goal removed: {result}", LogCategory.TRAINING)
            
            # Refresh brains panel goals
            if hasattr(app, 'brains_panel') and app.brains_panel and hasattr(app.brains_panel, '_on_tree_select'):
                app.brains_panel._on_tree_select()
        except Exception as e:
            logger.error(f"Failed to remove goal: {e}")
            app._set_error(f"Failed to remove goal: {e}")
    
    # Attach handlers to app
    app._on_goal_add_for_brain = _on_goal_add_for_brain
    app._on_goals_list_for_brain = _on_goals_list_for_brain
    app._on_goal_remove = _on_goal_remove
