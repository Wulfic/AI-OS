"""Brain operations module for AI-OS GUI.

This module handles:
- Brain management (create, delete, list)
- Brain training operations
- Brain model operations
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    pass

from ..services import LogCategory

logger = logging.getLogger(__name__)


def setup_brain_operations(app: Any) -> None:
    """
    Set up brain operation handlers.
    
    Args:
        app: AiosTkApp instance with _run_cli available
    """
    
    def _on_brain_create(name: str, model_type: str) -> None:
        """
        Create a new brain.
        
        Args:
            name: Brain name
            model_type: Model type/architecture
        """
        try:
            app._log_router.log(f"Creating brain: {name} ({model_type})", LogCategory.TRAINING)
            result = app._run_cli(["brains", "create", name, "--type", model_type])
            app._log_router.log(f"Brain created: {result}", LogCategory.TRAINING)
            
            # Refresh brains panel
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
        except Exception as e:
            logger.error(f"Failed to create brain: {e}")
            app._set_error(f"Failed to create brain: {e}")
    
    def _on_brain_delete(name: str) -> None:
        """
        Delete a brain.
        
        Args:
            name: Brain name
        """
        try:
            app._log_router.log(f"Deleting brain: {name}", LogCategory.TRAINING)
            result = app._run_cli(["brains", "delete", name])
            app._log_router.log(f"Brain deleted: {result}", LogCategory.TRAINING)
            
            # Refresh brains panel
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
        except Exception as e:
            logger.error(f"Failed to delete brain: {e}")
            app._set_error(f"Failed to delete brain: {e}")
    
    def _on_brain_train(name: str, dataset: str, **kwargs) -> None:
        """
        Train a brain.
        
        Args:
            name: Brain name
            dataset: Dataset path/name
            **kwargs: Additional training parameters
        """
        try:
            args = ["brains", "train", name, "--dataset", dataset]
            
            # Add optional parameters
            for key, value in kwargs.items():
                if value is not None:
                    args.extend([f"--{key.replace('_', '-')}", str(value)])
            
            app._log_router.log(f"Training brain: {name}", LogCategory.TRAINING)
            result = app._run_cli(args)
            app._log_router.log(f"Training complete: {result}", LogCategory.TRAINING)
        except Exception as e:
            logger.error(f"Failed to train brain: {e}")
            app._set_error(f"Failed to train brain: {e}")
    
    # Attach handlers to app
    app._on_brain_create = _on_brain_create
    app._on_brain_delete = _on_brain_delete
    app._on_brain_train = _on_brain_train
