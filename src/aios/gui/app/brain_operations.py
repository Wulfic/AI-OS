"""Brain operations module for AI-OS GUI.

This module handles:
- Brain management (create, delete, list)
- Brain training operations
- Brain model operations
"""

from __future__ import annotations
from concurrent.futures import Future
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
    
    worker_pool = getattr(app, "_worker_pool", None)
    ui_dispatcher = getattr(app, "_ui_dispatcher", None)

    def _on_brain_create(name: str, model_type: str) -> Future[str]:
        """
        Create a new brain.
        
        Args:
            name: Brain name
            model_type: Model type/architecture
        """
        app._log_router.log(f"Creating brain: {name} ({model_type})", LogCategory.TRAINING)

        def _on_success(result: str) -> None:
            app._log_router.log(f"Brain created: {result}", LogCategory.TRAINING)
            if hasattr(app, "brains_panel") and app.brains_panel:
                try:
                    app.brains_panel.refresh(force=True)
                except Exception:
                    app.brains_panel.refresh()

        def _on_error(exc: Exception) -> None:
            logger.error(f"Failed to create brain: {exc}")
            app._set_error(f"Failed to create brain: {exc}")

        try:
            return app._run_cli_async(
                ["brains", "create", name, "--type", model_type],
                use_cache=False,
                worker_pool=worker_pool,
                ui_dispatcher=ui_dispatcher,
                on_success=_on_success,
                on_error=_on_error,
                description=f"brains create {name}",
            )
        except Exception as e:
            logger.error(f"Failed to queue brain creation: {e}", exc_info=True)
            app._set_error(f"Failed to create brain: {e}")
            future: Future[str] = Future()
            future.set_exception(e)
            return future
    
    def _on_brain_delete(name: str) -> Future[str]:
        """
        Delete a brain.
        
        Args:
            name: Brain name
        """
        app._log_router.log(f"Deleting brain: {name}", LogCategory.TRAINING)

        def _on_success(result: str) -> None:
            app._log_router.log(f"Brain deleted: {result}", LogCategory.TRAINING)
            if hasattr(app, "brains_panel") and app.brains_panel:
                app.brains_panel.refresh(force=True)

        def _on_error(exc: Exception) -> None:
            logger.error(f"Failed to delete brain: {exc}")
            app._set_error(f"Failed to delete brain: {exc}")

        try:
            return app._run_cli_async(
                ["brains", "delete", name],
                use_cache=False,
                worker_pool=worker_pool,
                ui_dispatcher=ui_dispatcher,
                on_success=_on_success,
                on_error=_on_error,
                description=f"brains delete {name}",
            )
        except Exception as e:
            logger.error(f"Failed to queue brain deletion: {e}", exc_info=True)
            app._set_error(f"Failed to delete brain: {e}")
            future: Future[str] = Future()
            future.set_exception(e)
            return future
    
    def _on_brain_train(name: str, dataset: str, **kwargs) -> Future[str]:
        """
        Train a brain.
        
        Args:
            name: Brain name
            dataset: Dataset path/name
            **kwargs: Additional training parameters
        """
        args = ["brains", "train", name, "--dataset", dataset]

        for key, value in kwargs.items():
            if value is not None:
                args.extend([f"--{key.replace('_', '-')}", str(value)])

        app._log_router.log(f"Training brain: {name}", LogCategory.TRAINING)

        def _on_success(result: str) -> None:
            app._log_router.log(f"Training complete: {result}", LogCategory.TRAINING)

        def _on_error(exc: Exception) -> None:
            logger.error(f"Failed to train brain: {exc}")
            app._set_error(f"Failed to train brain: {exc}")

        try:
            return app._run_cli_async(
                args,
                use_cache=False,
                worker_pool=worker_pool,
                ui_dispatcher=ui_dispatcher,
                on_success=_on_success,
                on_error=_on_error,
                description=f"brains train {name}",
            )
        except Exception as e:
            logger.error(f"Failed to queue brain training: {e}", exc_info=True)
            app._set_error(f"Failed to train brain: {e}")
            future: Future[str] = Future()
            future.set_exception(e)
            return future
    
    # Attach handlers to app
    app._on_brain_create = _on_brain_create
    app._on_brain_delete = _on_brain_delete
    app._on_brain_train = _on_brain_train
