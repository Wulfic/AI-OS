"""Safe Tkinter Variable wrappers to prevent RuntimeError during cleanup.

This module provides wrapped versions of Tkinter Variable classes that handle
cleanup gracefully when the main loop is not running, preventing the common
"RuntimeError: main thread is not in main loop" error during Variable.__del__.
"""

import logging
import tkinter as tk
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _SafeVarMixin:
    """Mixin that skips Tk calls once the interpreter is torn down."""

    __slots__ = ()

    def _can_cleanup(self) -> bool:
        tkapp = getattr(self, "_tk", None)
        name = getattr(self, "_name", None)
        if not tkapp or not name:
            return False

        # When the interpreter is shutting down Tk replaces _tclCommands with None.
        if getattr(tkapp, "_tclCommands", None) is None:
            return False

        return True

    def __del__(self) -> None:  # type: ignore[override]
        try:
            if self._can_cleanup():
                try:
                    super().__del__()
                except (RuntimeError, tk.TclError):
                    # Happens when cleanup runs off the Tk thread or interpreter is closing.
                    pass
                except Exception:
                    pass
        except Exception:
            pass


class SafeStringVar(_SafeVarMixin, tk.StringVar):
    """StringVar that handles cleanup safely."""
    
    def __init__(self, master: Any = None, value: Optional[str] = None, name: Optional[str] = None):
        """Initialize SafeStringVar.
        
        Args:
            master: Parent widget
            value: Initial value
            name: Variable name
        """
        super().__init__(master, value, name)
        self._cleanup_safe = True


class SafeIntVar(_SafeVarMixin, tk.IntVar):
    """IntVar that handles cleanup safely."""
    
    def __init__(self, master: Any = None, value: Optional[int] = None, name: Optional[str] = None):
        """Initialize SafeIntVar.
        
        Args:
            master: Parent widget
            value: Initial value
            name: Variable name
        """
        super().__init__(master, value, name)
        self._cleanup_safe = True


class SafeBooleanVar(_SafeVarMixin, tk.BooleanVar):
    """BooleanVar that handles cleanup safely."""
    
    def __init__(self, master: Any = None, value: Optional[bool] = None, name: Optional[str] = None):
        """Initialize SafeBooleanVar.
        
        Args:
            master: Parent widget
            value: Initial value
            name: Variable name
        """
        super().__init__(master, value, name)
        self._cleanup_safe = True


class SafeDoubleVar(_SafeVarMixin, tk.DoubleVar):
    """DoubleVar that handles cleanup safely."""
    
    def __init__(self, master: Any = None, value: Optional[float] = None, name: Optional[str] = None):
        """Initialize SafeDoubleVar.
        
        Args:
            master: Parent widget
            value: Initial value
            name: Variable name
        """
        super().__init__(master, value, name)
        self._cleanup_safe = True


# For convenience, provide factory functions with the same signature as tk.*Var
def StringVar(master: Any = None, value: Optional[str] = None, name: Optional[str] = None) -> SafeStringVar:
    """Create a safe StringVar.
    
    Args:
        master: Parent widget
        value: Initial value
        name: Variable name
        
    Returns:
        SafeStringVar instance
    """
    return SafeStringVar(master, value, name)


def IntVar(master: Any = None, value: Optional[int] = None, name: Optional[str] = None) -> SafeIntVar:
    """Create a safe IntVar.
    
    Args:
        master: Parent widget
        value: Initial value
        name: Variable name
        
    Returns:
        SafeIntVar instance
    """
    return SafeIntVar(master, value, name)


def BooleanVar(master: Any = None, value: Optional[bool] = None, name: Optional[str] = None) -> SafeBooleanVar:
    """Create a safe BooleanVar.
    
    Args:
        master: Parent widget
        value: Initial value
        name: Variable name
        
    Returns:
        SafeBooleanVar instance
    """
    return SafeBooleanVar(master, value, name)


def DoubleVar(master: Any = None, value: Optional[float] = None, name: Optional[str] = None) -> SafeDoubleVar:
    """Create a safe DoubleVar.
    
    Args:
        master: Parent widget
        value: Initial value
        name: Variable name
        
    Returns:
        SafeDoubleVar instance
    """
    return SafeDoubleVar(master, value, name)
