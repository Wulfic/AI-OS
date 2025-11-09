"""Enhanced logging helpers for AI-OS GUI.

Provides decorators and utilities to wrap functions with comprehensive logging.
"""
from __future__ import annotations

import logging
import functools
import time
import traceback
from typing import Callable, Any, TypeVar, ParamSpec

# Type variables for generic function signatures
P = ParamSpec('P')
R = TypeVar('R')

logger = logging.getLogger(__name__)


def log_function_call(
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = True,
    log_exceptions: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function calls with arguments and results.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        log_exceptions: Whether to log exceptions
        
    Example:
        @log_function_call(level=logging.INFO)
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            if log_args and (args or kwargs):
                args_str = ", ".join([repr(a) for a in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                params = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.log(level, f"🔵 Calling {func_name}({params})")
            else:
                logger.log(level, f"🔵 Calling {func_name}()")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000  # Convert to ms
                
                # Log successful completion
                if log_result:
                    result_str = repr(result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    logger.log(level, f"✅ {func_name} completed in {duration:.2f}ms → {result_str}")
                else:
                    logger.log(level, f"✅ {func_name} completed in {duration:.2f}ms")
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                if log_exceptions:
                    logger.error(
                        f"❌ {func_name} failed after {duration:.2f}ms: {type(e).__name__}: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                raise
        
        return wrapper
    return decorator


def log_method_call(
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log method calls (similar to log_function_call but for class methods).
    
    Automatically excludes 'self' from logging.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Skip 'self' in method calls
            display_args = args[1:] if args else args
            
            func_name = f"{func.__qualname__}"
            
            # Log method entry
            if log_args and (display_args or kwargs):
                args_str = ", ".join([repr(a) for a in display_args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                params = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.log(level, f"🔷 {func_name}({params})")
            else:
                logger.log(level, f"🔷 {func_name}()")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                if log_result:
                    result_str = repr(result)
                    if len(result_str) > 150:
                        result_str = result_str[:150] + "..."
                    logger.log(level, f"✅ {func_name} → {result_str} ({duration:.2f}ms)")
                else:
                    logger.log(level, f"✅ {func_name} completed ({duration:.2f}ms)")
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"❌ {func_name} failed: {type(e).__name__}: {str(e)} ({duration:.2f}ms)")
                raise
        
        return wrapper
    return decorator


def log_performance(threshold_ms: float = 100.0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log performance warnings for slow functions.
    
    Args:
        threshold_ms: Time threshold in milliseconds. Functions taking longer will be logged as warnings.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            if duration > threshold_ms:
                func_name = f"{func.__module__}.{func.__qualname__}"
                logger.warning(
                    f"⚠️ SLOW: {func_name} took {duration:.2f}ms (threshold: {threshold_ms}ms)"
                )
            
            return result
        return wrapper
    return decorator


class LogContext:
    """Context manager for structured logging blocks.
    
    Example:
        with LogContext("Loading brain", level=logging.INFO):
            # Your code here
            brain = load_brain()
    """
    
    def __init__(
        self,
        operation: str,
        level: int = logging.INFO,
        log_success: bool = True,
        log_duration: bool = True,
    ):
        self.operation = operation
        self.level = level
        self.log_success = log_success
        self.log_duration = log_duration
        self.start_time = 0.0
        
    def __enter__(self) -> 'LogContext':
        logger.log(self.level, f"▶️ Starting: {self.operation}")
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        duration = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            logger.error(
                f"❌ {self.operation} failed after {duration:.2f}ms: "
                f"{exc_type.__name__}: {str(exc_val)}"
            )
            return False  # Re-raise exception
        
        if self.log_success:
            if self.log_duration:
                logger.log(self.level, f"✅ Completed: {self.operation} ({duration:.2f}ms)")
            else:
                logger.log(self.level, f"✅ Completed: {self.operation}")
        
        return False


def log_gui_event(event_name: str, details: dict[str, Any] | None = None) -> None:
    """Log a GUI event with optional details.
    
    Args:
        event_name: Name of the GUI event (e.g., "button_click", "tab_switch")
        details: Optional dictionary of event details
    """
    if details:
        details_str = ", ".join([f"{k}={repr(v)}" for k, v in details.items()])
        logger.info(f"🖱️ GUI Event: {event_name} | {details_str}")
    else:
        logger.info(f"🖱️ GUI Event: {event_name}")


def log_state_change(component: str, old_state: Any, new_state: Any) -> None:
    """Log a state change in a component.
    
    Args:
        component: Name of the component
        old_state: Previous state
        new_state: New state
    """
    logger.info(f"🔄 State Change in {component}: {repr(old_state)} → {repr(new_state)}")


def log_error_with_context(
    error: Exception,
    context: str,
    additional_info: dict[str, Any] | None = None
) -> None:
    """Log an error with full context and traceback.
    
    Args:
        error: The exception object
        context: Description of what was being done when error occurred
        additional_info: Optional additional information to help debugging
    """
    error_msg = f"❌ Error in {context}: {type(error).__name__}: {str(error)}"
    
    if additional_info:
        info_str = "\n".join([f"  {k}: {repr(v)}" for k, v in additional_info.items()])
        error_msg += f"\nAdditional Info:\n{info_str}"
    
    error_msg += f"\nTraceback:\n{traceback.format_exc()}"
    
    logger.error(error_msg)


# Example usage documentation
if __name__ == "__main__":
    # Example 1: Function logging
    @log_function_call(level=logging.INFO, log_args=True, log_result=True)
    def fetch_data(url: str, timeout: int = 30) -> dict:
        """Fetch data from URL."""
        return {"data": "example"}
    
    # Example 2: Method logging
    class MyClass:
        @log_method_call(level=logging.DEBUG)
        def process(self, data: str) -> str:
            return data.upper()
    
    # Example 3: Context manager
    with LogContext("Processing user request", level=logging.INFO):
        # Do something
        pass
    
    # Example 4: GUI event
    log_gui_event("refresh_button_clicked", {"tab": "brains", "user": "admin"})
    
    # Example 5: State change
    log_state_change("BrainsPanel", old_state="loading", new_state="ready")
