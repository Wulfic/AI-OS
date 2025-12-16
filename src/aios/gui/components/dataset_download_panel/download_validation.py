"""
Download Validation

Pre-download checks to ensure datasets fit within storage limits and constraints.
All dialog functions are thread-safe - they schedule dialogs on the main thread.
"""

import logging
import shutil
import threading
from pathlib import Path
from tkinter import messagebox
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _show_dialog_threadsafe(parent_widget: Any, dialog_func, *args, **kwargs) -> Any:
    """
    Show a tkinter dialog in a thread-safe manner.
    
    If called from main thread, runs dialog directly.
    If called from background thread, schedules dialog on main thread and waits for result.
    
    Args:
        parent_widget: Parent tkinter widget to schedule on
        dialog_func: The dialog function (e.g., messagebox.askyesno)
        *args, **kwargs: Arguments to pass to dialog_func
        
    Returns:
        Result from the dialog
    """
    result = [None]  # Use list to allow modification in nested function
    error = [None]
    event = threading.Event()
    
    dialog_name = getattr(dialog_func, '__name__', 'dialog')
    logger.debug(f"Scheduling thread-safe dialog: {dialog_name}")
    
    def _run_dialog():
        try:
            logger.debug(f"Running dialog: {dialog_name}")
            result[0] = dialog_func(*args, **kwargs)
            logger.debug(f"Dialog result: {result[0]}")
        except Exception as e:
            error[0] = e
            logger.error(f"Dialog error: {e}")
        finally:
            event.set()
    
    # Try to schedule on main thread via parent widget's after()
    try:
        if parent_widget:
            # Don't call winfo_exists() from background thread - just try to schedule
            logger.debug(f"Scheduling dialog on main thread via after()")
            try:
                parent_widget.after(0, _run_dialog)
            except Exception as after_error:
                logger.warning(f"after() scheduling failed: {after_error}, running directly")
                _run_dialog()
                return result[0] if not error[0] else None
            
            # Wait for dialog to complete (with timeout to prevent deadlock)
            logger.debug(f"Waiting for dialog response...")
            if not event.wait(timeout=120):  # 2 minute timeout
                logger.warning("Dialog timeout - continuing without response")
                return None
            logger.debug(f"Dialog completed, result: {result[0]}")
        else:
            # No parent widget - try running directly
            logger.debug("No parent widget - running dialog directly")
            _run_dialog()
    except Exception as e:
        logger.error(f"Failed to schedule dialog: {e}")
        # Try running directly as fallback
        try:
            _run_dialog()
        except Exception as e2:
            logger.error(f"Dialog fallback also failed: {e2}")
            return None
    
    if error[0]:
        return None
    return result[0]

# Import storage management functions
try:
    from ....data.datasets.storage import (
        datasets_storage_cap_gb,
        datasets_storage_usage_gb,
        can_store_additional_gb,
    )
    STORAGE_MANAGEMENT_AVAILABLE = True
except ImportError:
    logger.warning("Storage management functions not available")
    datasets_storage_cap_gb = None  # type: ignore
    datasets_storage_usage_gb = None  # type: ignore
    can_store_additional_gb = None  # type: ignore
    STORAGE_MANAGEMENT_AVAILABLE = False


def check_disk_space(path: Path, required_gb: float) -> Tuple[bool, Optional[str]]:
    """
    Check if there's enough disk space at the target path.
    
    Args:
        path: Target directory path
        required_gb: Required space in GB
        
    Returns:
        (is_sufficient, error_message)
    """
    try:
        # Create directory if it doesn't exist (to get the correct disk)
        path.mkdir(parents=True, exist_ok=True)
        
        # Get disk usage stats
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        
        # Leave at least 5GB free for system operations
        SAFETY_MARGIN_GB = 5.0
        available_gb = free_gb - SAFETY_MARGIN_GB
        
        if available_gb < required_gb:
            return False, (
                f"Insufficient disk space.\n\n"
                f"Required: {required_gb:.1f} GB\n"
                f"Available: {available_gb:.1f} GB\n"
                f"(keeping {SAFETY_MARGIN_GB:.1f} GB safety margin)\n\n"
                f"Please free up disk space and try again."
            )
        
        return True, None
        
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        # Don't block download if we can't check - just warn
        return True, None


def check_storage_cap(required_gb: float) -> Tuple[bool, Optional[str]]:
    """
    Check if download fits within the configured dataset storage cap.
    
    Args:
        required_gb: Required storage space in GB
        
    Returns:
        (fits_within_cap, error_message)
    """
    if not STORAGE_MANAGEMENT_AVAILABLE:
        # If storage management isn't available, don't block
        return True, None
    
    try:
        # Get current cap and usage
        cap_gb = datasets_storage_cap_gb()
        current_usage_gb = datasets_storage_usage_gb()
        
        # Cap of 0 means unlimited
        if cap_gb == 0:
            return True, None
        
        # Check if dataset fits within cap
        if not can_store_additional_gb(required_gb):
            return False, (
                f"Dataset exceeds storage limit.\n\n"
                f"Dataset size: {required_gb:.1f} GB\n"
                f"Current usage: {current_usage_gb:.1f} GB / {cap_gb:.1f} GB\n"
                f"Available: {max(0, cap_gb - current_usage_gb):.1f} GB\n\n"
                f"Go to Settings → Dataset Storage to increase the limit."
            )
        
        return True, None
        
    except Exception as e:
        logger.error(f"Failed to check storage cap: {e}")
        # Don't block download if we can't check
        return True, None


def validate_download_prerequisites(
    dataset: Dict[str, Any],
    output_path: Path,
    parent_widget: Any = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a dataset download can proceed.
    
    Checks:
    1. Dataset size is known (not "Unknown")
    2. Fits within configured storage cap
    3. Sufficient disk space available
    
    Args:
        dataset: Dataset dictionary with size_gb field
        output_path: Target download directory
        parent_widget: Parent tkinter widget for dialogs
        
    Returns:
        (can_proceed, error_message)
    """
    dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
    
    # Get dataset size
    size_gb = dataset.get("size_gb", 0.0)
    num_rows = dataset.get("num_rows", 0)
    
    # If size is unknown and rows are unknown, warn but allow
    if size_gb == 0 and num_rows == 0:
        logger.warning(f"Dataset size unknown for {dataset_name}, allowing download")
        if parent_widget:
            response = _show_dialog_threadsafe(
                parent_widget,
                messagebox.askyesno,
                "Unknown Dataset Size",
                f"The size of '{dataset_name}' could not be determined.\n\n"
                f"This may result in a very large download that exceeds your storage limits.\n\n"
                f"Do you want to proceed anyway?",
                parent=parent_widget,
                icon="warning"
            )
            if response is False:  # Explicit False check (None means dialog failed)
                return False, "Download cancelled by user (unknown size)"
        return True, None
    
    # If we only have row count, estimate size
    if size_gb == 0 and num_rows > 0:
        from .hf_size_detection import estimate_download_size_gb
        modality = dataset.get("modality", "text")
        feature_types = dataset.get("feature_types", [])
        size_gb = estimate_download_size_gb(num_rows, modality, feature_types)
        logger.info(f"Estimated size for {dataset_name}: {size_gb:.2f} GB ({num_rows:,} rows)")
    
    # Check storage cap
    cap_ok, cap_error = check_storage_cap(size_gb)
    if not cap_ok and cap_error:
        logger.warning(f"Dataset {dataset_name} exceeds storage cap: {cap_error}")
        if parent_widget:
            _show_dialog_threadsafe(
                parent_widget,
                messagebox.showerror,
                "Storage Limit Exceeded",
                cap_error,
                parent=parent_widget
            )
        return False, cap_error
    
    # Check disk space
    space_ok, space_error = check_disk_space(output_path, size_gb)
    if not space_ok and space_error:
        logger.warning(f"Insufficient disk space for {dataset_name}: {space_error}")
        if parent_widget:
            _show_dialog_threadsafe(
                parent_widget,
                messagebox.showerror,
                "Insufficient Disk Space",
                space_error,
                parent=parent_widget
            )
        return False, space_error
    
    # All checks passed
    logger.info(f"Download validation passed for {dataset_name} ({size_gb:.2f} GB)")
    return True, None


def show_download_confirmation_dialog(
    dataset: Dict[str, Any],
    parent_widget: Any = None,
) -> bool:
    """
    Show a confirmation dialog with dataset details before download.
    
    Args:
        dataset: Dataset dictionary
        parent_widget: Parent tkinter widget
        
    Returns:
        True if user confirms, False otherwise
    """
    dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
    size_gb = dataset.get("size_gb", 0.0)
    num_rows = dataset.get("num_rows", 0)
    modality = dataset.get("modality", "Unknown")
    
    # Build confirmation message
    msg = f"Download '{dataset_name}'?\n\n"
    
    if num_rows > 0:
        msg += f"Rows: {num_rows:,}\n"
    else:
        msg += "Rows: Unknown\n"
    
    if size_gb > 0:
        msg += f"Size: ~{size_gb:.2f} GB\n"
    elif num_rows > 0:
        from .hf_size_detection import estimate_download_size_gb
        est_size = estimate_download_size_gb(num_rows, modality)
        msg += f"Size: Estimated ~{est_size:.2f} GB\n"
    else:
        msg += "Size: Unknown\n"
    
    msg += f"Type: {modality}\n\n"
    
    # Add warnings
    if dataset.get("gated", False):
        msg += "⚠️ This is a gated dataset. You may need to accept terms on HuggingFace.\n\n"
    if dataset.get("private", False):
        msg += "⚠️ This is a private dataset. You need appropriate access.\n\n"
    
    # Add streaming info
    if dataset.get("max_samples", 0) == 0:
        msg += "The dataset will be downloaded in 100k-sample blocks for efficient training."
    else:
        msg += f"Up to {dataset.get('max_samples', 0):,} samples will be downloaded."
    
    if not parent_widget:
        # No dialog available, default to yes
        return True
    
    result = _show_dialog_threadsafe(
        parent_widget,
        messagebox.askyesno,
        "Confirm Download",
        msg,
        parent=parent_widget
    )
    
    if result is None:
        logger.error("Confirmation dialog failed - proceeding with download")
        return True
    return result
