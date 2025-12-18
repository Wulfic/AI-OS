"""System tray utilities for AI-OS.

This module provides functionality to create and manage a system tray icon
for the AI-OS application. The tray icon allows users to:
- Quickly show/hide the main window
- Access settings
- Exit the application

The implementation uses the pystray library for cross-platform support
(Windows, Linux, macOS).

Example:
    >>> from aios.gui.utils.tray import TrayManager
    >>> 
    >>> manager = TrayManager(root, icon_path="AI-OS.ico")
    >>> manager.create_tray()  # Creates and starts tray icon
    >>> manager.hide_window()  # Hides main window
    >>> manager.show_window()  # Shows main window
    >>> manager.destroy()      # Cleanup on exit
"""

import sys
import threading
from pathlib import Path
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

# Optional imports with graceful degradation
try:
    import pystray  # type: ignore
    from PIL import Image  # type: ignore
    HAS_TRAY_SUPPORT = True
except ImportError:
    pystray = None  # type: ignore
    Image = None  # type: ignore
    HAS_TRAY_SUPPORT = False


class TrayManager:
    """Manages system tray icon and window visibility.
    
    This class encapsulates all system tray functionality including:
    - Creating and managing the tray icon
    - Thread-safe window show/hide operations
    - Tray menu creation and handling
    - Graceful cleanup on exit
    
    The tray icon runs in a separate thread to avoid blocking the main GUI.
    """
    
    def __init__(
        self,
        root: Any,
        icon_path: Optional[Path] = None,
        app_name: str = "AI-OS",
        on_settings: Optional[Callable[[], None]] = None,
    ):
        """Initialize TrayManager.
        
        Args:
            root: Tkinter root window
            icon_path: Path to icon file (.ico or .png)
            app_name: Application name for tray tooltip
            on_settings: Callback for Settings menu item
        """
        self.root = root
        self.icon_path = icon_path
        self.app_name = app_name
        self.on_settings = on_settings
        
        self._tray_icon: Any = None
        self._tray_thread: Optional[threading.Thread] = None
        self._is_visible = True  # Track window visibility state
        
        # Find icon if not provided
        if self.icon_path is None or not self.icon_path.exists():
            self.icon_path = self._find_icon()
    
    def _find_icon(self) -> Optional[Path]:
        """Find the application icon file.
        
        Returns:
            Path to icon file, or None if not found
        """
        try:
            # Get installers directory (same as app.py logic)
            icon_dir = Path(__file__).parent.parent.parent.parent.parent / "installers"
            
            # Try .ico first (Windows native)
            ico_path = icon_dir / "AI-OS.ico"
            if ico_path.exists():
                return ico_path
            
            # Try .png as fallback
            png_path = icon_dir / "AI-OS.png"
            if png_path.exists():
                return png_path
            
            logger.warning(f"Icon not found in {icon_dir}")
            return None
        except Exception as e:
            logger.error(f"Error finding icon: {e}")
            return None
    
    def has_tray_support(self) -> bool:
        """Check if system tray support is available.
        
        Returns:
            True if pystray and PIL are available, False otherwise
        """
        return HAS_TRAY_SUPPORT and self.icon_path is not None
    
    def create_tray(self) -> bool:
        """Create and start the system tray icon.
        
        Returns:
            True if tray was created successfully, False otherwise
        """
        if not self.has_tray_support():
            logger.error("Tray icon creation failed: missing dependencies or icon")
            logger.warning("System tray not supported")
            logger.info("System tray not supported (missing pystray/PIL or icon)")
            return False
        
        try:
            # Load icon image in background thread to avoid blocking
            def _load_and_create():
                try:
                    icon_image = Image.open(str(self.icon_path))  # type: ignore[union-attr]
                    
                    # Create menu
                    menu = pystray.Menu(  # type: ignore[union-attr]
                        pystray.MenuItem(  # type: ignore[union-attr]
                            "Show/Hide",
                            self._on_toggle_window,
                            default=True,
                            visible=True
                        ),
                        pystray.MenuItem(  # type: ignore[union-attr]
                            "Settings",
                            self._on_settings_click,
                            visible=bool(self.on_settings)
                        ),
                        pystray.Menu.SEPARATOR,  # type: ignore[union-attr]
                        pystray.MenuItem(  # type: ignore[union-attr]
                            "Exit",
                            self._on_exit_click
                        )
                    )
                    
                    # Create tray icon
                    self._tray_icon = pystray.Icon(  # type: ignore[union-attr]
                        self.app_name,
                        icon_image,
                        self.app_name,
                        menu
                    )
                    
                    # Run the tray icon (this blocks within this thread)
                    logger.debug("Starting system tray")
                    self._tray_icon.run()  # type: ignore[union-attr]
                except Exception as e:
                    logger.error(f"Tray thread error: {e}", exc_info=True)
            
            # Start tray loading and running in a daemon thread
            logger.debug("Starting system tray thread")
            self._tray_thread = threading.Thread(
                target=_load_and_create,
                daemon=True,
                name="TrayThread"
            )
            self._tray_thread.start()
            logger.debug("System tray thread started")
            
            logger.info("System tray icon creation initiated")
            return True
            
        except Exception as e:
            logger.error(f"Tray icon creation failed: {e}", exc_info=True)
            logger.warning("System tray not supported")
            return False
    
    def _on_toggle_window(self, icon: Any = None, item: Any = None) -> None:
        """Toggle window visibility (tray menu callback).
        
        This is called from the tray thread, so we must use root.after()
        for thread safety.
        """
        try:
            if self._is_visible:
                self.hide_window()
            else:
                self.show_window()
        except Exception as e:
            logger.error(f"Tray click handler failed: {e}", exc_info=True)
    
    def _on_settings_click(self, icon: Any = None, item: Any = None) -> None:
        """Open settings (tray menu callback)."""
        if self.on_settings:
            # Show window first, then switch to settings tab
            self.show_window()
            self.root.after(100, self.on_settings)
    
    def _on_exit_click(self, icon: Any = None, item: Any = None) -> None:
        """Exit application (tray menu callback)."""
        logger.info("Application quit via tray icon")
        # Stop tray icon first
        if self._tray_icon:
            try:
                self._tray_icon.stop()  # type: ignore[union-attr]
            except Exception as e:
                logger.error(f"Quit handler failed: {e}", exc_info=True)
        
        # Quit application (thread-safe)
        self.root.after(0, self._quit_app)
    
    def _quit_app(self) -> None:
        """Quit the application (called in main thread)."""
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
    
    def show_window(self) -> None:
        """Show the main window (thread-safe)."""
        def _show():
            try:
                self.root.deiconify()  # Restore from iconified state
                self.root.lift()       # Bring to front
                self.root.focus_force()  # Give focus
                self._is_visible = True
            except Exception as e:
                logger.error(f"Error showing window: {e}")
        
        # Schedule in main thread
        self.root.after(0, _show)
    
    def hide_window(self) -> None:
        """Hide the main window (thread-safe)."""
        def _hide():
            try:
                self.root.withdraw()  # Hide window
                self._is_visible = False
            except Exception as e:
                logger.error(f"Error hiding window: {e}")
        
        # Schedule in main thread
        self.root.after(0, _hide)
    
    def destroy(self) -> None:
        """Clean up tray resources.
        
        This method is designed to be non-blocking. If the tray icon
        doesn't stop within a reasonable time, we continue anyway.
        """
        if not HAS_TRAY_SUPPORT:
            return
        
        # Stop the tray icon - this signals the tray thread to exit
        if self._tray_icon:
            try:
                # pystray's stop() is generally fast but we wrap it just in case
                self._tray_icon.stop()  # type: ignore[union-attr]
                logger.debug("Tray icon stop called")
            except Exception as e:
                logger.error(f"Error stopping tray icon: {e}")
        
        # Wait for tray thread to finish with timeout
        # Reduced timeout from 2.0 to 1.0 for faster shutdown
        if self._tray_thread and self._tray_thread.is_alive():
            try:
                self._tray_thread.join(timeout=1.0)
                if self._tray_thread.is_alive():
                    logger.warning("Tray thread did not stop within timeout - continuing anyway")
                else:
                    logger.debug("Tray thread stopped successfully")
            except Exception as e:
                logger.error(f"Error joining tray thread: {e}")
        
        # Clear references
        self._tray_icon = None
        self._tray_thread = None
        
        logger.info("Tray manager destroyed")
    
    def is_visible(self) -> bool:
        """Check if main window is currently visible.
        
        Returns:
            True if window is visible, False if hidden
        """
        return self._is_visible


def has_tray_support() -> bool:
    """Check if system tray support is available globally.
    
    Returns:
        True if pystray and PIL are installed, False otherwise
    """
    return HAS_TRAY_SUPPORT


# Module summary docstring for reference
r"""
Module Functions:
-----------------
- has_tray_support() -> bool
  Check if pystray and PIL are available

- TrayManager class
  Main interface for tray icon management

TrayManager Methods:
-------------------
- __init__(root, icon_path, app_name, on_settings)
  Initialize tray manager

- create_tray() -> bool
  Create and start the tray icon

- show_window() -> None
  Show the main window (thread-safe)

- hide_window() -> None
  Hide the main window (thread-safe)

- destroy() -> None
  Clean up tray resources

- is_visible() -> bool
  Check if window is currently visible

Dependencies:
-------------
- pystray: System tray icon library
- PIL (Pillow): Image handling
- threading: Run tray in separate thread

Cross-Platform:
---------------
- Windows: Full support via Windows API
- Linux: Support via AppIndicator/StatusNotifier
- macOS: Support via NSStatusBar
- Graceful degradation if pystray not available

Thread Safety:
--------------
- Tray icon runs in separate thread
- All window operations use root.after() for thread safety
- Proper cleanup on exit
"""
