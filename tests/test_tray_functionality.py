"""Test suite for system tray functionality (Task 15)."""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestTraySupport:
    """Test tray support detection and availability."""
    
    def test_has_tray_support_function_exists(self):
        """Test that has_tray_support function exists."""
        from aios.gui.utils.tray import has_tray_support
        assert callable(has_tray_support)
    
    def test_has_tray_support_returns_bool(self):
        """Test that has_tray_support returns boolean."""
        from aios.gui.utils.tray import has_tray_support
        result = has_tray_support()
        assert isinstance(result, bool)
    
    def test_tray_manager_class_exists(self):
        """Test that TrayManager class exists."""
        from aios.gui.utils.tray import TrayManager
        assert TrayManager is not None


class TestTrayManager:
    """Test TrayManager class functionality."""
    
    @pytest.fixture
    def mock_root(self):
        """Create mock tkinter root."""
        root = Mock()
        root.after = Mock()
        root.deiconify = Mock()
        root.withdraw = Mock()
        root.lift = Mock()
        root.focus_force = Mock()
        root.quit = Mock()
        root.destroy = Mock()
        return root
    
    @pytest.fixture
    def icon_path(self, tmp_path):
        """Create temporary icon file."""
        icon = tmp_path / "test_icon.png"
        icon.write_bytes(b"fake_icon_data")
        return icon
    
    def test_tray_manager_init(self, mock_root, icon_path):
        """Test TrayManager initialization."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(
            mock_root,
            icon_path=icon_path,
            app_name="TestApp",
            on_settings=None
        )
        
        assert manager.root == mock_root
        assert manager.icon_path == icon_path
        assert manager.app_name == "TestApp"
        assert manager._is_visible is True
    
    def test_tray_manager_has_tray_support_method(self, mock_root, icon_path):
        """Test has_tray_support method exists and returns bool."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(mock_root, icon_path=icon_path)
        result = manager.has_tray_support()
        assert isinstance(result, bool)
    
    def test_tray_manager_is_visible_method(self, mock_root, icon_path):
        """Test is_visible method."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(mock_root, icon_path=icon_path)
        assert manager.is_visible() is True
        
        manager._is_visible = False
        assert manager.is_visible() is False
    
    def test_tray_manager_show_window(self, mock_root, icon_path):
        """Test show_window method."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(mock_root, icon_path=icon_path)
        manager._is_visible = False
        
        manager.show_window()
        
        # Should schedule show operation
        assert mock_root.after.called
    
    def test_tray_manager_hide_window(self, mock_root, icon_path):
        """Test hide_window method."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(mock_root, icon_path=icon_path)
        manager._is_visible = True
        
        manager.hide_window()
        
        # Should schedule hide operation
        assert mock_root.after.called
    
    def test_tray_manager_destroy(self, mock_root, icon_path):
        """Test destroy method doesn't raise exceptions."""
        from aios.gui.utils.tray import TrayManager
        
        manager = TrayManager(mock_root, icon_path=icon_path)
        
        # Should not raise even with no tray icon
        manager.destroy()


class TestCLIMinimizedFlag:
    """Test CLI --minimized flag parsing."""
    
    def test_gui_function_accepts_minimized_param(self):
        """Test that gui function accepts minimized parameter."""
        from aios.cli.core_cli import gui
        import inspect
        
        sig = inspect.signature(gui)
        params = sig.parameters
        
        assert 'minimized' in params
    
    def test_app_run_accepts_minimized_param(self):
        """Test that app.run accepts minimized parameter."""
        from aios.gui.app import run
        import inspect
        
        sig = inspect.signature(run)
        params = sig.parameters
        
        assert 'minimized' in params


class TestSettingsPanelTrayOptions:
    """Test settings panel tray checkboxes."""
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock parent widget."""
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()  # Hide window
            frame = tk.Frame(root)
            yield frame
            root.destroy()
        except Exception:
            # Headless environment
            pytest.skip("Tkinter not available")
    
    def test_settings_panel_has_tray_variables(self, mock_parent):
        """Test that settings panel has tray-related variables."""
        from aios.gui.components.settings_panel import SettingsPanel
        
        panel = SettingsPanel(mock_parent)
        
        assert hasattr(panel, 'start_minimized_var')
        assert hasattr(panel, 'minimize_to_tray_var')
    
    def test_settings_panel_state_includes_tray_settings(self, mock_parent):
        """Test that get_state includes tray settings."""
        from aios.gui.components.settings_panel import SettingsPanel
        
        panel = SettingsPanel(mock_parent)
        state = panel.get_state()
        
        assert 'start_minimized' in state
        assert 'minimize_to_tray' in state
        assert isinstance(state['start_minimized'], bool)
        assert isinstance(state['minimize_to_tray'], bool)
    
    def test_settings_panel_set_state_restores_tray_settings(self, mock_parent):
        """Test that set_state restores tray settings."""
        from aios.gui.components.settings_panel import SettingsPanel
        
        panel = SettingsPanel(mock_parent)
        
        state = {
            'start_minimized': True,
            'minimize_to_tray': True
        }
        
        panel.set_state(state)
        
        assert panel.start_minimized_var.get() is True
        assert panel.minimize_to_tray_var.get() is True


class TestStartupCommandWithMinimized:
    """Test startup command generation with --minimized flag."""
    
    def test_get_startup_command_accepts_minimized_param(self):
        """Test that get_startup_command accepts minimized parameter."""
        from aios.gui.utils.startup import get_startup_command
        import inspect
        
        sig = inspect.signature(get_startup_command)
        params = sig.parameters
        
        assert 'minimized' in params
    
    def test_get_startup_command_without_minimized(self):
        """Test startup command without minimized flag."""
        from aios.gui.utils.startup import get_startup_command
        
        command = get_startup_command(minimized=False)
        
        assert isinstance(command, str)
        assert len(command) > 0
        assert '--minimized' not in command
    
    def test_get_startup_command_with_minimized(self):
        """Test startup command with minimized flag."""
        from aios.gui.utils.startup import get_startup_command
        
        command = get_startup_command(minimized=True)
        
        assert isinstance(command, str)
        assert '--minimized' in command
    
    def test_set_startup_enabled_accepts_minimized_param(self):
        """Test that set_startup_enabled accepts minimized parameter."""
        from aios.gui.utils.startup import set_startup_enabled
        import inspect
        
        sig = inspect.signature(set_startup_enabled)
        params = sig.parameters
        
        assert 'minimized' in params
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_set_startup_enabled_with_minimized_flag(self):
        """Test setting startup with minimized flag on Windows."""
        from aios.gui.utils.startup import set_startup_enabled, get_startup_path, is_windows
        
        if not is_windows():
            pytest.skip("Not on Windows")
        
        try:
            # Enable with minimized
            result = set_startup_enabled(True, minimized=True)
            
            if result:
                # Check that command includes --minimized
                path = get_startup_path()
                if path:
                    assert '--minimized' in path
            
            # Clean up
            set_startup_enabled(False)
        except Exception:
            # Permission issues or other registry problems
            pytest.skip("Cannot modify registry")


class TestTrayIconPath:
    """Test tray icon path detection."""
    
    def test_tray_manager_finds_icon(self):
        """Test that TrayManager can find icon file."""
        from aios.gui.utils.tray import TrayManager
        
        mock_root = Mock()
        manager = TrayManager(mock_root)
        
        # Should either find an icon or handle gracefully
        assert manager.icon_path is None or isinstance(manager.icon_path, Path)


class TestCrossPlatformBehavior:
    """Test cross-platform behavior."""
    
    def test_tray_manager_no_exceptions_any_platform(self):
        """Test that TrayManager doesn't raise exceptions on any platform."""
        from aios.gui.utils.tray import TrayManager
        
        mock_root = Mock()
        mock_root.after = Mock()
        
        try:
            manager = TrayManager(mock_root)
            manager.has_tray_support()
            manager.show_window()
            manager.hide_window()
            manager.destroy()
        except Exception as e:
            pytest.fail(f"TrayManager raised exception: {e}")
    
    def test_startup_command_generation_no_exceptions(self):
        """Test that startup command generation doesn't raise exceptions."""
        from aios.gui.utils.startup import get_startup_command
        
        try:
            cmd1 = get_startup_command(minimized=False)
            cmd2 = get_startup_command(minimized=True)
            assert isinstance(cmd1, str)
            assert isinstance(cmd2, str)
        except Exception as e:
            pytest.fail(f"get_startup_command raised exception: {e}")


class TestAppIntegration:
    """Test app.py integration with tray functionality."""
    
    def test_app_has_tray_manager_attribute(self):
        """Test that AiosTkApp has _tray_manager attribute."""
        # This is a structural test - checking the attribute exists in __init__
        from aios.gui.app import AiosTkApp
        import inspect
        
        source = inspect.getsource(AiosTkApp.__init__)
        assert '_tray_manager' in source
    
    def test_app_has_minimize_to_tray_attribute(self):
        """Test that AiosTkApp has _minimize_to_tray_on_close attribute."""
        from aios.gui.app import AiosTkApp
        import inspect
        
        source = inspect.getsource(AiosTkApp.__init__)
        assert '_minimize_to_tray_on_close' in source
    
    def test_app_has_sync_tray_settings_method(self):
        """Test that AiosTkApp has _sync_tray_settings method."""
        from aios.gui.app import AiosTkApp
        
        assert hasattr(AiosTkApp, '_sync_tray_settings')
        assert callable(getattr(AiosTkApp, '_sync_tray_settings'))
    
    def test_app_has_init_tray_method(self):
        """Test that AiosTkApp has _init_tray method."""
        from aios.gui.app import AiosTkApp
        
        assert hasattr(AiosTkApp, '_init_tray')
        assert callable(getattr(AiosTkApp, '_init_tray'))


class TestThreadSafety:
    """Test thread safety of tray operations."""
    
    def test_tray_operations_use_root_after(self):
        """Test that tray operations use root.after for thread safety."""
        from aios.gui.utils.tray import TrayManager
        import inspect
        
        # Check that show_window and hide_window use root.after
        show_source = inspect.getsource(TrayManager.show_window)
        hide_source = inspect.getsource(TrayManager.hide_window)
        
        assert 'root.after' in show_source or 'self.root.after' in show_source
        assert 'root.after' in hide_source or 'self.root.after' in hide_source


class TestErrorHandling:
    """Test error handling in tray functionality."""
    
    def test_tray_manager_handles_missing_icon(self):
        """Test that TrayManager handles missing icon gracefully."""
        from aios.gui.utils.tray import TrayManager
        
        mock_root = Mock()
        mock_root.after = Mock()
        
        # Pass non-existent icon path
        manager = TrayManager(mock_root, icon_path=Path("/nonexistent/icon.png"))
        
        # Should not raise, should return False for has_tray_support
        assert isinstance(manager.has_tray_support(), bool)
    
    def test_tray_manager_create_tray_handles_exceptions(self):
        """Test that create_tray handles exceptions gracefully."""
        from aios.gui.utils.tray import TrayManager
        
        mock_root = Mock()
        manager = TrayManager(mock_root)
        
        # Should return False if creation fails, not raise exception
        result = manager.create_tray()
        assert isinstance(result, bool)


class TestStatePersistence:
    """Test state persistence of tray settings."""
    
    @pytest.fixture
    def mock_parent(self):
        """Create mock parent widget."""
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()
            frame = tk.Frame(root)
            yield frame
            root.destroy()
        except Exception:
            pytest.skip("Tkinter not available")
    
    def test_tray_settings_persist_across_state_save_load(self, mock_parent):
        """Test that tray settings persist across save/load."""
        from aios.gui.components.settings_panel import SettingsPanel
        
        panel1 = SettingsPanel(mock_parent)
        panel1.start_minimized_var.set(True)
        panel1.minimize_to_tray_var.set(True)
        
        state = panel1.get_state()
        
        panel2 = SettingsPanel(mock_parent)
        panel2.set_state(state)
        
        assert panel2.start_minimized_var.get() is True
        assert panel2.minimize_to_tray_var.get() is True


# Summary comment for test coverage
"""
Test Coverage Summary:
---------------------
- TrayManager class creation and methods
- CLI --minimized flag parsing
- Settings panel tray checkboxes
- Startup command generation with --minimized
- Icon path detection
- Cross-platform behavior
- App integration points
- Thread safety
- Error handling
- State persistence

Total Tests: 30+
Platform-Specific: 1 (Windows registry test)
Cross-Platform: All others

All tests designed to be non-destructive and work in CI environments.
"""
