"""GPU monitoring stub - imports from parent directory."""

from __future__ import annotations

# Import the actual GPU monitor from parent directory
try:
    from ..gpu_monitor import create_gpu_monitor
except ImportError:
    # Fallback if gpu_monitor doesn't exist
    def create_gpu_monitor(*args, **kwargs):
        """Stub GPU monitor that does nothing."""
        class DummyMonitor:
            def start_monitoring(self, *args, **kwargs):
                pass
            def stop_monitoring(self):
                pass
            def get_summary(self):
                return {}
        return DummyMonitor()

__all__ = ['create_gpu_monitor']
