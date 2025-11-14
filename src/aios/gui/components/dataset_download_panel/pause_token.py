"""Download pause token with resume notifications."""

from __future__ import annotations

import threading
from typing import Callable


class PauseToken:
    """Cooperative pause primitive used by dataset downloads."""

    def __init__(self) -> None:
        self._pause_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()

    def set(self) -> None:
        """Signal that the download should pause."""
        self._resume_event.clear()
        self._pause_event.set()

    def clear(self) -> None:
        """Signal that the download may resume."""
        self._pause_event.clear()
        self._resume_event.set()

    def is_set(self) -> bool:
        """Return True if a pause is requested."""
        return self._pause_event.is_set()

    def wait_for_resume(self, should_cancel: Callable[[], bool], timeout: float = 5.0) -> bool:
        """Block until the download may resume or cancellation requested.

        Args:
            should_cancel: Callback to check if caller wishes to cancel.
            timeout: Maximum seconds to wait per iteration before rechecking.

        Returns:
            True if resumed, False if cancelled while waiting.
        """
        while self.is_set():
            if should_cancel():
                return False
            resumed = self._resume_event.wait(timeout)
            if resumed and not self.is_set():
                return True
        return True

    # Make PauseToken quack like threading.Event for stream manager compatibility
    def wait(self, timeout: float | None = None) -> bool:
        return self._pause_event.wait(timeout)


__all__ = ["PauseToken"]
