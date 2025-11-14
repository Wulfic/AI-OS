from __future__ import annotations

class DebugMixin:
    def _debug_write(self, msg: str) -> None:
        try:
            self.post_to_ui(lambda: self.debug_panel.write(msg))  # type: ignore[attr-defined]
        except Exception:
            pass

    def _debug_set_error(self, text: str) -> None:
        try:
            self.post_to_ui(lambda: self.debug_panel.set_error(text))  # type: ignore[attr-defined]
        except Exception:
            pass
