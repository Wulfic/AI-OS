"""Reusable GUI components for AI-OS Tk interface.

Each component encapsulates a specific feature (output, chat, goals) and exposes
simple methods the main app can call. Components avoid hard dependencies on the
rest of the app and interact via small callables passed in the constructor.
"""

from __future__ import annotations

__all__ = [
    "OutputPanel",
    "ChatPanel",
    "RichChatPanel",
    "GoalsPanel",
    "DatasetsPanel",
    "DatasetBuilderPanel",
    "DatasetDownloadPanel",
    "CrawlPanel",
    "ResourcesPanel",
    "StatusBar",
    "DebugPanel",
    "TrainingPanel",
    "HRMTrainingPanel",
    "EvaluationPanel",
    "BrainsPanel",
    "SettingsPanel",
    "SubbrainsManagerPanel",
    "MCPManagerPanel",
    "HelpPanel",
    "MessageParser",
    "CodeBlockWidget",
    "ImageWidget",
    "VideoWidget",
    "LinkWidget",
]

from .output_panel import OutputPanel  # noqa: E402
from .chat_panel import ChatPanel  # noqa: E402
# Help panel (new)
from .help_panel.panel_main import HelpPanel  # noqa: E402
from .rich_chat_panel import RichChatPanel  # noqa: E402
from .hrm_training_panel import HRMTrainingPanel  # noqa: E402
# EvaluationPanel uses lazy import to avoid loading heavy lm_eval dependencies at startup
# from .evaluation_panel import EvaluationPanel  # noqa: E402
from .message_parser import MessageParser  # noqa: E402
from .rich_message_widgets import (  # noqa: E402
    CodeBlockWidget,
    ImageWidget,
    VideoWidget,
    LinkWidget,
)
from .goals_panel import GoalsPanel  # noqa: E402
from .datasets_panel import DatasetsPanel  # noqa: E402
from .dataset_builder_panel import DatasetBuilderPanel  # noqa: E402
from .dataset_download_panel import DatasetDownloadPanel  # noqa: E402
from .crawl_panel import CrawlPanel  # noqa: E402
from .resources_panel import ResourcesPanel  # noqa: E402
from .status_bar import StatusBar  # noqa: E402
from .debug_panel import DebugPanel  # noqa: E402
from .training_panel import TrainingPanel  # noqa: E402
from .brains_panel import BrainsPanel  # noqa: E402
from .settings_panel import SettingsPanel  # noqa: E402
from .subbrains_manager_panel import SubbrainsManagerPanel  # noqa: E402
from .mcp_manager_panel import MCPManagerPanel  # noqa: E402


def __getattr__(name: str):
    """Lazy import for heavy components like EvaluationPanel."""
    if name == "EvaluationPanel":
        from .evaluation_panel import EvaluationPanel
        return EvaluationPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
