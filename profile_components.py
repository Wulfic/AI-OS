"""Find which component import is slow."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

start = time.time()
last = start

def log_time(msg):
    global last
    now = time.time()
    duration = now - last
    total = now - start
    if duration > 0.1:  # Only show slow imports
        print(f"[{duration:6.3f}s] {msg:<60} (total: {total:.3f}s)")
    last = now

print("=" * 90)
print("COMPONENT IMPORT PROFILING (only showing imports > 0.1s)")
print("=" * 90)

log_time("START")

# Import each component one by one
components = [
    ("OutputPanel", "aios.gui.components.output_panel"),
    ("ChatPanel", "aios.gui.components.chat_panel"),
    ("HelpPanel", "aios.gui.components.help_panel.panel_main"),
    ("RichChatPanel", "aios.gui.components.rich_chat_panel"),
    ("HRMTrainingPanel", "aios.gui.components.hrm_training_panel"),
    ("EvaluationPanel", "aios.gui.components.evaluation_panel"),
    ("MessageParser", "aios.gui.components.message_parser"),
    ("GoalsPanel", "aios.gui.components.goals_panel"),
    ("DatasetsPanel", "aios.gui.components.datasets_panel"),
    ("DatasetBuilderPanel", "aios.gui.components.dataset_builder_panel"),
    ("DatasetDownloadPanel", "aios.gui.components.dataset_download_panel"),
    ("CrawlPanel", "aios.gui.components.crawl_panel"),
    ("ResourcesPanel", "aios.gui.components.resources_panel"),
    ("StatusBar", "aios.gui.components.status_bar"),
    ("DebugPanel", "aios.gui.components.debug_panel"),
    ("TrainingPanel", "aios.gui.components.training_panel"),
    ("BrainsPanel", "aios.gui.components.brains_panel"),
    ("SettingsPanel", "aios.gui.components.settings_panel"),
    ("SubbrainsManagerPanel", "aios.gui.components.subbrains_manager_panel"),
    ("MCPManagerPanel", "aios.gui.components.mcp_manager_panel"),
]

for name, module in components:
    try:
        log_time(f"Importing {name}...")
        __import__(module)
        log_time(f"✓ {name}")
    except Exception as e:
        log_time(f"✗ {name}: {str(e)[:50]}")

print("=" * 90)
print(f"TOTAL TIME: {time.time() - start:.3f}s")
print("=" * 90)
