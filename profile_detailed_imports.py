"""Detailed import profiling to find the bottleneck."""
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
    print(f"[{duration:6.3f}s] {msg:<60} (total: {total:.3f}s)")
    last = now

print("=" * 90)
print("DETAILED IMPORT PROFILING")
print("=" * 90)

log_time("START")

# Test individual imports to find the slow one
log_time("Importing tkinter...")
import tkinter as tk
from tkinter import ttk
log_time("✓ tkinter")

log_time("Importing aios.gui.app.app_main...")
from aios.gui.app import app_main
log_time("✓ app_main")

log_time("Importing panel_setup (this imports all components)...")
from aios.gui.app import panel_setup
log_time("✓ panel_setup")

log_time("Importing components...")
from aios.gui import components
log_time("✓ components")

log_time("Importing evaluation_panel...")
try:
    from aios.gui.components import evaluation_panel
    log_time("✓ evaluation_panel")
except Exception as e:
    log_time(f"✗ evaluation_panel: {e}")

log_time("Importing EvaluationPanel directly...")
try:
    from aios.gui.components.evaluation_panel import EvaluationPanel
    log_time("✓ EvaluationPanel")
except Exception as e:
    log_time(f"✗ EvaluationPanel: {e}")

log_time("Importing aios.core.evaluation...")
try:
    from aios.core import evaluation
    log_time("✓ aios.core.evaluation")
except Exception as e:
    log_time(f"✗ aios.core.evaluation: {e}")

print("=" * 90)
print(f"TOTAL TIME: {time.time() - start:.3f}s")
print("=" * 90)
