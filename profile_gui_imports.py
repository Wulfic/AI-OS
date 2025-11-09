"""Profile GUI startup to find slow imports."""
import sys
import time
from pathlib import Path

# Ensure project is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("GUI IMPORT PROFILING")
print("=" * 80)

start = time.time()
last = start

def log_time(msg):
    global last
    now = time.time()
    print(f"[{now - last:.3f}s] {msg} (total: {now - start:.3f}s)")
    last = now

log_time("Starting import profiling...")

log_time("Importing aios.gui.app...")
from aios.gui import app
log_time("✓ aios.gui.app imported")

log_time("Importing AiosTkApp...")
from aios.gui.app import AiosTkApp
log_time("✓ AiosTkApp imported")

log_time("Importing run...")
from aios.gui import run
log_time("✓ run imported")

print("=" * 80)
print(f"TOTAL IMPORT TIME: {time.time() - start:.3f}s")
print("=" * 80)
