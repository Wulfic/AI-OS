"""Find what OutputPanel imports that's slow."""
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
print("TRACING OUTPUT_PANEL IMPORT CHAIN")
print("=" * 90)

log_time("START")

# Check what gets imported BEFORE OutputPanel
log_time("Checking sys.modules before import...")
modules_before = set(sys.modules.keys())
log_time(f"Modules loaded: {len(modules_before)}")

log_time("Importing OutputPanel...")
from aios.gui.components.output_panel import OutputPanel
log_time("✓ OutputPanel imported")

log_time("Checking new modules...")
modules_after = set(sys.modules.keys())
new_modules = modules_after - modules_before
log_time(f"New modules loaded: {len(new_modules)}")

# Show the heavy ones
heavy_modules = [m for m in new_modules if any(x in m.lower() for x in ['torch', 'transform', 'numpy', 'tensor', 'cuda', 'lm_eval', 'evaluate'])]
if heavy_modules:
    print("\nHeavy modules imported:")
    for m in sorted(heavy_modules)[:20]:
        print(f"  - {m}")

print("=" * 90)
print(f"TOTAL TIME: {time.time() - start:.3f}s")
print("=" * 90)
