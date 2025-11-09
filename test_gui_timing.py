"""Test script to measure GUI startup timing.

Run this to see timing information printed to console.
Close the GUI window manually when it appears.
"""
import sys
from pathlib import Path

# Ensure project is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("GUI STARTUP TIMING TEST")
print("=" * 80)
print("Watch for [TIMING] and [PANEL TIMING] messages below...")
print("Close the GUI window manually when it appears.")
print("=" * 80)
print()

# Import and run
from aios.gui import run

run()
