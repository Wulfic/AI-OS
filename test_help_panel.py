#!/usr/bin/env python3
"""Quick test to verify HelpPanel link handler attachment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import tkinter as tk
from aios.gui.components.help_panel.panel_main import HelpPanel

# Create minimal GUI
root = tk.Tk()
root.title("Help Panel Test")
root.geometry("1000x700")

# Create HelpPanel
help_panel = HelpPanel(root)

print("\n" + "="*60)
print("Help Panel Test Started")
print("="*60)
print(f"HelpPanel created: {help_panel}")
print(f"html_view: {help_panel.html_view}")
print("\nNow open the Help panel and click a link.")
print("Watch this console for debug output.")
print("="*60 + "\n")

root.mainloop()
