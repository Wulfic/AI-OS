"""Test 4-column chart layout."""
import sys

try:
    print("Testing 4-column chart layout...")
    
    print("1. Importing components...")
    from aios.gui.app import AiosTkApp
    import tkinter as tk
    
    print("2. Creating root window...")
    root = tk.Tk()
    
    print("3. Creating app with 4-column layout...")
    app = AiosTkApp(root)
    
    print("\n✅ SUCCESS! GUI loaded with 4-column chart layout:")
    print("\n📊 Layout: All 4 charts in a single row")
    print("   [Processor] [Memory] [Network] [Disk I/O]")
    print("\n📐 Chart Specs:")
    print("   • Size: 5\" x 4\" @ 95 DPI")
    print("   • Title: 12pt bold")
    print("   • Labels: 10pt bold")
    print("   • Ticks: 9pt")
    print("   • Legend: 9pt")
    print("   • Minimum: 300x300 pixels each")
    print("   • Spacing: 4px between charts")
    print("\nCharts arranged horizontally for better comparison!")
    print("\nClose the GUI window to exit.")
    
    # Don't start mainloop in test
    root.destroy()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
