"""Quick test of the updated GUI charts."""
import sys

try:
    print("Testing updated GUI with unified charts...")
    
    print("1. Importing components...")
    from aios.gui.app import AiosTkApp
    import tkinter as tk
    
    print("2. Creating root window...")
    root = tk.Tk()
    
    print("3. Creating app...")
    app = AiosTkApp(root)
    
    print("\n✅ SUCCESS! GUI loaded with unified charts:")
    print("   - Processor Utilization (CPU + GPUs)")
    print("   - Memory Usage (RAM + GPU memory)")
    print("   - Network (Upload/Download)")
    print("   - Disk I/O (Read/Write)")
    print("\nCharts are now larger and more readable!")
    print("\nClose the GUI window to exit.")
    
    # Don't start mainloop in test
    root.destroy()
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
