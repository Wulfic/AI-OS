"""Test GUI initialization to find the issue."""
import traceback
import sys

try:
    print("1. Importing tkinter...")
    import tkinter as tk
    print("   ✓ tkinter imported")
    
    print("2. Importing LogRouter...")
    from aios.gui.services import LogRouter, LogCategory
    print(f"   ✓ LogRouter imported, categories: {[c.value for c in LogCategory]}")
    
    print("3. Importing DebugPanel...")
    from aios.gui.components import DebugPanel
    print("   ✓ DebugPanel imported")
    
    print("4. Creating tkinter root...")
    root = tk.Tk()
    print("   ✓ root created")
    
    print("5. Creating DebugPanel...")
    tab = tk.Frame(root)
    debug_panel = DebugPanel(tab)
    print("   ✓ DebugPanel created")
    
    print("6. Creating LogRouter...")
    log_router = LogRouter()
    print("   ✓ LogRouter created")
    
    print("7. Registering handlers...")
    for category in LogCategory:
        if category != LogCategory.DATASET:
            log_router.register_handler(category, lambda msg, cat=category: debug_panel.write(msg, cat.value))
        else:
            log_router.register_handler(category, lambda msg: debug_panel.write(msg, "debug"))
    print("   ✓ Handlers registered")
    
    print("8. Testing log routing...")
    log_router.log("Test message", LogCategory.DEBUG)
    print("   ✓ Log routing works")
    
    print("9. Importing full app...")
    from aios.gui.app import AiosTkApp
    print("   ✓ App imported")
    
    print("10. Creating app...")
    root2 = tk.Tk()
    app = AiosTkApp(root2)
    print("   ✓ App created successfully!")
    
    print("\n✅ ALL TESTS PASSED")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
