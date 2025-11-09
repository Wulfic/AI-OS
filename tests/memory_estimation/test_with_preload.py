"""Pre-load all libraries to avoid import interrupts during tests."""

import sys
import os

# Set minimal logging BEFORE importing anything
os.environ["AIOS_MINIMAL_LOGGING"] = "1"

print("Pre-loading libraries (this may take 30-60 seconds)...", flush=True)

# Pre-load all heavy libraries
try:
    import torch
    print("  ✓ PyTorch loaded", flush=True)
    
    import transformers
    from transformers import AutoTokenizer, AutoModel
    print("  ✓ Transformers loaded", flush=True)
    
    import numpy as np
    print("  ✓ NumPy loaded", flush=True)
    
    import safetensors
    print("  ✓ Safetensors loaded", flush=True)
    
    # Import our training modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from aios.cli.hrm_hf.train_actv1 import train_actv1_impl
    print("  ✓ AIOS HRM modules loaded", flush=True)
    
    print("All libraries pre-loaded successfully!\n", flush=True)
    
except Exception as e:
    print(f"Warning: Failed to pre-load some libraries: {e}", flush=True)
    print("Continuing anyway...\n", flush=True)

# Now run the actual test
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_with_preload.py <tokenizer_name>")
        sys.exit(1)
    
    tokenizer_name = sys.argv[1]
    
    # Import and run the test
    from test_single_tokenizer import main as run_test
    
    # Override sys.argv for the test script
    sys.argv = ["test_single_tokenizer.py", tokenizer_name]
    
    run_test()
