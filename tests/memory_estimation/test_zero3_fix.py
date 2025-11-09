"""Quick test to verify DeepSpeed ZeRO-3 works on single GPU."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set environment for C: drive storage
os.environ["HF_HOME"] = "C:\\Users\\tyler\\.cache\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\tyler\\.cache\\huggingface\\transformers"
os.environ["AIOS_MINIMAL_LOGGING"] = "1"

from test_harness_real import RealTestConfig, run_real_training_test

# Test ZeRO-3 with smallest context
test_config = RealTestConfig(
    model_name="artifacts/hf_implant/base_model",
    h_layers=3,
    l_layers=3,
    hidden_size=512,
    num_heads=8,
    context_size=128,
    batch_size=1,
    use_moe=False,
    gradient_checkpointing=False,
    use_amp=False,
    deepspeed_stage=3,  # ZeRO-3
    steps=3,  # Just 3 steps to verify it works
    device="cuda:1",
    model_cache_dir="C:\\Users\\tyler\\AppData\\Local\\Temp\\aios_opt_test_cache",
    temp_dir_base="C:\\Users\\tyler\\AppData\\Local\\Temp\\aios_opt_test_temp",
)

print("=" * 80)
print("TESTING DEEPSPEED ZERO-3 FIX")
print("=" * 80)
print("Configuration: ZeRO-3, ctx=128, 3 steps, no other optimizations")
print("Expected: No device mismatch errors\n")

result = run_real_training_test(test_config, verbose=True)

print("\n" + "=" * 80)
if result.success:
    print(f"✅ SUCCESS! VRAM: {result.actual_vram_bytes / (1024**3):.2f} GB")
    print("ZeRO-3 is working correctly on single GPU!")
else:
    print(f"❌ FAILED: {result.error_message}")
    print("ZeRO-3 fix did not resolve the issue")
print("=" * 80)
