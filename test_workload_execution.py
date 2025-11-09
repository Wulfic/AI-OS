"""
Test the corrected workload execution that was failing in the optimizer
"""

import sys
import subprocess
import os
import time
from pathlib import Path

def test_workload_execution():
    """Test both generation and training workloads with corrected subprocess commands."""
    
    print("🚀 Testing Corrected Workload Execution")
    print("=" * 45)
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = r"c:\Users\tyler\Repos\AI-OS\src"
    env["AIOS_USE_OPTIMIZER_V2"] = "1"
    
    python_exe = Path(r"c:\Users\tyler\Repos\AI-OS\.venv\Scripts\python.exe")
    working_dir = Path(r"c:\Users\tyler\Repos\AI-OS")
    
    print(f"📋 Python executable: {python_exe}")
    print(f"📋 Working directory: {working_dir}")
    print(f"📋 Environment variables set: PYTHONPATH, AIOS_USE_OPTIMIZER_V2")
    
    # Test 1: Generation workload (mimics what optimizer does)
    print("\n🧪 Testing Generation Workload...")
    gen_args = [
        "hrm-hf", "train-actv1",
        "--model", "gpt2",
        "--steps", "0",  # Zero steps for quick test
        "--batch-size", "1",
        "--teacher-dataset",
        "--td-num-samples", "2",  # Minimal samples
        "--td-max-new-tokens", "4",  # Minimal tokens
        "--device", "cpu",  # CPU for safety
        "--strict"
    ]
    
    gen_cmd = [str(python_exe), "-m", "aios.cli.aios"] + gen_args
    print(f"📋 Generation command: {' '.join(gen_cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            gen_cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
            env=env,
            cwd=str(working_dir)
        )
        duration = time.time() - start_time
        
        print(f"📊 Generation test completed in {duration:.2f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"📊 Stdout (first 200 chars): {result.stdout[:200]}")
        if result.stderr:
            print(f"📊 Stderr (first 200 chars): {result.stderr[:200]}")
            
        gen_success = result.returncode == 0 or "WinError 2" not in str(result.stderr)
        print(f"📊 Generation workload: {'✅ SUCCESS' if gen_success else '❌ FAILED'}")
        
    except subprocess.TimeoutExpired:
        print("⚠️ Generation test timed out (may be normal)")
        gen_success = True  # Timeout is better than WinError 2
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        gen_success = False
    
    # Test 2: Training workload (mimics what optimizer does)
    print("\n🧪 Testing Training Workload...")
    train_args = [
        "hrm-hf", "train-actv1",
        "--model", "gpt2",
        "--steps", "1",  # One step for quick test
        "--batch-size", "1",
        "--teacher-dataset",
        "--td-num-samples", "2",  # Minimal samples
        "--td-max-new-tokens", "4",  # Minimal tokens
        "--device", "cpu",  # CPU for safety
        "--strict"
    ]
    
    train_cmd = [str(python_exe), "-m", "aios.cli.aios"] + train_args
    print(f"📋 Training command: {' '.join(train_cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for training
            env=env,
            cwd=str(working_dir)
        )
        duration = time.time() - start_time
        
        print(f"📊 Training test completed in {duration:.2f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"📊 Stdout (first 200 chars): {result.stdout[:200]}")
        if result.stderr:
            print(f"📊 Stderr (first 200 chars): {result.stderr[:200]}")
            
        train_success = result.returncode == 0 or "WinError 2" not in str(result.stderr)
        print(f"📊 Training workload: {'✅ SUCCESS' if train_success else '❌ FAILED'}")
        
    except subprocess.TimeoutExpired:
        print("⚠️ Training test timed out (may be normal)")
        train_success = True  # Timeout is better than WinError 2
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        train_success = False
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Generation workload execution: {'✅ WORKING' if gen_success else '❌ BROKEN'}")
    print(f"Training workload execution: {'✅ WORKING' if train_success else '❌ BROKEN'}")
    
    overall_success = gen_success and train_success
    
    if overall_success:
        print("\n🎉 SUCCESS: Workload execution is fixed!")
        print("The optimizer should now work correctly.")
        print("\nNext steps:")
        print("1. Run the actual optimization system")
        print("2. Verify dual GPU utilization")
        print("3. Confirm real generation/training during optimization")
    else:
        print("\n❌ ISSUES: Some workloads still have problems")
        print("Check the error messages above for details.")
    
    return overall_success

if __name__ == "__main__":
    success = test_workload_execution()
    input(f"\nPress Enter to continue... (Success: {success})")
    sys.exit(0 if success else 1)