#!/usr/bin/env python3
"""
Quick verification that the new optimizer is active and ready to use.
"""

import os
import sys

def check_optimizer_status():
    """Check if the new optimizer is enabled and available."""
    
    print("🔍 Checking Advanced Optimizer v2 Status")
    print("=" * 45)
    
    # Check environment variable
    env_var = os.environ.get("AIOS_USE_OPTIMIZER_V2", "0")
    print(f"📝 Environment Variable: AIOS_USE_OPTIMIZER_V2 = {env_var}")
    
    if env_var == "1":
        print("✅ New optimizer is ENABLED")
    else:
        print("❌ New optimizer is DISABLED")
        print("   Set: $env:AIOS_USE_OPTIMIZER_V2 = \"1\"")
        return False
    
    # Check if files exist
    base_path = r"c:\Users\tyler\Repos\AI-OS\src\aios\gui\components\hrm_training"
    files_to_check = [
        "optimizer.py",
        "optimizer_v2.py", 
        "gpu_monitor.py"
    ]
    
    print("\n📁 Checking Required Files:")
    all_files_exist = True
    
    for filename in files_to_check:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            all_files_exist = False
            
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
        
    # Check GPU availability
    print("\n🖥️ Checking GPU Availability:")
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--list-gpus"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.split('\n') if line.strip()]
            print(f"✅ {len(gpu_lines)} GPU(s) detected")
            for i, line in enumerate(gpu_lines):
                print(f"   GPU {i}: {line.strip()}")
        else:
            print("⚠️ nvidia-smi available but returned error")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found - GPU monitoring may not work")
        return False
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")
        
    print("\n🎯 Status Summary:")
    print("✅ Advanced Optimizer v2 is READY TO USE!")
    print("\n📋 To use:")
    print("1. Open AIOS GUI")
    print("2. Go to HRM Training panel") 
    print("3. Click 'Optimize' button")
    print("4. Watch for multi-GPU verification in logs")
    
    return True

if __name__ == "__main__":
    success = check_optimizer_status()
    if success:
        print("\n🎉 Everything looks good! The new optimizer should work perfectly.")
    else:
        print("\n⚠️ Some issues detected. Please check the requirements above.")
        
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)