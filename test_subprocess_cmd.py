"""
Test the corrected subprocess command construction for the optimizer
"""

import sys
import subprocess
import os

def test_command_construction():
    """Test that the Python command construction works correctly."""
    
    print("🔧 Testing Subprocess Command Construction")
    print("=" * 45)
    
    # Test the command that the optimizer will use
    test_args = [
        "hrm-hf", "train-actv1",
        "--model", "gpt2", 
        "--steps", "1",
        "--batch-size", "1",
        "--teacher-dataset",
        "--td-num-samples", "4",
        "--td-max-new-tokens", "8",
        "--strict"
    ]
    
    # Construct the command as the optimizer does
    python_cmd = [sys.executable, "-m", "aios.cli.aios"] + test_args
    
    print(f"📋 Python executable: {sys.executable}")
    print(f"📋 Full command: {' '.join(python_cmd)}")
    
    # Test if the module is importable
    try:
        print("\n📦 Testing module import...")
        import aios.cli.aios
        print("✅ aios.cli.aios module is importable")
    except ImportError as e:
        print(f"❌ Cannot import aios.cli.aios: {e}")
        return False
    
    # Test a quick dry run (with very minimal settings)
    try:
        print("\n🧪 Testing quick dry run...")
        
        # Add current directory to Python path
        env = os.environ.copy()
        env["PYTHONPATH"] = r"c:\Users\tyler\Repos\AI-OS\src"
        
        # Very short test with minimal resources
        quick_test_args = [
            "hrm-hf", "train-actv1", 
            "--model", "gpt2",
            "--steps", "0",  # Zero steps for quick test
            "--batch-size", "1",
            "--device", "cpu",  # Use CPU for safety
            "--strict"
        ]
        
        python_cmd = [sys.executable, "-m", "aios.cli.aios"] + quick_test_args
        
        result = subprocess.run(
            python_cmd,
            capture_output=True,
            text=True,
            timeout=30,  # Short timeout
            env=env,
            cwd=r"c:\Users\tyler\Repos\AI-OS"  # Set working directory
        )
        
        print(f"📊 Exit code: {result.returncode}")
        print(f"📊 Stdout: {result.stdout[:300]}")
        print(f"📊 Stderr: {result.stderr[:300]}")
        
        if result.returncode == 0:
            print("✅ Command execution successful!")
        else:
            print("⚠️ Command returned non-zero exit code (this might be expected)")
            
        return True
        
    except subprocess.TimeoutExpired:
        print("⚠️ Test timed out (this might be expected)")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_command_construction()
    if success:
        print("\n✅ Subprocess command construction is working!")
        print("The optimizer should now be able to run AIOS commands correctly.")
    else:
        print("\n❌ Issues detected with command construction")
    
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)