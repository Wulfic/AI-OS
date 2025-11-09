"""
Extract information from .pyc files to help reconstruct source
"""
import dis
import marshal
import importlib.util
from pathlib import Path

def analyze_pyc(pyc_path):
    """Analyze a .pyc file and extract what information we can"""
    try:
        # Load the .pyc file
        with open(pyc_path, 'rb') as f:
            # Skip header (16 bytes for Python 3.7+)
            f.read(16)
            code_obj = marshal.load(f)
        
        print(f"\n{'='*80}")
        print(f"File: {pyc_path.name}")
        print(f"Output: {pyc_path.parent.parent / (pyc_path.stem.split('.')[0] + '.py')}")
        print(f"{'='*80}")
        
        # Print basic info
        print(f"Function: {code_obj.co_name}")
        print(f"Args: {code_obj.co_argcount}")
        print(f"Variables: {code_obj.co_varnames}")
        print(f"Constants: {[c for c in code_obj.co_consts if isinstance(c, str)][:10]}")
        print(f"Names: {code_obj.co_names[:20]}")
        
        # Disassemble
        print("\n--- Bytecode ---")
        dis.dis(code_obj)
        
    except Exception as e:
        print(f"Error analyzing {pyc_path}: {e}")

# Test on one file
test_file = Path(r"C:\Users\tyler\Repos\AI-OS\src\aios\__pycache__\__init__.cpython-312.pyc")
if test_file.exists():
    analyze_pyc(test_file)
