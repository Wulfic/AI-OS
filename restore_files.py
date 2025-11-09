"""
Restore Python source files from .pyc bytecode files
"""
import os
import subprocess
from pathlib import Path

def restore_from_pyc():
    """Decompile all .pyc files back to .py files"""
    
    root = Path(r"c:\Users\tyler\Repos\AI-OS")
    
    # Find all .pyc files in src and tests (not .venv)
    pyc_files = []
    for pattern in ["src/**/*.pyc", "tests/**/*.pyc"]:
        pyc_files.extend(root.glob(pattern))
    
    print(f"Found {len(pyc_files)} .pyc files to decompile")
    
    success_count = 0
    fail_count = 0
    
    for pyc_file in pyc_files:
        # Determine output path (remove __pycache__ and .cpython-*.pyc)
        rel_path = pyc_file.relative_to(root)
        
        if '__pycache__' in pyc_file.parts:
            # Move up one directory and rename
            parent_dir = pyc_file.parent.parent
            filename = pyc_file.stem.split('.cpython-')[0] + '.py'
            output_file = parent_dir / filename
        else:
            output_file = pyc_file.with_suffix('.py')
        
        # Skip if .py already exists
        if output_file.exists():
            print(f"SKIP (exists): {output_file}")
            continue
        
        try:
            # Decompile using uncompyle6
            result = subprocess.run(
                ['uncompyle6', str(pyc_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Write decompiled code
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                print(f"✓ Restored: {output_file}")
                success_count += 1
            else:
                print(f"✗ Failed: {pyc_file} - {result.stderr[:100]}")
                fail_count += 1
                
        except Exception as e:
            print(f"✗ Error: {pyc_file} - {str(e)[:100]}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Restoration complete!")
    print(f"Successfully restored: {success_count} files")
    print(f"Failed: {fail_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    restore_from_pyc()
