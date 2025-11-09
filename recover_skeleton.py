"""
Advanced Python 3.12 .pyc to .py recovery using multiple strategies
"""
import os
import marshal
import dis
import types
from pathlib import Path
import inspect

def extract_strings_and_names(code_obj):
    """Extract useful information from code object"""
    info = {
        'docstring': code_obj.co_consts[0] if code_obj.co_consts and isinstance(code_obj.co_consts[0], str) else None,
        'imports': [],
        'functions': [],
        'classes': [],
        'variables': list(code_obj.co_varnames),
        'names': list(code_obj.co_names),
        'constants': [c for c in code_obj.co_consts if isinstance(c, (str, int, float, bool))],
    }
    
    # Look for nested code objects (functions/classes)
    for const in code_obj.co_consts:
        if isinstance(const, types.CodeType):
            if const.co_name != '<module>' and const.co_name != '<listcomp>':
                info['functions'].append(const.co_name)
    
    return info

def reconstruct_skeleton(pyc_path, output_path):
    """Create a skeleton .py file from .pyc"""
    try:
        with open(pyc_path, 'rb') as f:
            # Skip magic number and timestamp (16 bytes for Python 3.7+)
            f.read(16)
            code_obj = marshal.load(f)
        
        info = extract_strings_and_names(code_obj)
        
        # Build skeleton source
        lines = []
        
        # Add docstring if present
        if info['docstring']:
            lines.append(f'"""{info['docstring']}"""')
            lines.append('')
        
        lines.append(f"# Recovered from: {pyc_path.name}")
        lines.append(f"# Original path: {output_path}")
        lines.append("# This is a SKELETON - manual reconstruction needed")
        lines.append('')
        
        # Add imports based on names
        common_imports = {
            'os': 'import os',
            'sys': 'import sys',
            'Path': 'from pathlib import Path',
            'json': 'import json',
            'torch': 'import torch',
            'numpy': 'import numpy as np',
            'click': 'import click',
            'QWidget': 'from PyQt6.QtWidgets import *',
            'QApplication': 'from PyQt6.QtWidgets import QApplication',
        }
        
        added_imports = set()
        for name in info['names']:
            if name in common_imports and common_imports[name] not in added_imports:
                lines.append(common_imports[name])
                added_imports.add(common_imports[name])
        
        if added_imports:
            lines.append('')
        
        # Add function/class skeletons
        for func_name in info['functions']:
            if not func_name.startswith('<'):
                lines.append(f"def {func_name}(*args, **kwargs):")
                lines.append(f"    # TODO: Reconstruct {func_name}")
                lines.append("    pass")
                lines.append('')
        
        # Add variable hints
        if info['names']:
            lines.append("# Referenced names:")
            for name in sorted(set(info['names']))[:30]:
                lines.append(f"# - {name}")
        
        # Write skeleton
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True, info
        
    except Exception as e:
        return False, str(e)

def recover_all_files():
    """Recover all .pyc files to skeleton .py files"""
    root = Path(r"c:\Users\tyler\Repos\AI-OS")
    
    pyc_files = list(root.glob("src/**/*.pyc")) + list(root.glob("tests/**/*.pyc"))
    
    print(f"Found {len(pyc_files)} .pyc files\n")
    
    success = 0
    for pyc_file in pyc_files:
        if '__pycache__' in pyc_file.parts:
            parent_dir = pyc_file.parent.parent
            filename = pyc_file.stem.split('.cpython-')[0] + '.py'
            output_file = parent_dir / filename
        else:
            output_file = pyc_file.with_suffix('.py')
        
        if output_file.exists():
            continue
        
        ok, info = reconstruct_skeleton(pyc_file, output_file)
        if ok:
            print(f"✓ {output_file.relative_to(root)}")
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Created {success} skeleton files")
    print(f"{'='*60}")
    print("\nNOTE: These are SKELETONS only. You'll need to:")
    print("1. Review each file")
    print("2. Reconstruct logic from function/variable names")
    print("3. Check git commits for similar code patterns")
    print("4. Use IDE autocomplete to help rebuild")

if __name__ == "__main__":
    recover_all_files()
