"""
Import .pyc files and extract as much as possible using introspection
"""
import sys
import importlib.util
import inspect
from pathlib import Path

def import_pyc_and_inspect(pyc_path, module_name):
    """Import a .pyc file and inspect it"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, pyc_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Extract all members
            members = inspect.getmembers(module)
            
            # Build source skeleton
            lines = [f"# Recovered from {pyc_path.name}\n"]
            
            # Get docstring
            if module.__doc__:
                lines.append(f'"""{module.__doc__}"""\n\n')
            
            # Extract imports (look for modules)
            for name, obj in members:
                if inspect.ismodule(obj) and not name.startswith('_'):
                    lines.append(f"import {name}")
            
            lines.append("\n")
            
            # Extract classes
            for name, obj in members:
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    lines.append(f"\nclass {name}:")
                    if obj.__doc__:
                        lines.append(f'    """{obj.__doc__}"""')
                    
                    # Get methods
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith('_') or method_name in ['__init__', '__str__', '__repr__']:
                            sig = inspect.signature(method)
                            lines.append(f"    def {method_name}{sig}:")
                            if method.__doc__:
                                lines.append(f'        """{method.__doc__}"""')
                            lines.append("        pass\n")
            
            # Extract functions
            for name, obj in members:
                if inspect.isfunction(obj) and obj.__module__ == module_name and not name.startswith('_'):
                    sig = inspect.signature(obj)
                    lines.append(f"\ndef {name}{sig}:")
                    if obj.__doc__:
                        lines.append(f'    """{obj.__doc__}"""')
                    lines.append("    pass\n")
            
            # Extract constants
            for name, obj in members:
                if not name.startswith('_') and not inspect.isfunction(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                    lines.append(f"{name} = {repr(obj)}")
            
            return True, '\n'.join(lines)
    except Exception as e:
        return False, str(e)

# Test on one file
test_pyc = Path(r"C:\Users\tyler\Repos\AI-OS\src\aios\__pycache__\__init__.cpython-312.pyc")
output_py = Path(r"C:\Users\tyler\Repos\AI-OS\src\aios\__init__.py")

success, content = import_pyc_and_inspect(test_pyc, "aios_test")
if success:
    print("SUCCESS! Content:")
    print(content[:500])
    print(f"\n... (saving to {output_py})")
    with open(output_py, 'w') as f:
        f.write(content)
else:
    print(f"Failed: {content}")
