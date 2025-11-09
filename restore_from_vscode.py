"""
Restore Python files from VS Code local history
"""
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

def find_vscode_history():
    """Find all Python files in VS Code history"""
    history_path = Path(os.environ['APPDATA']) / 'Code' / 'User' / 'History'
    
    # Map of original paths to history entries
    file_map = defaultdict(list)
    
    for history_dir in history_path.iterdir():
        if not history_dir.is_dir():
            continue
        
        entries_file = history_dir / 'entries.json'
        if not entries_file.exists():
            continue
        
        try:
            with open(entries_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            
            for entry in entries.get('entries', []):
                resource = entry.get('resource', '')
                if 'AI-OS' in resource and resource.endswith('.py'):
                    # Find the actual file
                    py_files = list(history_dir.glob('*.py'))
                    if py_files:
                        # Get the most recent one
                        latest = max(py_files, key=lambda p: p.stat().st_mtime)
                        file_map[resource].append({
                            'history_file': latest,
                            'timestamp': latest.stat().st_mtime,
                            'resource': resource
                        })
        except Exception as e:
            print(f"Error processing {history_dir}: {e}")
    
    return file_map

def restore_files():
    """Restore files from VS Code history"""
    print("Scanning VS Code local history...")
    file_map = find_vscode_history()
    
    print(f"\nFound history for {len(file_map)} unique files")
    
    restored = 0
    skipped = 0
    
    for resource, versions in file_map.items():
        # Get the most recent version
        latest_version = max(versions, key=lambda v: v['timestamp'])
        
        # Parse the resource URI to get the file path
        # Format: file:///c%3A/Users/tyler/Repos/AI-OS/src/...
        try:
            if resource.startswith('file:///'):
                # Remove file:/// and decode URL encoding
                import urllib.parse
                path_part = resource[8:]  # Remove 'file:///'
                decoded_path = urllib.parse.unquote(path_part)
                target_path = Path(decoded_path)
                
                if not target_path.exists() or target_path.stat().st_size < 100:
                    # Restore it
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(latest_version['history_file'], target_path)
                    print(f"✓ Restored: {target_path.relative_to(Path.cwd())}")
                    restored += 1
                else:
                    # Check if it's one of the broken ones
                    with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(200)
                    
                    if 'Unsupported Python version' in content:
                        # Replace it
                        shutil.copy2(latest_version['history_file'], target_path)
                        print(f"✓ Fixed: {target_path.relative_to(Path.cwd())}")
                        restored += 1
                    else:
                        skipped += 1
        except Exception as e:
            print(f"✗ Error restoring {resource}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Restoration complete!")
    print(f"Restored: {restored} files")
    print(f"Skipped (already good): {skipped} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    restore_files()
