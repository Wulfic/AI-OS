"""
Restore ALL Python files from VS Code local history - COMPLETE VERSION
"""
import os
import json
import shutil
import urllib.parse
from pathlib import Path
from datetime import datetime

def restore_all_from_vscode_history():
    """Restore files from VS Code history using entries.json"""
    history_root = Path(os.environ['APPDATA']) / 'Code' / 'User' / 'History'
    
    print(f"Scanning: {history_root}\n")
    
    # Track what we restore
    stats = {
        'total_found': 0,
        'restored': 0,
        'fixed_broken': 0,
        'skipped_good': 0,
        'errors': 0
    }
    
    for hist_dir in history_root.iterdir():
        if not hist_dir.is_dir():
            continue
        
        entries_file = hist_dir / 'entries.json'
        if not entries_file.exists():
            continue
        
        try:
            with open(entries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            resource = data.get('resource', '')
            if not resource or not resource.endswith('.py'):
                continue
            
            if 'AI-OS' not in resource:
                continue
            
            stats['total_found'] += 1
            
            # Decode the file path
            if resource.startswith('file:///'):
                path_part = resource[8:]  # Remove 'file:///'
                decoded_path = urllib.parse.unquote(path_part)
                target_path = Path(decoded_path)
                
                # Find the most recent version
                entries = data.get('entries', [])
                if not entries:
                    continue
                
                latest_entry = max(entries, key=lambda e: e.get('timestamp', 0))
                history_file = hist_dir / latest_entry['id']
                
                if not history_file.exists():
                    continue
                
                # Check if we should restore
                should_restore = False
                reason = ""
                
                if not target_path.exists():
                    should_restore = True
                    reason = "MISSING"
                else:
                    # Check if it's broken
                    try:
                        with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(300)
                        
                        if 'Unsupported Python version' in content or 'uncompyle6 version' in content:
                            should_restore = True
                            reason = "BROKEN"
                            stats['fixed_broken'] += 1
                        else:
                            stats['skipped_good'] += 1
                            continue
                    except:
                        should_restore = True
                        reason = "ERROR_READING"
                
                if should_restore:
                    try:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(history_file, target_path)
                        
                        rel_path = str(target_path.relative_to(Path.cwd())) if target_path.is_relative_to(Path.cwd()) else str(target_path)
                        print(f"✓ [{reason}] {rel_path}")
                        stats['restored'] += 1
                    except Exception as e:
                        print(f"✗ Error restoring {target_path}: {e}")
                        stats['errors'] += 1
                        
        except Exception as e:
            print(f"✗ Error processing {hist_dir.name}: {e}")
            stats['errors'] += 1
    
    print(f"\n{'='*70}")
    print(f"RESTORATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total files found in history: {stats['total_found']}")
    print(f"Files restored: {stats['restored']}")
    print(f"  - Fixed broken files: {stats['fixed_broken']}")
    print(f"  - Restored missing files: {stats['restored'] - stats['fixed_broken']}")
    print(f"Files already good (skipped): {stats['skipped_good']}")
    print(f"Errors: {stats['errors']}")
    print(f"{'='*70}")

if __name__ == "__main__":
    restore_all_from_vscode_history()
