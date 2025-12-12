#!/usr/bin/env python3
"""
AI-OS Version Updater Script

This script updates version strings across the entire codebase.
Run with: python scripts/update_version.py

Supports both Windows and Ubuntu.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Files and patterns to update
# Each entry is (file_path_relative_to_root, pattern, replacement_template)
# replacement_template uses {version} as placeholder
VERSION_LOCATIONS: List[Tuple[str, str, str]] = [
    # pyproject.toml - main package version
    (
        "pyproject.toml",
        r'version\s*=\s*"[^"]+"',
        'version = "{version}"'
    ),
    # src/aios/__init__.py - Python module version
    (
        "src/aios/__init__.py",
        r'__version__\s*=\s*"[^"]+"',
        '__version__ = "{version}"'
    ),
    # src/aios/core/orchestrator.py - orchestrator status version
    (
        "src/aios/core/orchestrator.py",
        r'"version":\s*"[^"]+"',
        '"version": "{version}"'
    ),
    # README.md - title
    (
        "README.md",
        r'# AI-OS v[\d.]+',
        '# AI-OS v{version}'
    ),
    # README.md - badge version
    (
        "README.md",
        r'!\[Version\]\(https://img\.shields\.io/badge/version-[\d.]+-blue\.svg\)',
        '![Version](https://img.shields.io/badge/version-{version}-blue.svg)'
    ),
    # README.md - key features version
    (
        "README.md",
        r'Key features in v[\d.]+:',
        'Key features in v{version}:'
    ),
]


def get_current_version() -> str:
    """Read current version from pyproject.toml."""
    pyproject_path = ROOT_DIR / "pyproject.toml"
    if not pyproject_path.exists():
        return "unknown"
    
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "unknown"


def validate_version(version: str) -> bool:
    """Validate version string format (semantic versioning)."""
    pattern = r'^\d+\.\d+\.\d+(-[\w.]+)?$'
    return bool(re.match(pattern, version))


def update_file(file_path: Path, pattern: str, replacement: str) -> Tuple[bool, str]:
    """
    Update a single file with the new version.
    
    Returns:
        Tuple of (success, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Check if pattern exists
        if not re.search(pattern, content):
            return False, f"Pattern not found in {file_path.name}"
        
        # Perform replacement
        new_content = re.sub(pattern, replacement, content)
        
        if new_content == content:
            return True, f"No changes needed in {file_path.name}"
        
        file_path.write_text(new_content, encoding="utf-8")
        return True, f"Updated {file_path.name}"
    
    except Exception as e:
        return False, f"Error updating {file_path.name}: {e}"


def update_all_versions(new_version: str, dry_run: bool = False) -> List[Tuple[str, bool, str]]:
    """
    Update all version strings in the codebase.
    
    Args:
        new_version: The new version string
        dry_run: If True, only report what would be changed
    
    Returns:
        List of (file_path, success, message) tuples
    """
    results = []
    
    for rel_path, pattern, replacement_template in VERSION_LOCATIONS:
        file_path = ROOT_DIR / rel_path
        replacement = replacement_template.format(version=new_version)
        
        if dry_run:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                if re.search(pattern, content):
                    results.append((rel_path, True, f"Would update {rel_path}"))
                else:
                    results.append((rel_path, False, f"Pattern not found in {rel_path}"))
            else:
                results.append((rel_path, False, f"File not found: {rel_path}"))
        else:
            success, message = update_file(file_path, pattern, replacement)
            results.append((rel_path, success, message))
    
    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("AI-OS Version Updater")
    print("=" * 60)
    
    current_version = get_current_version()
    print(f"\nCurrent version: {current_version}")
    print(f"\nFiles that will be updated:")
    for rel_path, _, _ in VERSION_LOCATIONS:
        file_path = ROOT_DIR / rel_path
        status = "✓" if file_path.exists() else "✗ (not found)"
        print(f"  {status} {rel_path}")
    
    print()
    
    # Prompt for new version
    try:
        new_version = input("Enter new version (e.g., 1.0.15): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        sys.exit(0)
    
    if not new_version:
        print("No version entered. Exiting.")
        sys.exit(1)
    
    if not validate_version(new_version):
        print(f"Invalid version format: '{new_version}'")
        print("Version must be in format: X.Y.Z (e.g., 1.0.15)")
        sys.exit(1)
    
    if new_version == current_version:
        print(f"New version is the same as current version ({current_version}). No changes made.")
        sys.exit(0)
    
    # Confirm
    print(f"\nUpdating version: {current_version} → {new_version}")
    try:
        confirm = input("Proceed? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        sys.exit(0)
    
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Perform updates
    print("\nUpdating files...")
    results = update_all_versions(new_version)
    
    print()
    success_count = 0
    fail_count = 0
    
    for rel_path, success, message in results:
        if success:
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ✗ {message}")
            fail_count += 1
    
    print()
    print(f"Done! {success_count} files updated, {fail_count} failures.")
    
    if fail_count == 0:
        print("\nNext steps:")
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git add -A && git commit -m \"chore: bump version to {new_version}\"")
        print(f"  3. Push: git push origin main")
        print(f"  4. Update Official tag:")
        print(f"     git tag -d Official")
        print(f"     git push origin --delete Official")
        print(f"     git tag -a Official -m \"Official Release v{new_version}\"")
        print(f"     git push origin Official")


if __name__ == "__main__":
    main()
