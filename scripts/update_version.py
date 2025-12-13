#!/usr/bin/env python3
"""
AI-OS Version Updater Script

This script updates version strings across the entire codebase.
Run with: python scripts/update_version.py [version]

Supports both Windows and Ubuntu.

Features:
- Updates version in pyproject.toml, __init__.py, orchestrator.py, and README.md
- Automatically commits, pushes, and updates the Official tag
- Can be run with version as argument for non-interactive mode
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

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


def run_git_command(args: List[str], check: bool = True) -> Tuple[bool, str]:
    """
    Run a git command and return (success, output).
    
    Args:
        args: Git command arguments (without 'git' prefix)
        check: If True, treat non-zero exit as failure
    
    Returns:
        Tuple of (success, output/error message)
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if check and result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return False, error_msg
        
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "Git not found in PATH"
    except Exception as e:
        return False, str(e)


def get_current_branch() -> Optional[str]:
    """Get the current git branch name."""
    success, output = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    return output if success else None


def git_commit_and_push(version: str) -> List[Tuple[str, bool, str]]:
    """
    Commit changes and push to origin.
    
    Returns:
        List of (operation, success, message) tuples
    """
    results = []
    
    # Get current branch
    branch = get_current_branch()
    if not branch:
        results.append(("Get branch", False, "Could not determine current branch"))
        return results
    results.append(("Get branch", True, f"Current branch: {branch}"))
    
    # Stage all changes
    success, msg = run_git_command(["add", "-A"])
    if not success:
        results.append(("Stage changes", False, msg))
        return results
    results.append(("Stage changes", True, "Staged all changes"))
    
    # Commit
    commit_msg = f"chore: bump version to {version}"
    success, msg = run_git_command(["commit", "-m", commit_msg])
    if not success:
        # Check if it's just "nothing to commit"
        if "nothing to commit" in msg.lower():
            results.append(("Commit", True, "Nothing to commit (already up to date)"))
        else:
            results.append(("Commit", False, msg))
            return results
    else:
        results.append(("Commit", True, f"Committed: {commit_msg}"))
    
    # Push to origin
    success, msg = run_git_command(["push", "origin", branch])
    if not success:
        results.append(("Push", False, msg))
        return results
    results.append(("Push", True, f"Pushed to origin/{branch}"))
    
    return results


def git_update_official_tag(version: str) -> List[Tuple[str, bool, str]]:
    """
    Update the Official tag to point to the current commit.
    
    Returns:
        List of (operation, success, message) tuples
    """
    results = []
    
    # Delete local Official tag (ignore errors - tag may not exist)
    run_git_command(["tag", "-d", "Official"], check=False)
    
    # Delete remote Official tag (ignore errors - tag may not exist)
    run_git_command(["push", "origin", "--delete", "Official"], check=False)
    
    # Create new Official tag
    success, msg = run_git_command(["tag", "-a", "Official", "-m", f"Official Release v{version}"])
    if not success:
        results.append(("Create tag", False, msg))
        return results
    results.append(("Create tag", True, f"Created tag: Official (v{version})"))
    
    # Push the new tag
    success, msg = run_git_command(["push", "origin", "Official"])
    if not success:
        results.append(("Push tag", False, msg))
        return results
    results.append(("Push tag", True, "Pushed Official tag to origin"))
    
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
    
    # Check for command-line argument
    new_version = None
    auto_git = False
    
    if len(sys.argv) >= 2:
        new_version = sys.argv[1].strip()
        # Check for --auto flag for fully automated mode
        auto_git = "--auto" in sys.argv or "-y" in sys.argv
    
    # Prompt for new version if not provided
    if not new_version:
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
    
    # Confirm (skip if --auto flag)
    print(f"\nUpdating version: {current_version} → {new_version}")
    if not auto_git:
        try:
            confirm = input("Proceed? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled.")
            sys.exit(0)
        
        if confirm != 'y':
            print("Cancelled.")
            sys.exit(0)
    else:
        print("Auto mode enabled, proceeding...")
    
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
        # Ask about git operations (auto if --auto flag)
        print("\n" + "-" * 60)
        print("Git Operations")
        print("-" * 60)
        
        # Step 1: Commit and push
        if auto_git:
            do_commit = 'y'
            print("\nAuto mode: Running commit and push...")
        else:
            try:
                do_commit = input("\nCommit and push changes? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\nSkipping git operations.")
                do_commit = 'n'
        
        commit_success = False
        if do_commit != 'n':
            print("\nCommitting and pushing...")
            git_results = git_commit_and_push(new_version)
            
            print()
            commit_success = True
            for operation, success, message in git_results:
                if success:
                    print(f"  ✓ {operation}: {message}")
                else:
                    print(f"  ✗ {operation}: {message}")
                    commit_success = False
            
            if not commit_success:
                print("\n⚠ Commit/push failed. Fix issues before updating tags.")
        else:
            print("\nCommit/push skipped. Manual steps:")
            print(f"  1. Review changes: git diff")
            print(f"  2. Commit: git add -A && git commit -m \"chore: bump version to {new_version}\"")
            print(f"  3. Push: git push origin <branch>")
        
        # Step 2: Update Official tag (only if commit succeeded or was skipped)
        if commit_success or do_commit == 'n':
            print("\n" + "-" * 60)
            
            if auto_git:
                do_tag = 'y'
                print("\nAuto mode: Updating Official tag...")
            else:
                try:
                    do_tag = input("\nUpdate Official tag? [Y/n]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n\nSkipping tag update.")
                    do_tag = 'n'
            
            if do_tag != 'n':
                print("\nUpdating Official tag...")
                tag_results = git_update_official_tag(new_version)
                
                print()
                tag_success = True
                for operation, success, message in tag_results:
                    if success:
                        print(f"  ✓ {operation}: {message}")
                    else:
                        print(f"  ✗ {operation}: {message}")
                        tag_success = False
                
                if tag_success:
                    print(f"\n✓ Version {new_version} released successfully!")
                else:
                    print("\n⚠ Tag update failed. Please check and complete manually.")
            else:
                print("\nTag update skipped. Manual steps:")
                print(f"  git tag -d Official")
                print(f"  git push origin --delete Official")
                print(f"  git tag -a Official -m \"Official Release v{new_version}\"")
                print(f"  git push origin Official")


if __name__ == "__main__":
    main()
