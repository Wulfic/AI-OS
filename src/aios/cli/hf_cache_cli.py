"""CLI commands for managing Hugging Face cache location."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer


def get_hf_cache_config_file() -> Path:
    """Get the path to the HF cache configuration file."""
    config_dir = Path.home() / ".config" / "aios"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "hf_cache_path.txt"


def get_current_hf_cache() -> Optional[str]:
    """Get the currently configured HF cache directory."""
    # Check environment variable first
    env_val = os.environ.get("HF_HOME")
    if env_val:
        return env_val
    
    # Check config file
    config_file = get_hf_cache_config_file()
    if config_file.exists():
        try:
            return config_file.read_text().strip()
        except Exception:
            pass
    
    return None


def set_hf_cache(cache_dir: str) -> bool:
    """Save HF cache directory preference."""
    try:
        config_file = get_hf_cache_config_file()
        config_file.write_text(str(Path(cache_dir).resolve()))
        return True
    except Exception as e:
        print(f"Error saving cache location: {e}")
        return False


def show(config: Optional[str] = typer.Option(None, "--config", help="Unused, for consistency")):
    """Show current HF cache location."""
    current = get_current_hf_cache()
    
    print("\n" + "=" * 60)
    print("Hugging Face Cache Configuration")
    print("=" * 60)
    
    if current:
        p = Path(current)
        print(f"\nCurrent cache location: {current}")
        print(f"Drive: {p.drive if p.drive else 'N/A'}")
        print(f"Exists: {'Yes' if p.exists() else 'No (will be created on first use)'}")
        
        # Show disk space if available
        try:
            import shutil
            if p.exists():
                stat = shutil.disk_usage(p)
                total_gb = stat.total / (1024**3)
                free_gb = stat.free / (1024**3)
                used_pct = ((stat.total - stat.free) / stat.total) * 100
                print(f"\nDisk space:")
                print(f"  Total: {total_gb:.1f} GB")
                print(f"  Free: {free_gb:.1f} GB")
                print(f"  Used: {used_pct:.1f}%")
        except Exception:
            pass
    else:
        print("\n⚠️  No cache location configured.")
        print("Default location will be used (install root/training_datasets/hf_cache)")
    
    print("\n" + "=" * 60)
    print("\nTo change the cache location, use:")
    print("  aios hf-cache set <path>")
    print("\nExample:")
    print("  aios hf-cache set Z:/training_datasets/.hf_cache")
    print("=" * 60 + "\n")


def set_location(
    path: str = typer.Argument(..., help="Path to use for HF cache"),
    create: bool = typer.Option(True, "--create/--no-create", help="Create directory if it doesn't exist"),
):
    """Set HF cache location."""
    cache_path = Path(path).resolve()
    
    print(f"\nSetting HF cache location to: {cache_path}")
    
    # Validate/create the directory
    if not cache_path.exists():
        if create:
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {cache_path}")
            except Exception as e:
                print(f"❌ Error creating directory: {e}")
                return
        else:
            print(f"❌ Directory doesn't exist and --no-create was specified")
            return
    elif not cache_path.is_dir():
        print(f"❌ Path exists but is not a directory")
        return
    
    # Save the configuration
    if set_hf_cache(str(cache_path)):
        print(f"✅ Cache location saved successfully!")
        print(f"\n⚠️  IMPORTANT: Restart VS Code and terminals for changes to take effect")
        print(f"   New downloads will go to: {cache_path}")
    else:
        print(f"❌ Failed to save cache location")


def clear(
    config: Optional[str] = typer.Option(None, "--config", help="Unused, for consistency"),
):
    """Clear HF cache location configuration (will use smart defaults)."""
    config_file = get_hf_cache_config_file()
    
    if config_file.exists():
        try:
            config_file.unlink()
            print("✅ Cache configuration cleared")
            print("   Smart defaults will be used (checks D:, E:, F:, Z: drives)")
            print("\n⚠️  Restart VS Code and terminals for changes to take effect")
        except Exception as e:
            print(f"❌ Error clearing configuration: {e}")
    else:
        print("ℹ️  No cache configuration to clear (already using defaults)")


def register(app: typer.Typer):
    """Register hf-cache commands with the main app."""
    hf_cache = typer.Typer(help="Manage Hugging Face cache location")
    hf_cache.command("show")(show)
    hf_cache.command("set")(set_location)
    hf_cache.command("clear")(clear)
    app.add_typer(hf_cache, name="hf-cache")
