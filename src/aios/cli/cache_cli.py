"""
CLI commands for managing streaming dataset chunk cache.
"""

from __future__ import annotations

import typer
from rich import print
from rich.table import Table
from rich.console import Console

app = typer.Typer(help="Manage streaming dataset chunk cache")
console = Console()


@app.command()
def stats():
    """Show cache statistics."""
    try:
        from aios.data.streaming_cache import get_cache
        
        cache = get_cache()
        stats = cache.get_cache_stats()
        
        print("\n[bold cyan]Streaming Dataset Cache Statistics[/bold cyan]")
        print(f"  Cache Directory: {stats['cache_dir']}")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Datasets Cached: {stats['datasets_cached']}")
        print(f"  Cache Size: {stats['size_limit_status']}")
        print(f"  Max Chunks per Dataset: {stats['max_chunks_per_dataset']}")
        
        if stats['chunks_per_dataset']:
            print("\n[bold]Cached Datasets:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Dataset", style="cyan")
            table.add_column("Chunks", justify="right", style="green")
            
            for dataset, chunk_count in sorted(stats['chunks_per_dataset'].items()):
                table.add_row(dataset, str(chunk_count))
            
            console.print(table)
        else:
            print("\n  No cached datasets yet.")
        
    except Exception as e:
        print(f"[red]Error getting cache stats: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def clear(
    dataset: str = typer.Argument(None, help="Dataset path to clear (e.g., 'wikitext'). If not specified, clears all."),
    config: str = typer.Option(None, help="Specific config to clear"),
    split: str = typer.Option(None, help="Specific split to clear"),
):
    """Clear cached chunks for a dataset or all datasets."""
    try:
        from aios.data.streaming_cache import get_cache
        
        cache = get_cache()
        
        if dataset:
            removed = cache.clear_dataset_cache(dataset, config, split)
            filter_info = ""
            if config:
                filter_info += f" (config: {config})"
            if split:
                filter_info += f" (split: {split})"
            print(f"✓ Cleared {removed} chunk(s) for dataset: {dataset}{filter_info}")
        else:
            # Clear all by getting all datasets and clearing each
            stats = cache.get_cache_stats()
            total_removed = 0
            for ds in stats['chunks_per_dataset'].keys():
                removed = cache.clear_dataset_cache(ds)
                total_removed += removed
            print(f"✓ Cleared all caches: {total_removed} chunk(s) removed")
        
    except Exception as e:
        print(f"[red]Error clearing cache: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def cleanup(
    max_age_days: float = typer.Option(7.0, help="Maximum age in days (default: 7)"),
):
    """Remove cached chunks older than specified age."""
    try:
        from aios.data.streaming_cache import get_cache
        
        cache = get_cache()
        max_age_hours = max_age_days * 24
        
        removed = cache.cleanup_old_caches(max_age_hours)
        print(f"✓ Cleaned up {removed} chunk(s) older than {max_age_days} days")
        
    except Exception as e:
        print(f"[red]Error cleaning up cache: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def configure(
    max_size_mb: float = typer.Option(None, help="Maximum cache size in MB"),
    max_chunks: int = typer.Option(None, help="Maximum chunks per dataset"),
    max_age_hours: float = typer.Option(None, help="Maximum age in hours"),
):
    """Configure cache settings in config/default.yaml."""
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path.cwd() / "config" / "default.yaml"
        
        if not config_path.exists():
            print(f"[red]Config file not found: {config_path}[/red]")
            raise typer.Exit(code=1)
        
        # Load existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Ensure streaming_cache section exists
        if 'streaming_cache' not in config:
            config['streaming_cache'] = {}
        
        # Update values
        updated = []
        if max_size_mb is not None:
            config['streaming_cache']['max_size_mb'] = float(max_size_mb)
            updated.append(f"max_size_mb = {max_size_mb} MB")
        
        if max_chunks is not None:
            config['streaming_cache']['max_chunks_per_dataset'] = int(max_chunks)
            updated.append(f"max_chunks_per_dataset = {max_chunks}")
        
        if max_age_hours is not None:
            config['streaming_cache']['max_age_hours'] = float(max_age_hours)
            updated.append(f"max_age_hours = {max_age_hours}")
        
        if not updated:
            print("[yellow]No settings specified. Use --max-size-mb, --max-chunks, or --max-age-hours[/yellow]")
            return
        
        # Save config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Updated cache configuration:")
        for item in updated:
            print(f"  • {item}")
        print(f"\n[dim]Note: Restart any running processes to apply changes[/dim]")
        
    except Exception as e:
        print(f"[red]Error updating configuration: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
