#!/usr/bin/env python3
"""
Manual cleanup script for optimization artifacts.

This script can be run standalone to clean up the artifacts/optimization folder.
By default, it keeps the 3 most recent optimization runs and removes older ones.

Usage:
    python scripts/cleanup_optimization.py                    # Keep last 3 runs
    python scripts/cleanup_optimization.py --keep 5           # Keep last 5 runs
    python scripts/cleanup_optimization.py --dry-run          # Preview what would be deleted
    python scripts/cleanup_optimization.py --dir path/to/opt  # Custom directory
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path to import aios modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from aios.utils.optimization_cleanup import cleanup_old_optimization_runs


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old optimization runs from artifacts/optimization folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Keep last 3 runs (default)
  %(prog)s --keep 5                 # Keep last 5 runs
  %(prog)s --dry-run                # Preview without deleting
  %(prog)s --dir custom/path        # Use custom directory
  %(prog)s --keep 1 --verbose       # Keep only the most recent run, show debug info
        """
    )
    
    parser.add_argument(
        '--keep', '-k',
        type=int,
        default=3,
        help='Number of most recent runs to keep (default: 3)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='artifacts/optimization',
        help='Optimization directory to clean (default: artifacts/optimization)'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Resolve directory path
    opt_dir = Path(args.dir)
    if not opt_dir.is_absolute():
        opt_dir = repo_root / opt_dir
    
    # Run cleanup
    print("=" * 70)
    print("🧹 Optimization Folder Cleanup")
    print("=" * 70)
    print(f"Directory: {opt_dir}")
    print(f"Keep last: {args.keep} run(s)")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will delete files)'}")
    print("=" * 70)
    print()
    
    try:
        stats = cleanup_old_optimization_runs(
            opt_dir,
            keep_last_n=args.keep,
            dry_run=args.dry_run
        )
        
        # Print summary
        print()
        print("=" * 70)
        print("📊 Cleanup Summary")
        print("=" * 70)
        print(f"Sessions found:    {stats['sessions_found']}")
        print(f"Sessions kept:     {stats['sessions_kept']}")
        print(f"Sessions deleted:  {stats['sessions_deleted']}")
        print(f"Files deleted:     {stats['files_deleted']}")
        
        mb_freed = stats['bytes_freed'] / (1024 * 1024)
        print(f"Space freed:       {mb_freed:.2f} MB")
        
        if args.dry_run:
            print()
            print("💡 This was a dry run. Run without --dry-run to actually delete files.")
        
        print("=" * 70)
        
        if stats['sessions_deleted'] > 0 and not args.dry_run:
            print("\n✅ Cleanup completed successfully!")
        elif args.dry_run and stats['sessions_deleted'] > 0:
            print("\n✅ Dry run completed - files would be deleted")
        else:
            print("\n✅ No cleanup needed - folder is clean!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
