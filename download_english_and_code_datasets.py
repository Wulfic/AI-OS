#!/usr/bin/env python3
"""
Enhanced Dataset Downloader - English Language + Programming Datasets
Downloads comprehensive training data to Z:\\training_datasets

This includes:
- English language datasets (Wikipedia, books, web text, news, etc.)
- Programming datasets (code from GitHub, coding problems, documentation, etc.)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# CRITICAL: Set cache directories BEFORE any imports to prevent C drive usage
BASE_DOWNLOAD_PATH = Path("Z:\\training_datasets")
CACHE_DIR = BASE_DOWNLOAD_PATH / ".cache"
TEMP_DIR = BASE_DOWNLOAD_PATH / ".temp"

# Create directories early
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Force HuggingFace to use our cache/temp directories
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")
os.environ['HF_HUB_CACHE'] = str(CACHE_DIR / "hub")
os.environ['TMPDIR'] = str(TEMP_DIR)
os.environ['TEMP'] = str(TEMP_DIR)
os.environ['TMP'] = str(TEMP_DIR)
# Windows-specific temp variables
os.environ['TEMPDIR'] = str(TEMP_DIR)
os.environ['TEMP_DIR'] = str(TEMP_DIR)
os.environ['TMP_DIR'] = str(TEMP_DIR)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress symlink warning for network drives

# CRITICAL: Set Python's tempfile module to use our temp directory
import tempfile
tempfile.tempdir = str(TEMP_DIR)
# Also update environment for subprocess temp usage
if 'USERPROFILE' in os.environ:
    # Windows uses %USERPROFILE%\AppData\Local\Temp - redirect it
    os.environ['LOCALAPPDATA'] = str(TEMP_DIR.parent)  # Redirect local app data

try:
    from datasets import load_dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()
except ImportError:
    print("Installing datasets library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset, DownloadConfig
    from datasets.utils.logging import set_verbosity_info
    set_verbosity_info()

try:
    from huggingface_hub import login, whoami, HfFolder
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import login, whoami, HfFolder

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for better progress bars...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


# ENGLISH LANGUAGE DATASETS
ENGLISH_DATASETS = [
    # Tier 1: Small, Essential English Datasets
    {
        "name": "WikiText-103",
        "path": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "size_gb": 0.5,
        "description": "High-quality Wikipedia articles",
        "tier": 1,
        "category": "english",
    },
    {
        "name": "TinyStories",
        "path": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "size_gb": 2.5,
        "description": "Simple stories for language learning",
        "tier": 1,
        "category": "english",
    },
    {
        "name": "CC-News",
        "path": "cc_news",
        "config": None,
        "split": "train",
        "size_gb": 7,
        "description": "English news articles (Common Crawl)",
        "tier": 1,
        "category": "english",
        "streaming": True,
        "max_samples": 100000,
    },
    
    # Tier 2: Medium English Datasets
    {
        "name": "BookCorpus",
        "path": "bookcorpus",
        "config": "plain_text",
        "split": "train",
        "size_gb": 5,
        "description": "Books corpus - novels and literature",
        "tier": 2,
        "category": "english",
        "streaming": True,
        "max_samples": 50000,
    },
    {
        "name": "OpenWebText",
        "path": "Skylion007/openwebtext",
        "config": None,
        "split": "train",
        "size_gb": 38,
        "description": "Reddit-curated web content",
        "tier": 2,
        "category": "english",
        "streaming": True,
        "max_samples": 100000,
    },
    # C4 removed - full dataset is multiple terabytes
    {
        "name": "FineWeb-Edu-10BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "size_gb": 10,
        "description": "Educational web content (10B tokens)",
        "tier": 2,
        "category": "english",
    },
    
    # Tier 3: Large English Datasets
    {
        "name": "Wikipedia-20231101-en",
        "path": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "size_gb": 20,
        "description": "Full English Wikipedia (Nov 2023)",
        "tier": 3,
        "category": "english",
    },
    {
        "name": "FineWeb-Edu-100BT",
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "size_gb": 28,
        "description": "Educational web content (100B tokens)",
        "tier": 3,
        "category": "english",
    },
    {
        "name": "Gutenberg-Books",
        "path": "sedthh/gutenberg_english",
        "config": None,
        "split": "train",
        "size_gb": 12,
        "description": "Project Gutenberg books - classic literature",
        "tier": 3,
        "category": "english",
        "streaming": True,
        "max_samples": 50000,
    },
    {
        "name": "The-Pile-sample",
        "path": "EleutherAI/pile",
        "config": "all",
        "split": "train",
        "size_gb": 25,
        "description": "Diverse pile of text data (sample)",
        "tier": 3,
        "category": "english",
        "streaming": True,
        "max_samples": 200000,
    },
]


# PROGRAMMING/CODE DATASETS
PROGRAMMING_DATASETS = [
    # Tier 1: Small, Essential Programming Datasets
    {
        "name": "CodeSearchNet-Python",
        "path": "code_search_net",
        "config": "python",
        "split": "train",
        "size_gb": 2,
        "description": "Python code with docstrings",
        "tier": 1,
        "category": "programming",
    },
    {
        "name": "MBPP-Python-Problems",
        "path": "mbpp",
        "config": "sanitized",
        "split": "train",
        "size_gb": 0.01,
        "description": "Python programming problems (974 samples)",
        "tier": 1,
        "category": "programming",
    },
    {
        "name": "HumanEval",
        "path": "openai_humaneval",
        "config": None,
        "split": "test",
        "size_gb": 0.001,
        "description": "Python function synthesis problems (164 samples)",
        "tier": 1,
        "category": "programming",
    },
    
    # Tier 2: Medium Programming Datasets
    {
        "name": "CodeSearchNet-All-Languages",
        "path": "code_search_net",
        "config": "all",
        "split": "train",
        "size_gb": 8,
        "description": "6 languages: Python, Java, JS, PHP, Ruby, Go",
        "tier": 2,
        "category": "programming",
    },
    {
        "name": "APPS-Coding-Problems",
        "path": "codeparrot/apps",
        "config": "all",
        "split": "train",
        "size_gb": 1.5,
        "description": "10k Python coding problems with solutions",
        "tier": 2,
        "category": "programming",
    },
    {
        "name": "CodeContests",
        "path": "deepmind/code_contests",
        "config": None,
        "split": "train",
        "size_gb": 0.5,
        "description": "Competitive programming problems",
        "tier": 2,
        "category": "programming",
    },
    {
        "name": "GitHub-Code-Clean",
        "path": "codeparrot/github-code-clean",
        "config": None,
        "split": "train",
        "size_gb": 15,
        "description": "Clean GitHub code (multiple languages)",
        "tier": 2,
        "category": "programming",
        "streaming": True,
        "max_samples": 200000,
    },
    
    # Tier 3: Large Programming Datasets (require HuggingFace auth)
    {
        "name": "The-Stack-Python",
        "path": "bigcode/the-stack",
        "config": "python",
        "split": "train",
        "size_gb": 60,
        "description": "Python code from GitHub (requires HF auth)",
        "tier": 3,
        "category": "programming",
        "streaming": True,
        "max_samples": 500000,
        "requires_auth": True,
    },
    {
        "name": "The-Stack-JavaScript",
        "path": "bigcode/the-stack",
        "config": "javascript",
        "split": "train",
        "size_gb": 45,
        "description": "JavaScript code from GitHub (requires HF auth)",
        "tier": 3,
        "category": "programming",
        "streaming": True,
        "max_samples": 400000,
        "requires_auth": True,
    },
    {
        "name": "The-Stack-Java",
        "path": "bigcode/the-stack",
        "config": "java",
        "split": "train",
        "size_gb": 40,
        "description": "Java code from GitHub (requires HF auth)",
        "tier": 3,
        "category": "programming",
        "streaming": True,
        "max_samples": 300000,
        "requires_auth": True,
    },
    {
        "name": "CodeParrot-Clean",
        "path": "codeparrot/codeparrot-clean",
        "config": None,
        "split": "train",
        "size_gb": 50,
        "description": "Cleaned Python code for training",
        "tier": 3,
        "category": "programming",
        "streaming": True,
        "max_samples": 500000,
    },
]


# Combine all datasets
ALL_DATASETS = ENGLISH_DATASETS + PROGRAMMING_DATASETS


def check_hf_authentication() -> bool:
    """Check if user is authenticated with HuggingFace."""
    try:
        token = HfFolder.get_token()
        if token:
            try:
                user_info = whoami(token=token)
                return True
            except:
                return False
        # Try backup token file
        try:
            token_file = CACHE_DIR / ".hf_token"
            if token_file.exists():
                token = token_file.read_text().strip()
                if token:
                    login(token=token, add_to_git_credential=True)
                    HfFolder.save_token(token)
                    print("✅ Restored saved HuggingFace token")
                    return True
        except Exception:
            pass
        return False
    except:
        return False


def get_hf_username() -> str:
    """Get the currently logged-in HuggingFace username."""
    try:
        token = HfFolder.get_token()
        if token:
            user_info = whoami(token=token)
            # Handle different response formats
            if isinstance(user_info, dict):
                return user_info.get('name') or user_info.get('id') or user_info.get('username', 'Unknown')
            return str(user_info) if user_info else 'Unknown'
    except:
        pass
    return "Not logged in"


def login_to_huggingface() -> bool:
    """Prompt user to login to HuggingFace."""
    print("\n" + "="*80)
    print("🔐 HuggingFace Authentication Required")
    print("="*80)
    print("\nSome datasets require authentication to access.")
    print("\nTo get your access token:")
    print("  1. Go to: https://huggingface.co/settings/tokens")
    print("  2. Create a new token (or copy existing one)")
    print("  3. Paste it below (input will be hidden)\n")
    
    import getpass
    token = getpass.getpass("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided. Skipping authentication.")
        return False
    
    try:
        print("\n⏳ Validating token...")
        login(token=token, add_to_git_credential=True)
        
        # Save token to HuggingFace's default location
        try:
            HfFolder.save_token(token)
            print("💾 Token saved for future use")
        except Exception as save_error:
            print(f"⚠️  Warning: Could not save token ({save_error})")
        
        # Also save to our backup file
        try:
            token_file = CACHE_DIR / ".hf_token"
            token_file.write_text(token)
            # Make it readable only by owner (Unix permissions)
            try:
                token_file.chmod(0o600)
            except Exception:
                pass  # Windows doesn't support chmod
            print(f"💾 Backup token saved to {token_file}")
        except Exception as backup_error:
            print(f"⚠️  Warning: Could not save backup token ({backup_error})")
        
        # Verify token works
        try:
            user_info = whoami(token=token)
            if isinstance(user_info, dict):
                username = user_info.get('name') or user_info.get('id') or user_info.get('username', 'Unknown')
            else:
                username = str(user_info) if user_info else 'Unknown'
            print(f"✅ Successfully logged in as: {username}")
            return True
        except Exception as verify_error:
            print(f"⚠️  Token saved but verification failed: {verify_error}")
            print("   You may still be able to access datasets. Try downloading.")
            return True  # Token is saved, let user try
    except Exception as e:
        print(f"❌ Login failed ({type(e).__name__}): {e}")
        return False


def cleanup_c_drive_cache() -> None:
    """Clean up any HuggingFace cache and temp files that ended up on C drive."""
    print("\n" + "="*80)
    print("🧹 Checking for files on C drive...")
    print("="*80)
    
    locations_to_check = [
        (Path.home() / ".cache" / "huggingface", "HuggingFace cache"),
        (Path("C:\\Users") / os.environ.get('USERNAME', '') / "AppData" / "Local" / "Temp", "Windows Temp"),
        (Path("C:\\Temp"), "C:\\Temp"),
        (Path("C:\\tmp"), "C:\\tmp"),
        (Path(os.environ.get('TEMP', 'C:\\Temp')), "System TEMP"),
        (Path(os.environ.get('TMP', 'C:\\Temp')), "System TMP"),
    ]
    
    total_cleaned = 0
    total_size_gb = 0
    
    for location, name in locations_to_check:
        if not location.exists():
            continue
        
        try:
            # Look for HuggingFace or dataset-related files
            hf_files = list(location.rglob("*huggingface*"))
            hf_files += list(location.rglob("*datasets*"))
            hf_files += list(location.rglob("*.arrow"))
            hf_files += list(location.rglob("*hf_*"))
            
            if not hf_files:
                continue
            
            # Calculate size
            total_size = 0
            file_count = 0
            for item in hf_files:
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                        file_count += 1
                    except Exception:
                        pass
            
            size_gb = total_size / (1024**3)
            
            if size_gb < 0.01:  # Less than 10MB
                continue
            
            print(f"\n⚠️  Found {size_gb:.2f} GB in {name}")
            print(f"   Location: {location}")
            print(f"   Files: {file_count:,}")
            
            cleanup = input(f"   Clean up {name}? (y/n): ").strip().lower()
            if cleanup == 'y':
                print(f"   🗑️  Removing files...")
                removed_count = 0
                for item in hf_files:
                    try:
                        if item.is_file():
                            item.unlink()
                            removed_count += 1
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item, ignore_errors=True)
                            removed_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Could not remove {item.name}: {e}")
                
                print(f"   ✅ Cleaned up {size_gb:.2f} GB ({removed_count} items) from {name}")
                total_cleaned += removed_count
                total_size_gb += size_gb
            else:
                print(f"   ⏭️  Skipped {name}")
        
        except Exception as e:
            print(f"   ⚠️  Error checking {name}: {e}")
    
    if total_cleaned > 0:
        print(f"\n🎉 Total cleanup: {total_size_gb:.2f} GB freed from C drive!")
    else:
        print("\n✅ C drive is clean - no HuggingFace files found")


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def verify_no_c_drive_usage() -> bool:
    """Verify that no new files are being created on C drive."""
    c_cache = Path.home() / ".cache" / "huggingface"
    
    if not c_cache.exists():
        return True
    
    try:
        # Get modification times of files
        recent_files = []
        cutoff_time = time.time() - 300  # Last 5 minutes
        
        for item in c_cache.rglob("*"):
            if item.is_file():
                mtime = item.stat().st_mtime
                if mtime > cutoff_time:
                    recent_files.append((item, mtime))
        
        if recent_files:
            print(f"\n⚠️  WARNING: {len(recent_files)} recent files detected on C drive!")
            print(f"   This suggests temporary files are still being written to C:\\")
            for fpath, mtime in recent_files[:5]:  # Show first 5
                print(f"   - {fpath.name}")
            return False
        
        return True
    
    except Exception as e:
        print(f"⚠️  Error verifying C drive usage: {e}")
        return True  # Don't block on errors


def download_dataset(ds_info: Dict[str, Any], base_path: Path) -> bool:
    """Download a single dataset with enhanced progress tracking.
    
    Args:
        ds_info: Dataset information dictionary
        base_path: Base directory where datasets should be saved
    """
    output_path = base_path / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
    
    # Ensure cache and temp dirs exist
    cache_dir = base_path / ".cache"
    temp_dir = base_path / ".temp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if exists
    if output_path.exists():
        data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
        if data_files:
            size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
            print(f"   ✅ Already exists: {len(data_files)} files, {size_mb:.1f} MB")
            return True
    
    print(f"\n📥 {ds_info['name']} ({ds_info['size_gb']} GB)")
    print(f"   {ds_info['description']}")
    
    try:
        # Handle streaming datasets
        if ds_info.get("streaming", False):
            max_samples = ds_info.get("max_samples", 100000)
            print(f"   📦 Streaming first {max_samples:,} samples...")
            
            load_kwargs = {
                "path": ds_info["path"],
                "split": ds_info["split"],
                "streaming": True,
                "trust_remote_code": True,
                "cache_dir": str(cache_dir),
            }
            if ds_info["config"]:
                load_kwargs["name"] = ds_info["config"]
            
            start_time = time.time()
            dataset_stream = load_dataset(**load_kwargs)
            
            samples = []
            
            # Use tqdm for better progress tracking
            with tqdm(total=max_samples, unit=" samples", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                for i, sample in enumerate(dataset_stream):
                    samples.append(sample)
                    pbar.update(1)
                    
                    if i + 1 >= max_samples:
                        break
            
            # Convert and save
            from datasets import Dataset
            print(f"   💾 Saving {len(samples):,} samples to disk...")
            dataset = Dataset.from_dict({
                key: [s[key] for s in samples]
                for key in samples[0].keys()
            })
            
            output_path.mkdir(parents=True, exist_ok=True)
            if hasattr(dataset, 'save_to_disk'):
                dataset.save_to_disk(str(output_path))
            else:
                print(f"   ⚠️  Warning: Converted dataset missing save_to_disk")
                return False
            
            duration = time.time() - start_time
            size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
            print(f"   ✅ SUCCESS: {len(samples):,} samples, {size_mb:.1f} MB ({format_time(duration)})")
            return True
            
        else:
            # Direct download with progress tracking
            print(f"   ⏳ Downloading full dataset...")
            
            download_config = DownloadConfig(
                resume_download=True,
                max_retries=5,
                cache_dir=str(cache_dir),
            )
            
            load_kwargs = {
                "path": ds_info["path"],
                "split": ds_info["split"],
                "download_config": download_config,
                "trust_remote_code": True,
                "cache_dir": str(cache_dir),
            }
            if ds_info["config"]:
                load_kwargs["name"] = ds_info["config"]
            
            start = time.time()
            
            # Show progress
            print(f"   📡 Fetching dataset metadata...")
            # Force streaming=False to get regular Dataset instead of IterableDataset
            load_kwargs_copy = load_kwargs.copy()
            load_kwargs_copy.pop("streaming", None)  # Remove streaming if present
            dataset = load_dataset(**load_kwargs_copy)
            
            duration = time.time() - start
            
            print(f"   💾 Saving to disk...")
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Type narrowing: only Dataset and DatasetDict have save_to_disk
            from datasets import Dataset, DatasetDict
            if isinstance(dataset, (Dataset, DatasetDict)):
                dataset.save_to_disk(str(output_path))
            else:
                print(f"   ⚠️  Unexpected dataset type: {type(dataset)}")
                return False
            
            size_mb = sum(f.stat().st_size for f in output_path.glob("*.arrow")) / (1024 * 1024)
            # Safe len check for both Dataset and DatasetDict
            try:
                if isinstance(dataset, Dataset):
                    sample_count = len(dataset)
                elif isinstance(dataset, DatasetDict):
                    sample_count = len(dataset[list(dataset.keys())[0]])
                else:
                    sample_count = 0
            except Exception:
                sample_count = 0
            
            print(f"   ✅ SUCCESS: {sample_count:,} samples, {size_mb:.1f} MB ({format_time(duration)})")
            return True
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        print(f"   Stack trace: {traceback.format_exc()}")
        return False


def main():
    print("=" * 80)
    print("🚀 Enhanced Dataset Downloader - English + Programming")
    print("=" * 80)
    print("\nThis script downloads comprehensive training data for language and code!\n")
    
    # Check and clean C drive cache first
    cleanup_c_drive_cache()
    
    # Verify download paths
    print(f"\n📂 Download Configuration:")
    print(f"   Base Path: {BASE_DOWNLOAD_PATH}")
    print(f"   Cache Dir: {CACHE_DIR}")
    print(f"   Temp Dir:  {TEMP_DIR}")
    print(f"   Python tempfile.tempdir: {tempfile.tempdir}")
    print(f"   Environment TEMP: {os.environ.get('TEMP', 'not set')}")
    print(f"   Environment TMP: {os.environ.get('TMP', 'not set')}")
    
    # Check HuggingFace authentication
    is_authenticated = check_hf_authentication()
    if is_authenticated:
        username = get_hf_username()
        print(f"\n🔐 HuggingFace: Logged in as {username}")
    else:
        print("\n🔐 HuggingFace: Not logged in")
        print("   Note: Some datasets require authentication (e.g., The Stack)\n")
        
        # Ask if user wants to login
        login_choice = input("Would you like to login to HuggingFace now? (y/n): ").strip().lower()
        if login_choice == 'y':
            is_authenticated = login_to_huggingface()
            if is_authenticated:
                print("\n✅ You can now access authenticated datasets!")
        print()
    
    # Group by category and tier
    english_tier1 = [ds for ds in ENGLISH_DATASETS if ds["tier"] == 1]
    english_tier2 = [ds for ds in ENGLISH_DATASETS if ds["tier"] == 2]
    english_tier3 = [ds for ds in ENGLISH_DATASETS if ds["tier"] == 3]
    
    prog_tier1 = [ds for ds in PROGRAMMING_DATASETS if ds["tier"] == 1]
    prog_tier2 = [ds for ds in PROGRAMMING_DATASETS if ds["tier"] == 2]
    prog_tier3 = [ds for ds in PROGRAMMING_DATASETS if ds["tier"] == 3]
    
    print("📚 ENGLISH LANGUAGE DATASETS")
    print("-" * 80)
    
    print(f"\n  Tier 1 - Essential ({sum(ds['size_gb'] for ds in english_tier1):.1f} GB):")
    for ds in english_tier1:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print(f"\n  Tier 2 - Recommended ({sum(ds['size_gb'] for ds in english_tier2):.1f} GB):")
    for ds in english_tier2:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print(f"\n  Tier 3 - Large ({sum(ds['size_gb'] for ds in english_tier3):.1f} GB):")
    for ds in english_tier3:
        method = " [streaming]" if ds.get("streaming") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}")
    
    print("\n" + "=" * 80)
    print("💻 PROGRAMMING/CODE DATASETS")
    print("-" * 80)
    
    print(f"\n  Tier 1 - Essential ({sum(ds['size_gb'] for ds in prog_tier1):.1f} GB):")
    for ds in prog_tier1:
        method = " [streaming]" if ds.get("streaming") else ""
        auth = " 🔐" if ds.get("requires_auth") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}{auth}")
    
    print(f"\n  Tier 2 - Recommended ({sum(ds['size_gb'] for ds in prog_tier2):.1f} GB):")
    for ds in prog_tier2:
        method = " [streaming]" if ds.get("streaming") else ""
        auth = " 🔐" if ds.get("requires_auth") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}{auth}")
    
    print(f"\n  Tier 3 - Large ({sum(ds['size_gb'] for ds in prog_tier3):.1f} GB):")
    for ds in prog_tier3:
        method = " [streaming]" if ds.get("streaming") else ""
        auth = " 🔐" if ds.get("requires_auth") else ""
        print(f"    • {ds['name']}: {ds['description']} ({ds['size_gb']} GB){method}{auth}")
    
    print("\n" + "=" * 80)
    print("DOWNLOAD OPTIONS:")
    print("=" * 80)
    
    english_all_size = sum(ds['size_gb'] for ds in ENGLISH_DATASETS)
    prog_all_size = sum(ds['size_gb'] for ds in PROGRAMMING_DATASETS)
    
    print(f"1. English only - Tier 1 ({sum(ds['size_gb'] for ds in english_tier1):.1f} GB)")
    print(f"2. English only - Tier 1+2 ({sum(ds['size_gb'] for ds in english_tier1 + english_tier2):.1f} GB)")
    print(f"3. English only - All tiers ({english_all_size:.1f} GB)")
    print(f"4. Programming only - Tier 1 ({sum(ds['size_gb'] for ds in prog_tier1):.1f} GB)")
    print(f"5. Programming only - Tier 1+2 ({sum(ds['size_gb'] for ds in prog_tier1 + prog_tier2):.1f} GB)")
    print(f"6. Programming only - All tiers ({prog_all_size:.1f} GB)")
    print(f"7. Both English + Programming - Tier 1 ({sum(ds['size_gb'] for ds in english_tier1 + prog_tier1):.1f} GB)")
    print(f"8. Both English + Programming - Tier 1+2 ({sum(ds['size_gb'] for ds in english_tier1 + english_tier2 + prog_tier1 + prog_tier2):.1f} GB)")
    print(f"9. Everything! ({english_all_size + prog_all_size:.1f} GB) - Full collection")
    print("0. Exit")
    
    choice = input("\nSelect option (0-9): ").strip()
    
    datasets_to_download = []
    
    if choice == "0":
        print("Exiting...")
        return
    elif choice == "1":
        datasets_to_download = english_tier1
    elif choice == "2":
        datasets_to_download = english_tier1 + english_tier2
    elif choice == "3":
        datasets_to_download = ENGLISH_DATASETS
    elif choice == "4":
        datasets_to_download = prog_tier1
    elif choice == "5":
        datasets_to_download = prog_tier1 + prog_tier2
    elif choice == "6":
        datasets_to_download = PROGRAMMING_DATASETS
    elif choice == "7":
        datasets_to_download = english_tier1 + prog_tier1
    elif choice == "8":
        datasets_to_download = english_tier1 + english_tier2 + prog_tier1 + prog_tier2
    elif choice == "9":
        datasets_to_download = ALL_DATASETS
    else:
        print("❌ Invalid choice")
        return
    
    total_size = sum(ds["size_gb"] for ds in datasets_to_download)
    base_download_path = BASE_DOWNLOAD_PATH
    print(f"\n{'='*80}")
    print(f"📥 Downloading {len(datasets_to_download)} datasets (~{total_size:.1f} GB)")
    print(f"{'='*80}")
    print(f"⏰ This will take a while depending on your internet speed...")
    print(f"💾 All data will be saved to: {base_download_path}")
    print(f"🗂️  Cache location: {CACHE_DIR}")
    print(f"📁 Temp location: {TEMP_DIR}")
    print(f"\n⚠️  NO files will be saved to C drive!")
    
    confirm = input("\nConfirm and start download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Download cancelled.")
        return
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for i, ds_info in enumerate(datasets_to_download, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(datasets_to_download)}] {ds_info['name']} ({ds_info['category']})")
        print(f"{'='*80}")
        
        output_path = base_download_path / ds_info["name"].lower().replace(" ", "_").replace("-", "_")
        if output_path.exists():
            data_files = list(output_path.glob("*.arrow")) + list(output_path.glob("data/*.arrow"))
            if data_files:
                results["skipped"].append(ds_info["name"])
                size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                print(f"   ✅ Already downloaded: {size_mb:.1f} MB")
                continue
        
        success = download_dataset(ds_info, base_download_path)
        if success:
            results["success"].append(ds_info["name"])
        else:
            results["failed"].append(ds_info["name"])
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    
    if results["success"]:
        print(f"\n✅ Successfully Downloaded: {len(results['success'])}")
        for name in results["success"]:
            print(f"   • {name}")
    
    if results["skipped"]:
        print(f"\n⏭️  Skipped (already exist): {len(results['skipped'])}")
        for name in results["skipped"]:
            print(f"   • {name}")
    
    if results["failed"]:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name in results["failed"]:
            print(f"   • {name}")
    
    # Verify no C drive usage
    print(f"\n🔍 Verifying C drive is clean...")
    if verify_no_c_drive_usage():
        print("✅ No recent files on C drive - all downloads went to Z drive!")
    
    # Show final directory contents
    print(f"\n📁 Available datasets in {base_download_path}:")
    datasets_path = base_download_path
    if datasets_path.exists():
        total_size = 0
        total_samples = 0
        for path in sorted(datasets_path.iterdir()):
            if path.is_dir() and not path.name.startswith("."):
                data_files = list(path.glob("*.arrow")) + list(path.glob("data/*.arrow"))
                if data_files:
                    size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)
                    total_size += size_mb
                    sample_count = "?"
                    try:
                        from datasets import load_from_disk
                        ds = load_from_disk(str(path))
                        if hasattr(ds, '__len__'):
                            sample_count = f"{len(ds):,}"
                            total_samples += len(ds)
                    except:
                        pass
                    print(f"   • {path.name}: {size_mb:.1f} MB ({sample_count} samples)")
        
        print(f"\n📈 Total: {total_size/1024:.2f} GB across {len(list(datasets_path.glob('*')))-2} datasets")  # -2 for .cache and .temp
    
    print(f"\n✅ Complete! Ready to train your AI!")
    print(f"\n📖 To use these datasets:")
    print(f"   1. Open the AI-OS GUI: aios gui")
    print(f"   2. Go to HRM Training Panel")
    print(f"   3. Set dataset path to: {base_download_path}\\<dataset_name>")
    print(f"   4. Disable 'Teacher-as-dataset' option")
    print(f"   5. Configure other training parameters")
    print(f"   6. Click 'Start Training'!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
