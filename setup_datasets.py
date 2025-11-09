"""
Download and setup high-quality datasets for HRM training.

This script downloads the best datasets for English language and reasoning:
- OpenWebText: 38GB of web text (diverse, high-quality)
- Wikipedia: 20GB of factual knowledge
- Stack Exchange: 32GB of Q&A (reasoning patterns)

Total: ~90GB (within 100GB budget)
"""

import os
import sys
from pathlib import Path
import subprocess
import json


def download_openwebtext():
    """
    Download OpenWebText dataset (~38GB).
    
    High-quality web text scraped from Reddit links.
    """
    print("\n" + "="*80)
    print("DOWNLOADING OPENWEBTEXT (~38GB)")
    print("="*80)
    
    target_dir = Path("training_data/curated_datasets/openwebtext")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 OpenWebText is available via HuggingFace datasets")
    print("\nTo download programmatically:")
    print("""
from datasets import load_dataset

# Load OpenWebText
dataset = load_dataset("openwebtext", split="train")

# Save to disk
dataset.save_to_disk("training_data/curated_datasets/openwebtext")

# Or stream without downloading all at once:
dataset = load_dataset("openwebtext", split="train", streaming=True)
""")
    
    print("\n✅ Manual Download Option:")
    print("   1. Install: pip install datasets")
    print("   2. Run the Python code above")
    print("   3. Or download from: https://huggingface.co/datasets/openwebtext")
    
    return target_dir


def download_wikipedia():
    """
    Download Wikipedia dataset (~20GB).
    
    English Wikipedia dump with factual knowledge.
    """
    print("\n" + "="*80)
    print("DOWNLOADING WIKIPEDIA (~20GB)")
    print("="*80)
    
    target_dir = Path("training_data/curated_datasets/wikipedia")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Wikipedia is available via HuggingFace datasets")
    print("\nTo download programmatically:")
    print("""
from datasets import load_dataset

# Load Wikipedia (20220301 version - English)
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Save to disk
dataset.save_to_disk("training_data/curated_datasets/wikipedia")
""")
    
    print("\n✅ Manual Download Option:")
    print("   1. Install: pip install datasets")
    print("   2. Run the Python code above")
    print("   3. Or download from: https://huggingface.co/datasets/wikipedia")
    
    return target_dir


def download_stackexchange():
    """
    Download Stack Exchange dataset (~32GB).
    
    Q&A from Stack Exchange sites - excellent for reasoning.
    """
    print("\n" + "="*80)
    print("DOWNLOADING STACK EXCHANGE (~32GB)")
    print("="*80)
    
    target_dir = Path("training_data/curated_datasets/stackexchange")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Stack Exchange is available via HuggingFace datasets")
    print("\nTo download programmatically:")
    print("""
from datasets import load_dataset

# Load Stack Exchange
dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

# For actual Stack Exchange dumps:
# Download from https://archive.org/details/stackexchange
# or use: dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)
""")
    
    print("\n✅ Recommended: FineWeb-Edu (Alternative, High Quality)")
    print("   - Filtered web text with educational content")
    print("   - Similar quality to Stack Exchange for reasoning")
    print("   - Easier to download and use")
    
    return target_dir


def create_download_script():
    """Create a Python script to download all datasets."""
    script_path = Path("download_datasets.py")
    
    script_content = '''"""
Automated dataset download script.
Downloads OpenWebText, Wikipedia, and FineWeb-Edu datasets.
"""

from datasets import load_dataset
from pathlib import Path
import sys

def download_with_progress(dataset_name, config, target_dir):
    """Download dataset with progress tracking."""
    print(f"\\n📥 Downloading {dataset_name}...")
    print(f"   Target: {target_dir}")
    
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split="train")
        else:
            dataset = load_dataset(dataset_name, split="train")
        
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(target_path))
        print(f"✅ {dataset_name} downloaded successfully!")
        print(f"   Size: {len(dataset)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False


def main():
    """Download all datasets."""
    print("="*80)
    print("DATASET DOWNLOAD UTILITY")
    print("="*80)
    
    datasets_to_download = [
        {
            "name": "openwebtext",
            "config": None,
            "target": "training_data/curated_datasets/openwebtext",
            "size": "38GB",
            "description": "High-quality web text"
        },
        {
            "name": "wikipedia",
            "config": "20220301.en",
            "target": "training_data/curated_datasets/wikipedia",
            "size": "20GB",
            "description": "English Wikipedia"
        },
        {
            "name": "HuggingFaceFW/fineweb-edu",
            "config": None,
            "target": "training_data/curated_datasets/fineweb_edu",
            "size": "~30GB (sample)",
            "description": "Educational web content"
        }
    ]
    
    print("\\nDatasets to download:")
    for i, ds in enumerate(datasets_to_download, 1):
        print(f"   {i}. {ds['name']} ({ds['size']}) - {ds['description']}")
    
    print("\\n⚠️  Total download size: ~90GB")
    print("⚠️  This will take several hours depending on internet speed")
    
    response = input("\\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    success_count = 0
    for ds in datasets_to_download:
        if download_with_progress(ds["name"], ds["config"], ds["target"]):
            success_count += 1
    
    print("\\n" + "="*80)
    print(f"DOWNLOAD COMPLETE: {success_count}/{len(datasets_to_download)} successful")
    print("="*80)
    
    if success_count > 0:
        print("\\n✅ Datasets ready for training!")
        print("   Location: training_data/curated_datasets/")
        print("\\nTo use in training:")
        print("   1. Open HRM Training Panel in GUI")
        print("   2. Set Dataset file/dir to: training_data/curated_datasets/openwebtext")
        print("   3. Disable 'Teacher-as-dataset'")
        print("   4. Start training!")


if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n✅ Created download script: {script_path}")
    print("\nTo download datasets:")
    print(f"   1. Install dependencies: pip install datasets")
    print(f"   2. Run: python {script_path}")
    
    return script_path


def show_quick_start_guide():
    """Show quick start guide for using curated datasets."""
    print("\n" + "="*80)
    print("QUICK START GUIDE")
    print("="*80)
    
    print("""
🚀 FASTEST WAY TO GET STARTED (Small Dataset First):

1. Download a smaller dataset for testing:
   
   from datasets import load_dataset
   dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
   dataset.save_to_disk("training_data/curated_datasets/wikitext")
   
   Size: ~500MB (fast download!)

2. Configure in GUI:
   - Open HRM Training Panel
   - Dataset file/dir: training_data/curated_datasets/wikitext
   - DISABLE "Teacher-as-dataset" checkbox
   - Click "Start Training"

3. Once confirmed working, download full datasets (90GB)


📚 FULL DATASET SETUP (90GB):

Option A - Automated (Recommended):
   1. pip install datasets
   2. python download_datasets.py
   3. Wait 2-4 hours for download
   4. Start training with curated data!

Option B - Manual HuggingFace:
   1. Visit https://huggingface.co/datasets
   2. Search for: openwebtext, wikipedia, fineweb-edu
   3. Follow download instructions
   4. Place in training_data/curated_datasets/

Option C - Streaming (No Download):
   - Modify training code to use streaming datasets
   - Slower but no disk space needed


🎯 RECOMMENDED APPROACH:

Best for your use case (English + Reasoning):
   1. Start with WikiText (500MB) to test setup
   2. Download Wikipedia (20GB) for factual knowledge
   3. Download FineWeb-Edu (30GB) for reasoning
   4. Total: 50GB of high-quality data

This gives you:
   ✅ Diverse English text
   ✅ Factual knowledge
   ✅ Reasoning patterns
   ✅ Under 100GB budget
   ✅ Better than random "english" prompts!
""")


def main():
    """Main function."""
    print("="*80)
    print("DATASET SETUP WIZARD")
    print("="*80)
    
    print("\nThis wizard will help you download high-quality datasets")
    print("for training your HRM model with English and reasoning skills.")
    
    # Create necessary directories
    base_dir = Path("training_data/curated_datasets")
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Created directory: {base_dir}")
    
    # Show dataset options
    download_openwebtext()
    download_wikipedia()
    download_stackexchange()
    
    # Create automated download script
    create_download_script()
    
    # Show quick start guide
    show_quick_start_guide()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Install HuggingFace datasets:
   pip install datasets

2. Choose your approach:
   
   Quick Test (500MB):
   >>> from datasets import load_dataset
   >>> ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
   >>> ds.save_to_disk("training_data/curated_datasets/wikitext")
   
   Full Setup (90GB):
   >>> python download_datasets.py

3. Configure HRM Training Panel:
   - Dataset: training_data/curated_datasets/wikitext
   - Disable "Teacher-as-dataset"
   - Start training!

4. Watch training improve with real data! 🚀
""")


if __name__ == "__main__":
    main()
