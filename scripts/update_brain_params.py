#!/usr/bin/env python3
"""Update existing brain.json files to include total_params field.

This script:
1. Finds all brain.json files in artifacts/brains/
2. Reads architecture from each brain.json
3. Calculates total_params using calculate_actv1_params()
4. Updates brain.json with total_params and model_size_mb
5. Creates backup before modifying
"""

import json
import shutil
from pathlib import Path
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from aios.cli.hrm_hf.model_building import calculate_actv1_params


def update_brain_json(brain_path: Path) -> dict:
    """Update a single brain.json file with calculated params.
    
    Returns:
        Dict with keys: 'updated', 'skipped', 'error', 'message'
    """
    try:
        # Read existing brain.json
        with open(brain_path, 'r', encoding='utf-8') as f:
            brain_data = json.load(f)
        
        # Check if already has total_params
        if 'total_params' in brain_data:
            return {
                'updated': False,
                'skipped': True,
                'message': f"Already has total_params: {brain_data['total_params']:,}"
            }
        
        # Get architecture details
        vocab_size = brain_data.get('vocab_size')
        hidden_size = brain_data.get('hidden_size')
        h_layers = brain_data.get('h_layers')
        l_layers = brain_data.get('l_layers')
        expansion = brain_data.get('expansion', 2.0)
        use_moe = brain_data.get('use_moe', False)
        num_experts = brain_data.get('num_experts', 1)
        
        # Validate required fields
        if not all([vocab_size, hidden_size, h_layers is not None, l_layers is not None]):
            return {
                'updated': False,
                'error': True,
                'message': f"Missing required architecture fields"
            }
        
        # Calculate params
        total_params = calculate_actv1_params(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            h_layers=h_layers,
            l_layers=l_layers,
            expansion=expansion,
            use_moe=use_moe,
            num_experts=num_experts
        )
        
        # Calculate size in MB (4 bytes per param for float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Create backup
        backup_path = brain_path.with_suffix('.json.backup')
        shutil.copy2(brain_path, backup_path)
        
        # Add calculated fields
        brain_data['total_params'] = total_params
        brain_data['model_size_mb'] = round(model_size_mb, 2)
        
        # Write updated brain.json
        with open(brain_path, 'w', encoding='utf-8') as f:
            json.dump(brain_data, f, indent=2)
        
        return {
            'updated': True,
            'message': f"Added total_params: {total_params:,} ({model_size_mb:.2f} MB)",
            'backup': str(backup_path)
        }
    
    except Exception as e:
        return {
            'updated': False,
            'error': True,
            'message': f"Error: {e}"
        }


def main():
    """Find and update all brain.json files."""
    brains_dir = repo_root / "artifacts" / "brains"
    
    if not brains_dir.exists():
        print(f"❌ Brains directory not found: {brains_dir}")
        return 1
    
    # Find all brain.json files
    brain_files = list(brains_dir.rglob("brain.json"))
    
    if not brain_files:
        print(f"ℹ️  No brain.json files found in {brains_dir}")
        return 0
    
    print(f"Found {len(brain_files)} brain.json file(s)\n")
    
    # Update each brain
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for brain_path in brain_files:
        brain_name = brain_path.parent.name
        print(f"Processing: {brain_name}")
        
        result = update_brain_json(brain_path)
        
        if result.get('updated'):
            print(f"  ✅ {result['message']}")
            print(f"  📁 Backup: {result['backup']}")
            updated_count += 1
        elif result.get('skipped'):
            print(f"  ⏭️  {result['message']}")
            skipped_count += 1
        elif result.get('error'):
            print(f"  ❌ {result['message']}")
            error_count += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"✅ Updated: {updated_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Errors:  {error_count}")
    print(f"📊 Total:   {len(brain_files)}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
