#!/usr/bin/env python3
"""
Dataset Tracking Analysis Script

This script demonstrates how to analyze dataset tracking data from brain.json files.
It provides several useful analysis functions for understanding training patterns.

Usage:
    python analyze_dataset_tracking.py <path_to_brain.json>
    python analyze_dataset_tracking.py artifacts/brains/actv1/my-brain/brain.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def load_brain_data(brain_json_path: str) -> Dict[str, Any]:
    """Load brain.json file and return parsed data."""
    with open(brain_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_dataset_summary(brain_data: Dict[str, Any]) -> None:
    """Print a summary of all datasets used to train this brain."""
    print("\n" + "="*70)
    print("DATASET TRAINING SUMMARY")
    print("="*70)
    
    brain_name = brain_data.get('name', 'Unknown')
    total_steps = brain_data.get('training_steps', 0)
    print(f"\nBrain: {brain_name}")
    print(f"Total Training Steps: {total_steps:,}")
    
    dataset_stats = brain_data.get('dataset_stats', {})
    if not dataset_stats:
        print("\nNo dataset statistics available.")
        return
    
    print(f"\nDatasets Used: {len(dataset_stats)}")
    print("\n" + "-"*70)
    
    # Sort by total steps (most used first)
    sorted_datasets = sorted(
        dataset_stats.items(),
        key=lambda x: x[1].get('total_steps', 0),
        reverse=True
    )
    
    for dataset_name, stats in sorted_datasets:
        times_used = stats.get('times_used', 0)
        total_ds_steps = stats.get('total_steps', 0)
        first_used = stats.get('first_used', 0)
        last_used = stats.get('last_used', 0)
        
        first_dt = datetime.fromtimestamp(first_used).strftime('%Y-%m-%d %H:%M')
        last_dt = datetime.fromtimestamp(last_used).strftime('%Y-%m-%d %H:%M')
        
        pct_of_total = (total_ds_steps / total_steps * 100) if total_steps > 0 else 0
        
        print(f"\n📊 {dataset_name}")
        print(f"   Times Used: {times_used}")
        print(f"   Total Steps: {total_ds_steps:,} ({pct_of_total:.1f}% of total)")
        print(f"   First Used: {first_dt}")
        print(f"   Last Used:  {last_dt}")
        print(f"   Path: {stats.get('dataset_path', 'Unknown')}")


def print_training_timeline(brain_data: Dict[str, Any]) -> None:
    """Print a chronological timeline of all training sessions."""
    print("\n" + "="*70)
    print("TRAINING TIMELINE")
    print("="*70)
    
    dataset_history = brain_data.get('dataset_history', [])
    if not dataset_history:
        print("\nNo training history available.")
        return
    
    print(f"\nTotal Training Sessions: {len(dataset_history)}")
    print("\n" + "-"*70)
    
    # Sort by timestamp (chronological order)
    sorted_history = sorted(dataset_history, key=lambda x: x.get('timestamp', 0))
    
    cumulative_steps = 0
    for i, session in enumerate(sorted_history, 1):
        dataset_name = session.get('dataset_name', 'Unknown')
        steps = session.get('steps', 0)
        timestamp = session.get('timestamp', 0)
        
        cumulative_steps += steps
        dt = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{i}. {dt}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Steps: {steps:,}")
        print(f"   Cumulative: {cumulative_steps:,} steps")


def print_dataset_effectiveness(brain_data: Dict[str, Any]) -> None:
    """Analyze and print dataset effectiveness metrics."""
    print("\n" + "="*70)
    print("DATASET EFFECTIVENESS ANALYSIS")
    print("="*70)
    
    dataset_stats = brain_data.get('dataset_stats', {})
    if not dataset_stats:
        print("\nNo dataset statistics available.")
        return
    
    print("\nMetrics per training session:")
    print("-"*70)
    
    for dataset_name, stats in dataset_stats.items():
        times_used = stats.get('times_used', 0)
        total_steps = stats.get('total_steps', 0)
        
        if times_used > 0:
            avg_steps_per_session = total_steps / times_used
            print(f"\n📈 {dataset_name}")
            print(f"   Average Steps per Session: {avg_steps_per_session:.0f}")
            print(f"   Total Sessions: {times_used}")
            print(f"   Total Steps: {total_steps:,}")


def print_recommendations(brain_data: Dict[str, Any]) -> None:
    """Print recommendations based on training patterns."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    dataset_stats = brain_data.get('dataset_stats', {})
    dataset_history = brain_data.get('dataset_history', [])
    
    if not dataset_stats or not dataset_history:
        print("\nInsufficient data for recommendations.")
        return
    
    total_steps = brain_data.get('training_steps', 0)
    
    # Find most used dataset
    most_used = max(dataset_stats.items(), key=lambda x: x[1].get('total_steps', 0))
    most_used_name = most_used[0]
    most_used_pct = (most_used[1].get('total_steps', 0) / total_steps * 100) if total_steps > 0 else 0
    
    print("\n💡 Training Insights:")
    
    if most_used_pct > 70:
        print(f"\n   ⚠️  '{most_used_name}' accounts for {most_used_pct:.1f}% of training.")
        print("   Consider diversifying training data to improve generalization.")
    
    if len(dataset_stats) == 1:
        print("\n   ℹ️  Only one dataset has been used for training.")
        print("   Consider adding more diverse datasets to improve capabilities.")
    
    if len(dataset_stats) > 1:
        print(f"\n   ✅ Good diversity: {len(dataset_stats)} different datasets used.")
        print("   This helps the brain learn from varied examples.")
    
    # Check training recency
    last_session = max(dataset_history, key=lambda x: x.get('timestamp', 0))
    last_timestamp = last_session.get('timestamp', 0)
    days_since = (datetime.now().timestamp() - last_timestamp) / 86400
    
    if days_since > 7:
        print(f"\n   📅 Last training was {days_since:.0f} days ago.")
        print("   Consider resuming training to keep the brain up-to-date.")


def main():
    """Main function to run analysis."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_dataset_tracking.py <path_to_brain.json>")
        print("\nExample:")
        print("  python analyze_dataset_tracking.py artifacts/brains/actv1/my-brain/brain.json")
        sys.exit(1)
    
    brain_json_path = sys.argv[1]
    
    if not Path(brain_json_path).exists():
        print(f"Error: File not found: {brain_json_path}")
        sys.exit(1)
    
    try:
        brain_data = load_brain_data(brain_json_path)
        
        # Run all analyses
        print_dataset_summary(brain_data)
        print_training_timeline(brain_data)
        print_dataset_effectiveness(brain_data)
        print_recommendations(brain_data)
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70 + "\n")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {brain_json_path}")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing brain data: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
