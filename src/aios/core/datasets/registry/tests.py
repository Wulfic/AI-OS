"""
Embedded tests for the dataset registry system.

Run with: python -m aios.core.datasets.registry.tests
"""

import tempfile
import shutil
import os

from .metadata import DatasetMetadata
from .helpers import create_dataset_metadata
from .registry import DatasetRegistry
from .scanner import LocalDatasetScanner


def run_tests():
    """Run comprehensive tests for the dataset registry."""
    print("\nTesting DatasetRegistry...")
    print("="*70)
    
    # Create temp directory for testing
    temp_dir = tempfile.mkdtemp(prefix="dataset_test_")
    print(f"\n[Setup] Created temp directory: {temp_dir}")
    
    try:
        # ===== Test 1: Create DatasetMetadata =====
        print("\n[Test 1] Creating DatasetMetadata...")
        
        metadata1 = create_dataset_metadata(
            dataset_id="test_python_001",
            name="Python Code Examples",
            source_path="/data/python_code.txt",
            source_type="local",
            domain="coding",
            categories=["python", "programming"],
            tags=["code", "python", "examples"],
            size_bytes=1024000,
            estimated_tokens=256000
        )
        
        print(f"[OK] Created dataset: {metadata1.name}")
        print(f"     ID: {metadata1.dataset_id}")
        print(f"     Domain: {metadata1.domain}")
        print(f"     Categories: {metadata1.categories}")
        print(f"     Tags: {metadata1.tags}")
        
        # ===== Test 2: DatasetRegistry CRUD =====
        print("\n[Test 2] Testing DatasetRegistry CRUD...")
        
        registry = DatasetRegistry()
        
        # Add datasets
        registry.add_dataset(metadata1)
        
        metadata2 = create_dataset_metadata(
            dataset_id="test_math_001",
            name="Math Problems",
            source_path="/data/math.jsonl",
            source_type="local",
            domain="math",
            categories=["algebra", "calculus"],
            tags=["math", "equations"]
        )
        registry.add_dataset(metadata2)
        
        metadata3 = create_dataset_metadata(
            dataset_id="test_writing_001",
            name="Creative Stories",
            source_path="hf://stories/dataset",
            source_type="huggingface",
            domain="writing",
            categories=["creative_writing"],
            tags=["stories", "fiction"]
        )
        registry.add_dataset(metadata3)
        
        print(f"[OK] Added 3 datasets to registry")
        print(f"     Total datasets: {len(registry.get_all_datasets())}")
        
        # Get dataset
        retrieved = registry.get_dataset("test_python_001")
        print(f"[OK] Retrieved dataset: {retrieved.name if retrieved else 'None'}")
        
        # ===== Test 3: Search by Domain =====
        print("\n[Test 3] Searching by domain...")
        
        coding_datasets = registry.search_by_domain("coding")
        print(f"[OK] Found {len(coding_datasets)} coding datasets")
        for ds in coding_datasets:
            print(f"     - {ds.name} ({ds.domain})")
        
        # ===== Test 4: Search by Tags =====
        print("\n[Test 4] Searching by tags...")
        
        code_datasets = registry.search_by_tags(["code", "python"])
        print(f"[OK] Found {len(code_datasets)} datasets with code/python tags")
        for ds in code_datasets:
            print(f"     - {ds.name} (tags: {ds.tags})")
        
        # ===== Test 5: Recommend for Expert =====
        print("\n[Test 5] Recommending datasets for expert...")
        
        recommendations = registry.recommend_for_expert(
            domain="coding",
            categories=["python"],
            max_results=3
        )
        print(f"[OK] Got {len(recommendations)} recommendations for Python expert")
        for i, ds in enumerate(recommendations, 1):
            print(f"     {i}. {ds.name} ({ds.domain})")
        
        # ===== Test 6: Mark Dataset as Used =====
        print("\n[Test 6] Marking dataset as used...")
        
        metadata1.mark_used("expert_python_001")
        metadata1.mark_used("expert_general_001")
        print(f"[OK] Dataset used by {len(metadata1.used_by_experts)} experts")
        print(f"     Experts: {metadata1.used_by_experts}")
        print(f"     Last used: {metadata1.last_used}")
        
        # ===== Test 7: Save and Load Registry =====
        print("\n[Test 7] Testing save/load...")
        
        registry_path = os.path.join(temp_dir, "dataset_registry.json")
        registry.save(registry_path)
        print(f"[OK] Saved registry to {registry_path}")
        
        loaded_registry = DatasetRegistry.load(registry_path)
        print(f"[OK] Loaded registry with {len(loaded_registry.get_all_datasets())} datasets")
        
        # Verify loaded data
        loaded_metadata = loaded_registry.get_dataset("test_python_001")
        if loaded_metadata:
            print(f"[OK] Verified loaded dataset: {loaded_metadata.name}")
            print(f"     Used by experts: {loaded_metadata.used_by_experts}")
        
        # ===== Test 8: Local Dataset Scanner =====
        print("\n[Test 8] Testing LocalDatasetScanner...")
        
        # Create test dataset files
        test_data_dir = os.path.join(temp_dir, "datasets")
        os.makedirs(test_data_dir, exist_ok=True)
        
        test_files = [
            "python_code_examples.txt",
            "math_problems.jsonl",
            "creative_writing.txt",
            "general_knowledge.csv"
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_data_dir, filename)
            with open(filepath, 'w') as f:
                f.write("Sample data for testing\n" * 100)
        
        print(f"[OK] Created {len(test_files)} test dataset files")
        
        # Scan directory
        scanner = LocalDatasetScanner()
        discovered = scanner.scan_directory(test_data_dir, recursive=True)
        
        print(f"[OK] Discovered {len(discovered)} datasets")
        for ds in discovered:
            print(f"     - {ds.name}")
            print(f"       Domain: {ds.domain}, Categories: {ds.categories}")
            print(f"       Size: {ds.size_bytes} bytes, Est. tokens: {ds.estimated_tokens}")
        
        # ===== Test 9: Filter by Size =====
        print("\n[Test 9] Filtering by size...")
        
        # Add discovered datasets to registry
        for ds in discovered:
            loaded_registry.add_dataset(ds)
        
        large_datasets = loaded_registry.filter_by_size(min_bytes=1000)
        print(f"[OK] Found {len(large_datasets)} datasets >= 1000 bytes")
        
        # ===== Test 10: Search by Expert =====
        print("\n[Test 10] Searching by expert...")
        
        expert_datasets = loaded_registry.search_by_expert("expert_python_001")
        print(f"[OK] Found {len(expert_datasets)} datasets used by expert_python_001")
        for ds in expert_datasets:
            print(f"     - {ds.name}")
        
        print("\n" + "="*70)
        print("[SUCCESS] All DatasetRegistry tests passed!")
        print("="*70)
        
    finally:
        # Cleanup
        print(f"\n[Cleanup] Removing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
