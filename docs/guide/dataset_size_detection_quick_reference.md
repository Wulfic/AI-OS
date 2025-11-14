# Quick Reference: HuggingFace Dataset Size Detection

## For Developers

### Import and Use

```python
from src.aios.gui.components.dataset_download_panel.hf_size_detection import (
    get_hf_dataset_metadata,
    enrich_dataset_with_size,
    format_size_display
)

# Get metadata for any HuggingFace dataset
info = get_hf_dataset_metadata("stanfordnlp/imdb")
print(f"Rows: {info['num_rows']:,}")
print(f"Size: {info['size_gb']:.2f} GB")
print(f"Blocks: {info['total_blocks']}")

# Enrich existing dataset dictionary
dataset = {"path": "stanfordnlp/imdb", "name": "IMDB"}
enriched = enrich_dataset_with_size(dataset)
# Now has: size_gb, num_rows, total_blocks, etc.

# Format for display
size_str, rows_str, blocks_str = format_size_display(enriched)
print(f"{size_str} | {rows_str} | {blocks_str}")
```

## CLI Testing

```bash
# Test a specific dataset
python src/aios/gui/components/dataset_download_panel/hf_size_detection.py stanfordnlp/imdb

# Test with config and split
python src/aios/gui/components/dataset_download_panel/hf_size_detection.py wikipedia 20220301.en train

# Run built-in tests
python src/aios/gui/components/dataset_download_panel/hf_size_detection.py
```

## Understanding the Output

### Dataset Metadata Fields

When you call `get_hf_dataset_metadata()`, you get:

```python
{
    "num_rows": 25000,              # Total rows in dataset
    "num_rows_estimated": False,     # True if estimated from file size
    "num_bytes": 20971520,          # Size in bytes
    "size_mb": 20.0,                # Size in megabytes
    "size_gb": 0.02,                # Size in gigabytes
    "total_blocks": 1,              # Number of 100k-sample blocks
    "samples_per_block": 100000,    # Block size constant
    "is_partial": False,            # True if data incomplete
    "source": "dataset_viewer_api"  # Which API provided data
}
```

### Block Calculation

```
Blocks = ceil(total_samples / 100,000)

Examples:
- 25,000 rows = 1 block
- 150,000 rows = 2 blocks
- 6,458,670 rows = 65 blocks
```

## Integration Examples

### In Search Results

The Dataset Download Panel automatically enriches search results:

```python
# This happens automatically in search_operations.py
for ds in results:
    enrich_dataset_with_size(ds)  # Adds size info
```

### In Custom Code

```python
# Get size for a specific dataset
from src.aios.gui.components.dataset_download_panel.hf_size_detection import get_hf_dataset_metadata

dataset_name = "stanfordnlp/imdb"
metadata = get_hf_dataset_metadata(dataset_name)

if metadata:
    print(f"Dataset: {dataset_name}")
    print(f"  Size: {metadata['size_gb']:.2f} GB")
    print(f"  Rows: {metadata['num_rows']:,}")
    print(f"  Blocks: {metadata['total_blocks']}")
    print(f"  Source: {metadata['source']}")
    
    if metadata['num_rows_estimated']:
        print("  Note: Row count is estimated")
else:
    print(f"Could not get size info for {dataset_name}")
```

## API Details

### Dataset Viewer API (Primary)

**Endpoint**: `https://datasets-server.huggingface.co/size?dataset={name}`

**Pros**:
- Fast (100-500ms)
- Exact row counts
- No auth needed
- No library required

**Cons**:
- Not all datasets supported
- Some return partial data

### Hub API (Fallback)

**Method**: `HfApi().dataset_info(repo_id=name, files_metadata=True)`

**Pros**:
- Works for any dataset
- Reliable
- Comprehensive file info

**Cons**:
- Slower (200-800ms)
- Only file sizes (must estimate rows)
- Requires huggingface_hub library

> **Auto fallback:** When the Dataset Viewer APIs return errors, AI-OS now automatically falls back to the Hub API (if `huggingface_hub` is installed). Disable this behavior with `AIOS_AUTO_ENABLE_HF_HUB_FALLBACK=0` or force-enable/disable it explicitly with `AIOS_ENABLE_HF_HUB_FALLBACK`.

## Troubleshooting

### "Unknown" size displayed

**Cause**: Both APIs failed to get size info

**Solutions**:
1. Check dataset name is correct
2. Verify network connection
3. Check if dataset was renamed on HuggingFace
4. Try accessing the dataset directly on HuggingFace Hub
5. If you intentionally disabled the Hub fallback, consider re-enabling it when Dataset Viewer outages occur (`AIOS_AUTO_ENABLE_HF_HUB_FALLBACK=1`).

### Row count marked "(est.)"

**Cause**: Dataset Viewer API unavailable, using Hub API with estimation

**Info**: 
- Estimated as: `total_bytes / 500` (assumes 500 bytes per row)
- Generally accurate within 20-30%
- More accurate for text datasets

### Slow size detection

**Cause**: Network latency or API slowdown

**Solutions**:
1. Results are cached with search results
2. Size detection runs in background thread
3. Individual failures don't block UI

## Constants

```python
SAMPLES_PER_BLOCK = 100000       # Standard block size
DEFAULT_BYTES_PER_ROW = 500      # For estimation
API_TIMEOUT = 10                 # Seconds
```

## Return Values

### Success

```python
metadata = {
    "num_rows": int,
    "size_gb": float,
    "total_blocks": int,
    # ... other fields
}
```

### Failure

```python
metadata = None  # All methods failed
```

When integrated into datasets:
```python
dataset = {
    "size_gb": 0.0,        # Default if detection fails
    "num_rows": 0,         # Default
    "total_blocks": 0,     # Default
    # ... other fields
}
```
