# Dataset Download Validation System

## Overview

The dataset download system now includes comprehensive pre-download validation to prevent storage issues and provide users with detailed information before downloads begin.

## Validation Flow

When a user initiates a dataset download, the system performs the following checks in order:

### 1. Size Detection & Estimation

**Location:** `hf_size_detection.py`

- Attempts to fetch exact size and row count from HuggingFace Dataset Viewer API
- If exact size unavailable, estimates based on modality and row count:
  - **Text datasets:** ~500 bytes per row
  - **Image datasets:** ~100KB per row
  - **Audio datasets:** ~500KB per row
  - **Video datasets:** ~10MB per row
- Warns user if size is unknown but allows proceeding with confirmation

### 2. Storage Cap Validation

**Location:** `download_validation.py` → `check_storage_cap()`

- Checks dataset size against configured storage limit (`datasets_storage_cap_gb` in Settings)
- Includes current storage usage in calculation
- **Blocks download** if size exceeds available cap
- Directs user to Settings to increase limit if needed

**Error Message:**
```
This dataset (X.XX GB) would exceed your dataset storage cap.

Current usage: X.XX GB
Storage cap: X.XX GB
Available: X.XX GB

Please increase the storage limit in Settings > Datasets.
```

### 3. Disk Space Validation

**Location:** `download_validation.py` → `check_disk_space()`

- Checks available disk space on target drive
- Adds 5GB safety margin to required space
- **Blocks download** if insufficient space
- Considers both dataset size and safety margin

**Error Message:**
```
Insufficient disk space on C:\.

Required: X.XX GB (including 5 GB safety margin)
Available: X.XX GB

Please free up disk space before downloading.
```

### 4. User Confirmation Dialog

**Location:** `download_validation.py` → `show_download_confirmation_dialog()`

Shows detailed information before download:
- Dataset name and author
- Dataset size (or "Unknown" with warning)
- Number of rows/samples
- Data type/modality
- Gated dataset warning (if applicable)

User must explicitly confirm to proceed.

## Integration Points

### Download Entry Points

Both download paths go through `download_dataset()` in `download_core.py`:

1. **Search Results → Download Button**
   - Path: `panel_main.py` → `_download_selected()` → `download_core.download_dataset()`
   
2. **Favorites → Direct Download**
   - Path: `panel_main.py` → `_download_dataset_direct()` → `download_core.download_dataset()`

### Validation Execution

**In `download_core.download_dataset()`:**
```python
# Run all validation checks
can_proceed, error_msg = validate_download_prerequisites(dataset, output_path, panel.frame)
if not can_proceed:
    panel.log(f"❌ Download blocked: {error_msg}")
    return

# Show confirmation dialog with full details
if not show_download_confirmation_dialog(dataset, parent_widget=panel.frame):
    panel.log(f"❌ Download cancelled by user")
    return

# Proceed with download only if all checks pass
# ... download logic ...
```

## Configuration

### Storage Limit

Set in Settings panel (stored in application config):
- **Key:** `datasets_storage_cap_gb`
- **Default:** 50 GB (configurable)
- **Access:** Settings → Datasets section

### Safety Margins

**Disk Space Check:**
- Fixed 5GB safety margin
- Helps prevent disk from filling completely
- Located in `download_validation.py:check_disk_space()`

## Error Handling

### Graceful Degradation

If validation systems are unavailable:
- Logs warning but allows download to proceed
- Example: If storage cap system not available, skips that check
- Ensures downloads can proceed even if validation partially fails

### Unknown Sizes

When dataset size cannot be determined:
- Shows warning in confirmation dialog
- Estimates size if row count available
- Allows user to proceed at their own risk
- Suggests checking HuggingFace page for more info

## Testing Scenarios

### Test Case 1: Storage Cap Exceeded
1. Set low storage cap in Settings (e.g., 5GB)
2. Attempt to download large dataset (>5GB)
3. **Expected:** Download blocked with error message directing to Settings

### Test Case 2: Insufficient Disk Space
1. Select drive with limited space
2. Attempt to download dataset larger than available space
3. **Expected:** Download blocked with error showing required vs available space

### Test Case 3: Unknown Size
1. Search for dataset with no size info
2. Attempt download
3. **Expected:** Warning shown, user can proceed after confirmation

### Test Case 4: Normal Download
1. Select reasonably-sized dataset
2. Sufficient space and within cap
3. **Expected:** Confirmation dialog shows details, download proceeds after confirmation

## Files Modified

- **download_validation.py** (NEW) - All validation logic
- **download_core.py** - Integrated validation before download
- **hf_size_detection.py** - Enhanced size estimation
- **panel_main.py** - Removed duplicate confirmation dialog

## Future Improvements

1. **Enhanced Size Detection:**
   - Add more fallback strategies for HF API
   - Cache size information to reduce API calls
   - Better estimation models per modality

2. **Progressive Validation:**
   - Show real-time validation status as user selects datasets
   - Disable download button if validation would fail
   - Color-code datasets by feasibility

3. **Storage Management:**
   - Auto-cleanup of old/unused datasets
   - Storage usage visualization
   - Per-dataset size tracking

4. **User Preferences:**
   - Configurable safety margins
   - Option to skip certain validations
   - Warning thresholds
