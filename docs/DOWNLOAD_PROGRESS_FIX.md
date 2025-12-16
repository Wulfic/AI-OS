# Dataset Download Progress - Block-Based Tracking Fix

## Issue
The dataset download progress percentage was showing values over 100% (e.g., "101%") during downloads. This occurred because the progress calculation was based on the number of samples downloaded rather than the number of blocks completed.

## Root Cause
- Downloads use a block-based structure (100k samples per block)
- Progress tracker was counting individual samples
- When a dataset had, for example, 250k samples, it would be split into 3 blocks
- The tracker would count samples up to 250k, but divide by the original estimate
- Rounding and estimation errors caused percentages to exceed 100%

## Solution
Updated the progress tracking system to use **blocks completed** as the primary progress metric:

### Changes Made

1. **Enhanced DownloadStats** ([download_progress.py](../src/aios/gui/components/dataset_download_panel/download_progress.py))
   - Added `blocks_completed: int` field
   - Added `total_blocks: int` field

2. **Updated DownloadProgressTracker** ([download_progress.py](../src/aios/gui/components/dataset_download_panel/download_progress.py))
   - Added `total_blocks` parameter to `__init__`
   - Added block tracking to `set_progress()` method
   - Added `total_blocks` to `set_totals()` method

3. **Modified Progress Calculation** ([ui_builder.py](../src/aios/gui/components/dataset_download_panel/ui_builder.py))
   - Changed percentage calculation to prefer blocks over samples/bytes
   - Added clamping to prevent showing >100%
   - Priority order: blocks → bytes → samples

4. **Updated Download Core** ([download_core.py](../src/aios/gui/components/dataset_download_panel/download_core.py))
   - Calculate `total_blocks` from dataset metadata
   - Update `blocks_completed` when blocks are flushed to disk
   - Pass block information to progress tracker

## Behavior

### Before Fix
```
Downloading dataset...
Progress: 50,000/100,000 (50%)
Progress: 100,000/100,000 (100%)
Progress: 101,000/100,000 (101%)  ← Problem!
Progress: 102,000/100,000 (102%)  ← Problem!
```

### After Fix
```
Downloading dataset...
Block 0: 50,000 samples (50%)
Block 1: 50,000 samples (100%)
Block 2: 2,000 samples (100%)  ← Clamped
```

Progress percentage is now calculated as:
```
percentage = (blocks_completed / total_blocks) × 100
```

With a maximum cap of 100%.

## Edge Cases Handled

1. **Unknown Total Blocks**: Falls back to byte-based or sample-based calculation
2. **Partial Blocks**: Last block may have fewer samples, but still counts as 1 block
3. **Estimation Errors**: Percentage is clamped to max 100%
4. **Multiple Progress Metrics**: Prioritizes blocks > bytes > samples for accuracy

## Testing

To verify the fix works:

1. Start downloading a large dataset (>100k samples)
2. Observe the progress percentage
3. Verify it never exceeds 100%
4. Check that blocks are counted correctly in logs

Example log output:
```
📦 Downloading: tinystories
   📊 Tracking: 2,119,719 samples expected
   Total blocks: 22
   📦 Block 0 saved: 100,000 samples (4.5%)
   📦 Block 1 saved: 100,000 samples (9.1%)
   ...
   📦 Block 21 saved: 19,719 samples (100.0%)
   ✅ Download complete: 22 blocks
```

## Future Enhancements

Potential improvements for even better progress tracking:

1. **Real-time ETA based on blocks**: Adjust ETA as blocks complete
2. **Block verification**: Mark blocks as verified after write
3. **Resume support**: Track which blocks are already downloaded
4. **Parallel block downloads**: Download multiple blocks simultaneously

## Related Files

- `src/aios/gui/components/dataset_download_panel/download_progress.py`
- `src/aios/gui/components/dataset_download_panel/download_core.py`
- `src/aios/gui/components/dataset_download_panel/ui_builder.py`
- `src/aios/gui/components/dataset_download_panel/block_processor.py`
