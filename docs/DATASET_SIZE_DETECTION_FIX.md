# Dataset Size Detection Fix - December 15, 2025

## Problem

Many datasets were showing "Unknown" for size and row count in the search results, making it impossible to:
- Assess if a dataset would fit within storage limits
- Understand the dataset scale before downloading
- Validate storage prerequisites

## Root Cause

The size detection code was using `fetch_json_sync()` from `hf_async_client.py`, which requires an **active async event loop**. When the enrichment workers ran in background threads (outside the main GUI thread), the async loop wasn't available, causing:

```
RuntimeError: Async event loop is not initialized
```

This caused both the SIZE API and INFO API calls to fail, falling back to the slower Hub API which only provides estimated rows (not exact counts).

## Solution

Replaced `fetch_json_sync()` calls with direct synchronous HTTP requests using `httpx.Client()`:

### Before (Broken):
```python
from ....data.hf_async_client import fetch_json_sync

data = fetch_json_sync(url, timeout=timeout, max_attempts=max_attempts)
# ❌ Requires async loop - fails in worker threads
```

### After (Fixed):
```python
import httpx

_http_client = httpx.Client(timeout=10.0, follow_redirects=True)

response = _http_client.get(url, timeout=timeout)
response.raise_for_status()
data = response.json()
# ✅ Works everywhere - no async loop required
```

## Impact

### Before Fix:
- Most datasets: **"Unknown" size/rows**
- Only Hub API fallback worked (estimated rows)
- Quality: "low" (estimates based on file sizes)

### After Fix:
- Most datasets: **Exact row counts and sizes**
- SIZE API and INFO API working properly
- Quality: "exact" (from HuggingFace Dataset Viewer)

### Test Results:
```
stanfordnlp/imdb: 25,000 rows, 0.020 GB (dataset_viewer_api) ✅
squad: 32,573 rows, 0.015 GB (info_api) ✅
uoft-cs/cifar10: 50,000 rows, 0.111 GB (dataset_viewer_api) ✅
```

## Debug Logging Added

Comprehensive logging was added to track the entire detection pipeline:

1. **[METADATA]** - Main lookup orchestration
2. **[SIZE API]** - Dataset Viewer /size endpoint
3. **[INFO API]** - Dataset Viewer /info endpoint (with feature types)
4. **[ENRICH]** - Enrichment success/failure
5. **[WORKER]** - Background thread enrichment status

Log levels:
- `DEBUG`: API requests, responses, intermediate steps
- `INFO`: Successful enrichments with full details
- `WARNING`: Failures, fallbacks, missing data

## Files Modified

1. **hf_size_detection.py**
   - Removed `fetch_json_sync` dependency
   - Added synchronous `httpx.Client()`
   - Added comprehensive debug logging
   - All API calls now work in any thread context

2. **search_operations.py**
   - Enhanced worker logging
   - Added enrichment summary with failed dataset names
   - Better visibility into background enrichment process

## Verification

To verify the fix is working, check the logs when searching for datasets:

```
✅ [SIZE API] Success for <dataset>: X,XXX rows, X.XXX GB
✅ [INFO API] Success for <dataset>: X,XXX rows, X.XXX GB, features=[...]
✅ [ENRICH] <dataset>: X,XXX rows, X.XXX GB (quality: exact, source: dataset_viewer_api)
```

If you see many:
```
❌ [SIZE API] ... 
❌ [INFO API] ...
⚠️ [ENRICHMENT] Failed datasets: ...
```

This indicates API issues (rate limiting, network, or dataset not in Dataset Viewer).

## Next Steps

With exact size/row detection working:
1. Pre-download validation can now reliably check against storage caps
2. Disk space validation has accurate size estimates
3. Users see real dataset information in search results
4. Download confirmation dialogs show accurate details

## Configuration

No configuration changes needed. The fix is automatic.

The Hub API fallback is still available for datasets not in the Dataset Viewer, but now most datasets will use the faster, more accurate SIZE/INFO APIs.
