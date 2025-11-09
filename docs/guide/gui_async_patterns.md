# GUI Async Patterns & Worker Pool Usage Guide

**Author**: AI-OS Development Team  
**Last Updated**: November 2025  
**Applies to**: AI-OS GUI v2.0+

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Worker Pool Patterns](#worker-pool-patterns)
4. [Common Patterns](#common-patterns)
5. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
6. [Best Practices](#best-practices)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The AI-OS GUI uses an async worker pool architecture to keep the interface responsive while performing blocking operations like subprocess calls, file I/O, and network requests.

### Key Components

- **AsyncWorkerPool**: Thread pool for background operations
- **AsyncEventLoop**: Dedicated asyncio loop for async/await patterns
- **TimerManager**: Debounced timer management
- **ProcessReaper**: Subprocess cleanup

### Performance Characteristics

- **Worker Pool Size**: `(cpu_count * 4) + 1`, minimum 12 workers
- **CLI Cache TTL**: 2 seconds for read-only operations
- **Refresh Throttling**: 5 seconds for panel refreshes
- **Configurable**: Set `AIOS_WORKER_THREADS` environment variable

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tkinter Main Thread                  │
│  - Event Loop                                           │
│  - UI Updates                                           │
│  - Event Handlers                                       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Submit work via worker_pool.submit()
                   ↓
┌─────────────────────────────────────────────────────────┐
│              AsyncWorkerPool (12-50 threads)            │
│  - CLI subprocess calls                                 │
│  - File I/O operations                                  │
│  - Network requests                                     │
│  - Heavy computations                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Schedule UI updates via .after(0, callback)
                   ↓
┌─────────────────────────────────────────────────────────┐
│              Tkinter Main Thread (UI Update)            │
│  - Update widgets                                       │
│  - Refresh displays                                     │
│  - Show results                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Worker Pool Patterns

### Pattern 1: Simple Background Operation

**Use Case**: Single async operation with UI update

```python
def refresh(self):
    """Refresh panel data asynchronously."""
    
    def _do_work():
        # Background work (blocking I/O, subprocess, etc.)
        data = self._fetch_data()  # Blocking call OK here
        
        # Schedule UI update on main thread
        def _update_ui():
            self._display_data(data)
        
        try:
            self.after(0, _update_ui)
        except Exception:
            pass  # Widget destroyed
    
    # Submit to worker pool
    if self._worker_pool:
        self._worker_pool.submit(_do_work)
    else:
        # Fallback if worker pool unavailable
        import threading
        threading.Thread(target=_do_work, daemon=True).start()
```

### Pattern 2: Operation with Loading Indicator

**Use Case**: Long operation with user feedback

```python
def refresh(self):
    """Refresh with loading indicator."""
    
    # Prevent duplicate operations
    if hasattr(self, '_loading') and self._loading:
        return
    
    self._loading = True
    self.status_var.set("Loading...")
    
    def _do_work():
        try:
            data = self._fetch_data()
            
            def _update_ui():
                self._display_data(data)
                self.status_var.set("")
                self._loading = False
            
            self.after(0, _update_ui)
        except Exception as e:
            def _handle_error():
                self.status_var.set(f"Error: {e}")
                self._loading = False
            
            self.after(0, _handle_error)
    
    self._worker_pool.submit(_do_work)
```

### Pattern 3: Throttled Operation

**Use Case**: Prevent excessive calls (e.g., rapid refresh)

```python
def refresh(self, force: bool = False):
    """Refresh with throttling."""
    
    # Throttle unless forced
    if not force:
        if not hasattr(self, '_last_refresh'):
            self._last_refresh = 0.0
        
        import time
        now = time.time()
        if now - self._last_refresh < 5.0:  # 5-second throttle
            return  # Too soon, skip
        
        self._last_refresh = now
    
    # Proceed with refresh (use Pattern 2)
    # ...
```

### Pattern 4: Multiple Concurrent Operations

**Use Case**: Parallel data fetching

```python
def refresh_all(self):
    """Refresh multiple data sources concurrently."""
    
    results = {'brains': None, 'datasets': None, 'models': None}
    completed = {'count': 0}
    total = len(results)
    
    def _fetch_and_update(key, fetch_fn):
        try:
            data = fetch_fn()
            results[key] = data
        except Exception as e:
            results[key] = None
        
        # Update UI when all complete
        completed['count'] += 1
        if completed['count'] == total:
            def _update_all():
                self._display_brains(results['brains'])
                self._display_datasets(results['datasets'])
                self._display_models(results['models'])
            
            self.after(0, _update_all)
    
    # Submit all tasks concurrently
    self._worker_pool.submit(_fetch_and_update, 'brains', self._fetch_brains)
    self._worker_pool.submit(_fetch_and_update, 'datasets', self._fetch_datasets)
    self._worker_pool.submit(_fetch_and_update, 'models', self._fetch_models)
```

---

## Common Patterns

### CLI Subprocess Calls

```python
# ❌ WRONG - Blocks GUI thread
def load_brain(self, name):
    result = self._run_cli(['brains', 'load', name])
    self.display_result(result)

# ✅ CORRECT - Async with worker pool
def load_brain(self, name):
    def _load():
        result = self._run_cli(['brains', 'load', name])
        
        def _show_result():
            self.display_result(result)
        
        self.after(0, _show_result)
    
    self._worker_pool.submit(_load)
```

### File I/O Operations

```python
# ❌ WRONG - Blocks GUI
def save_data(self, data):
    with open('data.json', 'w') as f:
        json.dump(data, f)
    self.status_var.set("Saved")

# ✅ CORRECT - Async file I/O
def save_data(self, data):
    def _save():
        with open('data.json', 'w') as f:
            json.dump(data, f)
        
        def _update_status():
            self.status_var.set("Saved")
        
        self.after(0, _update_status)
    
    self._worker_pool.submit(_save)
```

### Network Requests

```python
# ❌ WRONG - Blocks during download
def download_dataset(self, url):
    response = requests.get(url)
    self.save_dataset(response.content)

# ✅ CORRECT - Async with progress
def download_dataset(self, url):
    def _download():
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            progress = (downloaded / total) * 100
            
            # Update progress bar
            def _update_progress(p=progress):
                self.progress_var.set(p)
            
            self.after(0, _update_progress)
        
        # Final update
        def _complete():
            self.status_var.set("Download complete")
        
        self.after(0, _complete)
    
    self._worker_pool.submit(_download)
```

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Direct UI Updates from Worker Thread

```python
# WRONG - Tkinter is not thread-safe!
def _worker_function():
    data = fetch_data()
    self.label.config(text=data)  # CRASHES or CORRUPTS UI
```

**Problem**: Tkinter widgets can only be modified from the main thread.

**Solution**: Always use `.after(0, callback)` to schedule UI updates.

### ❌ Anti-Pattern 2: Blocking in Main Thread

```python
# WRONG - Freezes GUI
def on_button_click(self):
    time.sleep(5)  # Blocks event loop
    self.label.config(text="Done")
```

**Problem**: Main thread is blocked, GUI becomes unresponsive.

**Solution**: Move blocking work to worker pool.

### ❌ Anti-Pattern 3: Creating Threads Directly

```python
# WRONG - Resource leak, no management
def do_work(self):
    threading.Thread(target=self._work).start()
```

**Problem**: Threads not managed, can accumulate and leak.

**Solution**: Use worker pool for automatic lifecycle management.

### ❌ Anti-Pattern 4: Missing Error Handling

```python
# WRONG - Crashes silently
def _worker():
    data = risky_operation()  # May raise exception
    self.after(0, lambda: self.display(data))
```

**Problem**: Exceptions in worker threads are silent.

**Solution**: Always wrap in try/except with error UI updates.

### ❌ Anti-Pattern 5: No Progress Feedback

```python
# WRONG - User thinks app is frozen
def long_operation(self):
    self._worker_pool.submit(self._do_long_work)
```

**Problem**: No visual feedback during long operations.

**Solution**: Add loading indicators and status updates.

---

## Best Practices

### 1. Always Update UI from Main Thread

```python
# Use .after(0, callback) for all widget updates
def _worker():
    result = process_data()
    self.after(0, lambda: self.label.config(text=result))
```

### 2. Provide User Feedback

```python
# Show loading state
self.status_var.set("Processing...")
self._worker_pool.submit(work)

# Update on completion
def _complete():
    self.status_var.set("Complete")
```

### 3. Handle Errors Gracefully

```python
def _worker():
    try:
        result = risky_operation()
        self.after(0, lambda: self._show_result(result))
    except Exception as e:
        self.after(0, lambda: self._show_error(str(e)))
```

### 4. Prevent Duplicate Operations

```python
if self._operation_in_progress:
    return  # Already running

self._operation_in_progress = True
# ... submit work ...
# Set to False in completion callback
```

### 5. Use Throttling for Frequent Operations

```python
# Only refresh if >5 seconds since last refresh
if time.time() - self._last_refresh < 5.0:
    return
```

### 6. Clean Up Resources

```python
def cleanup(self):
    """Called on widget destruction."""
    self._worker_pool.shutdown(wait=False, timeout=2.0)
```

### 7. Leverage CLI Caching

```python
# Caching is automatic for read-only CLI operations
result = self._run_cli(['brains', 'stats'])  # Cached for 2 seconds
```

---

## Examples

### Example 1: Complete Panel Refresh

```python
class MyPanel(ttk.LabelFrame):
    def __init__(self, parent, *, worker_pool=None):
        super().__init__(parent, text="My Panel")
        self._worker_pool = worker_pool
        self._loading = False
        self._last_refresh = 0.0
        
        # Build UI
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.pack()
        
        self.data_tree = ttk.Treeview(self)
        self.data_tree.pack()
        
        ttk.Button(self, text="Refresh", command=self.refresh).pack()
    
    def refresh(self, force: bool = False):
        """Refresh panel data with throttling and loading indicator."""
        
        # Throttle
        if not force:
            import time
            now = time.time()
            if now - self._last_refresh < 5.0:
                return
            self._last_refresh = now
        
        # Prevent duplicates
        if self._loading:
            return
        
        self._loading = True
        self.status_var.set("Loading...")
        
        def _fetch_data():
            """Background data fetching."""
            try:
                # Blocking I/O operations
                data = self._run_cli(['my-data', 'list'])
                parsed = self._parse_data(data)
                
                # Schedule UI update
                def _update_ui():
                    self._populate_tree(parsed)
                    self.status_var.set("")
                    self._loading = False
                
                self.after(0, _update_ui)
            except Exception as e:
                # Schedule error handling
                def _show_error():
                    self.status_var.set(f"Error: {e}")
                    self._loading = False
                
                self.after(0, _show_error)
        
        # Submit to worker pool
        if self._worker_pool:
            self._worker_pool.submit(_fetch_data)
        else:
            import threading
            threading.Thread(target=_fetch_data, daemon=True).start()
    
    def _populate_tree(self, data):
        """Populate treeview with data (main thread only)."""
        # Clear existing
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add new items
        for item in data:
            self.data_tree.insert('', 'end', values=(item['name'], item['value']))
```

### Example 2: User Action with Confirmation

```python
def delete_item(self, item_name):
    """Delete item with confirmation dialog."""
    from tkinter import messagebox
    
    # Confirmation dialog (main thread, blocking is OK)
    if not messagebox.askyesno("Confirm", f"Delete {item_name}?"):
        return
    
    # Show progress
    self.status_var.set(f"Deleting {item_name}...")
    
    def _do_delete():
        """Background deletion."""
        try:
            result = self._run_cli(['delete', item_name])
            success = 'error' not in result.lower()
            
            def _update_ui():
                if success:
                    self.status_var.set("Deleted successfully")
                    self.refresh(force=True)  # Refresh list
                else:
                    self.status_var.set("Delete failed")
                    messagebox.showerror("Error", result)
            
            self.after(0, _update_ui)
        except Exception as e:
            def _show_error():
                self.status_var.set("Error")
                messagebox.showerror("Error", str(e))
            
            self.after(0, _show_error)
    
    self._worker_pool.submit(_do_delete)
```

---

## Troubleshooting

### Problem: GUI Freezes

**Symptoms**: Window becomes unresponsive, "Not Responding" in title

**Causes**:
- Blocking operation in main thread
- Synchronous subprocess call
- Heavy computation without worker pool

**Solutions**:
```python
# Check for blocking calls:
grep -r "subprocess.run\|time.sleep\|\.wait()" src/aios/gui/

# Move to worker pool:
self._worker_pool.submit(blocking_operation)
```

### Problem: UI Updates Don't Appear

**Symptoms**: Changes happen but UI doesn't refresh

**Causes**:
- Direct widget updates from worker thread
- Missing `.after(0, callback)`

**Solutions**:
```python
# Always schedule UI updates:
def _worker():
    data = fetch()
    self.after(0, lambda: self.widget.config(text=data))
```

### Problem: "RuntimeError: main thread is not in main loop"

**Symptoms**: Crash when updating UI from worker

**Cause**: Direct UI modification from worker thread

**Solution**:
```python
# Wrap all UI updates in .after():
self.after(0, self._update_widgets)
```

### Problem: Worker Pool Exhaustion

**Symptoms**: Operations queue up, long delays

**Causes**:
- Too many concurrent operations
- Workers blocked on long-running tasks
- Insufficient worker count

**Solutions**:
```python
# Increase worker pool size:
export AIOS_WORKER_THREADS=32

# Check pool size:
print(f"Workers: {app._worker_pool._max_workers}")

# Optimize long-running tasks:
# - Break into chunks
# - Add timeouts
# - Use async/await for I/O
```

### Problem: Memory Leaks

**Symptoms**: Memory usage grows over time

**Causes**:
- Widget references in closures
- Unclosed file handles
- Accumulated cache

**Solutions**:
```python
# Use weak references:
import weakref
widget_ref = weakref.ref(widget)

# Clear cache periodically:
if len(self._cli_cache) > 50:
    oldest = min(self._cli_cache.keys(), 
                 key=lambda k: self._cli_cache[k][1])
    del self._cli_cache[oldest]

# Cleanup on destroy:
def cleanup(self):
    self._worker_pool.shutdown(wait=False)
```

---

## Performance Tuning

### Worker Pool Sizing

```bash
# For CPU-bound tasks:
export AIOS_WORKER_THREADS=$(python -c "import os; print(os.cpu_count())")

# For I/O-bound tasks (default, recommended):
export AIOS_WORKER_THREADS=$(python -c "import os; print((os.cpu_count() or 4) * 4 + 1)")

# For high-concurrency scenarios:
export AIOS_WORKER_THREADS=64
```

### Cache Configuration

```python
# Adjust cache TTL in CliBridgeMixin.__init__():
self._cache_ttl = 5.0  # Cache for 5 seconds instead of 2
```

### Throttling Intervals

```python
# Adjust per-panel in refresh() method:
if time.time() - self._last_refresh < 10.0:  # 10-second throttle
    return
```

---

## Testing Async Code

### Manual Testing Checklist

- [ ] Click buttons rapidly - UI stays responsive
- [ ] Switch tabs quickly - no freezing
- [ ] Trigger multiple refreshes - no crashes
- [ ] Close window during operations - clean shutdown
- [ ] Network down - graceful error handling
- [ ] Large datasets - progress feedback shown

### Automated Testing

Use the profiling script:

```bash
python scripts/profile_gui_responsiveness.py
```

This will:
- Measure startup time
- Test tab switching speed
- Monitor UI thread blocking
- Check refresh operation latency
- Generate performance report

---

## References

- **AsyncWorkerPool**: `src/aios/gui/utils/resource_management/async_pool.py`
- **CliBridgeMixin**: `src/aios/gui/mixins/cli_bridge.py`
- **Panel Setup**: `src/aios/gui/app/panel_setup.py`
- **Example Panel**: `src/aios/gui/components/brains_panel/panel_main.py`

---

## Changelog

### November 2025
- Initial version
- Added CLI caching (2s TTL)
- Increased worker pool size (4x CPU cores)
- Added loading indicators to brains panel
- Implemented async panel initialization

---

*For questions or improvements to this guide, please open an issue or PR.*
