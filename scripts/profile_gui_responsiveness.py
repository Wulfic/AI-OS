#!/usr/bin/env python3
"""GUI responsiveness profiling script.

This script helps identify performance bottlenecks in the AI-OS GUI by:
1. Measuring startup time
2. Testing panel switching responsiveness
3. Measuring refresh operation latency
4. Monitoring worker pool utilization
5. Detecting UI thread blocking

Usage:
    python scripts/profile_gui_responsiveness.py

Environment Variables:
    AIOS_WORKER_THREADS: Override default worker pool size
    AIOS_PROFILE_DURATION: How long to run profiling (default: 60 seconds)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GUIProfiler:
    """Profiler for GUI responsiveness metrics."""
    
    def __init__(self):
        self.metrics = {
            'startup_time': None,
            'panel_switches': [],
            'refresh_operations': [],
            'worker_pool_stats': {},
            'ui_blocks': [],
        }
        self.start_time = None
        self.profile_duration = int(os.environ.get('AIOS_PROFILE_DURATION', '60'))
    
    def profile_startup(self):
        """Measure GUI startup time."""
        logger.info("=== Profiling GUI Startup ===")
        
        try:
            import tkinter as tk
            from aios.gui.app import AiosTkApp
            
            start = time.time()
            
            # Create app instance
            root = tk.Tk()
            app = AiosTkApp(root)
            
            # Measure time until window is ready
            root.update()
            startup_time = time.time() - start
            
            self.metrics['startup_time'] = startup_time
            logger.info(f"✓ Startup time: {startup_time:.3f}s")
            
            # Check worker pool configuration
            if hasattr(app, '_worker_pool'):
                pool = app._worker_pool
                worker_count = pool._max_workers if hasattr(pool, '_max_workers') else 'unknown'
                logger.info(f"✓ Worker pool size: {worker_count}")
                self.metrics['worker_pool_stats']['size'] = worker_count
            
            # Schedule profiling tasks
            self.start_time = time.time()
            self._schedule_profiling_tasks(app, root)
            
            # Run main loop with timeout
            root.after(self.profile_duration * 1000, lambda: self._finish_profiling(root))
            root.mainloop()
            
        except Exception as e:
            logger.error(f"✗ Startup profiling failed: {e}", exc_info=True)
    
    def _schedule_profiling_tasks(self, app, root):
        """Schedule various profiling tasks."""
        
        # Test 1: Rapid tab switching (every 2 seconds for first 20 seconds)
        def switch_tabs():
            elapsed = time.time() - self.start_time
            if elapsed > 20:  # Stop after 20 seconds
                return
            
            if hasattr(app, 'notebook'):
                try:
                    tabs = app.notebook.tabs()
                    if tabs:
                        current = app.notebook.index('current')
                        next_tab = (current + 1) % len(tabs)
                        
                        switch_start = time.time()
                        app.notebook.select(next_tab)
                        root.update()
                        switch_time = time.time() - switch_start
                        
                        self.metrics['panel_switches'].append(switch_time)
                        
                        # Log if switch took too long
                        if switch_time > 0.1:
                            logger.warning(f"⚠ Slow tab switch: {switch_time:.3f}s")
                except Exception as e:
                    logger.debug(f"Tab switch error: {e}")
            
            # Schedule next switch
            root.after(2000, switch_tabs)
        
        # Test 2: Refresh operations (every 5 seconds starting at 25s)
        def test_refresh():
            elapsed = time.time() - self.start_time
            if elapsed < 25 or elapsed > 50:  # Run from 25s to 50s
                if elapsed < 50:
                    root.after(5000, test_refresh)
                return
            
            if hasattr(app, 'brains_panel') and app.brains_panel:
                try:
                    refresh_start = time.time()
                    app.brains_panel.refresh(force=True)
                    
                    # Check if refresh completed quickly (should be async)
                    immediate_time = time.time() - refresh_start
                    self.metrics['refresh_operations'].append({
                        'immediate': immediate_time,
                        'timestamp': elapsed
                    })
                    
                    if immediate_time > 0.05:
                        logger.warning(f"⚠ Refresh blocked UI for {immediate_time:.3f}s")
                    else:
                        logger.info(f"✓ Async refresh initiated in {immediate_time:.4f}s")
                except Exception as e:
                    logger.debug(f"Refresh test error: {e}")
            
            # Schedule next refresh test
            root.after(5000, test_refresh)
        
        # Test 3: Monitor UI thread responsiveness
        def monitor_ui_thread():
            elapsed = time.time() - self.start_time
            if elapsed > self.profile_duration - 5:
                return
            
            ping_start = time.time()
            
            def pong():
                latency = time.time() - ping_start
                if latency > 0.1:
                    logger.warning(f"⚠ UI thread latency: {latency:.3f}s")
                    self.metrics['ui_blocks'].append(latency)
            
            root.after(0, pong)
            root.after(1000, monitor_ui_thread)
        
        # Start profiling tasks
        logger.info("\n=== Starting Profiling Tasks ===")
        logger.info("Task 1: Rapid tab switching (0-20s)")
        logger.info("Task 2: Refresh operations (25-50s)")
        logger.info("Task 3: UI thread monitoring (continuous)")
        
        root.after(100, switch_tabs)
        root.after(25000, test_refresh)
        root.after(500, monitor_ui_thread)
    
    def _finish_profiling(self, root):
        """Complete profiling and generate report."""
        logger.info("\n=== Profiling Complete ===")
        self._generate_report()
        root.quit()
    
    def _generate_report(self):
        """Generate and display profiling report."""
        print("\n" + "="*70)
        print("GUI RESPONSIVENESS PROFILING REPORT")
        print("="*70)
        
        # Startup metrics
        if self.metrics['startup_time']:
            print(f"\n📊 STARTUP PERFORMANCE")
            print(f"   Total startup time: {self.metrics['startup_time']:.3f}s")
            if self.metrics['startup_time'] < 2.0:
                print("   ✓ EXCELLENT - Fast startup")
            elif self.metrics['startup_time'] < 5.0:
                print("   ⚠ ACCEPTABLE - Moderate startup")
            else:
                print("   ✗ SLOW - Needs optimization")
        
        # Worker pool
        if self.metrics['worker_pool_stats']:
            print(f"\n⚙️  WORKER POOL CONFIGURATION")
            size = self.metrics['worker_pool_stats'].get('size', 'unknown')
            print(f"   Worker threads: {size}")
            cpu_count = os.cpu_count() or 4
            if isinstance(size, int):
                ratio = size / cpu_count
                print(f"   CPU cores: {cpu_count}")
                print(f"   Workers per core: {ratio:.1f}x")
                if ratio >= 4:
                    print("   ✓ OPTIMAL - High concurrency support")
                elif ratio >= 2:
                    print("   ⚠ ACCEPTABLE - Moderate concurrency")
                else:
                    print("   ✗ LOW - Consider increasing workers")
        
        # Tab switching
        if self.metrics['panel_switches']:
            print(f"\n🔄 TAB SWITCHING PERFORMANCE")
            switches = self.metrics['panel_switches']
            avg_time = sum(switches) / len(switches)
            max_time = max(switches)
            min_time = min(switches)
            print(f"   Total switches: {len(switches)}")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Min: {min_time:.3f}s")
            print(f"   Max: {max_time:.3f}s")
            
            slow_switches = [t for t in switches if t > 0.1]
            if not slow_switches:
                print("   ✓ EXCELLENT - All switches < 100ms")
            elif len(slow_switches) < len(switches) * 0.1:
                print(f"   ⚠ ACCEPTABLE - {len(slow_switches)} slow switches")
            else:
                print(f"   ✗ SLOW - {len(slow_switches)} switches > 100ms")
        
        # Refresh operations
        if self.metrics['refresh_operations']:
            print(f"\n♻️  REFRESH OPERATION PERFORMANCE")
            refreshes = self.metrics['refresh_operations']
            print(f"   Total refreshes: {len(refreshes)}")
            
            immediate_times = [r['immediate'] for r in refreshes]
            avg_immediate = sum(immediate_times) / len(immediate_times)
            print(f"   Average initiation time: {avg_immediate:.4f}s")
            
            blocking_refreshes = [t for t in immediate_times if t > 0.05]
            if not blocking_refreshes:
                print("   ✓ EXCELLENT - All refreshes async (< 50ms)")
            elif len(blocking_refreshes) < len(refreshes) * 0.2:
                print(f"   ⚠ ACCEPTABLE - {len(blocking_refreshes)} blocking calls")
            else:
                print(f"   ✗ BLOCKING - {len(blocking_refreshes)} sync operations")
        
        # UI thread blocking
        if self.metrics['ui_blocks']:
            print(f"\n⏱️  UI THREAD RESPONSIVENESS")
            blocks = self.metrics['ui_blocks']
            print(f"   Detected blocks: {len(blocks)}")
            if blocks:
                avg_block = sum(blocks) / len(blocks)
                max_block = max(blocks)
                print(f"   Average block time: {avg_block:.3f}s")
                print(f"   Max block time: {max_block:.3f}s")
                
                if max_block > 1.0:
                    print("   ✗ CRITICAL - UI freezes detected")
                elif max_block > 0.5:
                    print("   ⚠ WARNING - Significant delays detected")
                else:
                    print("   ⚠ MINOR - Small delays detected")
            else:
                print("   ✓ EXCELLENT - No blocking detected")
        else:
            print(f"\n⏱️  UI THREAD RESPONSIVENESS")
            print("   ✓ EXCELLENT - No blocking detected")
        
        # Overall assessment
        print(f"\n📈 OVERALL ASSESSMENT")
        
        issues = []
        if self.metrics['startup_time'] and self.metrics['startup_time'] > 5.0:
            issues.append("Slow startup")
        
        if self.metrics['panel_switches']:
            slow_switches = [t for t in self.metrics['panel_switches'] if t > 0.1]
            if len(slow_switches) > len(self.metrics['panel_switches']) * 0.1:
                issues.append("Slow tab switching")
        
        if self.metrics['refresh_operations']:
            immediate_times = [r['immediate'] for r in self.metrics['refresh_operations']]
            blocking = [t for t in immediate_times if t > 0.05]
            if len(blocking) > len(self.metrics['refresh_operations']) * 0.2:
                issues.append("Blocking refresh operations")
        
        if self.metrics['ui_blocks'] and max(self.metrics['ui_blocks']) > 0.5:
            issues.append("UI thread blocking")
        
        if not issues:
            print("   ✓ EXCELLENT - No significant issues detected")
            print("   GUI is highly responsive with good async handling")
        else:
            print("   ⚠ Issues detected:")
            for issue in issues:
                print(f"      - {issue}")
        
        print("\n" + "="*70)
        
        # Save detailed report
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed metrics to file."""
        import json
        from datetime import datetime
        
        report_file = project_root / "artifacts" / "diagnostics" / f"gui_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"\n📝 Detailed report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save detailed report: {e}")


def main():
    """Run GUI responsiveness profiling."""
    print("\n" + "="*70)
    print("AI-OS GUI Responsiveness Profiler")
    print("="*70)
    print(f"\nProfile duration: {os.environ.get('AIOS_PROFILE_DURATION', '60')}s")
    print(f"Worker threads: {os.environ.get('AIOS_WORKER_THREADS', 'auto')}")
    print("\nStarting profiling... This will take about 1 minute.")
    print("The GUI will open and perform automated tests.")
    print("="*70 + "\n")
    
    profiler = GUIProfiler()
    profiler.profile_startup()


if __name__ == '__main__':
    main()
