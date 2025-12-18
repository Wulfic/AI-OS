"""Aggressive process management to prevent hanging."""

from __future__ import annotations

import os
import time
import threading
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any


class ProcessManager:
    """Aggressive process management to prevent hanging."""
    
    def __init__(self, timeout: int = 240):  # Increased default timeout for DDP init + training
        self.timeout = timeout
        self.processes = []
        self._heartbeat_file = None
        self._heartbeat_timeout = 60  # seconds without heartbeat = frozen (increased for slow init)
        self._stop_callback = None  # Callback to check if user requested stop
        
    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        heartbeat_file: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        stop_callback: Optional[Callable[[], bool]] = None
    ) -> subprocess.CompletedProcess:
        """Run command with aggressive timeout, cleanup, and heartbeat monitoring."""
        
        self._heartbeat_file = heartbeat_file
        self._stop_callback = stop_callback
        
        # Start the process in a new process group
        try:
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP for process control and CREATE_NO_WINDOW to prevent CMD popups
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            else:  # Unix-like
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=env,
                    preexec_fn=os.setsid
                )
                
            self.processes.append(process)
            
            # Wait with timeout and heartbeat monitoring
            try:
                stdout, stderr = self._communicate_with_monitoring(process, log_callback)
                return subprocess.CompletedProcess(
                    cmd, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                # Aggressively kill the process and all children
                self._kill_process_tree(process)
                return subprocess.CompletedProcess(
                    cmd, -1, "", f"Process killed after {self.timeout}s timeout"
                )
                
        except Exception as e:
            return subprocess.CompletedProcess(
                cmd, -1, "", f"Failed to start process: {e}"
            )
    
    def _communicate_with_monitoring(self, process, log_callback=None) -> Tuple[str, str]:
        """Monitor process with heartbeat detection to catch freezes.
        
        Drains stdout/stderr in real-time to prevent pipe buffer deadlocks on Windows,
        which can cause DDP workers to hang when they write more output than the pipe
        buffer can hold (typically 4-64KB).
        """
        start_time = time.time()
        last_heartbeat = start_time
        last_progress_log = start_time
        
        # Use threads to drain stdout/stderr in real-time to prevent deadlocks
        stdout_lines = []
        stderr_lines = []
        stdout_thread = None
        stderr_thread = None
        
        def drain_pipe(pipe, line_list):
            """Continuously drain a pipe to prevent buffer overflow."""
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line_list.append(line)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass
        
        # Start drain threads
        if process.stdout:
            stdout_thread = threading.Thread(target=drain_pipe, args=(process.stdout, stdout_lines), daemon=True)
            stdout_thread.start()
        
        if process.stderr:
            stderr_thread = threading.Thread(target=drain_pipe, args=(process.stderr, stderr_lines), daemon=True)
            stderr_thread.start()
        
        # Poll process with heartbeat checks
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            # Check if user requested stop
            if self._stop_callback:
                try:
                    if self._stop_callback():
                        if log_callback:
                            log_callback("  ... stop requested by user")
                        raise subprocess.TimeoutExpired(process.args, elapsed, output="User requested stop")
                except subprocess.TimeoutExpired:
                    raise  # Re-raise our own exception
                except Exception:
                    pass  # Ignore callback errors
            
            # Log progress every 10 seconds for long-running processes
            if time.time() - last_progress_log > 10:
                if log_callback:
                    log_callback(f"  ... still initializing ({int(elapsed)}s elapsed)")
                last_progress_log = time.time()
            
            # Check overall timeout
            if elapsed > self.timeout:
                raise subprocess.TimeoutExpired(process.args, self.timeout)
            
            # Check heartbeat if file provided
            if self._heartbeat_file:
                try:
                    if self._heartbeat_file.exists():
                        mtime = self._heartbeat_file.stat().st_mtime
                        if mtime > last_heartbeat:
                            last_heartbeat = mtime
                    
                    # Check for frozen process (no heartbeat)
                    if time.time() - last_heartbeat > self._heartbeat_timeout:
                        raise subprocess.TimeoutExpired(
                            process.args, self._heartbeat_timeout,
                            output=f"No heartbeat for {self._heartbeat_timeout}s"
                        )
                except Exception:
                    pass
            
            time.sleep(0.5)
        
        # Process completed - wait for drain threads to finish (with timeout)
        if stdout_thread is not None:
            stdout_thread.join(timeout=2.0)
        if stderr_thread is not None:
            stderr_thread.join(timeout=2.0)
        
        # Combine captured output
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        return stdout, stderr
    
    def _kill_process_tree(self, process):
        """Kill process and all its children."""
        try:
            if process.poll() is None:  # Process still running
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # Kill children first
                for child in children:
                    try:
                        child.kill()
                    except:
                        pass
                        
                # Kill parent
                try:
                    parent.kill()
                except:
                    pass
                    
                # Force terminate subprocess
                try:
                    process.kill()
                except:
                    pass
                    
        except Exception:
            # Last resort - try to terminate the subprocess directly
            try:
                process.terminate()
                process.kill()
            except:
                pass
    
    def cleanup(self):
        """Clean up any remaining processes."""
        for process in self.processes:
            self._kill_process_tree(process)
