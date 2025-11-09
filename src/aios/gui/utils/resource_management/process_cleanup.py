"""Process cleanup utilities to prevent zombie processes."""

from __future__ import annotations

import logging
import subprocess
import threading
import time


logger = logging.getLogger(__name__)


class ProcessReaper:
    """Ensures all child processes are cleaned up.
    
    Tracks processes and provides aggressive cleanup to prevent zombies.
    
    Usage:
        reaper = ProcessReaper()
        
        # Register processes
        proc = subprocess.Popen(...)
        reaper.register(proc)
        
        # On shutdown
        reaper.cleanup_all(timeout=5.0)
    """
    
    def __init__(self):
        self.processes: list[subprocess.Popen] = []
        self._lock = threading.Lock()
    
    def register(self, proc: subprocess.Popen):
        """Register a process for cleanup."""
        with self._lock:
            self.processes.append(proc)
            logger.debug(f"Registered process {proc.pid}")
    
    def unregister(self, proc: subprocess.Popen):
        """Unregister a process (e.g., after manual cleanup)."""
        with self._lock:
            if proc in self.processes:
                self.processes.remove(proc)
                logger.debug(f"Unregistered process {proc.pid}")
    
    def cleanup_all(self, timeout: float = 5.0):
        """Terminate all registered processes.
        
        Args:
            timeout: Maximum seconds to wait for graceful termination
        """
        with self._lock:
            if not self.processes:
                return
            
            logger.info(f"Cleaning up {len(self.processes)} processes...")
            
            # Step 1: Send terminate signal to all processes
            for proc in self.processes[:]:  # Copy list
                if proc.poll() is None:  # Still running
                    try:
                        proc.terminate()
                        logger.debug(f"Terminated process {proc.pid}")
                    except Exception as e:
                        logger.warning(f"Failed to terminate process {proc.pid}: {e}")
            
            # Step 2: Wait for graceful termination
            start = time.time()
            while time.time() - start < timeout:
                all_dead = True
                for proc in self.processes:
                    if proc.poll() is None:
                        all_dead = False
                        break
                
                if all_dead:
                    logger.info("All processes terminated gracefully")
                    break
                
                time.sleep(0.1)
            
            # Step 3: Force kill any survivors
            for proc in self.processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                        logger.warning(f"Force killed process {proc.pid}")
                    except Exception as e:
                        logger.error(f"Failed to kill process {proc.pid}: {e}")
            
            self.processes.clear()
            logger.info("Process cleanup complete")
