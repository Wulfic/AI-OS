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
        logger.debug("Acquiring process cleanup lock for register")
        with self._lock:
            logger.debug("Process cleanup lock acquired")
            self.processes.append(proc)
            logger.info(f"Registered process for cleanup: PID={proc.pid}")
            logger.debug("Process cleanup lock released")
    
    def unregister(self, proc: subprocess.Popen):
        """Unregister a process (e.g., after manual cleanup)."""
        logger.debug("Acquiring process cleanup lock for unregister")
        with self._lock:
            logger.debug("Process cleanup lock acquired")
            if proc in self.processes:
                self.processes.remove(proc)
                logger.debug(f"Unregistered process {proc.pid}")
            logger.debug("Process cleanup lock released")
    
    def cleanup_all(self, timeout: float = 5.0):
        """Terminate all registered processes.
        
        Args:
            timeout: Maximum seconds to wait for graceful termination
        """
        logger.debug("Acquiring process cleanup lock for cleanup_all")
        with self._lock:
            logger.debug("Process cleanup lock acquired")
            if not self.processes:
                logger.debug("No processes to clean up")
                logger.debug("Process cleanup lock released")
                return
            
            logger.info(f"Starting cleanup of {len(self.processes)} registered processes (timeout: {timeout}s)")
            
            # Step 1: Send terminate signal to all processes
            terminated_count = 0
            for proc in self.processes[:]:  # Copy list
                if proc.poll() is None:  # Still running
                    try:
                        proc.terminate()
                        logger.debug(f"Sent SIGTERM to process PID={proc.pid}")
                        terminated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to terminate process PID={proc.pid}: {e}")
            
            logger.info(f"Sent termination signal to {terminated_count} running processes")
            
            # Step 2: Wait for graceful termination
            start = time.time()
            while time.time() - start < timeout:
                all_dead = True
                for proc in self.processes:
                    if proc.poll() is None:
                        all_dead = False
                        break
                
                if all_dead:
                    elapsed = time.time() - start
                    logger.info(f"All processes terminated gracefully in {elapsed:.2f}s")
                    break
                
                time.sleep(0.1)
            
            # Step 3: Force kill any survivors
            killed_count = 0
            for proc in self.processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                        logger.warning(f"Force killed unresponsive process PID={proc.pid}")
                        killed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to kill process PID={proc.pid}: {e}")
            
            if killed_count > 0:
                logger.warning(f"Force killed {killed_count} processes that did not terminate gracefully")
            
            self.processes.clear()
            logger.info("Process cleanup complete - all processes cleared")
            logger.debug("Process cleanup lock released")
