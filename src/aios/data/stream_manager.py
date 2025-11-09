"""
Dataset Stream Manager - Coordinates concurrent access to HuggingFace datasets.

Prevents conflicts when the same dataset is being downloaded and trained on simultaneously.
Implements a queue system that pauses downloads when training starts, and resumes after.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple, Optional, Callable, Any
from datetime import datetime


class DatasetStreamManager:
    """
    Singleton manager for coordinating dataset streaming operations.
    
    Prevents HuggingFace API conflicts by ensuring only one operation
    (download OR training) streams a given dataset at a time.
    """
    
    _instance: Optional['DatasetStreamManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the stream manager."""
        self._registry_lock = threading.Lock()
        
        # Active downloads: {dataset_id: {"pause_event": Event, "started": timestamp, "status": str}}
        self._active_downloads: Dict[str, Dict[str, Any]] = {}
        
        # Active training: {dataset_id: {"started": timestamp}}
        self._active_training: Dict[str, Dict[str, Any]] = {}
        
        # Logging callback
        self._log_callback: Optional[Callable[[str], None]] = None
    
    @classmethod
    def get_instance(cls) -> 'DatasetStreamManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def set_log_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback for logging stream manager events."""
        self._log_callback = callback
    
    def _log(self, message: str) -> None:
        """Log a message if callback is set."""
        if self._log_callback:
            try:
                self._log_callback(message)
            except Exception:
                pass
    
    @staticmethod
    def normalize_dataset_id(dataset_id: str) -> str:
        """
        Normalize dataset ID for consistent comparison.
        
        Examples:
            - "wikitext" → "wikitext"
            - "Wikitext" → "wikitext"
            - "user/dataset" → "user/dataset"
            - "  Dataset  " → "dataset"
        """
        return dataset_id.strip().lower()
    
    @staticmethod
    def parse_dataset_id_from_hf_path(hf_path: str) -> Optional[str]:
        """
        Parse dataset ID from hf:// path format.
        
        Examples:
            - "hf://wikitext" → "wikitext"
            - "hf://wikitext:wikitext-2-raw-v1:train" → "wikitext"
            - "hf://user/dataset:config:split" → "user/dataset"
        
        Returns:
            Normalized dataset ID or None if invalid format
        """
        if not isinstance(hf_path, str) or not hf_path.startswith("hf://"):
            return None
        
        # Remove hf:// prefix
        path = hf_path[5:]
        
        # Extract dataset name (before first colon)
        dataset_name = path.split(":")[0] if ":" in path else path
        
        return DatasetStreamManager.normalize_dataset_id(dataset_name)
    
    def register_download(self, dataset_id: str, pause_event: threading.Event) -> bool:
        """
        Register that a download is starting.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            pause_event: Threading event to signal pause/resume
            
        Returns:
            True if registration successful, False if training is active
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            # Check if training is active on this dataset
            if normalized_id in self._active_training:
                training_info = self._active_training[normalized_id]
                self._log(f"⚠️ Cannot download {dataset_id}: training active since {training_info['started']}")
                return False
            
            # Register download
            self._active_downloads[normalized_id] = {
                "pause_event": pause_event,
                "started": datetime.now().strftime("%H:%M:%S"),
                "status": "active",
                "dataset_id": dataset_id,  # Keep original for display
            }
            
            self._log(f"📥 Download registered: {dataset_id}")
            return True
    
    def unregister_download(self, dataset_id: str) -> None:
        """
        Unregister a download (completed or cancelled).
        
        Args:
            dataset_id: HuggingFace dataset identifier
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            if normalized_id in self._active_downloads:
                del self._active_downloads[normalized_id]
                self._log(f"✅ Download unregistered: {dataset_id}")
    
    def register_training(self, dataset_id: str) -> Tuple[bool, str]:
        """
        Register that training is starting on a dataset.
        
        If a download is active, it will be paused.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            
        Returns:
            (success, message) tuple
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            # Check if download is active
            if normalized_id in self._active_downloads:
                download_info = self._active_downloads[normalized_id]
                
                # Pause the download
                pause_event = download_info["pause_event"]
                pause_event.set()  # Signal pause
                download_info["status"] = "paused_for_training"
                
                self._log(f"⏸️ Download paused for training: {dataset_id}")
            
            # Register training
            self._active_training[normalized_id] = {
                "started": datetime.now().strftime("%H:%M:%S"),
                "dataset_id": dataset_id,
            }
            
            self._log(f"🎯 Training registered: {dataset_id}")
            return True, "Training registered"
    
    def unregister_training(self, dataset_id: str) -> None:
        """
        Unregister training (completed or stopped).
        
        If a download was paused, it will be resumed.
        
        Args:
            dataset_id: HuggingFace dataset identifier
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            if normalized_id in self._active_training:
                del self._active_training[normalized_id]
                self._log(f"✅ Training unregistered: {dataset_id}")
            
            # Resume any paused downloads
            if normalized_id in self._active_downloads:
                download_info = self._active_downloads[normalized_id]
                if download_info["status"] == "paused_for_training":
                    pause_event = download_info["pause_event"]
                    pause_event.clear()  # Signal resume
                    download_info["status"] = "active"
                    self._log(f"▶️ Download resumed: {dataset_id}")
    
    def can_download(self, dataset_id: str) -> Tuple[bool, str]:
        """
        Check if a download can proceed.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            
        Returns:
            (allowed, reason) tuple
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            if normalized_id in self._active_training:
                return False, f"Training active on {dataset_id}"
            return True, "OK"
    
    def can_train(self, dataset_id: str) -> Tuple[bool, str]:
        """
        Check if training can proceed (always returns True, may pause downloads).
        
        Args:
            dataset_id: HuggingFace dataset identifier
            
        Returns:
            (allowed, reason) tuple
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            if normalized_id in self._active_downloads:
                return True, f"Will pause download of {dataset_id}"
            return True, "OK"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of all active operations.
        
        Returns:
            Dictionary with active_downloads and active_training lists
        """
        with self._registry_lock:
            return {
                "active_downloads": [
                    {
                        "dataset_id": info["dataset_id"],
                        "started": info["started"],
                        "status": info["status"],
                    }
                    for info in self._active_downloads.values()
                ],
                "active_training": [
                    {
                        "dataset_id": info["dataset_id"],
                        "started": info["started"],
                    }
                    for info in self._active_training.values()
                ],
            }
    
    def is_dataset_busy(self, dataset_id: str) -> bool:
        """
        Check if a dataset has any active operations.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            
        Returns:
            True if download or training is active
        """
        normalized_id = self.normalize_dataset_id(dataset_id)
        
        with self._registry_lock:
            return (
                normalized_id in self._active_downloads
                or normalized_id in self._active_training
            )


# Global singleton instance getter
def get_stream_manager() -> DatasetStreamManager:
    """Get the global dataset stream manager instance."""
    return DatasetStreamManager.get_instance()
