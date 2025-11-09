"""
Lazy expert loader for memory-efficient expert management.

Implements LRU caching with GPU/CPU/disk tiers.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

from aios.core.hrm_models.moe_layer import FeedForward

logger = logging.getLogger(__name__)


class LazyExpertLoader:
    """
    Manages lazy loading of expert networks from disk.
    
    Keeps frequently-used experts on GPU, offloads inactive ones to CPU or disk.
    Uses LRU cache strategy for memory efficiency.
    """
    
    def __init__(
        self,
        max_gpu_experts: int = 4,
        max_cpu_experts: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            max_gpu_experts: Maximum experts to keep on GPU
            max_cpu_experts: Maximum experts to keep in CPU memory
            device: Default device for loaded experts
        """
        self.max_gpu_experts = max_gpu_experts
        self.max_cpu_experts = max_cpu_experts
        self.device = device
        
        # Cache: expert_id -> (expert_module, location)
        # location: "gpu", "cpu", or "disk"
        self.cache: Dict[str, Tuple[nn.Module, str]] = {}
        
        # LRU tracking
        self.access_order: List[str] = []
        
        # Statistics
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_loads = 0
    
    def load_expert(
        self,
        expert_id: str,
        checkpoint_path: Optional[str] = None,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
    ) -> nn.Module:
        """
        Load expert, using cache if available.
        
        Args:
            expert_id: Unique expert identifier
            checkpoint_path: Path to expert checkpoint (if loading from disk)
            hidden_size: Expert hidden size (for creating new expert)
            intermediate_size: Expert intermediate size
        
        Returns:
            Expert module on GPU
        """
        # Check cache
        if expert_id in self.cache:
            expert, location = self.cache[expert_id]
            
            # Update LRU
            self._update_access(expert_id)
            
            # Move to GPU if needed
            if location == "gpu":
                self.gpu_hits += 1
                return expert
            elif location == "cpu":
                self.cpu_hits += 1
                expert = expert.to(self.device)
                self.cache[expert_id] = (expert, "gpu")
                self._evict_if_needed()
                return expert
        
        # Load from disk
        self.disk_loads += 1
        
        if checkpoint_path and Path(checkpoint_path).exists():
            expert = FeedForward(hidden_size, intermediate_size)
            try:
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(str(checkpoint_path), device=str(self.device))
                expert.load_state_dict(state_dict)
                logger.info(f"[LazyExpertLoader] Loaded expert {expert_id} from {checkpoint_path} (safetensors)")
            except Exception as e:
                logger.warning(f"[LazyExpertLoader] Failed to load expert with safetensors: {e}")
                # Fallback to torch.load for backwards compatibility
                expert.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                logger.info(f"[LazyExpertLoader] Loaded expert {expert_id} from {checkpoint_path} (torch)")
        else:
            # Create new expert
            expert = FeedForward(hidden_size, intermediate_size)
            expert.to(self.device)
            logger.info(f"[LazyExpertLoader] Created new expert {expert_id}")
        
        # Add to cache
        self.cache[expert_id] = (expert, "gpu")
        self._update_access(expert_id)
        self._evict_if_needed()
        
        return expert
    
    def _update_access(self, expert_id: str):
        """Update LRU access order."""
        if expert_id in self.access_order:
            self.access_order.remove(expert_id)
        self.access_order.append(expert_id)
    
    def _evict_if_needed(self):
        """Evict least-recently-used experts to maintain cache limits."""
        # Count experts by location
        gpu_count = sum(1 for _, loc in self.cache.values() if loc == "gpu")
        cpu_count = sum(1 for _, loc in self.cache.values() if loc == "cpu")
        
        # Evict from GPU to CPU
        while gpu_count > self.max_gpu_experts and len(self.access_order) > 0:
            # Find LRU GPU expert
            for expert_id in self.access_order:
                if expert_id in self.cache:
                    expert, location = self.cache[expert_id]
                    if location == "gpu":
                        # Move to CPU
                        expert = expert.to("cpu")
                        self.cache[expert_id] = (expert, "cpu")
                        gpu_count -= 1
                        cpu_count += 1
                        logger.debug(f"[LazyExpertLoader] Evicted {expert_id} from GPU to CPU")
                        break
        
        # Evict from CPU to disk (remove from cache)
        while cpu_count > self.max_cpu_experts and len(self.access_order) > 0:
            # Find LRU CPU expert
            for expert_id in self.access_order:
                if expert_id in self.cache:
                    expert, location = self.cache[expert_id]
                    if location == "cpu":
                        # Remove from cache (will be loaded from disk next time)
                        del self.cache[expert_id]
                        self.access_order.remove(expert_id)
                        cpu_count -= 1
                        logger.debug(f"[LazyExpertLoader] Evicted {expert_id} from CPU to disk")
                        break
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        gpu_count = sum(1 for _, loc in self.cache.values() if loc == "gpu")
        cpu_count = sum(1 for _, loc in self.cache.values() if loc == "cpu")
        
        return {
            "gpu_experts": float(gpu_count),
            "cpu_experts": float(cpu_count),
            "total_cached": float(len(self.cache)),
            "gpu_hits": float(self.gpu_hits),
            "cpu_hits": float(self.cpu_hits),
            "disk_loads": float(self.disk_loads),
            "hit_rate": (self.gpu_hits + self.cpu_hits) / max(1, self.gpu_hits + self.cpu_hits + self.disk_loads),
        }
