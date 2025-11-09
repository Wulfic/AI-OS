"""Dynamic Mixture of Experts with runtime expert management.

This module extends the basic MoE layer to support:
- Runtime expert addition/removal without destroying base model
- Expert freezing/unfreezing for selective training
- Lazy loading for memory-efficient expert management
- Recursive submodels (experts can have child experts)
- Integration with ExpertMetadata registry

Author: AI-OS Team
Date: January 2025
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

from aios.core.hrm_models.moe_layer import FeedForward, TopKRouter
from aios.core.hrm_models.expert_metadata import ExpertMetadata, ExpertRegistry

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
            expert.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"[LazyExpertLoader] Loaded expert {expert_id} from {checkpoint_path}")
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


class DynamicMoELayer(nn.Module):
    """
    Dynamic Mixture of Experts layer with runtime expert management.
    
    Extends basic MoE to support:
    - Adding/removing experts without retraining base model
    - Freezing/unfreezing experts for selective training
    - Lazy loading for memory efficiency
    - Integration with ExpertRegistry for metadata
    
    Key differences from MoELayer:
    - Uses nn.ModuleDict instead of nn.ModuleList for dynamic keys
    - Supports non-contiguous expert IDs (can have gaps)
    - Maintains ExpertRegistry synchronization
    - Optional lazy loading for large expert counts
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        num_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
        registry_path: Optional[str] = None,
        lazy_loading: bool = False,
        max_gpu_experts: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Expert intermediate size (default: hidden_size * 4)
            num_experts_per_tok: Top-k experts to activate per token
            capacity_factor: Expert capacity factor for load balancing
            registry_path: Path to expert registry JSON file
            lazy_loading: Enable lazy loading of experts
            max_gpu_experts: Maximum experts on GPU (if lazy loading)
            device: Device for experts
        """
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.capacity_factor = capacity_factor
        self.device = device
        self.lazy_loading = lazy_loading
        
        # Expert storage: ModuleDict for dynamic keys
        self.experts = nn.ModuleDict()
        
        # Registry integration
        self.registry_path = registry_path
        if registry_path and Path(registry_path).exists():
            self.registry = ExpertRegistry.load(registry_path)
            logger.info(f"[DynamicMoE] Loaded registry with {len(self.registry.experts)} experts")
        else:
            self.registry = ExpertRegistry()
            logger.info("[DynamicMoE] Created new expert registry")
        
        # Lazy loader (optional)
        self.lazy_loader: Optional[LazyExpertLoader] = None
        if lazy_loading:
            self.lazy_loader = LazyExpertLoader(
                max_gpu_experts=max_gpu_experts,
                max_cpu_experts=max_gpu_experts * 2,
                device=device,
            )
        
        # Router: dynamically sized based on active experts
        # Initialize with 1 expert (will be updated when experts are added)
        self.router = TopKRouter(hidden_size, num_experts=1)
        
        # Frozen experts (excluded from training)
        self.frozen_experts: set = set()
        
        # Track last router logits for load balancing loss
        self.last_router_logits: Optional[torch.Tensor] = None
    
    def add_expert(
        self,
        expert_id: str,
        metadata: Optional[ExpertMetadata] = None,
        expert_module: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Add a new expert to the layer.
        
        Args:
            expert_id: Unique expert identifier
            metadata: ExpertMetadata object (will be added to registry)
            expert_module: Pre-initialized expert module (optional)
            checkpoint_path: Path to load expert weights from (optional)
        """
        if expert_id in self.experts:
            logger.warning(f"[DynamicMoE] Expert {expert_id} already exists, skipping")
            return
        
        # Create or load expert
        if expert_module is not None:
            expert = expert_module
        elif checkpoint_path and Path(checkpoint_path).exists():
            expert = FeedForward(self.hidden_size, self.intermediate_size)
            expert.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"[DynamicMoE] Loaded expert {expert_id} from {checkpoint_path}")
        else:
            expert = FeedForward(self.hidden_size, self.intermediate_size)
            logger.info(f"[DynamicMoE] Created new expert {expert_id}")
        
        # Add to ModuleDict
        if not self.lazy_loading:
            expert.to(self.device)
            self.experts[expert_id] = expert
        else:
            # For lazy loading, store metadata but don't load module yet
            # (will be loaded on-demand during forward pass)
            pass
        
        # Update registry
        if metadata is not None:
            self.registry.add_expert(metadata)
        elif expert_id not in self.registry.experts:
            # Create default metadata
            from aios.core.hrm_models.expert_metadata import create_expert_metadata
            metadata = create_expert_metadata(
                expert_id=expert_id,
                name=f"Expert {expert_id}",
                description=f"Auto-created expert {expert_id}",
                category="general",
                checkpoint_path=checkpoint_path,
            )
            self.registry.add_expert(metadata)
        
        # Update router size
        self._update_router_size()
        
        logger.info(f"[DynamicMoE] Added expert {expert_id} (total: {len(self.get_active_expert_ids())})")
    
    def remove_expert(self, expert_id: str, remove_from_registry: bool = True) -> bool:
        """
        Remove an expert from the layer.
        
        Args:
            expert_id: Expert to remove
            remove_from_registry: Also remove from registry (default: True)
        
        Returns:
            True if expert was removed, False if not found
        """
        if expert_id not in self.experts and expert_id not in self.registry.experts:
            logger.warning(f"[DynamicMoE] Expert {expert_id} not found")
            return False
        
        # Remove from ModuleDict
        if expert_id in self.experts:
            del self.experts[expert_id]
        
        # Remove from registry
        if remove_from_registry:
            self.registry.remove_expert(expert_id)
        
        # Remove from frozen set
        if expert_id in self.frozen_experts:
            self.frozen_experts.remove(expert_id)
        
        # Update router size
        self._update_router_size()
        
        logger.info(f"[DynamicMoE] Removed expert {expert_id} (remaining: {len(self.get_active_expert_ids())})")
        return True
    
    def freeze_expert(self, expert_id: str) -> None:
        """Freeze expert (exclude from gradient updates)."""
        if expert_id not in self.get_active_expert_ids():
            logger.warning(f"[DynamicMoE] Expert {expert_id} not found, cannot freeze")
            return
        
        if expert_id in self.experts:
            for param in self.experts[expert_id].parameters():
                param.requires_grad = False
        
        self.frozen_experts.add(expert_id)
        logger.info(f"[DynamicMoE] Froze expert {expert_id}")
    
    def unfreeze_expert(self, expert_id: str) -> None:
        """Unfreeze expert (include in gradient updates)."""
        if expert_id not in self.get_active_expert_ids():
            logger.warning(f"[DynamicMoE] Expert {expert_id} not found, cannot unfreeze")
            return
        
        if expert_id in self.experts:
            for param in self.experts[expert_id].parameters():
                param.requires_grad = True
        
        if expert_id in self.frozen_experts:
            self.frozen_experts.remove(expert_id)
        
        logger.info(f"[DynamicMoE] Unfroze expert {expert_id}")
    
    def get_active_expert_ids(self) -> List[str]:
        """Get list of active expert IDs."""
        # Combine experts in ModuleDict and registry
        expert_ids = set(self.experts.keys())
        registry_experts = [e.expert_id for e in self.registry.get_active_experts()]
        expert_ids.update(registry_experts)
        return sorted(list(expert_ids))
    
    def _update_router_size(self):
        """Update router to match current number of experts."""
        expert_ids = self.get_active_expert_ids()
        num_experts = len(expert_ids)
        if num_experts == 0:
            num_experts = 1  # Minimum 1 for initialization
        
        logger.info(f"[DynamicMoE] _update_router_size: Current router.num_experts={self.router.num_experts}, target={num_experts}, expert_ids={expert_ids}")
        
        if self.router.num_experts != num_experts:
            # Recreate router with new size
            old_router = self.router
            old_device = next(old_router.parameters()).device
            old_dtype = next(old_router.parameters()).dtype
            
            logger.info(f"[DynamicMoE] Creating new router: hidden_size={self.hidden_size}, num_experts={num_experts}, device={old_device}, dtype={old_dtype}")
            self.router = TopKRouter(self.hidden_size, num_experts)
            self.router.to(device=old_device, dtype=old_dtype)  # Move to same device AND dtype
            
            # Try to preserve old routing weights (partial transfer)
            if num_experts >= old_router.num_experts:
                with torch.no_grad():
                    self.router.gate.weight[:old_router.num_experts] = old_router.gate.weight
            
            # Verify router dimensions
            actual_output_size = self.router.gate.weight.shape[0]
            logger.info(f"[DynamicMoE] Router updated: num_experts={self.router.num_experts}, gate.weight.shape={self.router.gate.weight.shape}, actual_output_size={actual_output_size}")
            
            if actual_output_size != num_experts:
                logger.error(f"[DynamicMoE] MISMATCH! Router gate output size {actual_output_size} != num_experts {num_experts}")
        else:
            logger.debug(f"[DynamicMoE] Router size already correct at {num_experts} experts")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic expert routing.
        
        Args:
            hidden_states: [batch, seq, hidden]
        
        Returns:
            output: [batch, seq, hidden]
            router_logits: [batch, seq, num_experts] for load balancing loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get active expert IDs
        expert_ids = self.get_active_expert_ids()
        
        if len(expert_ids) == 0:
            logger.warning("[DynamicMoE] No active experts, returning input unchanged")
            return hidden_states, torch.zeros(batch_size, seq_len, 1, device=hidden_states.device)
        
        # Route tokens
        top_k = min(self.num_experts_per_tok, len(expert_ids))
        
        # Log routing info
        logger.debug(f"[DynamicMoE] Forward: expert_ids={expert_ids}, len={len(expert_ids)}, router.num_experts={self.router.num_experts}, top_k={top_k}")
        
        # Validate router size matches expert count
        if self.router.num_experts != len(expert_ids):
            logger.error(f"[DynamicMoE] ROUTER SIZE MISMATCH! router.num_experts={self.router.num_experts} != len(expert_ids)={len(expert_ids)}")
            logger.error(f"[DynamicMoE] Calling _update_router_size() to fix...")
            self._update_router_size()
        
        # Ensure router is on same device as input to prevent device mismatch
        input_device = hidden_states.device
        try:
            router_device = next(self.router.parameters()).device
            if router_device != input_device:
                logger.debug(f"[DynamicMoE] Moving router from {router_device} to {input_device}")
                self.router = self.router.to(input_device)
        except StopIteration:
            pass
        
        top_k_weights, top_k_indices, router_logits = self.router(hidden_states, top_k)
        
        # Validate router output indices
        max_idx = top_k_indices.max().item()
        min_idx = top_k_indices.min().item()
        logger.debug(f"[DynamicMoE] Router output: top_k_indices range=[{min_idx}, {max_idx}], expected range=[0, {len(expert_ids)-1}]")
        
        if max_idx >= len(expert_ids):
            logger.error(f"[DynamicMoE] INVALID ROUTER OUTPUT! max_idx={max_idx} >= len(expert_ids)={len(expert_ids)}")
            logger.error(f"[DynamicMoE] router.gate.weight.shape={self.router.gate.weight.shape}")
            logger.error(f"[DynamicMoE] This will cause CUDA indexing error!")
            # Clamp indices to valid range as emergency fix
            top_k_indices = torch.clamp(top_k_indices, 0, len(expert_ids) - 1)
            logger.warning(f"[DynamicMoE] Clamped indices to valid range [0, {len(expert_ids)-1}]")
        
        # Store for load balancing loss
        self.last_router_logits = router_logits
        
        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
        flat_top_k_weights = top_k_weights.view(-1, top_k)
        flat_top_k_indices = top_k_indices.view(-1, top_k)
        
        # Prepare output
        output = torch.zeros_like(flat_hidden)
        
        # Process each expert
        for expert_idx, expert_id in enumerate(expert_ids):
            # Find tokens routed to this expert
            expert_mask = (flat_top_k_indices == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
            
            # Get expert module
            if self.lazy_loading and self.lazy_loader is not None:
                metadata = self.registry.get_expert(expert_id)
                checkpoint_path = metadata.checkpoint_path if metadata else None
                expert = self.lazy_loader.load_expert(
                    expert_id,
                    checkpoint_path,
                    self.hidden_size,
                    self.intermediate_size,
                )
            else:
                expert = self.experts[expert_id]
            
            # Get tokens for this expert
            expert_input = flat_hidden[expert_mask]
            
            # Ensure expert is on the same device and dtype as input to prevent mismatch errors
            input_device = expert_input.device
            input_dtype = expert_input.dtype
            needs_update = False
            
            try:
                expert_device = next(expert.parameters()).device
                expert_dtype = next(expert.parameters()).dtype
                
                # Synchronize device
                if expert_device != input_device:
                    expert = expert.to(input_device)
                    needs_update = True
                
                # Synchronize dtype to prevent "expected mat1 and mat2 to have the same dtype" error
                if expert_dtype != input_dtype:
                    expert = expert.to(input_dtype)
                    needs_update = True
                    
            except StopIteration:
                # Expert has no parameters, try moving it anyway
                expert = expert.to(device=input_device, dtype=input_dtype)
                needs_update = True
            
            # Update the stored expert reference if we converted it
            if needs_update and not self.lazy_loading:
                self.experts[expert_id] = expert
            
            # Process with expert
            expert_output = expert(expert_input)
            
            # Get routing weights for this expert
            expert_positions = (flat_top_k_indices[expert_mask] == expert_idx)
            expert_weights = flat_top_k_weights[expert_mask][expert_positions].unsqueeze(-1)
            
            # Apply weights and accumulate
            output[expert_mask] += expert_output * expert_weights
        
        # Reshape to original
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, router_logits
    
    def save_checkpoint(self, save_dir: str) -> None:
        """
        Save all experts and registry to disk.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save registry
        registry_path = save_path / "expert_registry.json"
        self.registry.save(str(registry_path))
        logger.info(f"[DynamicMoE] Saved registry to {registry_path}")
        
        # Save router
        router_path = save_path / "router.pt"
        torch.save(self.router.state_dict(), router_path)
        logger.info(f"[DynamicMoE] Saved router to {router_path}")
        
        # Save each expert
        experts_dir = save_path / "experts"
        experts_dir.mkdir(exist_ok=True)
        
        for expert_id in self.get_active_expert_ids():
            if expert_id in self.experts:
                expert_path = experts_dir / f"expert_{expert_id}.pt"
                torch.save(self.experts[expert_id].state_dict(), expert_path)
                
                # Update metadata checkpoint path
                metadata = self.registry.get_expert(expert_id)
                if metadata:
                    metadata.checkpoint_path = str(expert_path)
                
                logger.info(f"[DynamicMoE] Saved expert {expert_id} to {expert_path}")
        
        # Save updated registry with checkpoint paths
        self.registry.save(str(registry_path))
        
        logger.info(f"[DynamicMoE] Checkpoint complete: {len(self.get_active_expert_ids())} experts saved")
    
    def load_checkpoint(self, load_dir: str, strict: bool = False) -> None:
        """
        Load experts and registry from disk.
        
        Args:
            load_dir: Directory containing checkpoints
            strict: Raise error if checkpoint files missing (default: False)
        """
        load_path = Path(load_dir)
        
        # Load registry first
        registry_path = load_path / "expert_registry.json"
        if registry_path.exists():
            self.registry = ExpertRegistry.load(str(registry_path))
            print(f"[DEBUG] Loaded registry with {len(self.registry.experts)} experts")
        elif strict:
            raise FileNotFoundError(f"Registry not found: {registry_path}")
        
        # Load router weights FIRST to check compatibility
        router_path = load_path / "router.pt"
        router_state_dict = None
        if router_path.exists():
            router_state_dict = torch.load(router_path, map_location=self.device)
            logger.info(f"[DynamicMoE] Found router checkpoint at {router_path}")
        elif strict:
            raise FileNotFoundError(f"Router not found: {router_path}")
        
        # Check if router size matches registry
        num_experts_to_load = len(self.get_active_expert_ids())
        print(f"[DEBUG] Active expert IDs after loading registry: {self.get_active_expert_ids()}")
        print(f"[DEBUG] Current router has {self.router.num_experts} experts, registry has {num_experts_to_load} experts")
        
        # Check router checkpoint size vs registry
        checkpoint_num_experts = None
        if router_state_dict and 'gate.weight' in router_state_dict:
            checkpoint_num_experts = router_state_dict['gate.weight'].shape[0]  # Output dimension = num_experts
            print(f"[DEBUG] Checkpoint router has {checkpoint_num_experts} experts")
        
        # If sizes match, load the checkpoint router
        if router_state_dict and checkpoint_num_experts == num_experts_to_load:
            print(f"[DEBUG] Loading router from checkpoint (sizes match: {checkpoint_num_experts} experts)")
            self.router.load_state_dict(router_state_dict)
        else:
            # Sizes don't match - reinitialize router to match registry
            if checkpoint_num_experts:
                logger.warning(
                    f"[DynamicMoE] Router size mismatch! Checkpoint has {checkpoint_num_experts} experts "
                    f"but registry has {num_experts_to_load} experts. Creating new router."
                )
            print(f"[DEBUG] Reinitializing router to {num_experts_to_load} experts")
            self._update_router_size()
            print(f"[DEBUG] After resize, router has {self.router.num_experts} experts")
        
        # Load experts (if not lazy loading)
        if not self.lazy_loading:
            experts_dir = load_path / "experts"
            if experts_dir.exists():
                for expert_metadata in self.registry.get_active_experts():
                    expert_id = expert_metadata.expert_id
                    expert_path = experts_dir / f"expert_{expert_id}.pt"
                    if expert_path.exists():
                        expert = FeedForward(self.hidden_size, self.intermediate_size or self.hidden_size * 4)
                        expert.load_state_dict(torch.load(expert_path, map_location=self.device))
                        expert.to(self.device)
                        self.experts[expert_id] = expert
                        logger.info(f"[DynamicMoE] Loaded expert {expert_id}")
                    elif strict:
                        raise FileNotFoundError(f"Expert checkpoint not found: {expert_path}")
        
        logger.info(f"[DynamicMoE] Loaded {len(self.get_active_expert_ids())} experts from {load_dir}")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing DynamicMoELayer...")
    
    # Force CPU for testing to avoid device issues
    device = "cpu"
    
    # Test 1: Basic creation
    print("\n[Test 1] Creating DynamicMoELayer...")
    moe = DynamicMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts_per_tok=2,
        lazy_loading=False,
        device=device,
    )
    print(f"[OK] Created DynamicMoELayer with 0 experts initially")
    
    # Test 2: Add experts
    print("\n[Test 2] Adding experts...")
    from aios.core.hrm_models.expert_metadata import create_expert_metadata
    
    for i in range(4):
        expert_id = f"expert_{i}"
        metadata = create_expert_metadata(
            expert_id=expert_id,
            name=f"Expert {i}",
            description=f"Test expert {i}",
            category="general" if i < 2 else "specialized",
        )
        moe.add_expert(expert_id, metadata=metadata)
    
    active_ids = moe.get_active_expert_ids()
    print(f"[OK] Added 4 experts: {active_ids}")
    assert len(active_ids) == 4
    
    # Test 3: Forward pass
    print("\n[Test 3] Testing forward pass...")
    x = torch.randn(2, 10, 256)
    output, router_logits = moe(x)
    print(f"[OK] Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Router logits shape: {router_logits.shape}")
    assert output.shape == (2, 10, 256)
    assert router_logits.shape == (2, 10, 4)
    
    # Test 4: Freeze/unfreeze expert
    print("\n[Test 4] Testing freeze/unfreeze...")
    moe.freeze_expert("expert_0")
    assert "expert_0" in moe.frozen_experts
    print(f"[OK] Froze expert_0")
    
    moe.unfreeze_expert("expert_0")
    assert "expert_0" not in moe.frozen_experts
    print(f"[OK] Unfroze expert_0")
    
    # Test 5: Remove expert
    print("\n[Test 5] Testing expert removal...")
    moe.remove_expert("expert_3")
    active_ids = moe.get_active_expert_ids()
    print(f"[OK] Removed expert_3, remaining: {active_ids}")
    assert len(active_ids) == 3
    assert "expert_3" not in active_ids
    
    # Test 6: Save/load checkpoint
    print("\n[Test 6] Testing save/load checkpoint...")
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        moe.save_checkpoint(tmpdir)
        print(f"[OK] Saved checkpoint to {tmpdir}")
        
        # Create new layer and load
        moe2 = DynamicMoELayer(
            hidden_size=256,
            intermediate_size=1024,
            num_experts_per_tok=2,
            device=device,
        )
        moe2.load_checkpoint(tmpdir)
        
        loaded_ids = moe2.get_active_expert_ids()
        print(f"[OK] Loaded checkpoint, experts: {loaded_ids}")
        assert len(loaded_ids) == 3
        assert set(loaded_ids) == set(active_ids)
    
    # Test 7: Lazy loading
    print("\n[Test 7] Testing lazy loading...")
    moe_lazy = DynamicMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts_per_tok=2,
        lazy_loading=True,
        max_gpu_experts=2,
        device=device,
    )
    
    # Add experts with lazy loading
    for i in range(4):
        expert_id = f"lazy_expert_{i}"
        metadata = create_expert_metadata(
            expert_id=expert_id,
            name=f"Lazy Expert {i}",
            description=f"Lazy test expert {i}",
            category="general"
        )
        moe_lazy.add_expert(expert_id, metadata=metadata)
    
    print(f"[OK] Created lazy MoE with 4 experts")
    
    # Get lazy loader stats
    if moe_lazy.lazy_loader:
        stats = moe_lazy.lazy_loader.get_stats()
        print(f"   Lazy loader stats: {stats}")
    
    print("\n[OK] All tests passed!")

