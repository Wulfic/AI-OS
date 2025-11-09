"""Goal-aware router for dynamic expert selection.

This module extends the basic top-k router to incorporate goal-based biasing,
allowing experts linked to active goals to receive higher routing probabilities.

Key Features:
- Bias routing toward experts linked to active goals
- Configurable bias strength (0.0-5.0)
- Routing history tracking for debugging
- Compatible with existing TopKRouter interface

Author: AI-OS Team
Date: January 2025
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from datetime import datetime
import logging

from aios.core.hrm_models.moe_layer import TopKRouter
from aios.core.hrm_models.expert_metadata import ExpertRegistry

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Record of a single routing decision for debugging/analysis."""
    
    timestamp: str
    active_goals: List[str]
    expert_logits_before_bias: Dict[str, float]
    expert_logits_after_bias: Dict[str, float]
    selected_experts: List[str]
    expert_weights: List[float]
    bias_applied: Dict[str, float]
    
    def __repr__(self) -> str:
        return (
            f"RoutingDecision("
            f"goals={self.active_goals}, "
            f"selected={self.selected_experts}, "
            f"weights={[f'{w:.3f}' for w in self.expert_weights]})"
        )


class GoalAwareRouter(nn.Module):
    """
    Router that biases expert selection toward those linked to active goals.
    
    This extends the basic TopKRouter to incorporate goal information, allowing
    the model to preferentially route to experts that are relevant to the user's
    current objectives.
    
    Example:
        User has active goal "learn_python"
        Expert "Python Programming" is linked to goal "learn_python"
        When routing, this expert receives a logit boost, making it more likely
        to be selected even if the base router wouldn't have chosen it.
    
    Args:
        hidden_size: Hidden dimension size
        num_experts: Number of experts (can change dynamically)
        expert_registry: ExpertRegistry for goal-expert mappings
        bias_strength: Multiplier for goal bias (0.0-5.0)
        track_history: Whether to track routing decisions
        max_history: Maximum routing decisions to keep in memory
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_registry: Optional[ExpertRegistry] = None,
        bias_strength: float = 1.0,
        track_history: bool = True,
        max_history: int = 100,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_registry = expert_registry or ExpertRegistry()
        self.bias_strength = bias_strength
        self.track_history = track_history
        self.max_history = max_history
        
        # Base router (standard learned gating)
        self.base_router = TopKRouter(hidden_size, num_experts)
        
        # Routing history for debugging
        self.routing_history: List[RoutingDecision] = []
        
        # Statistics
        self.total_routings = 0
        self.goal_biased_routings = 0
    
    def compute_goal_bias(
        self,
        active_goal_ids: List[str],
        expert_ids: List[str],
    ) -> torch.Tensor:
        """
        Compute bias values for each expert based on goal matching.
        
        Args:
            active_goal_ids: List of currently active goal IDs
            expert_ids: List of expert IDs in same order as logits
        
        Returns:
            bias: Tensor of shape [num_experts] with bias values
                  0.0 for experts not linked to any active goal
                  1.0 * bias_strength for experts linked to active goals
        """
        num_experts = len(expert_ids)
        bias = torch.zeros(num_experts)
        
        if not active_goal_ids or not expert_ids:
            return bias
        
        # Check each expert for goal matches
        for idx, expert_id in enumerate(expert_ids):
            expert_meta = self.expert_registry.get_expert(expert_id)
            if expert_meta is None:
                continue
            
            # Check if expert is linked to any active goal
            expert_goals = set(expert_meta.goals)
            active_goals = set(active_goal_ids)
            
            if expert_goals & active_goals:  # Intersection
                # Expert matches at least one active goal
                bias[idx] = self.bias_strength
        
        return bias
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: int,
        active_goal_ids: Optional[List[str]] = None,
        expert_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with goal-aware routing.
        
        Args:
            hidden_states: [batch, seq, hidden]
            top_k: Number of experts to select per token
            active_goal_ids: List of active goal IDs (optional)
            expert_ids: List of expert IDs matching router experts (optional)
        
        Returns:
            top_k_weights: [batch, seq, top_k] - Routing weights for selected experts
            top_k_indices: [batch, seq, top_k] - Indices of selected experts
            logits: [batch, seq, num_experts] - Router logits (after bias)
        """
        # Get base routing logits
        logits = self.base_router.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Store pre-bias logits for history tracking
        logits_before_bias = logits.clone() if self.track_history else None
        
        # Apply goal bias if goals are active
        bias_applied = None
        if active_goal_ids and expert_ids:
            # Compute bias for each expert
            bias = self.compute_goal_bias(active_goal_ids, expert_ids)
            
            if bias.sum() > 0:
                # Apply bias to logits
                # bias shape: [num_experts]
                # logits shape: [batch, seq, num_experts]
                bias = bias.to(logits.device)
                logits = logits + bias.unsqueeze(0).unsqueeze(0)  # Broadcast to [batch, seq, num_experts]
                
                bias_applied = bias
                self.goal_biased_routings += 1
        
        # Top-k selection with biased logits
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Track routing decision
        if self.track_history and expert_ids and logits_before_bias is not None:
            self._record_routing_decision(
                active_goal_ids=active_goal_ids or [],
                expert_ids=expert_ids,
                logits_before_bias=logits_before_bias,
                logits_after_bias=logits,
                top_k_indices=top_k_indices,
                top_k_weights=top_k_weights,
                bias_applied=bias_applied,
            )
        
        self.total_routings += 1
        
        return top_k_weights, top_k_indices, logits
    
    def _record_routing_decision(
        self,
        active_goal_ids: List[str],
        expert_ids: List[str],
        logits_before_bias: torch.Tensor,
        logits_after_bias: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        bias_applied: Optional[torch.Tensor],
    ):
        """Record a routing decision for debugging."""
        # Use first token of first batch for simplicity
        # [batch, seq, num_experts] -> [num_experts]
        logits_before = logits_before_bias[0, 0].detach().cpu().tolist()
        logits_after = logits_after_bias[0, 0].detach().cpu().tolist()
        
        # Selected experts for first token
        selected_indices = top_k_indices[0, 0].detach().cpu().tolist()
        selected_weights = top_k_weights[0, 0].detach().cpu().tolist()
        
        # Create dictionaries mapping expert_id -> logit
        logits_before_dict = {expert_ids[i]: logits_before[i] for i in range(len(expert_ids))}
        logits_after_dict = {expert_ids[i]: logits_after[i] for i in range(len(expert_ids))}
        
        # Get selected expert IDs
        selected_expert_ids = [expert_ids[i] for i in selected_indices]
        
        # Compute bias applied to each expert
        bias_dict = {}
        if bias_applied is not None:
            bias_list = bias_applied.detach().cpu().tolist()
            bias_dict = {expert_ids[i]: bias_list[i] for i in range(len(expert_ids))}
        
        # Create routing decision record
        decision = RoutingDecision(
            timestamp=datetime.now().isoformat(),
            active_goals=active_goal_ids,
            expert_logits_before_bias=logits_before_dict,
            expert_logits_after_bias=logits_after_dict,
            selected_experts=selected_expert_ids,
            expert_weights=selected_weights,
            bias_applied=bias_dict,
        )
        
        # Add to history
        self.routing_history.append(decision)
        
        # Trim history if needed
        if len(self.routing_history) > self.max_history:
            self.routing_history = self.routing_history[-self.max_history:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        bias_rate = (
            self.goal_biased_routings / max(1, self.total_routings)
            if self.total_routings > 0
            else 0.0
        )
        
        return {
            "total_routings": self.total_routings,
            "goal_biased_routings": self.goal_biased_routings,
            "bias_rate": bias_rate,
            "bias_strength": self.bias_strength,
            "history_size": len(self.routing_history),
            "max_history": self.max_history,
        }
    
    def get_recent_history(self, n: int = 10) -> List[RoutingDecision]:
        """Get the n most recent routing decisions."""
        return self.routing_history[-n:]
    
    def clear_history(self):
        """Clear routing history and reset statistics."""
        self.routing_history.clear()
        self.total_routings = 0
        self.goal_biased_routings = 0
    
    def set_bias_strength(self, strength: float):
        """Update bias strength (0.0-5.0)."""
        if not 0.0 <= strength <= 5.0:
            logger.warning(f"Bias strength {strength} outside recommended range [0.0, 5.0]")
        self.bias_strength = strength
        logger.info(f"[GoalAwareRouter] Updated bias strength to {strength}")
    
    def update_expert_registry(self, registry: ExpertRegistry):
        """Update the expert registry reference."""
        self.expert_registry = registry
        logger.info(f"[GoalAwareRouter] Updated expert registry with {len(registry.experts)} experts")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing GoalAwareRouter...")
    
    # Force CPU for testing
    device = "cpu"
    
    # Test 1: Basic creation
    print("\n[Test 1] Creating GoalAwareRouter...")
    from aios.core.hrm_models.expert_metadata import create_expert_metadata, ExpertRegistry
    
    # Create registry with experts linked to goals
    registry = ExpertRegistry()
    
    # Expert 0: Python expert linked to "learn_python" goal
    python_expert = create_expert_metadata(
        expert_id="expert_0",
        name="Python Programming",
        description="Expert in Python programming",
        category="programming",
        goals=["learn_python", "coding_tasks"],
    )
    registry.add_expert(python_expert)
    
    # Expert 1: Math expert linked to "learn_math" goal
    math_expert = create_expert_metadata(
        expert_id="expert_1",
        name="Mathematics",
        description="Expert in mathematical concepts",
        category="mathematics",
        goals=["learn_math", "problem_solving"],
    )
    registry.add_expert(math_expert)
    
    # Expert 2: General expert with no specific goals
    general_expert = create_expert_metadata(
        expert_id="expert_2",
        name="General Knowledge",
        description="General knowledge expert",
        category="general",
        goals=[],
    )
    registry.add_expert(general_expert)
    
    # Expert 3: Writing expert linked to "creative_writing" goal
    writing_expert = create_expert_metadata(
        expert_id="expert_3",
        name="Creative Writing",
        description="Expert in creative writing",
        category="writing",
        goals=["creative_writing", "storytelling"],
    )
    registry.add_expert(writing_expert)
    
    router = GoalAwareRouter(
        hidden_size=256,
        num_experts=4,
        expert_registry=registry,
        bias_strength=2.0,
        track_history=True,
    )
    print(f"[OK] Created GoalAwareRouter with {router.num_experts} experts")
    print(f"     Bias strength: {router.bias_strength}")
    
    # Test 2: Routing without goals (baseline)
    print("\n[Test 2] Routing without active goals...")
    x = torch.randn(1, 5, 256)  # [batch=1, seq=5, hidden=256]
    expert_ids = ["expert_0", "expert_1", "expert_2", "expert_3"]
    
    weights, indices, logits = router(x, top_k=2, active_goal_ids=None, expert_ids=expert_ids)
    
    print(f"[OK] Routing completed")
    print(f"     Output shapes: weights={weights.shape}, indices={indices.shape}, logits={logits.shape}")
    print(f"     Selected experts (token 0): {[expert_ids[i] for i in indices[0, 0].tolist()]}")
    print(f"     Weights (token 0): {weights[0, 0].tolist()}")
    
    # Test 3: Routing with active goal (should bias toward Python expert)
    print("\n[Test 3] Routing with active goal 'learn_python'...")
    active_goals = ["learn_python"]
    
    weights_biased, indices_biased, logits_biased = router(
        x, top_k=2, active_goal_ids=active_goals, expert_ids=expert_ids
    )
    
    selected_experts = [expert_ids[i] for i in indices_biased[0, 0].tolist()]
    print(f"[OK] Routing with goal bias completed")
    print(f"     Active goals: {active_goals}")
    print(f"     Selected experts (token 0): {selected_experts}")
    print(f"     Weights (token 0): {weights_biased[0, 0].tolist()}")
    
    # Check if Python expert was selected more often
    if "expert_0" in selected_experts:
        print(f"     [BIAS EFFECT] Python expert selected (linked to active goal)")
    else:
        print(f"     [NOTE] Python expert not selected despite bias (base router very confident in others)")
    
    # Test 4: Compute bias directly
    print("\n[Test 4] Testing bias computation...")
    bias = router.compute_goal_bias(["learn_python"], expert_ids)
    print(f"[OK] Bias computed: {bias.tolist()}")
    print(f"     expert_0 (Python, has goal): {bias[0].item()}")
    print(f"     expert_1 (Math, no match): {bias[1].item()}")
    print(f"     expert_2 (General, no goals): {bias[2].item()}")
    print(f"     expert_3 (Writing, no match): {bias[3].item()}")
    
    expected_bias = [2.0, 0.0, 0.0, 0.0]  # Only expert_0 matches
    assert bias.tolist() == expected_bias, f"Expected {expected_bias}, got {bias.tolist()}"
    print(f"     [OK] Bias values correct!")
    
    # Test 5: Multiple active goals
    print("\n[Test 5] Routing with multiple active goals...")
    multi_goals = ["learn_python", "creative_writing"]
    
    weights_multi, indices_multi, logits_multi = router(
        x, top_k=2, active_goal_ids=multi_goals, expert_ids=expert_ids
    )
    
    selected_experts_multi = [expert_ids[i] for i in indices_multi[0, 0].tolist()]
    print(f"[OK] Routing with multiple goals completed")
    print(f"     Active goals: {multi_goals}")
    print(f"     Selected experts (token 0): {selected_experts_multi}")
    print(f"     Weights (token 0): {weights_multi[0, 0].tolist()}")
    
    # Should bias toward expert_0 (Python) AND expert_3 (Writing)
    bias_multi = router.compute_goal_bias(multi_goals, expert_ids)
    print(f"     Bias applied: {bias_multi.tolist()}")
    
    # Test 6: Routing history
    print("\n[Test 6] Checking routing history...")
    history = router.get_recent_history(n=3)
    print(f"[OK] Retrieved {len(history)} recent routing decisions")
    
    for i, decision in enumerate(history):
        print(f"\n     Decision {i+1}:")
        print(f"       Active goals: {decision.active_goals}")
        print(f"       Selected experts: {decision.selected_experts}")
        print(f"       Expert weights: {[f'{w:.3f}' for w in decision.expert_weights]}")
        if decision.bias_applied:
            print(f"       Bias applied: {[(k, f'{v:.2f}') for k, v in decision.bias_applied.items() if v > 0]}")
    
    # Test 7: Statistics
    print("\n[Test 7] Routing statistics...")
    stats = router.get_routing_stats()
    print(f"[OK] Statistics retrieved:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    # Test 8: Bias strength adjustment
    print("\n[Test 8] Testing bias strength adjustment...")
    router.set_bias_strength(3.5)
    assert router.bias_strength == 3.5
    print(f"[OK] Bias strength updated to {router.bias_strength}")
    
    # Route again with new strength
    bias_new = router.compute_goal_bias(["learn_python"], expert_ids)
    print(f"     New bias for Python expert: {bias_new[0].item()}")
    assert bias_new[0].item() == 3.5
    print(f"     [OK] Bias strength change reflected in computation")
    
    # Test 9: Clear history
    print("\n[Test 9] Testing history clearing...")
    initial_size = len(router.routing_history)
    router.clear_history()
    print(f"[OK] History cleared (was {initial_size}, now {len(router.routing_history)})")
    assert len(router.routing_history) == 0
    assert router.total_routings == 0
    assert router.goal_biased_routings == 0
    print(f"     [OK] Statistics reset")
    
    print("\n[OK] All tests passed!")
