"""Build comprehensive brain details text for display.

Functions to build formatted text sections for brain information display.
Each section builder returns a list of strings (lines of text).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional


def build_model_info_section(
    brain_stats: dict[str, Any],
    brain_metadata: dict[str, Any]
) -> list[str]:
    """Build model information section with size, parameters, status, timestamps.
    
    Args:
        brain_stats: Brain statistics from registry
        brain_metadata: Brain metadata from brain.json
        
    Returns:
        List of formatted text lines
    """
    lines = []
    lines.append("📦 MODEL INFORMATION")
    lines.append("─" * 70)
    
    brain_type = brain_metadata.get("type", "ActV1")
    lines.append(f"Type:              {brain_type}")
    
    # Calculate theoretical max parameters from architecture
    arch_config = brain_metadata.get("arch", {})
    h_layers = (arch_config.get("H_layers") or brain_metadata.get("h_layers") or 
                arch_config.get("h_layers"))
    l_layers = (arch_config.get("L_layers") or brain_metadata.get("l_layers") or 
                arch_config.get("l_layers"))
    hidden_size = arch_config.get("hidden_size") or brain_metadata.get("hidden_size")
    vocab_size = arch_config.get("vocab_size") or brain_metadata.get("vocab_size")
    expansion = arch_config.get("expansion") or brain_metadata.get("expansion") or 2.0
    use_moe = brain_metadata.get("use_moe", False)
    num_experts = brain_metadata.get("num_experts", 8)
    
    # Calculate theoretical max parameters
    max_params = None
    if h_layers and l_layers and hidden_size and vocab_size:
        # Embedding layer: vocab_size * hidden_size
        embed_params = vocab_size * hidden_size
        
        # Attention params per layer: 4 * hidden_size^2 (Q, K, V, O projections)
        attn_params_per_layer = 4 * hidden_size * hidden_size
        
        # FFN params per layer
        ffn_hidden = int(hidden_size * expansion)
        if use_moe:
            # MoE: num_experts * (up + down projections) + gate
            ffn_params_per_layer = (num_experts * (hidden_size * ffn_hidden + ffn_hidden * hidden_size) + 
                                   hidden_size * num_experts)
        else:
            # Dense: up + down projections
            ffn_params_per_layer = hidden_size * ffn_hidden + ffn_hidden * hidden_size
        
        # Layer norm params (2 per layer: post-attn, post-ffn)
        ln_params_per_layer = 2 * hidden_size * 2  # scale + bias for each LN
        
        total_layers = h_layers + l_layers
        layer_params = total_layers * (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer)
        
        # Output head
        output_params = hidden_size * vocab_size
        
        max_params = embed_params + layer_params + output_params
    
    # Size and current parameters from file
    size_bytes = brain_stats.get("size_bytes", 0) or brain_metadata.get("size_bytes", 0)
    if size_bytes:
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_mb / 1024
        # Assuming fp32 = 4 bytes per param
        current_params = size_bytes / 4
        
        if size_gb >= 1.0:
            lines.append(f"File Size:         {size_gb:.2f} GB ({size_mb:.1f} MB)")
        else:
            lines.append(f"File Size:         {size_mb:.2f} MB")
        
        # Show current parameters
        if current_params >= 1_000_000_000:
            params_b = current_params / 1_000_000_000
            lines.append(f"Current Params:    {params_b:.2f}B ({current_params:,.0f})")
        elif current_params >= 1_000_000:
            params_m = current_params / 1_000_000
            lines.append(f"Current Params:    {params_m:.2f}M ({current_params:,.0f})")
        else:
            lines.append(f"Current Params:    {current_params:,.0f}")
    else:
        lines.append("File Size:         Unknown")
        lines.append("Current Params:    Unknown")
    
    # Show max theoretical parameters
    if max_params:
        if max_params >= 1_000_000_000:
            max_params_b = max_params / 1_000_000_000
            lines.append(f"Max Params:        {max_params_b:.2f}B ({max_params:,.0f})")
        elif max_params >= 1_000_000:
            max_params_m = max_params / 1_000_000
            lines.append(f"Max Params:        {max_params_m:.2f}M ({max_params:,.0f})")
        else:
            lines.append(f"Max Params:        {max_params:,.0f}")
    else:
        lines.append("Max Params:        Unknown")
    
    # Status
    pinned = brain_stats.get("pinned", False)
    master = brain_stats.get("master", False)
    status_parts = []
    if master:
        status_parts.append("🌟 Master")
    if pinned:
        status_parts.append("📌 Pinned")
    if not status_parts:
        status_parts.append("Regular")
    lines.append(f"Status:            {' | '.join(status_parts)}")
    
    # Timestamps
    created_at = brain_metadata.get("created_at")
    if created_at:
        created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        lines.append(f"Created:           {created_str}")
    
    last_used = brain_stats.get("last_used")
    if last_used:
        lines.append(f"Last Used:         {last_used}")
    
    lines.append("")
    return lines


def build_config_section(brain_metadata: dict[str, Any]) -> list[str]:
    """Build configuration section with architecture details.
    
    Args:
        brain_metadata: Brain metadata from brain.json
        
    Returns:
        List of formatted text lines
    """
    lines = []
    
    # Extract architecture from arch dict or root level metadata
    arch_config = brain_metadata.get("arch", {}) or {}
    
    # Try to get values from arch dict first, fallback to root level
    h_layers = arch_config.get("H_layers") or brain_metadata.get("h_layers") or arch_config.get("h_layers")
    l_layers = arch_config.get("L_layers") or brain_metadata.get("l_layers") or arch_config.get("l_layers")
    hidden_size = arch_config.get("hidden_size") or brain_metadata.get("hidden_size")
    num_heads = arch_config.get("num_heads") or brain_metadata.get("num_heads")
    expansion = arch_config.get("expansion") or brain_metadata.get("expansion")
    h_cycles = arch_config.get("H_cycles") or brain_metadata.get("h_cycles") or arch_config.get("h_cycles")
    l_cycles = arch_config.get("L_cycles") or brain_metadata.get("l_cycles") or arch_config.get("l_cycles")
    pos_enc = (arch_config.get("pos_encodings") or brain_metadata.get("pos_encodings") or 
               brain_metadata.get("position_encoding"))
    vocab_size = arch_config.get("vocab_size") or brain_metadata.get("vocab_size")
    
    # MoE configuration
    use_moe = brain_metadata.get("use_moe", False)
    num_experts = brain_metadata.get("num_experts", 8)
    active_per_tok = brain_metadata.get("num_experts_per_tok", 2)
    
    # Tokenizer
    tokenizer_id = brain_metadata.get("tokenizer_id") or brain_metadata.get("tokenizer_model")
    max_seq = brain_metadata.get("max_seq_len")
    
    if not any([h_layers, l_layers, hidden_size, num_heads, expansion, h_cycles, l_cycles, 
                pos_enc, use_moe, tokenizer_id]):
        return lines
    
    lines.append("⚙️  CONFIGURATION")
    lines.append("─" * 70)
    
    # Layer configuration line
    h_str = str(h_layers) if h_layers is not None else "-"
    l_str = str(l_layers) if l_layers is not None else "-"
    hidden_str = str(hidden_size) if hidden_size else "-"
    heads_str = str(num_heads) if num_heads else "-"
    lines.append(f"H/L layers: {h_str:>3} / {l_str:<3}     Hidden: {hidden_str:<6}   Heads: {heads_str}")
    
    # Expansion and cycles line
    exp_str = str(expansion) if expansion is not None else "-"
    h_cycle_str = str(h_cycles) if h_cycles is not None else "-"
    l_cycle_str = str(l_cycles) if l_cycles is not None else "-"
    pos_str = str(pos_enc).lower() if pos_enc else "-"
    lines.append(f"Expansion: {exp_str:<4}        H/L cycles: {h_cycle_str} / {l_cycle_str:<3}   PosEnc: {pos_str}")
    
    # MoE and Tokenizer line
    moe_experts_str = str(num_experts) if use_moe else "-"
    moe_active_str = str(active_per_tok) if use_moe else "-"
    tok_str = str(tokenizer_id) if tokenizer_id else "-"
    
    # Shorten tokenizer path if it's too long
    if tok_str and len(tok_str) > 30:
        tok_str = os.path.basename(tok_str)
    if tok_str and len(tok_str) > 30:
        tok_str = tok_str[:27] + "..."
    
    lines.append(f"MoE Experts: {moe_experts_str:<5}    Active: {moe_active_str:<5}       Tokenizer: {tok_str}")
    
    # Additional detailed information
    lines.append("")
    lines.append("Detailed Architecture:")
    
    if h_layers is not None or l_layers is not None:
        total_layers = (h_layers or 0) + (l_layers or 0)
        lines.append(f"  Total Layers:      {total_layers} ({h_layers or 0} H-Level + {l_layers or 0} L-Level)")
    
    if hidden_size:
        lines.append(f"  Hidden Size:       {hidden_size} dimensions")
    
    if num_heads:
        if hidden_size:
            head_dim = hidden_size // num_heads
            lines.append(f"  Attention Heads:   {num_heads} heads ({head_dim} dim each)")
        else:
            lines.append(f"  Attention Heads:   {num_heads}")
    
    if expansion:
        if hidden_size:
            ffn_size = int(hidden_size * expansion)
            lines.append(f"  FFN Expansion:     {expansion}x (→ {ffn_size} dimensions)")
        else:
            lines.append(f"  FFN Expansion:     {expansion}x")
    
    if h_cycles or l_cycles:
        lines.append(f"  Processing Cycles: {h_cycles or 0} H-cycles, {l_cycles or 0} L-cycles")
    
    if pos_enc:
        lines.append(f"  Position Encoding: {pos_enc.upper()}")
    
    if vocab_size:
        lines.append(f"  Vocabulary Size:   {vocab_size:,} tokens")
    
    if max_seq:
        lines.append(f"  Max Sequence:      {max_seq:,} tokens")
    
    if use_moe:
        compute_reduction = (1 - active_per_tok / num_experts) * 100
        lines.append(f"  MoE Configuration: {num_experts} experts, {active_per_tok} active per token")
        lines.append(f"  Compute Reduction: ~{compute_reduction:.0f}% vs. dense model")
    
    lines.append("")
    return lines


def build_training_section(
    brain_stats: dict[str, Any],
    brain_metadata: dict[str, Any],
    training_steps: int
) -> list[str]:
    """Build training section with steps, rates, batch sizes.
    
    Args:
        brain_stats: Brain statistics from registry
        brain_metadata: Brain metadata from brain.json
        training_steps: Total training steps (may be loaded from metrics)
        
    Returns:
        List of formatted text lines
    """
    lines = []
    lines.append("🎓 TRAINING")
    lines.append("─" * 70)
    
    lines.append(f"Total Trained Steps: {training_steps:,}")
    
    # Show last trained timestamp
    last_trained = brain_stats.get("last_trained") or brain_metadata.get("last_trained")
    if last_trained:
        last_trained_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_trained))
        lines.append(f"Last Trained:      {last_trained_str}")
    
    learning_rate = brain_metadata.get("learning_rate")
    if learning_rate:
        # Convert scientific notation to regular decimal format
        try:
            lr_float = float(learning_rate)
            lr_formatted = f"{lr_float:.6f}".rstrip('0').rstrip('.')
            lines.append(f"Learning Rate:     {lr_formatted}")
        except (ValueError, TypeError):
            lines.append(f"Learning Rate:     {learning_rate}")
    
    batch_size = brain_metadata.get("batch_size")
    if batch_size:
        lines.append(f"Batch Size:        {batch_size}")
    
    halt_max_steps = brain_metadata.get("halt_max_steps")
    if halt_max_steps:
        lines.append(f"Halt Max Steps:    {halt_max_steps}")
    
    log_file = brain_metadata.get("log_file")
    if log_file:
        lines.append(f"Log File:          {log_file}")
    
    lines.append("")
    return lines


def build_dataset_section(
    brain_stats: dict[str, Any],
    brain_metadata: dict[str, Any],
    training_steps: int
) -> list[str]:
    """Build dataset tracking section with usage statistics.
    
    Args:
        brain_stats: Brain statistics from registry
        brain_metadata: Brain metadata from brain.json
        training_steps: Total training steps for percentage calculations
        
    Returns:
        List of formatted text lines
    """
    lines = []
    
    dataset_stats = brain_stats.get("dataset_stats") or brain_metadata.get("dataset_stats", {})
    dataset_history = brain_stats.get("dataset_history") or brain_metadata.get("dataset_history", [])
    
    if not dataset_stats and not dataset_history:
        # Fallback: show old-style dataset field if present
        dataset_file = brain_metadata.get("dataset_file")
        if dataset_file:
            lines.append("📊 DATASET TRACKING")
            lines.append("─" * 70)
            # Show only filename if it's a path
            dataset_name = (os.path.basename(dataset_file) if ('/' in dataset_file or '\\' in dataset_file) 
                          else dataset_file)
            lines.append(f"Dataset:           {dataset_name}")
            lines.append("")
        return lines
    
    lines.append("📊 DATASET TRACKING")
    lines.append("─" * 70)
    
    dataset_stats = dict(dataset_stats or {})
    if dataset_stats:
        num_datasets = len(dataset_stats)
        lines.append(f"Datasets Used:     {num_datasets}")
        lines.append("")
        
        # Sort datasets by total steps (most used first)
        sorted_datasets = sorted(
            dataset_stats.items(),
            key=lambda x: x[1].get('total_steps', 0),
            reverse=True
        )
        
        for i, (dataset_name, stats) in enumerate(sorted_datasets[:5], 1):  # Show top 5
            times_used = stats.get('times_used', 0)
            total_ds_steps = stats.get('total_steps', 0)
            last_used_ts = stats.get('last_used', 0)
            
            # Calculate percentage of total training
            if training_steps > 0:
                pct = (total_ds_steps / training_steps) * 100
                pct_str = f" ({pct:.1f}%)"
            else:
                pct_str = ""
            
            lines.append(f"  {i}. {dataset_name}")
            lines.append(f"     Sessions: {times_used}  |  Steps: {total_ds_steps:,}{pct_str}")
            
            if last_used_ts:
                last_used_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_used_ts))
                lines.append(f"     Last Used: {last_used_str}")
            lines.append("")
        
        if len(dataset_stats) > 5:
            lines.append(f"  ... and {len(dataset_stats) - 5} more datasets")
            lines.append("")
    
    # Show recent training history
    if dataset_history:
        lines.append("Recent Training Sessions:")
        lines.append("")
        
        # Show last 5 sessions
        recent_sessions = dataset_history[-5:] if len(dataset_history) > 5 else dataset_history
        recent_sessions = list(reversed(recent_sessions))  # Most recent first
        
        for i, session in enumerate(recent_sessions, 1):
            dataset_name = session.get('dataset_name', 'Unknown')
            steps = session.get('steps', 0)
            timestamp = session.get('timestamp', 0)
            
            if timestamp:
                session_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
                lines.append(f"  {i}. {session_time}")
            else:
                lines.append(f"  {i}. Unknown Date")
            
            lines.append(f"     {dataset_name} - {steps:,} steps")
            lines.append("")
        
        if len(dataset_history) > 5:
            lines.append(f"  ... and {len(dataset_history) - 5} earlier sessions")
            lines.append("")
    
    return lines


def build_goals_section(
    brain_metadata: dict[str, Any],
    goals_list_callback: Optional[Callable[[str], list[str]]],
    brain_name: str
) -> list[str]:
    """Build goals section with default goal and active goals.
    
    Args:
        brain_metadata: Brain metadata from brain.json
        goals_list_callback: Optional callback to get active goals list
        brain_name: Name of the brain
        
    Returns:
        List of formatted text lines
    """
    lines = []
    
    default_goal = brain_metadata.get("default_goal", "")
    active_goals = []
    
    if goals_list_callback:
        try:
            active_goals = list(goals_list_callback(brain_name) or [])
        except Exception:
            pass
    
    if not default_goal and not active_goals:
        return lines
    
    lines.append("🎯 GOALS")
    lines.append("─" * 70)
    
    if default_goal:
        # Word wrap the goal text
        goal_lines = [default_goal[i:i+64] for i in range(0, len(default_goal), 64)]
        lines.append("Default Goal:")
        for line in goal_lines:
            lines.append(f"  {line}")
    
    if active_goals:
        lines.append("")
        lines.append(f"Active Goals ({len(active_goals)}):")
        for goal in active_goals[:10]:  # Show max 10 goals
            goal_text = str(goal)[:60]
            lines.append(f"  • {goal_text}")
        if len(active_goals) > 10:
            lines.append(f"  ... and {len(active_goals) - 10} more")
    
    lines.append("")
    return lines


def build_relationships_section(brain_stats: dict[str, Any]) -> list[str]:
    """Build relationships section with parent and children.
    
    Args:
        brain_stats: Brain statistics from registry
        
    Returns:
        List of formatted text lines
    """
    lines = []
    
    parent = brain_stats.get("parent")
    children = brain_stats.get("children", [])
    
    if not parent and not children:
        return lines
    
    lines.append("🔗 RELATIONSHIPS")
    lines.append("─" * 70)
    
    if parent:
        lines.append(f"Parent Brain:      {parent}")
    
    if children:
        lines.append(f"Child Brains:      {len(children)} total")
        for child in children[:5]:
            lines.append(f"  • {child}")
        if len(children) > 5:
            lines.append(f"  ... and {len(children) - 5} more")
    
    lines.append("")
    return lines


def build_brain_details_text(
    brain_name: str,
    store_dir: str,
    run_cli: Optional[Callable[[list[str]], str]] = None,
    parse_cli_dict: Optional[Callable[[str], dict[str, Any]]] = None,
    goals_list_callback: Optional[Callable[[str], list[str]]] = None,
    brain_stats_data: Optional[dict[str, Any]] = None
) -> str:
    """Build complete brain details text.
    
    Orchestrates all section builders to create comprehensive brain information.
    
    Args:
        brain_name: Name of the brain
        store_dir: Path to brains store directory
        run_cli: Optional callback to run CLI commands (deprecated path for stats)
        parse_cli_dict: Optional function to parse CLI dict output
        goals_list_callback: Optional callback to get goals list
        brain_stats_data: Optional pre-fetched stats dict to avoid spawning CLI
        
    Returns:
        Complete formatted details text
    """
    from .helpers import load_training_steps
    
    # Load brain metadata and stats
    brain_json_path = os.path.join(store_dir, "actv1", brain_name, "brain.json")
    brain_metadata = {}
    
    if os.path.exists(brain_json_path):
        try:
            with open(brain_json_path, 'r', encoding='utf-8') as f:
                brain_metadata = json.load(f)
        except Exception:
            pass
    
    # Get current brain stats from registry
    stats_data: dict[str, Any] = brain_stats_data or {}

    if not stats_data:
        if run_cli is not None:
            stats_out = run_cli(["brains", "stats", "--store-dir", store_dir])
            if parse_cli_dict is not None:
                stats_data = parse_cli_dict(stats_out)
            else:
                try:
                    stats_data = json.loads(stats_out)
                except Exception:
                    stats_data = {}
        else:
            stats_data = {}

    brain_stats: dict[str, Any] = {}
    if isinstance(stats_data, dict):
        direct_entry = stats_data.get(brain_name)
        if isinstance(direct_entry, dict):
            brain_stats = direct_entry
        else:
            brains_section = stats_data.get("brains")
            if isinstance(brains_section, dict):
                nested_entry = brains_section.get(brain_name)
                if isinstance(nested_entry, dict):
                    brain_stats = nested_entry
    
    # Load training steps (may need to read metrics.jsonl)
    brain_path = os.path.join(store_dir, "actv1", brain_name)
    training_steps = int(brain_stats.get("training_steps", 0) or brain_metadata.get("training_steps", 0) or 0)
    if training_steps <= 0:
        training_steps = load_training_steps(brain_path, brain_metadata)
    
    # Build all sections
    details = []
    details.append(f"╔{'═'*68}╗")
    details.append(f"║ {brain_name:^66} ║")
    details.append(f"╚{'═'*68}╝")
    details.append("")
    
    details.extend(build_model_info_section(brain_stats, brain_metadata))
    details.extend(build_config_section(brain_metadata))
    details.extend(build_training_section(brain_stats, brain_metadata, training_steps))
    details.extend(build_dataset_section(brain_stats, brain_metadata, training_steps))
    details.extend(build_goals_section(brain_metadata, goals_list_callback, brain_name))
    details.extend(build_relationships_section(brain_stats))
    
    return "\n".join(details)
