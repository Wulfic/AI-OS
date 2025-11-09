from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Set, TYPE_CHECKING
import hashlib
import time
import os
import json
from pathlib import Path


class Brain(Protocol):
    """Protocol for a sub-brain. Implement minimal run() and size() for storage checks."""

    name: str
    modalities: List[str]

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def size_bytes(self) -> int:
        ...


if TYPE_CHECKING:
    # For type checking only (avoid import cycles at runtime)
    from aios.core.train import Trainer, TrainConfig  # pragma: no cover - typing only


@dataclass
class NumpyMLPBrain:
    """Thin wrapper around aios.core.train.Trainer as a 'brain'.

    This provides a uniform interface and allows dynamic instantiation with different widths.
    """

    name: str
    modalities: List[str]
    cfg: "TrainConfig"
    _trainer: Optional["Trainer"] = None

    def _trainer_ready(self) -> "Trainer":
        if self._trainer is None:
            from aios.core.train import Trainer

            self._trainer = Trainer(self.cfg)
        return self._trainer

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        tr = self._trainer_ready()
        # Ensure minimal dims for our featurizer expectations
        if getattr(tr.cfg, "input_dim", 0) < 2:
            tr.cfg.input_dim = 2
            # rebuild to apply new input dim
            from aios.core.train import NumpyMLP  # type: ignore
            tr.model_np = NumpyMLP(tr.cfg.input_dim, tr.cfg.hidden, tr.cfg.output_dim)
        if getattr(tr.cfg, "output_dim", 0) < 1:
            tr.cfg.output_dim = 1
            from aios.core.train import NumpyMLP  # type: ignore
            tr.model_np = NumpyMLP(tr.cfg.input_dim, tr.cfg.hidden, tr.cfg.output_dim)
        # Minimal forward: do one training step on synthetic replay for now
        from aios.core.replay import ReplayBuffer
        rb = ReplayBuffer(capacity=max(64, int(self.cfg.batch_size) * 2))
        for i in range(16):
            rb.push([0], i % 3, float(i % 5), [0], False)
        loss = tr.train(rb, steps=min(5, int(self.cfg.max_steps)))
        return {"ok": True, "loss": float(loss)}

    def size_bytes(self) -> int:
        # Conservative FP32 param size estimate for the tiny MLP
        in_d = int(self.cfg.input_dim)
        out_d = int(self.cfg.output_dim)
        h = int(self.cfg.hidden)
        params = in_d * h + h + h * out_d + out_d
        return int(params * 4)


@dataclass
class HFHRMBrain:
    """HRM-style brain backed by a Hugging Face LM adapter (starter brain).

    Loads a starter_brain.json (q_head, optional LoRA adapter) and exposes a minimal
    Brain interface. Training here is represented by a tiny HRM-style rollout on a
    toy batch so callers and GUI remain unchanged.
    """

    name: str
    modalities: List[str]
    starter_config: str = "artifacts/hf_starter/starter_brain.json"
    max_seq_len: Optional[int] = None  # Explicit max_seq_len override (takes precedence over config file)
    inference_device: Optional[str] = None  # Specific device for inference (e.g., "cuda:1" for multi-GPU)
    # Sparse MoE configuration for efficient expert routing
    use_moe: bool = True  # Enable sparse Mixture of Experts architecture by default for 75% compute reduction
    num_experts: int = 8  # Total number of expert networks (default 8 for good specialization)
    num_experts_per_tok: int = 2  # Top-k experts activated per token (sparse activation)
    moe_capacity_factor: float = 1.25  # Expert capacity factor for load balancing
    # Generation controls and lightweight chat memory
    system_prompt: Optional[str] = None
    history_max_turns: int = 20  # Increased from 3 to support longer conversations
    history: List[Dict[str, str]] = field(default_factory=list)
    gen_max_new_tokens: int = 256  # Increased from 48 for longer responses
    gen_temperature: float = 0.7  # Slightly higher for more varied output
    gen_top_p: float = 0.9
    gen_top_k: int = 50
    gen_repetition_penalty: float = 1.2  # Higher to prevent repetition loops
    max_response_chars: int = 2048  # User-configurable max response length (min 256, max based on model context)
    _adapter: Any = None
    _loaded_max_seq_len: int = 0  # Track actual loaded context window

    def _ensure_loaded(self) -> None:
        if self._adapter is not None:
            return
        
        # Try loading from starter config with progressive context reduction
        model_path = None
        
        # Use explicit max_seq_len if provided, otherwise read from config
        if self.max_seq_len is not None:
            base_max_seq_len = max(1024, self.max_seq_len)
            print(f"[Brain] Using explicit context length: {base_max_seq_len} tokens")
        else:
            base_max_seq_len = 2048  # Default to 2k tokens (matches config default)
            
            # Read model path and max_seq_len from config
            try:
                with open(self.starter_config, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                    mp = str(data.get("model") or "").strip()
                    model_path = mp or None
                    # Also read max_seq_len from config if present
                    try:
                        config_max_seq = int(data.get("max_seq_len", 2048))
                        base_max_seq_len = max(1024, config_max_seq)  # Ensure at least 1k
                        print(f"[Brain] Using context length from config: {base_max_seq_len} tokens")
                    except Exception:
                        pass
            except Exception:
                model_path = None
        
        from aios.core.hrm_models import build_hf_adapter  # type: ignore
        
        # Progressive context window reduction starting from base_max_seq_len
        max_seq_attempts = []
        current = base_max_seq_len
        while current >= 1024:
            max_seq_attempts.append(current)
            current -= 1024  # Reduce by 1k per attempt
        
        # Ensure we have at least one attempt
        if not max_seq_attempts:
            max_seq_attempts = [2048, 1024]
        
        last_error = None
        
        for max_seq_len in max_seq_attempts:
            try:
                self._adapter = build_hf_adapter(
                    model_name_or_path=model_path or "gpt2",  # Use valid HF model as fallback
                    max_seq_len=max_seq_len,
                    halt_max_steps=1,
                    halt_exploration_prob=0.0,
                    device=self.inference_device,  # Use specific device if provided for multi-GPU support
                    forward_dtype="bfloat16",
                    use_lora=False,
                    # Sparse MoE configuration for efficient inference
                    use_moe=self.use_moe,
                    num_experts=self.num_experts if self.use_moe else 1,
                    num_experts_per_tok=self.num_experts_per_tok if self.use_moe else 1,
                    moe_capacity_factor=self.moe_capacity_factor if self.use_moe else 1.0,
                )
                self._adapter.train(True)
                # Success! Store the loaded context window and use 50% for generation (no hard cap)
                self._loaded_max_seq_len = max_seq_len
                # Use up to 50% of context window for generation, respecting max_response_chars setting
                default_gen_tokens = max_seq_len // 2
                # If max_response_chars is set (not default), use it to calculate token limit
                if self.max_response_chars != 2048:  # User has customized it
                    tokens_from_chars = self.max_response_chars // 4  # Estimate tokens from chars
                    self.gen_max_new_tokens = min(default_gen_tokens, max(256, tokens_from_chars))
                else:
                    # Use default 50% of context
                    self.gen_max_new_tokens = default_gen_tokens
                # Update max_response_chars based on actual token limit
                self.max_response_chars = self.gen_max_new_tokens * 4
                print(f"[Brain] Loaded with context window: {max_seq_len} tokens (max generation: {self.gen_max_new_tokens} tokens, ~{self.max_response_chars} chars)")
                return
            except Exception as e:
                last_error = e
                continue  # Try next smaller context size
        
        # All attempts failed - provide helpful guidance
        raise RuntimeError(
            f"Failed to load brain with any context size (tried {len(max_seq_attempts)} sizes from {base_max_seq_len} to 1024 tokens).\n"
            f"Last error: {last_error}\n"
            f"Suggestion: The model may not support the requested context length. Try reducing the max_seq_len in your brain configuration."
        )

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle either chat generation or a minimal HRM-style rollout for liveness.

        If payload contains a user message (e.g., {"user": str} or {"text": str}),
        generate a short reply using the HF model. Otherwise, run a tiny rollout to
        produce a loss metric (used by idle/self-improvement probes).
        """
        self._ensure_loaded()
        payload = task.get("payload") if isinstance(task, dict) else None
        # Chat mode: generate text when a user message is present
        try:
            if isinstance(payload, str):
                user_msg = payload
            elif isinstance(payload, dict):
                user_msg = payload.get("user") or payload.get("text")
            else:
                user_msg = None
        except Exception:
            user_msg = None

        if isinstance(user_msg, str) and user_msg.strip():
            try:
                import torch
                tok = getattr(self._adapter, "tokenizer", None)
                model = getattr(self._adapter, "model", None)
                if tok is None or model is None:
                    return {"ok": False, "error": "adapter_incomplete"}
                # Build an instruction-like prompt suited for base models (non-chat-tuned)
                def _build_prompt(msg: str) -> str:
                    m = msg.strip()
                    lower = m.lower()
                    # Extract topic after 'about' if present
                    def _extract_topic(text: str) -> str:
                        lo = text.lower()
                        if " about " in f" {lo} ":
                            start = lo.find(" about ") + len(" about ")
                            topic_orig = text[start:]
                            return topic_orig.strip().strip(".?! ")
                        return text.strip().strip(".?! ")

                    # Specialize for jokes to steer model to concise, topical one-liners using few-shot
                    if "joke" in lower:
                        topic = _extract_topic(m)
                        return (
                            "Write a one-line, family-friendly joke.\n\n"
                            "Example:\n"
                            "Topic: dogs\n"
                            "Joke: My dog thinks he’s a developer—he keeps barking at bugs.\n\n"
                            f"Topic: {topic}\n"
                            "Joke:"
                        )
                    # Specialize for short story asks
                    if "story" in lower:
                        topic = _extract_topic(m)
                        return (
                            "Write a very short, coherent story in 3–4 sentences.\n\n"
                            "Example:\n"
                            "Prompt: a brave mouse and a storm\n"
                            "Story: The wind howled, but the tiny mouse braced himself and pushed the door shut. \n"
                            "He found a crumb, shared it with a trembling sparrow, and the storm felt smaller. \n"
                            "By morning, the sky cleared; the mouse and sparrow watched the sunrise like old friends.\n\n"
                            f"Prompt: {topic}\n"
                            "Story:"
                        )
                    # Build prompt with system prompt and conversation history
                    segments: List[str] = []
                    if isinstance(self.system_prompt, str) and self.system_prompt.strip():
                        segments.append(self.system_prompt.strip())
                    try:
                        hist = self.history[-int(max(0, self.history_max_turns)):] if self.history_max_turns > 0 else []
                    except Exception:
                        hist = []
                    for turn in hist:
                        u = str(turn.get("user", "")).strip()
                        a = str(turn.get("assistant", "")).strip()
                        if u:
                            segments.append(u)
                        if a:
                            segments.append(a)
                    segments.append(m)
                    segments.append("")
                    return "\n\n".join(segments)

                prompt = _build_prompt(user_msg)
                # Ensure pad token id exists to avoid warnings
                pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                # Tokenize without truncation, then left-truncate to ensure tail labels (e.g., 'Joke:') are present
                tokens = tok(prompt, return_tensors="pt", truncation=False)
                input_ids = tokens["input_ids"]
                # Use actual loaded context window, leaving room for generation
                max_input_len = max(16, self._loaded_max_seq_len - self.gen_max_new_tokens)
                if input_ids.shape[1] > max_input_len:
                    input_ids = input_ids[:, -max_input_len:]
                
                # Get the device from the model directly (most reliable method)
                device = model.device
                
                # Move input_ids to the model's device
                input_ids = input_ids.to(device)
                
                # Create attention mask on the same device
                attn = torch.ones_like(input_ids)
                
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=int(self.gen_max_new_tokens),
                        min_length=input_ids.shape[1] + 10,  # Ensure at least 10 new tokens to prevent early stopping
                        do_sample=True,
                        temperature=float(self.gen_temperature),
                        top_p=float(self.gen_top_p),
                        top_k=int(self.gen_top_k),
                        repetition_penalty=float(self.gen_repetition_penalty),
                        no_repeat_ngram_size=3,  # Prevent 3-gram repetition loops
                        pad_token_id=pad_id,
                        eos_token_id=tok.eos_token_id,
                    )
                # Decode only the new tokens appended after the prompt
                new_tokens = gen_ids[0, input_ids.shape[1]:]
                raw = tok.decode(new_tokens, skip_special_tokens=True)
                
                # Clean up common issues
                text = raw.strip()
                
                # Remove URLs which models often hallucinate
                try:
                    import re as _re
                    text = _re.sub(r"https?://\S+", "", text)
                    text = _re.sub(r"\bwww\.[^\s]+", "", text)
                except Exception:
                    pass
                
                # Remove system prompt echo if present at start
                if self.system_prompt and text.startswith(self.system_prompt[:50]):
                    # Model is echoing the system prompt - skip past it
                    for end_marker in ["\n\n", ". ", "! ", "? "]:
                        idx = text.find(end_marker, len(self.system_prompt) // 2)
                        if idx != -1:
                            text = text[idx + len(end_marker):].strip()
                            break
                
                # Trim at conversation boundaries to prevent model from continuing the dialogue
                for boundary in ["\nUser:", "\nAssistant:", "<|endoftext|>"]:
                    if boundary in text:
                        text = text[:text.find(boundary)].strip()
                
                # Ensure we have something to return
                if not text:
                    text = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
                
                # Normalize whitespace
                text = " ".join(text.split())
                
                # Respect user's max_response_chars setting (min 256, max based on model)
                max_chars = max(256, min(self.max_response_chars, self.gen_max_new_tokens * 4))
                if len(text) > max_chars:
                    # Truncate at word boundary
                    text = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:!?")
                    if not text:  # Safety fallback if rsplit fails
                        text = tok.decode(gen_ids[0], skip_special_tokens=True).strip()[:max_chars]
                # Update short history
                try:
                    self.history.append({"user": user_msg.strip(), "assistant": text})
                    if self.history_max_turns > 0 and len(self.history) > int(self.history_max_turns):
                        # Keep last N
                        self.history = self.history[-int(self.history_max_turns):]
                except Exception:
                    pass
                return {"ok": True, "text": text}
            except Exception as e:
                # Fall back to loss path on generation failure
                import traceback
                error_msg = str(e)
                if "dtype" in error_msg.lower():
                    # Provide detailed dtype mismatch debugging
                    try:
                        model_ref = self._adapter.model if hasattr(self._adapter, 'model') else None
                        model_dtype = next(model_ref.parameters()).dtype if model_ref else "unknown"
                        error_msg = f"{e}\n[DEBUG] Model dtype: {model_dtype}. Try restarting the application to reload experts with correct dtype."
                    except Exception:
                        pass
                print(f"[ERROR] Generation failed: {error_msg}")
                print(f"[ERROR] Traceback: {traceback.format_exc()[:500]}")
                return {"ok": False, "error": f"gen_error: {e}"}

        # Non-chat path: run a minimal rollout to return a scalar loss
        try:
            import torch
            from aios.core.hrm_models.train_utils import segment_rollout  # type: ignore
            tok = getattr(self._adapter, "tokenizer", None)
            if tok is None:
                return {"ok": False, "error": "tokenizer_missing"}
            text = json.dumps(payload or {"ping": 1})[:128]
            enc = tok([text], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            inp = enc["input_ids"].to(self._adapter.device)
            labels = inp.clone()
            if tok.pad_token_id is not None:
                labels[enc["attention_mask"] == 0] = -100
            batch = {
                "inputs": inp,
                "targets": labels,
                "puzzle_identifiers": torch.zeros((inp.shape[0],), dtype=torch.int64, device=self._adapter.device),
            }
            loss, _ = segment_rollout(self._adapter, batch, max_segments=1, epsilon=0.0)
            return {"ok": True, "loss": float(loss.detach().cpu().item())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def size_bytes(self) -> int:
        # Estimate size by small adapter head + optional LoRA files if present.
        # We deliberately keep this modest so budgets permit creation by default.
        base = 512 * 1024  # ~0.5 MB for q_head
        try:
            # If starter dir has lora/, include its on-disk size
            from pathlib import Path
            p = Path(self.starter_config).parent / "lora"
            if p.exists():
                total = 0
                for root, _, files in os.walk(p):
                    for f in files:
                        total += os.path.getsize(os.path.join(root, f))
                return int(base + total)
        except Exception:
            pass
        return int(base)


@dataclass
class BrainRegistry:
    """Keeps track of sub-brains and supports dynamic creation under a global storage budget."""

    brains: Dict[str, Brain] = field(default_factory=dict)
    total_storage_limit_mb: Optional[float] = None
    storage_limit_mb_by_modality: Dict[str, float] = field(default_factory=dict)
    store_dir: Optional[str] = field(default_factory=lambda: "artifacts/brains")  # Default store directory
    # usage stats
    usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pinned: set[str] = field(default_factory=set)
    masters: set[str] = field(default_factory=set)
    parent: Dict[str, Optional[str]] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)

    def _used_bytes(self) -> int:
        return sum(max(0, int(b.size_bytes())) for b in self.brains.values())

    def _within_budget(self, add_bytes: int) -> bool:
        if self.total_storage_limit_mb is None or self.total_storage_limit_mb <= 0:
            return True
        limit = int(self.total_storage_limit_mb * 1024 * 1024)
        return (self._used_bytes() + max(0, int(add_bytes))) <= limit

    def _within_modality_budget(self, modalities: List[str], add_bytes: int) -> bool:
        if not self.storage_limit_mb_by_modality:
            return True
        add_b = max(0, int(add_bytes))
        # Use first modality as the key for caps (simple policy)
        key = str(modalities[0]).strip().lower() if modalities else None
        if not key or key not in self.storage_limit_mb_by_modality:
            return True
        cap_mb = float(self.storage_limit_mb_by_modality.get(key, 0) or 0)
        if cap_mb <= 0:
            return True
        cap_bytes = int(cap_mb * 1024 * 1024)
        # sum bytes of brains matching this modality
        used = 0
        for n, b in self.brains.items():
            mods = (self.usage.get(n, {}).get("modalities") or [])
            if mods and str(mods[0]).strip().lower() == key:
                used += max(0, int(b.size_bytes()))
        return (used + add_b) <= cap_bytes

    def get(self, name: str) -> Optional[Brain]:
        b = self.brains.get(name)
        if b is not None:
            return b
        return self._try_load_offloaded(name)

    def list(self) -> List[str]:
        """List all brains, including loaded brains and offloaded ACTV1 brain bundles."""
        names = set(self.brains.keys())
        # Also include pinned and master brains even if not loaded
        names.update(self.pinned)
        names.update(self.masters)
        # Scan for ACTV1 brain bundles on disk
        try:
            if self.store_dir:
                actv1_base = os.path.join(self.store_dir, "actv1")
                if os.path.isdir(actv1_base):
                    for entry in os.listdir(actv1_base):
                        entry_path = os.path.join(actv1_base, entry)
                        if os.path.isdir(entry_path):
                            # Check if it looks like a brain bundle (has brain.json or actv1_student.safetensors)
                            has_brain = os.path.exists(os.path.join(entry_path, "brain.json"))
                            has_model = os.path.exists(os.path.join(entry_path, "actv1_student.safetensors"))
                            if has_brain or has_model:
                                names.add(entry)
        except Exception:
            pass
        return sorted(list(names))

    def record_use(self, name: str, modalities: Optional[List[str]] = None) -> None:
        now = time.time()
        meta = self.usage.get(name) or {}
        meta.setdefault("created_at", now)
        meta["last_used"] = now
        meta["hits"] = int(meta.get("hits", 0)) + 1
        if modalities is not None:
            meta.setdefault("modalities", list(modalities))
        self.usage[name] = meta

    def record_training_steps(self, name: str, steps: int) -> None:
        """Record training steps completed for a brain model.
        
        Args:
            name: Brain name
            steps: Number of training steps completed in this session
        """
        meta = self.usage.get(name) or {}
        current_steps = int(meta.get("training_steps", 0))
        meta["training_steps"] = current_steps + int(steps)
        meta["last_trained"] = time.time()
        self.usage[name] = meta

    def stats(self) -> Dict[str, Any]:
        # Helper to peek offloaded checkpoint size when not loaded
        def _offloaded_size(name: str) -> int:
            try:
                if not self.store_dir:
                    return 0
                import os as _os
                npz, _meta = self._store_paths(name)
                return int(_os.path.getsize(npz)) if _os.path.exists(npz) else 0
            except Exception:
                return 0

        # Helper to discover ACTV1 bundles and approximate their size from on-disk files
        def _actv1_dir_entries() -> Dict[str, int]:
            out: Dict[str, int] = {}
            try:
                if not self.store_dir:
                    return out
                base = os.path.join(self.store_dir, "actv1")
                if not os.path.isdir(base):
                    return out
                for entry in sorted(os.listdir(base)):
                    try:
                        p = os.path.join(base, entry)
                        if not os.path.isdir(p):
                            continue
                        # Prefer size of actv1_student.safetensors; fall back to total dir size
                        pt = os.path.join(p, "actv1_student.safetensors")
                        if os.path.exists(pt):
                            sz = int(os.path.getsize(pt))
                        else:
                            total = 0
                            for r, _d, files in os.walk(p):
                                for f in files:
                                    try:
                                        total += int(os.path.getsize(os.path.join(r, f)))
                                    except Exception:
                                        continue
                            sz = int(total)
                        out[str(entry)] = max(0, int(sz))
                    except Exception:
                        continue
            except Exception:
                return out
            return out

        # Start with loaded brains
        entries: Dict[str, Any] = {}
        for n, b in self.brains.items():
            try:
                sz = int(b.size_bytes())
            except Exception:
                sz = 0
            entries[n] = {
                **self.usage.get(n, {}),
                "size_bytes": sz,
                "pinned": n in self.pinned,
                "master": n in self.masters,
                "parent": self.parent.get(n),
                "children": sorted(self.children.get(n, [])),
            }
        # Ensure pinned/masters also appear even if not loaded
        def _ensure(name: str) -> None:
            if name not in entries:
                peek = _offloaded_size(name)
                entries[name] = {
                    **self.usage.get(name, {}),
                    "size_bytes": int(peek),
                    "pinned": name in self.pinned,
                    "master": name in self.masters,
                    "parent": self.parent.get(name),
                    "children": sorted(self.children.get(name, [])),
                }
        for n in sorted(self.pinned | self.masters):
            _ensure(n)

        # Also surface ACTV1 brain bundles found on disk under store_dir/actv1
        try:
            actv1_sizes = _actv1_dir_entries()
            for n, sz in actv1_sizes.items():
                if n not in entries:
                    # Load training_steps and dataset info from brain.json if available
                    meta_from_disk = {}
                    try:
                        if self.store_dir:
                            brain_json = os.path.join(self.store_dir, "actv1", n, "brain.json")
                            if os.path.exists(brain_json):
                                with open(brain_json, "r", encoding="utf-8") as f:
                                    disk_data = json.load(f) or {}
                                    # Extract relevant training metadata
                                    if "training_steps" in disk_data:
                                        meta_from_disk["training_steps"] = int(disk_data.get("training_steps", 0))
                                    if "last_trained" in disk_data:
                                        meta_from_disk["last_trained"] = float(disk_data.get("last_trained", 0))
                                    # Extract dataset tracking information
                                    if "dataset_stats" in disk_data:
                                        meta_from_disk["dataset_stats"] = disk_data.get("dataset_stats", {})
                                    if "dataset_history" in disk_data:
                                        # Only include recent history (last 20 sessions) to avoid bloat
                                        history = disk_data.get("dataset_history", [])
                                        meta_from_disk["dataset_history"] = history[-20:] if isinstance(history, list) else []
                    except Exception:
                        pass
                    entries[n] = {
                        **self.usage.get(n, {}),
                        **meta_from_disk,  # Override with disk metadata if present
                        "size_bytes": int(sz),
                        "pinned": n in self.pinned,
                        "master": n in self.masters,
                        "parent": self.parent.get(n),
                        "children": sorted(self.children.get(n, [])),
                    }
                else:
                    # If present but size unknown, fill it in
                    try:
                        cur = int(entries[n].get("size_bytes", 0) or 0)
                        if cur <= 0 and sz > 0:
                            entries[n]["size_bytes"] = int(sz)
                        # Also load training_steps and dataset info from disk if not already present
                        if "training_steps" not in entries[n] or entries[n].get("training_steps", 0) == 0:
                            try:
                                if self.store_dir:
                                    brain_json = os.path.join(self.store_dir, "actv1", n, "brain.json")
                                    if os.path.exists(brain_json):
                                        with open(brain_json, "r", encoding="utf-8") as f:
                                            disk_data = json.load(f) or {}
                                            if "training_steps" in disk_data:
                                                entries[n]["training_steps"] = int(disk_data.get("training_steps", 0))
                                            if "last_trained" in disk_data and "last_trained" not in entries[n]:
                                                entries[n]["last_trained"] = float(disk_data.get("last_trained", 0))
                                            # Load dataset tracking information
                                            if "dataset_stats" in disk_data and "dataset_stats" not in entries[n]:
                                                entries[n]["dataset_stats"] = disk_data.get("dataset_stats", {})
                                            if "dataset_history" in disk_data and "dataset_history" not in entries[n]:
                                                history = disk_data.get("dataset_history", [])
                                                entries[n]["dataset_history"] = history[-20:] if isinstance(history, list) else []
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass
        # Compute total used bytes from entries, falling back to live sum
        try:
            total_used = sum(int(v.get("size_bytes", 0) or 0) for v in entries.values())
        except Exception:
            total_used = int(self._used_bytes())
        return {"used_bytes": int(total_used), "brains": entries}

    def create_numpy_mlp(self, name: str, modalities: List[str], cfg_overrides: Optional[Dict[str, Any]] = None) -> Optional[Brain]:
        from aios.core.train import TrainConfig

        cfg = TrainConfig()
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        brain = NumpyMLPBrain(name=name, modalities=list(modalities), cfg=cfg)
        if not self._within_budget(brain.size_bytes()):
            return None
        if not self._within_modality_budget(modalities, brain.size_bytes()):
            return None
        self.brains[name] = brain
        self.record_use(name, modalities)
        return brain

    def create_hf_hrm(
        self, 
        name: str, 
        modalities: List[str], 
        starter_config: Optional[str] = None, 
        max_seq_len: Optional[int] = None,
        inference_device: Optional[str] = None,
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_capacity_factor: float = 1.25,
    ) -> Optional[Brain]:
        brain = HFHRMBrain(
            name=name, 
            modalities=list(modalities), 
            starter_config=starter_config or HFHRMBrain.starter_config,
            max_seq_len=max_seq_len,
            inference_device=inference_device,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_capacity_factor=moe_capacity_factor,
        )
        # Respect storage budgets
        if not self._within_budget(brain.size_bytes()):
            return None
        if not self._within_modality_budget(modalities, brain.size_bytes()):
            return None
        self.brains[name] = brain
        self.record_use(name, modalities)
        # Establish parent/child relations if a master exists for this modality
        try:
            # Choose a canonical parent: the first master for the same modality prefix
            parent_name = next((m for m in sorted(self.masters) if (self.usage.get(m, {}).get("modalities") or []) == list(modalities)), None)
            self.parent[name] = parent_name
            if parent_name:
                self.children.setdefault(parent_name, [])
                if name not in self.children[parent_name]:
                    self.children[parent_name].append(name)
        except Exception:
            pass
        return brain

    # --- Offload/Restore support ---
    def _store_paths(self, name: str) -> Tuple[str, str]:
        if not self.store_dir:
            raise RuntimeError("store_dir not configured")
        d = os.path.abspath(self.store_dir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{name}.npz"), os.path.join(d, f"{name}.json")

    def offload(self, name: str) -> bool:
        b = self.brains.get(name)
        if b is None or not isinstance(b, NumpyMLPBrain):
            return False
        try:
            npz, meta_path = self._store_paths(name)
            # Save checkpoint
            tr = b._trainer_ready()
            tr.save_checkpoint(npz, {"name": name})
            # Save meta
            meta = {
                "name": name,
                "modalities": list(b.modalities),
                "cfg": {k: getattr(b.cfg, k) for k in ("input_dim", "hidden", "output_dim", "dynamic_width", "width_storage_limit_mb") if hasattr(b.cfg, k)},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            # Remove from memory after successful save
            del self.brains[name]
            return True
        except Exception:
            return False

    # --- Pinning helpers (persisted in store_dir if set) ---
    def _pins_path(self) -> Optional[str]:
        if not self.store_dir:
            return None
        d = os.path.abspath(self.store_dir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "pinned.json")

    def _masters_path(self) -> Optional[str]:
        if not self.store_dir:
            return None
        d = os.path.abspath(self.store_dir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "masters.json")

    def load_pinned(self) -> bool:
        p = self._pins_path()
        if not p or not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = set(str(n) for n in (data or []))
            self.pinned = names
            return True
        except Exception:
            return False

    def load_masters(self) -> bool:
        p = self._masters_path()
        if not p or not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = set(str(n) for n in (data or []))
            self.masters = names
            # Ensure masters are pinned as well
            self.pinned |= self.masters
            return True
        except Exception:
            return False

    def save_pinned(self) -> bool:
        p = self._pins_path()
        if not p:
            return False
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(sorted(self.pinned), f)
            return True
        except Exception:
            return False

    def save_masters(self) -> bool:
        p = self._masters_path()
        if not p:
            return False
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(sorted(self.masters), f)
            return True
        except Exception:
            return False

    def pin(self, name: str) -> None:
        self.pinned.add(name)
        self.save_pinned()

    def unpin(self, name: str) -> None:
        # Do not allow unpinning a master brain
        if name in self.masters:
            return
        if name in self.pinned:
            self.pinned.remove(name)
            self.save_pinned()

    def mark_master(self, name: str) -> None:
        self.masters.add(name)
        # Masters are always pinned
        self.pinned.add(name)
        self.save_masters()
        self.save_pinned()

    def _try_load_offloaded(self, name: str) -> Optional[Brain]:
        if not self.store_dir:
            return None
        try:
            npz, meta_path = self._store_paths(name)
            if not (os.path.exists(npz) and os.path.exists(meta_path)):
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            modalities = list(meta.get("modalities") or [])
            cfg_dict = dict(meta.get("cfg") or {})
            from aios.core.train import TrainConfig
            cfg = TrainConfig()
            for k, v in cfg_dict.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            brain = NumpyMLPBrain(name=name, modalities=modalities or ["text"], cfg=cfg)
            tr = brain._trainer_ready()
            tr.load_checkpoint(npz)
            self.brains[name] = brain
            self.record_use(name, modalities)
            return brain
        except Exception:
            return None

    

    def prune(self, target_mb: Optional[float] = None, offload: bool = False) -> List[str]:
        """Evict brains (LRU) until within target_mb or global limit. Returns evicted names.

        If offload=True and store_dir is configured, checkpoints brains before eviction."""
        evicted: List[str] = []
        def over_limit() -> bool:
            if target_mb is not None and target_mb > 0:
                return self._used_bytes() > int(target_mb * 1024 * 1024)
            return not self._within_budget(0)
        if not over_limit():
            return evicted
        # Sort by last_used ascending (oldest first)
        order = sorted(self.brains.keys(), key=lambda n: float(self.usage.get(n, {}).get("last_used", 0)))
        for n in order:
            if not over_limit():
                break
            # Skip pinned brains
            if n in self.pinned:
                continue
            try:
                if offload and self.store_dir:
                    self.offload(n)
                else:
                    del self.brains[n]
                evicted.append(n)
            except Exception:
                pass
        return evicted

    # --- Management helpers: rename, master/parent relations ---
    def rename(self, old: str, new: str) -> bool:
        """Rename a brain in memory and on disk (offloaded files) when present.

        Returns True on success. If the name is not found, returns False. If 'new'
        already exists, returns False to avoid accidental overwrite.
        """
        old = str(old)
        new = str(new)
        if old == new:
            return True
        if new in self.brains:
            return False
        b = self.brains.pop(old, None)
        if b is not None:
            self.brains[new] = b
        # move usage/meta
        if old in self.usage:
            self.usage[new] = self.usage.pop(old)
        # pin/master sets
        if old in self.pinned:
            self.pinned.remove(old)
            self.pinned.add(new)
            self.save_pinned()
        if old in self.masters:
            self.masters.remove(old)
            self.masters.add(new)
            self.save_masters()
        # parent/children maps
        if old in self.parent:
            self.parent[new] = self.parent.pop(old)
        for p, kids in list(self.children.items()):
            if old in kids:
                kids = [new if k == old else k for k in kids]
                self.children[p] = kids
        if old in self.children:
            self.children[new] = self.children.pop(old)
        # rename offloaded files (best-effort)
        try:
            if self.store_dir:
                from os import path as _p, rename as _mv
                onpz, ojson = self._store_paths(old)
                nnpz, njson = self._store_paths(new)
                # Only rename if source exists and dest doesn't
                if _p.exists(onpz) and not _p.exists(nnpz):
                    _mv(onpz, nnpz)
                if _p.exists(ojson) and not _p.exists(njson):
                    _mv(ojson, njson)
        except Exception:
            pass
        return True

    def unmark_master(self, name: str) -> None:
        """Remove master flag (does not unpin)."""
        if name in self.masters:
            self.masters.remove(name)
            self.save_masters()

    def set_parent(self, child: str, parent: Optional[str]) -> None:
        """Set or clear the parent of a brain; updates children map accordingly."""
        c = str(child)
        p = str(parent) if parent else None
        # Remove from existing parent's children list
        try:
            cur = self.parent.get(c)
            if cur and cur in self.children:
                self.children[cur] = [k for k in self.children[cur] if k != c]
        except Exception:
            pass
        # Set new parent
        self.parent[c] = p
        if p:
            self.children.setdefault(p, [])
            if c not in self.children[p]:
                self.children[p].append(c)

    def clear_parent(self, child: str) -> None:
        self.set_parent(child, None)


@dataclass
class Router:
    """Simple router that picks sub-brains by modality and task hash. Creates on-demand."""

    registry: BrainRegistry
    default_modalities: List[str] = field(default_factory=lambda: ["text"])
    brain_prefix: str = "brain"
    create_cfg: Dict[str, Any] = field(default_factory=dict)
    modality_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategy: str = "hash"  # 'hash' | 'round_robin'
    expert_registry_path: Optional[str] = None
    _rr_idx: Dict[str, int] = field(default_factory=dict)
    expert_registry: Optional[Any] = field(default=None, init=False, repr=False)
    _active_goal_ids: Set[str] = field(default_factory=set, init=False, repr=False)
    _loaded_expert_ids: Set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self):
        """Load expert registry if path provided."""
        if self.expert_registry_path:
            try:
                registry_path = Path(self.expert_registry_path)
                if registry_path.exists():
                    # Lazy import to avoid circular dependencies
                    from aios.core.hrm_models.expert_metadata import ExpertRegistry
                    self.expert_registry = ExpertRegistry.load(str(registry_path))
                else:
                    # Create empty registry
                    from aios.core.hrm_models.expert_metadata import ExpertRegistry
                    self.expert_registry = ExpertRegistry()
                    # Ensure directory exists
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    self.expert_registry.save(str(registry_path))
            except Exception:
                # Log but don't crash - expert registry is optional
                from aios.core.hrm_models.expert_metadata import ExpertRegistry
                self.expert_registry = ExpertRegistry()

    def _brain_name_for(self, modalities: Iterable[str], payload: Any) -> str:
        # Stable but cheap name based on modalities and a hash of a payload sketch
        mods = ",".join(sorted({str(m).strip().lower() for m in modalities if str(m).strip()})) or ",".join(self.default_modalities)
        h = hashlib.sha1(str(type(payload)).encode("utf-8")).hexdigest()[:8]
        return f"{self.brain_prefix}-{mods}-{h}"

    def _select_existing_rr(self, modalities: List[str]) -> Optional[str]:
        mods_key = ",".join(sorted({str(m).strip().lower() for m in modalities if str(m).strip()})) or ",".join(self.default_modalities)
        names = [n for n in self.registry.list() if (self.registry.usage.get(n, {}).get("modalities") or []) == list(modalities)]
        if not names:
            return None
        i = int(self._rr_idx.get(mods_key, 0)) % len(names)
        self._rr_idx[mods_key] = i + 1
        return names[i]

    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        modalities = task.get("modalities") or self.default_modalities
        payload = task.get("payload")
        
        # Check if there's a master brain for these modalities - use it directly if available
        master_name = None
        if self.registry.masters:
            # Find first master matching these modalities
            for m in sorted(self.registry.masters):
                m_mods = self.registry.usage.get(m, {}).get("modalities", [])
                if m_mods == list(modalities):
                    master_name = m
                    break
        
        # If we have a master, try to load it directly
        if master_name:
            brain = self.registry.get(master_name)
            if brain is None:
                # Try to load master from disk (ACTV1 bundle)
                try:
                    if self.registry.store_dir:
                        brain_json_path = os.path.join(self.registry.store_dir, "actv1", master_name, "brain.json")
                        if os.path.exists(brain_json_path):
                            # Read brain.json and convert to starter config format
                            import json
                            with open(brain_json_path, "r", encoding="utf-8") as f:
                                brain_data = json.load(f)
                            
                            # Get max_seq_len from brain.json (this is the trained context length)
                            trained_max_seq_len = brain_data.get("max_seq_len", 2048)
                            
                            # Create a temporary starter config with proper format
                            starter_cfg = {
                                "model": brain_data.get("tokenizer_model", "artifacts/hf_implant/base_model"),
                                "q_head": os.path.join(self.registry.store_dir, "actv1", master_name, "actv1_student.safetensors"),
                                "max_seq_len": trained_max_seq_len,  # Use brain's trained context length
                                "halt_max_steps": brain_data.get("halt_max_steps", 1),
                                "device": None,  # auto-detect
                                "forward_dtype": "bfloat16",
                                "peft_dir": None
                            }
                            
                            # Write temporary starter config
                            temp_config_path = os.path.join(self.registry.store_dir, "actv1", master_name, "starter_brain.json")
                            with open(temp_config_path, "w", encoding="utf-8") as f:
                                json.dump(starter_cfg, f, indent=2)
                            
                            # Pass max_seq_len explicitly to ensure it uses the brain's configured context window
                            brain = self.registry.create_hf_hrm(
                                name=master_name,
                                modalities=list(modalities),
                                starter_config=temp_config_path,
                                max_seq_len=trained_max_seq_len,  # Use brain's trained context length
                            )
                except Exception:
                    pass
            if brain is not None:
                try:
                    res = brain.run(task)
                    self.registry.record_use(master_name, list(modalities))
                    return res
                except Exception as e:
                    return {"ok": False, "error": str(e)}
        
        # Fallback to original behavior: hash-based or round-robin brain selection
        if self.strategy == "round_robin":
            name = self._select_existing_rr(list(modalities)) or self._brain_name_for(modalities, payload)
        else:
            name = self._brain_name_for(modalities, payload)
        brain = self.registry.get(name)
        if brain is None:
            # merge overrides: global create_cfg + first-modality override (if any)
            ov = dict(self.create_cfg or {})
            first_mod = str(list(modalities)[0]).strip().lower() if modalities else None
            if first_mod and first_mod in self.modality_overrides:
                for k, v in (self.modality_overrides.get(first_mod) or {}).items():
                    ov[k] = v
            kind = str(ov.get("kind", "numpy")).strip().lower()
            if kind == "hf":
                # Extract MoE configuration from overrides
                moe_cfg = dict(ov.get("moe") or {})
                # Extract inference_device from overrides (set by GUI from resources panel)
                inference_device = ov.get("inference_device")
                
                # Extract max_seq_len from config, but check brain.json if starter_config points to one
                max_seq_len = ov.get("max_seq_len")  # Config value
                starter_config_path = str(ov.get("starter_config") or "artifacts/hf_starter/starter_brain.json")
                
                # If starter_config points to a brain.json, read the trained max_seq_len from it
                if starter_config_path.endswith("brain.json") and os.path.exists(starter_config_path):
                    try:
                        import json
                        with open(starter_config_path, "r", encoding="utf-8") as f:
                            brain_data = json.load(f)
                        # Use brain's trained context length instead of config override
                        brain_max_seq = brain_data.get("max_seq_len")
                        if brain_max_seq is not None:
                            max_seq_len = brain_max_seq
                            print(f"[Brain] Using trained context length from {starter_config_path}: {max_seq_len} tokens")
                    except Exception as e:
                        print(f"[Brain] Warning: Could not read max_seq_len from {starter_config_path}: {e}")
                
                if max_seq_len is not None:
                    try:
                        max_seq_len = int(max_seq_len)
                    except (ValueError, TypeError):
                        max_seq_len = None
                
                brain = self.registry.create_hf_hrm(
                    name=name,
                    modalities=list(modalities),
                    starter_config=starter_config_path,
                    max_seq_len=max_seq_len,
                    inference_device=inference_device,
                    use_moe=bool(moe_cfg.get("enabled", True)),
                    num_experts=int(moe_cfg.get("num_experts", 8)),
                    num_experts_per_tok=int(moe_cfg.get("experts_per_tok", 2)),
                    moe_capacity_factor=float(moe_cfg.get("capacity_factor", 1.25)),
                )
                # Pin the primary text HF brain as a master to prevent eviction
                # ONLY if it's not a temporary router-generated brain (pattern: brain-text-<8-char-hash>)
                try:
                    if brain is not None and list(modalities) == ["text"]:
                        # Check if this is a temporary brain by pattern matching
                        import re
                        is_temporary = bool(re.match(r'^brain-[a-z]+-[0-9a-f]{8}$', name))
                        if not is_temporary:
                            # Mark master (implies pin) and persist
                            self.registry.mark_master(name)
                            # Mark usage meta for visibility
                            meta = self.registry.usage.get(name, {})
                            meta["master"] = True
                            self.registry.usage[name] = meta
                except Exception:
                    pass
                # Apply generation/system prompt overrides if provided
                try:
                    if brain is not None and isinstance(brain, HFHRMBrain):
                        # Ensure brain is loaded first to calculate proper defaults
                        brain._ensure_loaded()
                        
                        gen = dict(ov.get("generation") or {})
                        # Only apply max_new_tokens if explicitly set and not 0 (0 means auto)
                        if "max_new_tokens" in gen:
                            tokens = int(gen["max_new_tokens"])
                            if tokens > 0:  # 0 means use auto-calculated value
                                brain.gen_max_new_tokens = tokens
                        if "temperature" in gen:
                            brain.gen_temperature = float(gen["temperature"])
                        if "top_p" in gen:
                            brain.gen_top_p = float(gen["top_p"])
                        if "top_k" in gen:
                            brain.gen_top_k = int(gen["top_k"])
                        if "repetition_penalty" in gen:
                            brain.gen_repetition_penalty = float(gen["repetition_penalty"])
                        if "history_max_turns" in gen:
                            brain.history_max_turns = int(gen["history_max_turns"])
                        # Handle max_response_chars from generation config or top-level
                        max_resp_chars = None
                        if "max_response_chars" in gen:
                            max_resp_chars = int(gen["max_response_chars"])
                        elif "max_response_chars" in ov:
                            max_resp_chars = int(ov["max_response_chars"])
                        
                        if max_resp_chars is not None:
                            if max_resp_chars <= 0:  # 0 or negative means auto (use calculated)
                                # Keep the auto-calculated value from _ensure_loaded
                                pass
                            else:
                                # User set specific limit - respect it but ensure min 256
                                tokens_from_chars = max_resp_chars // 4
                                # Cap at model's generation capacity
                                brain.gen_max_new_tokens = min(brain.gen_max_new_tokens, max(256, tokens_from_chars))
                                brain.max_response_chars = brain.gen_max_new_tokens * 4
                        
                        # Also accept top-level values in overrides
                        if "system_prompt" in ov:
                            brain.system_prompt = str(ov.get("system_prompt")) if ov.get("system_prompt") is not None else None
                        if "history_max_turns" in ov:
                            v = ov.get("history_max_turns")
                            if v is not None:
                                brain.history_max_turns = int(v)
                except Exception:
                    pass
            else:
                brain = self.registry.create_numpy_mlp(name=name, modalities=list(modalities), cfg_overrides=ov)
            if brain is None:
                return {"ok": False, "error": "storage_budget_exceeded", "modalities": modalities}
        try:
            res = brain.run(task)
            # update usage on success
            self.registry.record_use(name, list(modalities))
            return res
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def update_active_goals(self, goal_ids: List[str]) -> Dict[str, Any]:
        """Update active goals and manage expert loading/unloading.
        
        Args:
            goal_ids: List of active goal IDs (as strings)
        
        Returns:
            Dict with:
                - newly_activated: List of goal IDs that became active
                - newly_deactivated: List of goal IDs that were deactivated
                - linked_experts: List of expert IDs linked to active goals
                - loaded_experts: List of currently loaded expert IDs
        """
        goal_ids_set = set(goal_ids)
        
        # Find changes
        newly_activated = goal_ids_set - self._active_goal_ids
        newly_deactivated = self._active_goal_ids - goal_ids_set
        
        # Update active goals
        self._active_goal_ids = goal_ids_set
        
        # Get experts linked to all active goals
        linked_experts = self.get_experts_for_goals(list(goal_ids_set))
        
        # Update loaded expert IDs (placeholder for Task 12 when experts are actually loaded)
        # For now, just track what SHOULD be loaded based on goals
        self._loaded_expert_ids = set(linked_experts)
        
        return {
            "newly_activated": list(newly_activated),
            "newly_deactivated": list(newly_deactivated),
            "linked_experts": linked_experts,
            "loaded_experts": list(self._loaded_expert_ids),
        }

    def get_loaded_experts(self) -> List[str]:
        """Get list of currently loaded expert IDs.
        
        Returns:
            List of expert IDs that are currently loaded in memory
        """
        return list(self._loaded_expert_ids)

    def get_experts_for_goals(self, goal_ids: List[str]) -> List[str]:
        """Get expert IDs linked to given goals.
        
        Args:
            goal_ids: List of goal IDs
        
        Returns:
            List of unique expert IDs linked to any of the given goals
        """
        if not self.expert_registry:
            return []
        
        expert_ids = set()
        for goal_id in goal_ids:
            # Get experts linked to this goal
            experts = self.expert_registry.get_experts_by_goal(goal_id)
            for expert in experts:
                expert_ids.add(expert.expert_id)
        
        return list(expert_ids)

    def get_active_goals(self) -> List[str]:
        """Get currently active goal IDs.
        
        Returns:
            List of active goal IDs
        """
        return list(self._active_goal_ids)
