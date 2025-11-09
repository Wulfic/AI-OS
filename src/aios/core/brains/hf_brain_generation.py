"""HF brain text generation utilities - prompt building and response generation."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.hf_brain import HFHRMBrain


def build_prompt(brain: "HFHRMBrain", user_msg: str) -> str:
    """Build generation prompt from user message and conversation history.
    
    Args:
        brain: HFHRMBrain instance with history and system prompt
        user_msg: User's input message
        
    Returns:
        Complete prompt string ready for tokenization
    """
    m = user_msg.strip()
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
            "Joke: My dog thinks he's a developer—he keeps barking at bugs.\n\n"
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
    if isinstance(brain.system_prompt, str) and brain.system_prompt.strip():
        segments.append(brain.system_prompt.strip())
    try:
        hist = brain.history[-int(max(0, brain.history_max_turns)):] if brain.history_max_turns > 0 else []
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


def generate_response(brain: "HFHRMBrain", user_msg: str) -> Dict[str, Any]:
    """Generate text response using HF model.
    
    Args:
        brain: HFHRMBrain instance with loaded adapter
        user_msg: User's input message
        
    Returns:
        Response dict with ok/text or ok/error
    """
    try:
        import torch
        tok = getattr(brain._adapter, "tokenizer", None)
        model = getattr(brain._adapter, "model", None)
        if tok is None or model is None:
            return {"ok": False, "error": "adapter_incomplete"}
        
        prompt = build_prompt(brain, user_msg)
        
        # Ensure pad token id exists to avoid warnings
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # Tokenize without truncation, then left-truncate to ensure tail labels (e.g., 'Joke:') are present
        tokens = tok(prompt, return_tensors="pt", truncation=False)
        input_ids = tokens["input_ids"]
        # Use actual loaded context window, leaving room for generation
        max_input_len = max(16, brain._loaded_max_seq_len - brain.gen_max_new_tokens)
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
                max_new_tokens=int(brain.gen_max_new_tokens),
                min_length=input_ids.shape[1] + 10,  # Ensure at least 10 new tokens to prevent early stopping
                do_sample=True,
                temperature=float(brain.gen_temperature),
                top_p=float(brain.gen_top_p),
                top_k=int(brain.gen_top_k),
                repetition_penalty=float(brain.gen_repetition_penalty),
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
        if brain.system_prompt and text.startswith(brain.system_prompt[:50]):
            # Model is echoing the system prompt - skip past it
            for end_marker in ["\n\n", ". ", "! ", "? "]:
                idx = text.find(end_marker, len(brain.system_prompt) // 2)
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
        # Only truncate if explicitly set by user (not default 8192)
        if brain.max_response_chars < 8192 or len(text) > brain.max_response_chars:
            max_chars = max(256, min(brain.max_response_chars, brain.gen_max_new_tokens * 4))
            if len(text) > max_chars:
                # Truncate at word boundary
                text = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:!?")
                if not text:  # Safety fallback if rsplit fails
                    text = tok.decode(gen_ids[0], skip_special_tokens=True).strip()[:max_chars]
        
        # Update short history
        try:
            brain.history.append({"user": user_msg.strip(), "assistant": text})
            if brain.history_max_turns > 0 and len(brain.history) > int(brain.history_max_turns):
                # Keep last N
                brain.history = brain.history[-int(brain.history_max_turns):]
        except Exception:
            pass
        
        return {"ok": True, "text": text}
    except Exception as e:
        # Provide detailed dtype mismatch debugging
        import traceback
        error_msg = str(e)
        if "dtype" in error_msg.lower():
            try:
                model_ref = brain._adapter.model if hasattr(brain._adapter, 'model') else None
                model_dtype = next(model_ref.parameters()).dtype if model_ref else "unknown"
                error_msg = f"{e}\n[DEBUG] Model dtype: {model_dtype}. Try restarting the application to reload experts with correct dtype."
            except Exception:
                pass
        print(f"[ERROR] Generation failed: {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()[:500]}")
        return {"ok": False, "error": f"gen_error: {e}"}


def run_minimal_rollout(brain: "HFHRMBrain", payload: Any) -> Dict[str, Any]:
    """Run minimal HRM rollout for liveness check.
    
    Args:
        brain: HFHRMBrain instance with loaded adapter
        payload: Task payload
        
    Returns:
        Response dict with ok/loss or ok/error
    """
    try:
        import json
        import torch
        from aios.core.hrm_models.train_utils import segment_rollout  # type: ignore
        
        tok = getattr(brain._adapter, "tokenizer", None)
        if tok is None:
            return {"ok": False, "error": "tokenizer_missing"}
        text = json.dumps(payload or {"ping": 1})[:128]
        enc = tok([text], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inp = enc["input_ids"].to(brain._adapter.device)
        labels = inp.clone()
        if tok.pad_token_id is not None:
            labels[enc["attention_mask"] == 0] = -100
        batch = {
            "inputs": inp,
            "targets": labels,
            "puzzle_identifiers": torch.zeros((inp.shape[0],), dtype=torch.int64, device=brain._adapter.device),
        }
        loss, _ = segment_rollout(brain._adapter, batch, max_segments=1, epsilon=0.0)
        return {"ok": True, "loss": float(loss.detach().cpu().item())}
    except Exception as e:
        return {"ok": False, "error": str(e)}
