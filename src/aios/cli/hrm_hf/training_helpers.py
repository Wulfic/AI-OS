from __future__ import annotations

from typing import Optional, Dict, Any

import json
from pathlib import Path


def write_last_safe_batches(train_bs: Optional[int] = None) -> None:
    """Persist last safe batch size to help GUI pre-fill known-good defaults.

    Writes to artifacts/brains/actv1/last_safe.json. Missing directories are created.
    Silently ignores IO errors to avoid breaking training.
    """
    # Always write last_safe.json
    try:
        base_dir = Path("artifacts/brains/actv1")
        base_dir.mkdir(parents=True, exist_ok=True)
        p = base_dir / "last_safe.json"
        data: dict = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
        if train_bs is not None:
            data["train_batch"] = int(max(1, int(train_bs)))
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def write_jsonl(
    *,
    log_file: Optional[str],
    payload: Dict[str, Any],
    is_distributed: bool = False,
    rank_id: int = 0,
) -> None:
    """Append a JSON object to a JSONL log file.

    - Adds a 'ts' field if missing.
    - When distributed, writes only from rank 0 if torch.distributed is initialized.
    """
    if not log_file:
        return
    
    # Skip verbose logging during memory tests to reduce file I/O interrupts
    import os
    if os.environ.get("AIOS_MINIMAL_LOGGING") == "1":
        # Only log critical events during testing
        if not any(key in payload for key in ["finalization", "checkpoint_save", "training_start", "training_complete"]):
            return
    
    # If running in a distributed context, only allow rank 0 to write logs.
    # This applies even when dist is not initialized (e.g., init failure) to avoid duplicate logs.
    if is_distributed and int(rank_id) != 0:
        return
    
    # Retry logic for file operations (handles interrupts and locking)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            p = Path(str(log_file))
            p.parent.mkdir(parents=True, exist_ok=True)
            if "ts" not in payload:
                import time as _t
                payload = {**payload, "ts": float(_t.time())}
            # Use unbuffered write to avoid interrupts during flush
            with p.open("a", encoding="utf-8", buffering=1) as f:
                f.write(json.dumps(payload) + "\n")
            break  # Success, exit retry loop
        except (KeyboardInterrupt, SystemExit):
            # Don't retry on deliberate interrupts, just silently fail
            break
        except Exception:
            if attempt < max_retries - 1:
                import time as _t
                _t.sleep(0.1)  # Brief delay before retry
            pass


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def should_stop(stop_file: Optional[str]) -> bool:
    if not stop_file or not isinstance(stop_file, str):
        return False
    try:
        return Path(stop_file).exists()
    except Exception:
        return False



def eval_once(
    *,
    model_student,
    eval_ids,
    eval_labels,
    batch_size: int,
    device_obj,
    dml_device,
    halt_max_steps: int,
    eval_batches: int,
    segment_rollout,
    write_jsonl,
    tokenizer=None,
    enable_english_logic_eval: bool = True,
) -> None:
    """Run a quick evaluation loop and log metrics via write_jsonl.

    Mirrors the inline implementation in train_actv1_impl; accepts dependencies as parameters
    to avoid circular imports.
    
    Args:
        model_student: The student model to evaluate
        eval_ids: Evaluation input IDs tensor
        eval_labels: Evaluation labels tensor
        batch_size: Batch size for evaluation
        device_obj: Device to run evaluation on
        dml_device: DirectML device (if applicable)
        halt_max_steps: Maximum halting steps
        eval_batches: Number of evaluation batches
        segment_rollout: Function for segment rollout
        write_jsonl: Function to write metrics
        tokenizer: Optional tokenizer for English logic evaluation (generates samples)
        enable_english_logic_eval: Whether to run English logic compliance tests
    """
    if eval_ids is None or eval_labels is None:
        return
    try:
        import torch
        import torch.nn.functional as F
        model_student.train(False)
        total_ce = 0.0
        total_tok = 0
        total_correct = 0
        total_exact = 0
        EN = eval_ids.shape[0]
        batches = 0
        while batches < int(max(1, eval_batches)):
            idx = torch.randint(0, EN, (int(batch_size),))
            einp = eval_ids.index_select(0, idx)
            elbl = eval_labels.index_select(0, idx)
            if dml_device is not None:
                einp = einp.to(dml_device)
                elbl = elbl.to(dml_device)
            else:
                einp = einp.to(device_obj)
                elbl = elbl.to(device_obj)
            ebatch = {
                "inputs": einp,
                "targets": elbl,
                "puzzle_identifiers": torch.zeros((einp.shape[0],), dtype=torch.int64, device=(dml_device or device_obj)),
            }
            with torch.no_grad():
                # Unwrap DDP model if needed to access custom methods
                model_unwrapped = model_student.module if hasattr(model_student, 'module') else model_student
                c = model_unwrapped.initial_carry(ebatch)
                _, out = model_unwrapped(c, ebatch)
                logits = out["logits"]
                B, S, V = logits.shape
                ce = F.cross_entropy(logits.view(B * S, V), elbl.view(B * S), ignore_index=-100, reduction="sum")
                total_ce += float(ce.detach().cpu().item())
                pred = logits.argmax(dim=-1)
                mask = (elbl != -100)
                total_tok += int(mask.sum().detach().cpu().item())
                total_correct += int(((pred == elbl) & mask).sum().detach().cpu().item())
                eq = ((pred == elbl) | (~mask)).all(dim=-1)
                total_exact += int(eq.sum().detach().cpu().item())
            batches += 1
        avg_ce = total_ce / max(1, total_tok)
        ppl = float(torch.exp(torch.tensor(avg_ce)).item()) if avg_ce < 100.0 else float("inf")
        tok_acc = (float(total_correct) / float(max(1, total_tok)))
        exact = (float(total_exact) / float(max(1, int(eval_batches) * int(batch_size))))
        payload = {
            "event": "eval",
            "ce_token": round(avg_ce, 6),
            "ppl": round(ppl, 4) if ppl != float("inf") else ppl,
            "token_acc": round(tok_acc, 6),
            "exact_match": round(exact, 6),
        }
        
        # English logic compliance evaluation (if enabled and tokenizer provided)
        if enable_english_logic_eval and tokenizer is not None:
            try:
                from .english_logic_eval import evaluate_generated_samples
                
                # Generate samples and evaluate English quality
                english_eval = evaluate_generated_samples(
                    model=model_student,
                    tokenizer=tokenizer,
                    device=device_obj if dml_device is None else dml_device,
                    num_samples=3,  # Keep it small for speed
                )
                
                # Add English metrics to payload
                if "error" not in english_eval:
                    payload["english_logic"] = {
                        "quality_score": english_eval.get("avg_english_quality_score", 0.0),
                        "flesch_ease": english_eval.get("avg_flesch_reading_ease", 0.0),
                        "flesch_grade": english_eval.get("avg_flesch_kincaid_grade", 0.0),
                        "lexical_diversity": english_eval.get("avg_lexical_diversity", 0.0),
                        "grammar_issues_per_sample": english_eval.get("avg_grammar_issue_count", 0.0),
                        "samples_with_repetition": english_eval.get("samples_with_excessive_repetition", 0),
                        "contradiction_patterns": english_eval.get("total_contradiction_patterns", 0),
                    }
                    # Also add top-level keys for easier tracking
                    payload["english_quality_score"] = english_eval.get("avg_english_quality_score", 0.0)
                    payload["flesch_reading_ease"] = english_eval.get("avg_flesch_reading_ease", 0.0)
                    payload["flesch_kincaid_grade"] = english_eval.get("avg_flesch_kincaid_grade", 0.0)
            except Exception as e:
                payload["english_logic_error"] = str(e)
        
        print(payload)
        write_jsonl(payload)
    except Exception:
        pass
    finally:
        try:
            model_student.train(True)
        except Exception:
            pass

