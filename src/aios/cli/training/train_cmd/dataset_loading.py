"""
Dataset file loading for training.

Loads text lines from various formats and prepares them for training.
"""

import json
import re
from typing import List, Optional, Dict, Tuple

from aios.core.replay import ReplayBuffer
from aios.ml.english_metrics import summarize_corpus

from .instruction_parsing import parse_instruction_spec, check_instruction_adherence


def load_dataset_file(
    dataset_file: str,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    feature_dim: Optional[int] = None,
    task: Optional[str] = None,
    instr: Optional[str] = None,
) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
    """Load text dataset from file and populate replay buffer.
    
    Args:
        dataset_file: Path to dataset file
        batch_size: Batch size for determining sample count
        replay_buffer: ReplayBuffer to populate
        feature_dim: Optional feature dimension for text hashing
        task: Optional task specification (e.g., 'extract:key=REGEX')
        instr: Optional instruction adherence spec
    
    Returns:
        Tuple of (used_dataset, english_corpus_metrics, adherence_metrics)
    """
    try:
        from aios.data.datasets import read_text_lines_sample_any
        from aios.ml.text_features import featurize_bow_hashing, featurize_hashing
        
        lines = read_text_lines_sample_any(dataset_file, max_lines=max(2000, batch_size * 20))
        
        # Extract text from JSON if needed
        texts: List[str] = []
        for ln in lines:
            t = ln
            if ln and ln.strip().startswith("{"):
                try:
                    rec = json.loads(ln)
                    if isinstance(rec, dict) and rec.get("text"):
                        t = str(rec.get("text"))
                except Exception:
                    pass
            if t:
                texts.append(t)
        
        if texts:
            lines = texts
        
        # Compute corpus metrics
        english_corpus_metrics: Optional[Dict] = None
        try:
            if lines:
                english_corpus_metrics = summarize_corpus(lines, max_samples=5000)
        except Exception:
            english_corpus_metrics = None
        
        # Parse instruction spec if provided
        instr_spec: Optional[Dict] = None
        if instr:
            try:
                instr_spec = parse_instruction_spec(instr)
            except Exception:
                instr_spec = {}
        
        # Track adherence metrics
        ad_pass = 0
        ad_tot = 0
        ad_passive = 0
        
        # Track task metrics
        task_match_count = 0
        task_total = 0
        
        # Process each line
        dim = int(feature_dim) if feature_dim and feature_dim > 0 else None
        
        for i, ln in enumerate(lines):
            a = (i % 5)
            r = float(1.0 + (len(ln) % 3) * 0.1)
            
            # Apply task filter
            if task:
                try:
                    spec = str(task).strip()
                    if spec.lower().startswith("extract:"):
                        rhs = spec.split(":", 1)[1].strip()
                        if "=" in rhs:
                            _, rgx = rhs.split("=", 1)
                        else:
                            rgx = rhs
                        m = re.search(rgx, ln or "")
                        r = 1.0 if m else 0.0
                        task_total += 1
                        if m:
                            task_match_count += 1
                except Exception:
                    pass
            
            # Apply instruction adherence filter
            if instr and instr_spec is not None:
                try:
                    ok, meta = check_instruction_adherence(ln or "", instr_spec)
                    r = 1.0 if ok else 0.0
                    ad_pass += 1 if ok else 0
                    ad_tot += 1
                    if meta.get("passive"):
                        ad_passive += 1
                except Exception:
                    pass
            
            # Featurize and add to replay buffer
            if dim:
                try:
                    x = featurize_bow_hashing(ln, dim=dim).astype("float32")
                except Exception:
                    x = featurize_hashing(ln, dim=dim).astype("float32")
                replay_buffer.push(x, a, r, x, False)
            else:
                replay_buffer.push([0], a, r, [0], False)
        
        used_dataset = len(lines) > 0
        
        # Build adherence metrics
        adherence_metrics: Optional[Dict] = None
        if instr and ad_tot > 0:
            adherence_metrics = {
                "spec": instr,
                "count": int(ad_tot),
                "pass": int(ad_pass),
                "rate": float(ad_pass / ad_tot),
                "passive": int(ad_passive),
            }
        
        return used_dataset, english_corpus_metrics, adherence_metrics
        
    except Exception:
        return False, None, None
