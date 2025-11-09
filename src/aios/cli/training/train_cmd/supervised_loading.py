"""
Supervised CSV loading for training.

Loads labeled CSV data for supervised learning tasks.
"""

import re
from typing import Optional, Dict, Tuple

from aios.core.replay import ReplayBuffer
from aios.ml.english_metrics import summarize_corpus

from .instruction_parsing import parse_instruction_spec, check_instruction_adherence


def load_supervised_csv(
    supervised_csv: str,
    csv_text_col: str,
    csv_label_col: str,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    feature_dim: int,
    task: Optional[str] = None,
    instr: Optional[str] = None,
) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
    """Load supervised CSV data and populate replay buffer.
    
    Args:
        supervised_csv: Path to CSV file
        csv_text_col: Text column name or index
        csv_label_col: Label column name or index
        batch_size: Batch size for determining sample count
        replay_buffer: ReplayBuffer to populate
        feature_dim: Feature dimension for text hashing
        task: Optional task specification
        instr: Optional instruction adherence spec
    
    Returns:
        Tuple of (used_dataset, english_corpus_metrics, adherence_metrics)
    """
    try:
        from aios.data.datasets import read_csv_text_label_samples
        from aios.ml.text_features import featurize_bow_hashing, featurize_hashing
        
        # Read CSV pairs
        pairs = read_csv_text_label_samples(
            supervised_csv,
            text_col=(csv_text_col if csv_text_col.isdigit() else csv_text_col),
            label_col=(csv_label_col if csv_label_col.isdigit() else csv_label_col),
            max_rows=max(5000, batch_size * 50)
        )
        
        # Compute corpus metrics
        english_corpus_metrics: Optional[Dict] = None
        try:
            if pairs:
                english_corpus_metrics = summarize_corpus([t for (t, _) in pairs], max_samples=5000)
        except Exception:
            pass
        
        dim = int(feature_dim)
        
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
        
        # Track task metrics for regex extraction
        tp = fp = tn = fn = 0
        thr = 0.5
        rx: Optional[str] = None
        
        if task:
            try:
                spec = str(task).strip()
                if spec.lower().startswith("extract:"):
                    rhs = spec.split(":", 1)[1].strip()
                    rx = (rhs.split("=", 1)[1] if "=" in rhs else rhs)
            except Exception:
                rx = None
        
        # Process each text-label pair
        for i, (txt, y) in enumerate(pairs):
            # Featurize text
            try:
                x = featurize_bow_hashing(txt, dim=dim).astype("float32")
            except Exception:
                x = featurize_hashing(txt, dim=dim).astype("float32")
            
            a = (i % 5)
            r = float(y)
            
            # Apply regex task evaluation
            if rx is not None:
                try:
                    pred_pos = bool(re.search(rx, txt or ""))
                    try:
                        yv = float(y)
                    except Exception:
                        yv = 1.0 if str(y).strip() not in ("0", "", "false", "False") else 0.0
                    actual_pos = (yv >= thr)
                    
                    if pred_pos and actual_pos:
                        tp += 1
                    elif pred_pos and (not actual_pos):
                        fp += 1
                    elif (not pred_pos) and (not actual_pos):
                        tn += 1
                    else:
                        fn += 1
                except Exception:
                    pass
            
            # Apply instruction adherence filter
            if instr and instr_spec is not None:
                try:
                    ok, meta = check_instruction_adherence(txt or "", instr_spec)
                    r = 1.0 if ok else 0.0
                    ad_pass += 1 if ok else 0
                    ad_tot += 1
                    if meta.get("passive"):
                        ad_passive += 1
                except Exception:
                    pass
            
            replay_buffer.push(x, a, float(r), x, False)
        
        used_dataset = len(pairs) > 0
        
        # Build task metrics
        english_task_metrics: Optional[Dict] = None
        if rx is not None:
            try:
                prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                acc = ((tp + tn) / max(1, (tp + tn + fp + fn)))
                english_task_metrics = {
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "accuracy": float(acc),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                    "threshold": float(thr),
                }
            except Exception:
                english_task_metrics = None
        
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
