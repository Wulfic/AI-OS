from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich import print

from aios.core.train import Trainer, TrainConfig
from aios.memory.store import get_db, init_db


def english_eval(
    *,
    dataset_file: str,
    checkpoint: Optional[str] = None,
    feature_dim: Optional[int] = None,
    task: Optional[str] = None,
    instr: Optional[str] = None,
    max_lines: int = 2000,
    label: Optional[str] = None,
) -> None:
    """Evaluate English progress metrics on a dataset, optionally using a model checkpoint.

    Produces an 'english_eval' artifact with:
    - corpus readability summary
    - instruction adherence pass-rate (if --instr)
    - task extraction metrics and (if checkpoint provided) prediction quality
    """
    from aios.data.datasets import read_text_lines_sample_any
    from aios.ml.english_metrics import summarize_corpus
    from aios.ml.text_features import featurize_bow_hashing, featurize_hashing
    import numpy as _np

    def _normalize_lines(raw: list[str]) -> list[str]:
        out: list[str] = []
        for ln in raw:
            t = ln
            if ln and ln.strip().startswith("{"):
                try:
                    rec = json.loads(ln)
                    if isinstance(rec, dict) and rec.get("text"):
                        t = str(rec.get("text"))
                except Exception:
                    pass
            if t:
                out.append(t)
        return out

    def _parse_instr(spec: str) -> dict:
        d: dict = {}
        try:
            parts = [p.strip() for p in spec.split(";") if p.strip()]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    d[k.strip().lower()] = v.strip()
                else:
                    d[p.strip().lower()] = True
        except Exception:
            pass
        for list_key in ("require", "require_any", "forbid"):
            if list_key in d and isinstance(d[list_key], str):
                d[list_key] = [w.strip() for w in d[list_key].split(",") if w.strip()]
        try:
            for k in ("max_words", "min_words", "max_chars", "min_chars", "bullets_min", "sentences_max"):
                if k in d:
                    d[k] = int(d[k])
        except Exception:
            pass
        d["no_passive"] = str(d.get("no_passive", "false")).lower() in ("1", "true", "yes")
        return d

    def _pass_instr(line: str, spec: dict) -> tuple[bool, dict]:
        import re as _re
        lw = (line or "").lower()
        words = [w for w in lw.split() if w]
        wcount = len(words)
        chars = len(lw)
        required = list(spec.get("require", []) or [])
        require_any = list(spec.get("require_any", []) or [])
        forbid = list(spec.get("forbid", []) or [])
        req_ok = all((r.lower() in lw) for r in required)
        any_ok = (True if not require_any else any((r.lower() in lw) for r in require_any))
        forbid_ok = all((f.lower() not in lw) for f in forbid)
        maxw = spec.get("max_words")
        minw = spec.get("min_words")
        len_ok = True
        if isinstance(maxw, int) and maxw > 0:
            len_ok = len_ok and (wcount <= int(maxw))
        if isinstance(minw, int) and minw > 0:
            len_ok = len_ok and (wcount >= int(minw))
        maxc = spec.get("max_chars")
        minc = spec.get("min_chars")
        char_ok = True
        if isinstance(maxc, int) and maxc > 0:
            char_ok = char_ok and (chars <= int(maxc))
        if isinstance(minc, int) and minc > 0:
            char_ok = char_ok and (chars >= int(minc))
        bullets_min = spec.get("bullets_min")
        sentences_max = spec.get("sentences_max")
        bullets_ok = True
        sentences_ok = True
        try:
            bullet_count = 0
            for ln in (line or "").splitlines():
                if _re.match(r"^\s*(?:[-*•]|\d+[\.)])\s+", ln):
                    bullet_count += 1
            if isinstance(bullets_min, int) and bullets_min > 0:
                bullets_ok = (bullet_count >= int(bullets_min))
            sent_count = len([s for s in _re.split(r"[.!?]+", lw) if s.strip()])
            if isinstance(sentences_max, int) and sentences_max > 0:
                sentences_ok = (sent_count <= int(sentences_max))
        except Exception:
            pass
        passive = False
        if bool(spec.get("no_passive", False)):
            try:
                passive = bool(_re.search(r"\b(?:was|were|be|been|being|is|are|am)\b\s+\w+ed\b", lw))
            except Exception:
                passive = False
        ok = (req_ok and any_ok and forbid_ok and len_ok and char_ok and bullets_ok and sentences_ok and (not passive))
        return ok, {"words": wcount, "chars": chars, "req_ok": req_ok, "any_ok": any_ok, "forbid_ok": forbid_ok, "len_ok": len_ok, "char_ok": char_ok, "bullets_ok": bullets_ok, "sentences_ok": sentences_ok, "passive": passive}

    raw_lines = read_text_lines_sample_any(dataset_file, max_lines=max_lines)
    lines = _normalize_lines(raw_lines)
    if not lines:
        print({"evaluated": False, "error": "no lines found", "dataset_file": dataset_file})
        return

    try:
        from aios.ml.english_metrics import summarize_corpus
        corpus = summarize_corpus(lines, max_samples=max_lines)
    except Exception:
        corpus = {"count": len(lines)}

    rx: Optional[str] = None
    if task:
        try:
            spec = str(task).strip()
            if spec.lower().startswith("extract:"):
                rhs = spec.split(":", 1)[1].strip()
                rx = (rhs.split("=", 1)[1] if "=" in rhs else rhs)
            else:
                rx = spec
        except Exception:
            rx = None
    instr_spec: Optional[dict] = None
    if instr:
        try:
            instr_spec = _parse_instr(instr)
        except Exception:
            instr_spec = {}

    tr = None
    model_dim: Optional[int] = None
    if checkpoint:
        try:
            tcfg = TrainConfig()
            tr = Trainer(tcfg)
            if not tr.load_checkpoint(checkpoint):
                tr = None
            else:
                model_dim = int(tr.model_np.W1.shape[0])
        except Exception:
            tr = None

    dim: Optional[int] = None
    if tr and model_dim:
        dim = model_dim
    elif feature_dim and feature_dim > 0:
        dim = int(feature_dim)

    eval_out: dict = {
        "dataset_file": dataset_file,
        "checkpoint": checkpoint,
        "lines": len(lines),
        "corpus": corpus,
    }

    if instr_spec is not None:
        passes = 0
        passive = 0
        for ln in lines:
            try:
                ok, meta = _pass_instr(ln or "", instr_spec)
                passes += 1 if ok else 0
                if meta.get("passive"):
                    passive += 1
            except Exception:
                continue
        total = len(lines)
        eval_out["adherence"] = {"spec": instr, "count": total, "pass": passes, "rate": (passes / max(1, total)), "passive": passive}

    if rx is not None:
        import re as _re
        labels: list[int] = []
        for ln in lines:
            try:
                labels.append(1 if _re.search(rx or "", ln or "") else 0)
            except Exception:
                labels.append(0)
        matches = int(sum(labels))
        total = len(labels)
        eval_out.setdefault("task", {})
        eval_out["task"].update({"spec": task, "matches": matches, "total": total, "match_rate": (matches / max(1, total))})

        if tr and dim:
            X = []
            for ln in lines:
                try:
                    try:
                        x = featurize_bow_hashing(ln, dim=dim).astype("float32")
                    except Exception:
                        x = featurize_hashing(ln, dim=dim).astype("float32")
                except Exception:
                    x = _np.zeros((dim,), dtype="float32")
                X.append(x)
            Xb = _np.stack(X, axis=0)
            y_pred, _ = tr.model_np.forward(Xb)  # (N,1)
            preds = y_pred.reshape((-1,)).astype("float32")
            thr = float(_np.median(preds))
            tp = fp = tn = fn = 0
            for y, p in zip(labels, preds):
                pos = (p >= thr)
                if pos and y == 1:
                    tp += 1
                elif pos and y == 0:
                    fp += 1
                elif (not pos) and y == 0:
                    tn += 1
                else:
                    fn += 1
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = ((tp + tn) / max(1, (tp + tn + fp + fn)))
            eval_out["task"].update({
                "pred_threshold": thr,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            })

    try:
        conn = get_db()
        init_db(conn)
        from aios.memory.store import save_artifact
        lbl = label or Path(dataset_file).stem
        save_artifact(conn, kind="english_eval", label=str(lbl), data=eval_out)
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
    print({"evaluated": True, "label": (label or Path(dataset_file).stem), "lines": len(lines)})
