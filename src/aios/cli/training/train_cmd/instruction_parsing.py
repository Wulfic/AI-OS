"""
Instruction adherence parsing and validation.

Functions for parsing instruction specifications and checking if text meets criteria.
"""

import re
from typing import Dict, Tuple


def parse_instruction_spec(spec: str) -> Dict:
    """Parse instruction adherence specification string.
    
    Parses spec like: 'require=foo,bar;max_words=60;no_passive=true'
    
    Args:
        spec: Instruction spec string
    
    Returns:
        Dictionary with parsed specification
    """
    d: Dict = {}
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
    
    # Parse list fields
    for list_key in ("require", "require_any", "forbid"):
        if list_key in d and isinstance(d[list_key], str):
            d[list_key] = [w.strip() for w in d[list_key].split(",") if w.strip()]
    
    # Parse integer fields
    try:
        if "max_words" in d:
            d["max_words"] = int(d["max_words"])  # type: ignore[assignment]
        if "min_words" in d:
            d["min_words"] = int(d["min_words"])  # type: ignore[assignment]
        if "max_chars" in d:
            d["max_chars"] = int(d["max_chars"])  # type: ignore[assignment]
        if "min_chars" in d:
            d["min_chars"] = int(d["min_chars"])  # type: ignore[assignment]
        if "bullets_min" in d:
            d["bullets_min"] = int(d["bullets_min"])  # type: ignore[assignment]
        if "sentences_max" in d:
            d["sentences_max"] = int(d["sentences_max"])  # type: ignore[assignment]
    except Exception:
        d.pop("max_words", None)
        d.pop("min_words", None)
        d.pop("max_chars", None)
        d.pop("min_chars", None)
        d.pop("bullets_min", None)
        d.pop("sentences_max", None)
    
    # Parse boolean fields
    d["no_passive"] = str(d.get("no_passive", "false")).lower() in ("1", "true", "yes")
    
    return d


def check_instruction_adherence(line: str, spec: Dict) -> Tuple[bool, Dict]:
    """Check if a line passes instruction adherence criteria.
    
    Args:
        line: Text line to check
        spec: Parsed instruction specification
    
    Returns:
        Tuple of (passes, metadata_dict)
    """
    lw = (line or "").lower()
    words = [w for w in lw.split() if w]
    wcount = len(words)
    chars = len(lw)
    
    # Check required words
    required = list(spec.get("require", []) or [])
    req_ok = all((r.lower() in lw) for r in required)
    
    # Check require_any words
    require_any = list(spec.get("require_any", []) or [])
    any_ok = (True if not require_any else any((r.lower() in lw) for r in require_any))
    
    # Check forbidden words
    forbid = list(spec.get("forbid", []) or [])
    forbid_ok = all((f.lower() not in lw) for f in forbid)
    
    # Check word count limits
    maxw = spec.get("max_words")
    minw = spec.get("min_words")
    len_ok = True
    if isinstance(maxw, int) and maxw > 0:
        len_ok = len_ok and (wcount <= int(maxw))
    if isinstance(minw, int) and minw > 0:
        len_ok = len_ok and (wcount >= int(minw))
    
    # Check character count limits
    maxc = spec.get("max_chars")
    minc = spec.get("min_chars")
    char_ok = True
    if isinstance(maxc, int) and maxc > 0:
        char_ok = char_ok and (chars <= int(maxc))
    if isinstance(minc, int) and minc > 0:
        char_ok = char_ok and (chars >= int(minc))
    
    # Check bullets and sentences
    bullets_min = spec.get("bullets_min")
    sentences_max = spec.get("sentences_max")
    bullets_ok = True
    sentences_ok = True
    
    try:
        # Count bullet points
        bullet_count = 0
        for ln in (line or "").splitlines():
            if re.match(r"^\s*(?:[-*•]|\d+[\.)])\s+", ln):
                bullet_count += 1
        if isinstance(bullets_min, int) and bullets_min > 0:
            bullets_ok = (bullet_count >= int(bullets_min))
        
        # Count sentences
        sent_count = len([s for s in re.split(r"[.!?]+", lw) if s.strip()])
        if isinstance(sentences_max, int) and sentences_max > 0:
            sentences_ok = (sent_count <= int(sentences_max))
    except Exception:
        pass
    
    # Check passive voice
    passive = False
    if bool(spec.get("no_passive", False)):
        try:
            passive = bool(re.search(r"\b(?:was|were|be|been|being|is|are|am)\b\s+\w+ed\b", lw))
        except Exception:
            passive = False
    
    # Overall check
    ok = (
        req_ok and any_ok and forbid_ok and 
        len_ok and char_ok and 
        bullets_ok and sentences_ok and 
        (not passive)
    )
    
    # Build metadata
    metadata = {
        "words": wcount,
        "chars": chars,
        "req_ok": req_ok,
        "any_ok": any_ok,
        "forbid_ok": forbid_ok,
        "len_ok": len_ok,
        "char_ok": char_ok,
        "bullets_ok": bullets_ok,
        "sentences_ok": sentences_ok,
        "passive": passive,
    }
    
    return ok, metadata
