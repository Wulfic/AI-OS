from __future__ import annotations

from typing import Iterable, Dict, Any


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter (simple vowel-group method).

    Not perfect but good enough for relative readability tracking.
    """
    w = word.lower()
    if not w:
        return 0
    vowels = set("aeiouy")
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    # silent 'e'
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _split_sentences(text: str) -> list[str]:
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
    return parts or ([text] if text else [])


def compute_readability(text: str) -> Dict[str, float]:
    """Compute simple readability metrics (Flesch Reading Ease).

    Returns: { words, sentences, syllables, flesch }
    """
    sentences = _split_sentences(text)
    s_count = max(1, len(sentences))
    words = [w for w in text.replace("\n", " ").split() if w]
    w_count = max(1, len(words))
    sylls = 0
    for w in words:
        sylls += _count_syllables(w)
    # Flesch Reading Ease
    # 206.835 − 1.015 (words/sentences) − 84.6 (syllables/words)
    flesch = 206.835 - 1.015 * (w_count / s_count) - 84.6 * (sylls / w_count)
    return {
        "words": float(w_count),
        "sentences": float(s_count),
        "syllables": float(sylls),
        "flesch": float(flesch),
    }


def summarize_corpus(lines: Iterable[str], *, max_samples: int = 2000) -> Dict[str, Any]:
    total = 0
    sum_words = 0.0
    sum_sent = 0.0
    sum_syll = 0.0
    sum_flesch = 0.0
    sample_snippets: list[str] = []
    for i, ln in enumerate(lines):
        if not ln:
            continue
        m = compute_readability(ln)
        sum_words += m["words"]
        sum_sent += m["sentences"]
        sum_syll += m["syllables"]
        sum_flesch += m["flesch"]
        total += 1
        if len(sample_snippets) < 5:
            # store small sanitized snippet
            s = ln.strip().replace("\n", " ")
            if len(s) > 200:
                s = s[:200] + "…"
            sample_snippets.append(s)
        if total >= max_samples:
            break
    if total == 0:
        return {"count": 0}
    return {
        "count": int(total),
        "avg_words": float(sum_words / total),
        "avg_sentences": float(sum_sent / total),
        "avg_syllables": float(sum_syll / total),
        "avg_flesch": float(sum_flesch / total),
        "samples": sample_snippets,
    }
