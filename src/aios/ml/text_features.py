from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import re
import string


def featurize_hashing(
    text: str,
    dim: int = 512,
    *,
    lowercase: bool = True,
    normalize_l2: bool = True,
) -> np.ndarray:
    """Hashing featurizer to fixed dim using simple rolling hash over bytes.

    - Lowercases by default.
    - Uses a small multiplicative rolling hash to distribute bytes across buckets.
    - Different dim changes bucket mapping (not a simple prefix), so first 128 of 256 won't match 128-dim.
    - Returns shape (dim,) float32.
    """
    dim = int(max(8, dim))
    v = np.zeros((dim,), dtype=np.float32)
    if not text:
        return v
    try:
        s = text.lower() if lowercase else text
        bs = s.encode("utf-8", errors="ignore")
    except Exception:
        bs = b""
    if not bs:
        return v
    h = 2166136261  # FNV-like start
    for b in bs:
        # simple rolling: h = (h ^ b) * prime (mod 2**32)
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
        idx = h % dim
        v[idx] += 1.0
    if normalize_l2:
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
    return v


# --- English-first bag-of-words hashing encoder ---
_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def _tokenize_words(text: str, *, lowercase: bool = True, strip_punct: bool = True) -> list[str]:
    if not text:
        return []
    s = text.lower() if lowercase else text
    if strip_punct:
        # Replace punctuation with space, keep alnum and basic separators
        tbl = str.maketrans({c: " " for c in string.punctuation})
        s = s.translate(tbl)
    # Split on whitespace
    toks = [t for t in s.split() if t]
    return toks


def _gen_ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    L = len(tokens)
    if n <= 1:
        for t in tokens:
            yield t
        return
    for i in range(max(0, L - n + 1)):
        yield "_".join(tokens[i : i + n])


def featurize_bow_hashing(
    text: str,
    dim: int = 512,
    *,
    lowercase: bool = True,
    strip_punct: bool = True,
    stopwords: Optional[set[str]] = None,
    ngrams: Sequence[int] = (1, 2),
    use_char_ngrams: bool = False,
    char_ngram_range: tuple[int, int] = (3, 5),
    normalize_l2: bool = True,
) -> np.ndarray:
    """Hashing bag-of-words featurizer with optional word n-grams and char n-grams.

    - Tokens are lowercased (default) and punctuation stripped before splitting on whitespace.
    - Stopwords filtered (small built-in list by default; pass empty set() to disable).
    - Word n-grams included per `ngrams` (default: 1-grams and 2-grams).
    - Optional char n-grams (3..5 by default) appended to the multiset when enabled.
    - Features are counts bucketed by a 32-bit hash modulo dim; L2-normalized if requested.
    """
    dim = int(max(8, dim))
    v = np.zeros((dim,), dtype=np.float32)
    if not text:
        return v
    toks = _tokenize_words(text, lowercase=lowercase, strip_punct=strip_punct)
    sw = _DEFAULT_STOPWORDS if stopwords is None else stopwords
    if sw:
        toks = [t for t in toks if t not in sw]
    feats: list[str] = []
    # Word n-grams
    for n in ngrams:
        try:
            n_int = int(n)
        except Exception:
            n_int = 1
        if n_int <= 0:
            continue
        feats.extend(list(_gen_ngrams(toks, n_int)))
    # Optional char n-grams
    if use_char_ngrams:
        try:
            lo, hi = int(char_ngram_range[0]), int(char_ngram_range[1])
        except Exception:
            lo, hi = 3, 5
        s = (text.lower() if lowercase else text)
        s = re.sub(r"\s+", " ", s)
        for n in range(max(1, lo), max(lo, hi) + 1):
            for i in range(0, max(0, len(s) - n + 1)):
                feats.append(s[i : i + n])
    if not feats:
        return v
    # Hash into buckets
    for ft in feats:
        # Use a stable Python hash via hashlib to avoid per-run hash randomization
        # but keep it lightweight with a simple FNV-like scheme over bytes
        h = 2166136261
        bs = ft.encode("utf-8", errors="ignore")
        for b in bs:
            h ^= int(b)
            h = (h * 16777619) & 0xFFFFFFFF
        idx = h % dim
        v[idx] += 1.0
    if normalize_l2:
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
    return v
