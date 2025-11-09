"""Dataset fetchers for various public dataset sources.

Provides functions to fetch dataset lists from:
- GitHub NLP datasets repository
- Awesome Public Datasets (NLP section)
- AWS Open Data Registry
"""

from __future__ import annotations

import os
import re
import urllib.request as _urlreq
from typing import Any


def fetch_from_github(*, max_items: int = 200, max_size_gb: int = 15) -> list[dict]:
    """Fetch known datasets from GitHub NLP datasets repository.
    
    Args:
        max_items: Maximum number of datasets to return
        max_size_gb: Maximum size in GB for each dataset
        
    Returns:
        List of dataset dictionaries with 'name', 'url', and optional 'size_bytes'
    """
    RAW_URL = "https://raw.githubusercontent.com/niderhoff/nlp-datasets/master/README.md"
    resp = _urlreq.urlopen(RAW_URL, timeout=5)
    md = resp.read().decode("utf-8", errors="ignore")

    cand: list[tuple[str, str]] = []
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", md):
        title = m.group(1).strip()
        url = m.group(2).strip()
        if any(s in url.lower() for s in ["github.com/niderhoff/nlp-datasets", "#"]):
            continue
        if not re.search(r"\.(jsonl?|csv|tsv|txt|zip|tgz|tar\.gz|gz|bz2|xz)$", url, re.I):
            continue
        cand.append((title, url))
    
    items: list[dict] = []
    for title, url in cand[: max_items * 2]:
        ok = False
        size_ok = True
        size_val = None
        try:
            req = _urlreq.Request(url, method="HEAD")  # type: ignore[arg-type]
            with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                code = getattr(r2, "status", 200)
                ok = 200 <= int(code) < 400
                cl = r2.headers.get("Content-Length")
                if cl and cl.isdigit():
                    size = int(cl)
                    size_val = size
                    if size > max(1, max_size_gb) * (1024**3):
                        size_ok = False
            if not ok:
                req2 = _urlreq.Request(url, headers={"Range": "bytes=0-0"})
                with _urlreq.urlopen(req2, timeout=6) as r3:
                    code = getattr(r3, "status", 206)
                    ok = 200 <= int(code) < 400
        except Exception:
            ok = False
        if ok and size_ok:
            d: dict[str, Any] = {"name": title, "url": url}
            if isinstance(size_val, int):
                d["size_bytes"] = int(size_val)
            items.append(d)
        if len(items) >= max_items:
            break
    return items


def fetch_from_awesomedata_nlp(*, max_items: int = 120, max_size_gb: int = 15) -> list[dict]:
    """Fetch known datasets from Awesome Public Datasets (NLP section).
    
    Args:
        max_items: Maximum number of datasets to return
        max_size_gb: Maximum size in GB for each dataset
        
    Returns:
        List of dataset dictionaries with 'name', 'url', and optional 'size_bytes'
    """
    RAW_URL = "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/README.md"
    
    resp = _urlreq.urlopen(RAW_URL, timeout=6)
    md = resp.read().decode("utf-8", errors="ignore")
    m = re.search(r"^##\s*Natural\s+Language\b.*?(?=^##\s+|\Z)", md, re.M | re.S)
    block = m.group(0) if m else md
    
    cand: list[tuple[str, str]] = []
    for mm in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", block):
        title = mm.group(1).strip()
        url = mm.group(2).strip()
        if "github.com/awesomedata/awesome-public-datasets" in url.lower():
            continue
        if not re.search(r"\.(jsonl?|csv|tsv|txt|zip|tgz|tar\.gz|gz|bz2|xz)$", url, re.I):
            continue
        cand.append((title, url))
    
    items: list[dict] = []
    for title, url in cand[: max_items * 2]:
        ok = False
        size_ok = True
        size_val = None
        try:
            req = _urlreq.Request(url, method="HEAD")  # type: ignore[arg-type]
            with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                code = getattr(r2, "status", 200)
                ok = 200 <= int(code) < 400
                cl = r2.headers.get("Content-Length")
                if cl and cl.isdigit():
                    size = int(cl)
                    size_val = size
                    if size > max(1, max_size_gb) * (1024**3):
                        size_ok = False
            if not ok:
                req2 = _urlreq.Request(url, headers={"Range": "bytes=0-0"})
                with _urlreq.urlopen(req2, timeout=6) as r3:
                    code = getattr(r3, "status", 206)
                    ok = 200 <= int(code) < 400
        except Exception:
            ok = False
        if ok and size_ok:
            d: dict[str, Any] = {"name": title, "url": url}
            if isinstance(size_val, int):
                d["size_bytes"] = int(size_val)
            items.append(d)
        if len(items) >= max_items:
            break
    return items


def fetch_from_aws_open_data_registry(*, max_items: int = 60, max_size_gb: int = 15) -> list[dict]:
    """Fetch known datasets from AWS Open Data Registry.
    
    Args:
        max_items: Maximum number of datasets to return
        max_size_gb: Maximum size in GB for each dataset
        
    Returns:
        List of dataset dictionaries with 'name', 'url', and optional 'size_bytes'
    """
    RAW_URL = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/"
    # list of YAML files index (simple scrape)
    index_url = RAW_URL + "README.md"
    md = _urlreq.urlopen(index_url, timeout=6).read().decode("utf-8", errors="ignore")
    
    ymls = [m.group(1) for m in re.finditer(r"\((datasets/[^)]+\.ya?ml)\)", md)]
    items: list[dict] = []
    
    for rel in ymls[: max_items * 2]:
        url = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/" + rel
        try:
            txt = _urlreq.urlopen(url, timeout=6).read().decode("utf-8", errors="ignore")
            # crude scan for http/https URLs
            for mm in re.finditer(r"https?://[^\s'\"]+", txt):
                link = mm.group(0)
                # best-effort HEAD validate
                ok = False
                size_ok = True
                size_val = None
                try:
                    req = _urlreq.Request(link, method="HEAD")  # type: ignore[arg-type]
                    with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                        code = getattr(r2, "status", 200)
                        ok = 200 <= int(code) < 400
                        cl = r2.headers.get("Content-Length")
                        if cl and cl.isdigit():
                            size = int(cl)
                            size_val = size
                            if size > max(1, max_size_gb) * (1024**3):
                                size_ok = False
                    if not ok:
                        req2 = _urlreq.Request(link, headers={"Range": "bytes=0-0"})
                        with _urlreq.urlopen(req2, timeout=6) as r3:
                            code = getattr(r3, "status", 206)
                            ok = 200 <= int(code) < 400
                except Exception:
                    ok = False
                if ok and size_ok:
                    items.append({"name": os.path.basename(link), "url": link, "size_bytes": size_val})
                    if len(items) >= max_items:
                        break
        except Exception:
            pass
        if len(items) >= max_items:
            break
    return items
