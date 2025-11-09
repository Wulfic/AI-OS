from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import aiohttp
import bs4  # type: ignore
import typer

from aios.data.datasets import (
    datasets_base_dir,
    datasets_storage_cap_gb,
    can_store_additional_gb,
)
from aios.tools.web import ddg_search


def datasets_build_text(
    query: str = typer.Argument(..., help="What text/pages to collect, e.g., 'boats'"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="Dataset name; defaults to slugified query under 'text'"),
    limit: int = typer.Option(100, "--max-docs", help="Maximum number of text docs to collect"),
    per_site: int = typer.Option(20, "--per-site", help="Maximum docs per site"),
    search_results: int = typer.Option(10, "--search-results", help="Number of top search results (sites) to crawl"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset directory if present"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Emit JSONL progress events"),
    rps: float = typer.Option(2.0, "--rps", help="Requests per second cap (overall)"),
    min_chars: int = typer.Option(400, "--min-chars", help="Skip docs shorter than this many characters"),
    allow_ext: str = typer.Option("", "--allow-ext", help="Comma-separated allowed file extensions for text (e.g., txt,pdf,doc,docx). If set, only links with these extensions or content-types are saved."),
    file_prefix: Optional[str] = typer.Option(None, "--file-prefix", help="Filename prefix for saved docs; defaults to search query or store-dataset name."),
):
    """Search sites and collect textual content by extracting main text from pages.

    - Uses DuckDuckGo lite search to find sites for the query
    - Fetches each site front page and extracts main readable text
    - Stores docs under docs/ with a manifest.jsonl: path, label, url, title, chars, excerpt
    - Enforces dataset storage cap (approx by avg size estimation)
    """

    def _slug(s: str) -> str:
        import re
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        s2 = re.sub(r"[^a-z0-9_\-]", "", s2)
        return s2 or "dataset"

    dataset_name = (store_dataset or _slug(query))
    dataset_root = datasets_base_dir() / "text" / dataset_name
    docs_dir = dataset_root
    manifest_path = dataset_root / "manifest.jsonl"
    try:
        if dataset_root.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_root, ignore_errors=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print({"started": False, "error": f"cannot create dataset dir: {e}"})
        return

    sem = asyncio.Semaphore(max(1, int(rps)) if rps and rps > 0 else 5)

    async def _fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = (resp.headers.get("Content-Type", "") or "").lower()
                    if "text/html" in ctype:
                        return await resp.text(errors="ignore")
                    return None
        except Exception:
            return None

    def _extract_text(html: str) -> tuple[str, str]:
        """Return (title, text) best-effort using simple heuristics."""
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
        except Exception:
            return ("", "")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            try:
                tag.decompose()
            except Exception:
                pass
        # Title
        title = ""
        try:
            if soup.title and soup.title.string:
                title = str(soup.title.string).strip()
        except Exception:
            pass
        # Prefer article/main; else largest text block
        candidates = []
        try:
            for sel in ["article", "main"]:
                node = soup.select_one(sel)
                if node:
                    txt = node.get_text("\n", strip=True)
                    candidates.append(txt)
        except Exception:
            pass
        if not candidates:
            # Find largest <div>/<section> by text length
            try:
                best = ""
                best_len = 0
                for node in soup.find_all(["div", "section", "p"]):
                    try:
                        txt = node.get_text("\n", strip=True)
                    except Exception:
                        continue
                    n = len(txt)
                    if n > best_len:
                        best_len = n
                        best = txt
                if best:
                    candidates.append(best)
            except Exception:
                pass
        text = "\n\n".join([c for c in candidates if c]) if candidates else ""
        # Normalize whitespace
        if text:
            try:
                lines = [ln.strip() for ln in text.splitlines()]
                text = "\n".join([ln for ln in lines if ln])
            except Exception:
                pass
        return (title, text)

    # Parse allowed extensions and validate for text/doc types
    allowed_exts = [x.strip().lower() for x in (allow_ext or "").split(",") if x.strip()]
    allowed_exts = [x if x.startswith(".") else f".{x}" for x in allowed_exts]
    valid_text_exts = {".txt", ".pdf", ".doc", ".docx", ".rtf", ".md", ".html", ".htm"}
    if allowed_exts:
        invalid = [e for e in allowed_exts if e not in valid_text_exts]
        if invalid:
            print(json.dumps({
                "stored": False,
                "error": "invalid_extensions",
                "message": "Only text/document extensions are allowed for text dataset",
                "invalid": invalid,
                "allowed": sorted(list(valid_text_exts)),
            }, ensure_ascii=False))
            return

    async def _run():
        # Search
        sites = await ddg_search(query, limit=max(1, int(search_results)))
        if progress:
            try:
                print(json.dumps({"event": "search", "query": query, "sites": [s.url for s in sites]}, ensure_ascii=False), flush=True)
            except Exception:
                pass
        total_docs = 0
        written_bytes = 0
        manifest_lines: list[str] = []
        label = query.strip()
        # Rough capacity estimate: 8KB per doc
        est_gb = (limit * 8_000) / float(1024 ** 3)
        if not can_store_additional_gb(est_gb):
            print(json.dumps({"stored": False, "reason": "cap_exceeded", "cap_gb": float(datasets_storage_cap_gb())}, ensure_ascii=False))
            return
        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {"User-Agent": f"aios-bot/0.1{ua_suffix}", "Accept-Language": "en-US,en;q=0.7"}
        async with aiohttp.ClientSession(headers=headers) as session:
            doc_idx = 1
            # filename prefix
            prefix = (file_prefix or dataset_name or _slug(query)).strip()
            if prefix:
                import re as _re
                prefix = _re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix).strip("_-")
            for site in sites:
                if total_docs >= int(limit):
                    break
                html = await _fetch_url(session, site.url)
                if not html:
                    continue
                # If allow_ext is provided, we prefer downloading linked documents matching extensions
                if allowed_exts:
                    # Find candidate links
                    try:
                        soup = bs4.BeautifulSoup(html, "lxml")
                        from bs4.element import Tag  # type: ignore
                        from urllib.parse import urljoin
                        links = []
                        for el in soup.find_all("a"):
                            if not isinstance(el, Tag):
                                continue
                            href = el.get("href")
                            if not href:
                                continue
                            u = urljoin(site.url, str(href))
                            ln = u.lower()
                            if any(ln.endswith(ext) for ext in allowed_exts):
                                links.append(u)
                    except Exception:
                        links = []
                    # Try to fetch the first matching document (simple heuristic)
                    saved_doc = False
                    for u in links:
                        if total_docs >= int(limit) or saved_doc:
                            break
                        try:
                            async with session.get(u, timeout=aiohttp.ClientTimeout(total=30)) as resp2:
                                if resp2.status >= 400:
                                    continue
                                ctype2 = (resp2.headers.get("Content-Type", "") or "").lower()
                                data = await resp2.read()
                                if len(data) < 200:  # too small
                                    continue
                                # Decide extension
                                ext = ""
                                for e in allowed_exts:
                                    if u.lower().endswith(e):
                                        ext = e
                                        break
                                if not ext:
                                    if "pdf" in ctype2:
                                        ext = ".pdf"
                                    elif "msword" in ctype2 or "officedocument" in ctype2:
                                        ext = ".docx"
                                    elif "plain" in ctype2 or "text/" in ctype2:
                                        ext = ".txt"
                                    else:
                                        continue  # skip if can't map and filtering is on
                                import hashlib as _hl
                                h = _hl.sha256(data).hexdigest()[:16]
                                base = f"{h}{ext}"
                                fn = f"{prefix}_{base}" if prefix else base
                                fp = docs_dir / fn
                                with open(fp, "wb") as f:
                                    f.write(data)
                                total_docs += 1
                                doc_idx += 1
                                written_bytes += len(data)
                                rec = {
                                    "path": fn,
                                    "label": label,
                                    "url": u,
                                    "title": "",
                                    "chars": 0,
                                    "excerpt": "",
                                }
                                manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                                if progress:
                                    try:
                                        print(json.dumps({"event": "doc", "downloaded": total_docs, "target": int(limit), "file": fn}, ensure_ascii=False), flush=True)
                                    except Exception:
                                        pass
                                saved_doc = True
                        except Exception:
                            continue
                    if saved_doc:
                        continue
                    # Strict mode: if allowlist provided and .txt is NOT allowed, do not fallback to extracted text
                    if ".txt" not in allowed_exts:
                        continue

                title, text = _extract_text(html)
                if progress:
                    try:
                        nchar = len(text or "")
                        print(json.dumps({"event": "page", "url": site.url, "chars": nchar}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                if not text or len(text) < int(min_chars):
                    continue
                data = text.encode("utf-8", errors="ignore")
                import hashlib as _hl
                h = _hl.sha256(data).hexdigest()[:16]
                base = f"{h}.txt"
                fn = f"{prefix}_{base}" if prefix else base
                fp = docs_dir / fn
                try:
                    with open(fp, "wb") as f:
                        f.write(data)
                except Exception:
                    continue
                total_docs += 1
                doc_idx += 1
                written_bytes += len(data)
                rec = {
                    "path": fn,
                    "label": label,
                    "url": site.url,
                    "title": title or "",
                    "chars": len(text),
                    "excerpt": (text[:200] + ("…" if len(text) > 200 else "")),
                }
                manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                if progress:
                    try:
                        print(json.dumps({"event": "doc", "downloaded": total_docs, "target": int(limit), "file": fn}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            for ln in manifest_lines:
                f.write(ln)
        out = {
            "stored": True,
            "dataset_path": str(dataset_root),
            "docs": int(total_docs),
            "wrote_bytes": int(written_bytes),
            "manifest": str(manifest_path),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    asyncio.run(_run())
