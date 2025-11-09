from __future__ import annotations

import asyncio
import hashlib
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


def datasets_build_raw(
    query: str = typer.Argument(..., help="Generic data crawl keyed by query name"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="Dataset name; defaults to slugified query under 'raw'"),
    limit: int = typer.Option(100, "--max-files", help="Maximum number of files to collect"),
    per_site: int = typer.Option(20, "--per-site", help="Maximum files per site"),
    search_results: int = typer.Option(10, "--search-results", help="Number of top search results (sites) to crawl"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset directory if present"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Emit JSONL progress events"),
    rps: float = typer.Option(2.0, "--rps", help="Requests per second cap (overall)"),
    min_bytes: int = typer.Option(2_000, "--min-bytes", help="Skip files smaller than this many bytes"),
    allow_ext: str = typer.Option("pdf,csv,json,txt,zip,tar.gz,tar,bz2,xlsx,docx,md", "--allow-ext", help="Comma-separated list of allowed file extensions (lowercase, no spaces)"),
):
    # options defined in signature above

    def _slug(s: str) -> str:
        import re
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        s2 = re.sub(r"[^a-z0-9_\-]", "", s2)
        return s2 or "dataset"

    dataset_root = datasets_base_dir() / "raw" / (store_dataset or _slug(query))
    files_dir = dataset_root / "files"
    manifest_path = dataset_root / "manifest.jsonl"
    try:
        if dataset_root.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_root, ignore_errors=True)
        files_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print({"started": False, "error": f"cannot create dataset dir: {e}"})
        return

    # Normalize allowed extensions (support with or without leading dot, lowercase)
    allowed_exts_list = [x.strip().lower() for x in (allow_ext or "").split(",") if x.strip()]
    allowed_exts_list = [x if x.startswith(".") else f".{x}" for x in allowed_exts_list]

    sem = asyncio.Semaphore(max(1, int(rps)) if rps and rps > 0 else 5)

    async def _fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:
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

    def _extract_file_links(html: str, base_url: str, allowed_exts: list[str]) -> list[str]:
        out: list[str] = []
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
            from bs4.element import Tag  # type: ignore
            from urllib.parse import urljoin
            for el in soup.find_all("a"):
                if not isinstance(el, Tag):
                    continue
                href = el.get("href")
                if not href:
                    continue
                u = urljoin(base_url, str(href))
                ln = u.lower()
                if any(ln.endswith(ext) for ext in allowed_exts):
                    out.append(u)
        except Exception:
            return []
        # de-dup preserving order
        seen = set()
        uniq = []
        for u in out:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        return uniq

    async def _download_file(session: aiohttp.ClientSession, url: str, idx: int) -> Optional[tuple[str, int]]:
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=40)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = (resp.headers.get("Content-Type", "") or "").lower()
                    if "text/html" in ctype:
                        return None
                    # pick extension from URL if present
                    ln = url.lower()
                    ext = ""
                    for e in [".pdf", ".csv", ".json", ".txt", ".zip", ".tar.gz", ".tar", ".bz2", ".xlsx", ".docx", ".md"]:
                        if ln.endswith(e):
                            ext = e if e.startswith(".") else "." + e
                            break
                    if not ext:
                        # fallback from content-type
                        if "pdf" in ctype:
                            ext = ".pdf"
                        elif "json" in ctype:
                            ext = ".json"
                        elif "csv" in ctype:
                            ext = ".csv"
                        elif "zip" in ctype:
                            ext = ".zip"
                        elif "plain" in ctype or "text/" in ctype:
                            ext = ".txt"
                        else:
                            ext = ".bin"
                    tmp = files_dir / f"tmp_{idx:06d}.part"
                    h = hashlib.sha256()
                    size = 0
                    with open(tmp, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            if not chunk:
                                continue
                            f.write(chunk)
                            h.update(chunk)
                            size += len(chunk)
                    if size < int(min_bytes):
                        try:
                            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                        return None
                    digest = h.hexdigest()[:16]
                    # sanitize ext: remove leading dot in formatted filename if double-dot
                    ext_norm = ext if ext.startswith(".") else f".{ext}"
                    fn = f"{idx:06d}_{digest}{ext_norm}"
                    fp = files_dir / fn
                    if fp.exists():
                        try:
                            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                        return None
                    os.replace(tmp, fp)
                    return (fn, size)
        except Exception:
            return None

    async def _run():
        sites = await ddg_search(query, limit=max(1, int(search_results)))
        if progress:
            try:
                print(json.dumps({"event": "search", "query": query, "sites": [s.url for s in sites]}, ensure_ascii=False), flush=True)
            except Exception:
                pass
        total = 0
        written = 0
        manifest_lines: list[str] = []
        label = query.strip()
        # Rough capacity estimate: 1MB per file
        est_gb = (limit * 1_000_000) / float(1024 ** 3)
        if not can_store_additional_gb(est_gb):
            print(json.dumps({"stored": False, "reason": "cap_exceeded", "cap_gb": float(datasets_storage_cap_gb())}, ensure_ascii=False))
            return
        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {"User-Agent": f"aios-bot/0.1{ua_suffix}", "Accept-Language": "en-US,en;q=0.7"}
        async with aiohttp.ClientSession(headers=headers) as session:
            idx = 1
            for site in sites:
                if total >= int(limit):
                    break
                html = await _fetch_html(session, site.url)
                if not html:
                    continue
                links = _extract_file_links(html, site.url, allowed_exts_list)
                if progress:
                    try:
                        print(json.dumps({"event": "page", "url": site.url, "links_found": len(links)}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                count_this_site = 0
                for u in links:
                    if total >= int(limit):
                        break
                    if count_this_site >= int(per_site):
                        break
                    res = await _download_file(session, u, idx)
                    idx += 1
                    if not res:
                        continue
                    fn, nbytes = res
                    total += 1
                    count_this_site += 1
                    written += nbytes
                    rec = {
                        "path": str(Path("files") / fn),
                        "label": label,
                        "source_url": u,
                        "page_url": site.url,
                        "bytes": int(nbytes),
                    }
                    manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                    if progress:
                        try:
                            print(json.dumps({"event": "file", "downloaded": total, "target": int(limit), "file": fn}, ensure_ascii=False), flush=True)
                        except Exception:
                            pass
        with open(manifest_path, "w", encoding="utf-8") as f:
            for ln in manifest_lines:
                f.write(ln)
        out = {
            "stored": True,
            "dataset_path": str(dataset_root),
            "files": int(total),
            "wrote_bytes": int(written),
            "manifest": str(manifest_path),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    asyncio.run(_run())
