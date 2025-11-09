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


def datasets_build_websites(
    query: str = typer.Argument(..., help="What websites to crawl and snapshot, e.g., 'boats'"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="Dataset name; defaults to slugified query under 'websites'"),
    max_pages: int = typer.Option(50, "--max-pages", help="Maximum number of pages to collect"),
    per_site: int = typer.Option(10, "--per-site", help="Maximum pages per site"),
    search_results: int = typer.Option(10, "--search-results", help="Number of top search results (sites) to crawl"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset directory if present"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Emit JSONL progress events"),
    rps: float = typer.Option(2.0, "--rps", help="Requests per second cap (overall)"),
    min_bytes: int = typer.Option(2000, "--min-bytes", help="Skip pages smaller than this many bytes"),
):
    """Fetch top-site landing pages as HTML snapshots and write a manifest.

    - Uses DuckDuckGo lite search to find sites for the query
    - Fetches each site's landing page and stores raw HTML under pages/
    - Manifest includes: path, url, title, bytes, links
    """

    def _slug(s: str) -> str:
        import re
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        s2 = re.sub(r"[^a-z0-9_\-]", "", s2)
        return s2 or "dataset"

    dataset_root = datasets_base_dir() / "websites" / (store_dataset or _slug(query))
    pages_dir = dataset_root / "pages"
    manifest_path = dataset_root / "manifest.jsonl"
    try:
        if dataset_root.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_root, ignore_errors=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print({"started": False, "error": f"cannot create dataset dir: {e}"})
        return

    sem = asyncio.Semaphore(max(1, int(rps)) if rps and rps > 0 else 5)

    async def _fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = (resp.headers.get("Content-Type", "") or "").lower()
                    if "text/html" in ctype or "application/xhtml+xml" in ctype:
                        data = await resp.read()
                        return data
                    return None
        except Exception:
            return None

    def _title_and_links(html: str) -> tuple[str, int]:
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
        except Exception:
            return ("", 0)
        title = ""
        try:
            if soup.title and soup.title.string:
                title = str(soup.title.string).strip()
        except Exception:
            pass
        nlinks = 0
        try:
            nlinks = len(soup.find_all("a"))
        except Exception:
            nlinks = 0
        return (title, nlinks)

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
        est_gb = (max_pages * 100_000) / float(1024 ** 3)
        if not can_store_additional_gb(est_gb):
            print(json.dumps({"stored": False, "reason": "cap_exceeded", "cap_gb": float(datasets_storage_cap_gb())}, ensure_ascii=False))
            return
        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {"User-Agent": f"aios-bot/0.1{ua_suffix}", "Accept-Language": "en-US,en;q=0.7"}
        async with aiohttp.ClientSession(headers=headers) as session:
            page_idx = 1
            for site in sites:
                if total >= int(max_pages):
                    break
                # Just fetch landing page per site for now
                data = await _fetch_url(session, site.url)
                if not data or len(data) < int(min_bytes):
                    continue
                import hashlib as _hl
                h = _hl.sha256(data).hexdigest()[:16]
                fn = f"{page_idx:06d}_{h}.html"
                fp = pages_dir / fn
                try:
                    with open(fp, "wb") as f:
                        f.write(data)
                except Exception:
                    continue
                page_idx += 1
                total += 1
                written += len(data)
                title, nlinks = _title_and_links(data.decode("utf-8", errors="ignore"))
                rec = {
                    "path": str(Path("pages") / fn),
                    "url": site.url,
                    "title": title,
                    "bytes": len(data),
                    "links": int(nlinks),
                }
                manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                if progress:
                    try:
                        print(json.dumps({"event": "html", "downloaded": total, "target": int(max_pages), "file": fn}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
        with open(manifest_path, "w", encoding="utf-8") as f:
            for ln in manifest_lines:
                f.write(ln)
        out = {
            "stored": True,
            "dataset_path": str(dataset_root),
            "pages": int(total),
            "wrote_bytes": int(written),
            "manifest": str(manifest_path),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    asyncio.run(_run())
