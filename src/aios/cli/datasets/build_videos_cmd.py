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


def datasets_build_videos(
    query: str = typer.Argument(..., help="What videos to collect, e.g., 'boats'"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="Dataset name; defaults to slugified query under 'videos'"),
    limit: int = typer.Option(50, "--max-videos", help="Maximum number of videos to collect"),
    per_site: int = typer.Option(10, "--per-site", help="Maximum videos per site"),
    search_results: int = typer.Option(10, "--search-results", help="Number of top search results (sites) to crawl"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset directory if present"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Emit JSONL progress events"),
    rps: float = typer.Option(2.0, "--rps", help="Requests per second cap (overall)"),
    min_bytes: int = typer.Option(50_000, "--min-bytes", help="Skip videos smaller than this many bytes"),
    allow_ext: str = typer.Option("", "--allow-ext", help="Comma-separated allowed video extensions (e.g., mp4,webm,mov,m4v). If set, only these will be downloaded."),
    file_prefix: Optional[str] = typer.Option(None, "--file-prefix", help="Filename prefix for saved videos; defaults to search query or store-dataset name."),
):
    def _slug(s: str) -> str:
        import re
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        s2 = re.sub(r"[^a-z0-9_\-]", "", s2)
        return s2 or "dataset"

    dataset_name = (store_dataset or _slug(query))
    dataset_root = datasets_base_dir() / "videos" / dataset_name
    vid_dir = dataset_root
    manifest_path = dataset_root / "manifest.jsonl"
    # Determine filename prefix (default to dataset name or slug of query)
    prefix: str = (file_prefix or dataset_name or _slug(query)).strip()
    if prefix:
        import re as _re
        prefix = _re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix).strip("_-")
    try:
        if dataset_root.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_root, ignore_errors=True)
        vid_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print({"started": False, "error": f"cannot create dataset dir: {e}"})
        return

    # Parse and validate allowed extensions for video types
    allowed_exts = [x.strip().lower() for x in (allow_ext or "").split(",") if x.strip()]
    allowed_exts = [x if x.startswith(".") else f".{x}" for x in allowed_exts]
    valid_video_exts = {".mp4", ".webm", ".m4v", ".mov"}
    if allowed_exts:
        invalid = [e for e in allowed_exts if e not in valid_video_exts]
        if invalid:
            print(json.dumps({
                "stored": False,
                "error": "invalid_extensions",
                "message": "Only video extensions are allowed for videos dataset",
                "invalid": invalid,
                "allowed": sorted(list(valid_video_exts)),
            }, ensure_ascii=False))
            return

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

    def _extract_video_links(html: str, base_url: str) -> list[str]:
        out: list[str] = []
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
            from bs4.element import Tag  # type: ignore
            from urllib.parse import urljoin
            # <video src> and <source src>
            for el in soup.find_all(["video", "source", "a"]):
                if not isinstance(el, Tag):
                    continue
                src = el.get("src") or el.get("href")
                if not src:
                    continue
                u = urljoin(base_url, str(src))
                ln = u.lower()
                if any(ln.endswith(ext) for ext in (".mp4", ".webm", ".m4v", ".mov")):
                    if allowed_exts and not any(ln.endswith(ext) for ext in allowed_exts):
                        continue
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

    async def _download_video(session: aiohttp.ClientSession, url: str, idx: int) -> Optional[tuple[str, int]]:
        # Stream to temp file, then rename based on content hash
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = (resp.headers.get("Content-Type", "") or "").lower()
                    if not ("video/" in ctype or url.lower().endswith((".mp4", ".webm", ".m4v", ".mov"))):
                        return None
                    # choose extension
                    ext = ".mp4"
                    if ".webm" in ctype or url.lower().endswith(".webm"):
                        ext = ".webm"
                    elif "quicktime" in ctype or url.lower().endswith(".mov"):
                        ext = ".mov"
                    elif url.lower().endswith(".m4v"):
                        ext = ".m4v"
                    # Enforce allowlist if provided
                    if allowed_exts and ext not in allowed_exts:
                        return None
                    tmp = vid_dir / f"tmp_{idx:06d}.part"
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
                    base = f"{digest}{ext}"
                    fn = f"{prefix}_{base}" if prefix else base
                    fp = vid_dir / fn
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
        total_downloaded = 0
        written_bytes = 0
        manifest_lines: list[str] = []
        label = query.strip()
        # Rough capacity estimate: 5MB per video
        est_gb = (limit * 5_000_000) / float(1024 ** 3)
        if not can_store_additional_gb(est_gb):
            print(json.dumps({"stored": False, "reason": "cap_exceeded", "cap_gb": float(datasets_storage_cap_gb())}, ensure_ascii=False))
            return
        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {"User-Agent": f"aios-bot/0.1{ua_suffix}", "Accept-Language": "en-US,en;q=0.7"}
        async with aiohttp.ClientSession(headers=headers) as session:
            vid_idx = 1
            for site in sites:
                if total_downloaded >= int(limit):
                    break
                html = await _fetch_html(session, site.url)
                if not html:
                    continue
                links = _extract_video_links(html, site.url)
                if progress:
                    try:
                        print(json.dumps({"event": "page", "url": site.url, "links_found": len(links)}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                count_this_site = 0
                for u in links:
                    if total_downloaded >= int(limit):
                        break
                    if count_this_site >= int(per_site):
                        break
                    res = await _download_video(session, u, vid_idx)
                    vid_idx += 1
                    if not res:
                        continue
                    fn, nbytes = res
                    total_downloaded += 1
                    count_this_site += 1
                    written_bytes += nbytes
                    rec = {
                        "path": fn,
                        "label": label,
                        "source_url": u,
                        "page_url": site.url,
                        "bytes": int(nbytes),
                    }
                    manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                    if progress:
                        try:
                            print(json.dumps({"event": "video", "downloaded": total_downloaded, "target": int(limit), "file": fn}, ensure_ascii=False), flush=True)
                        except Exception:
                            pass
        with open(manifest_path, "w", encoding="utf-8") as f:
            for ln in manifest_lines:
                f.write(ln)
        out = {
            "stored": True,
            "dataset_path": str(dataset_root),
            "videos": int(total_downloaded),
            "wrote_bytes": int(written_bytes),
            "manifest": str(manifest_path),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    asyncio.run(_run())
