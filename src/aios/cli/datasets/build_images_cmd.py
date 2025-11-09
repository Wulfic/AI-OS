from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from collections import deque
from io import BytesIO
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


def datasets_build_images(
    query: str = typer.Argument(..., help="What images to collect, e.g., 'boats'"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="Dataset name; defaults to slugified query under 'images'"),
    limit: int = typer.Option(200, "--max-images", help="Maximum number of images to collect"),
    per_site: int = typer.Option(40, "--per-site", help="Maximum images per site"),
    pages_per_site: int = typer.Option(10, "--pages-per-site", help="Maximum pages to crawl per site (BFS)"),
    search_results: int = typer.Option(10, "--search-results", help="Number of top search results (sites) to crawl"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing dataset directory if present"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Emit JSONL progress events"),
    rps: float = typer.Option(2.0, "--rps", help="Requests per second cap (overall)"),
    min_bytes: int = typer.Option(8192, "--min-bytes", help="Skip images smaller than this many bytes"),
    allow_ext: str = typer.Option("", "--allow-ext", help="Comma-separated allowed image extensions (e.g., jpg,png,webp). If set, only these will be downloaded."),
    near_duplicate_threshold: int = typer.Option(0, "--near-duplicate-threshold", min=0, max=64, help="Perceptual near-duplicate Hamming distance threshold (0-64). 0 disables."),
    file_prefix: Optional[str] = typer.Option(None, "--file-prefix", help="Filename prefix for saved images; defaults to search query or store-dataset name."),
):
    """Autonomously search the web and build an image dataset for a single label.

    - Uses DuckDuckGo lite search to find sites for the query
    - Crawls each site front page for <img> tags and downloads images
    - Stores images and a manifest.jsonl with fields: path, label, source_url, page_url, title, alt
    - Enforces dataset storage cap and per-site limits
    """

    def _slug(s: str) -> str:
        import re
        s2 = re.sub(r"\s+", "_", s.strip().lower())
        s2 = re.sub(r"[^a-z0-9_\-]", "", s2)
        return s2 or "dataset"

    dataset_name = (store_dataset or _slug(query))
    dataset_root = datasets_base_dir() / "images" / dataset_name
    img_dir = dataset_root
    manifest_path = dataset_root / "manifest.jsonl"
    try:
        if dataset_root.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_root, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print({"started": False, "error": f"cannot create dataset dir: {e}"})
        return

    # Simple throttling via semaphore and sleep between requests
    sem = asyncio.Semaphore(max(1, int(rps)) if rps and rps > 0 else 5)

    async def _fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = resp.headers.get("Content-Type", "")
                    if "text/html" in ctype:
                        return await resp.text()
                    return None
        except Exception:
            return None

    # Parse allowed extensions and validate for image types
    allowed_exts = [x.strip().lower() for x in (allow_ext or "").split(",") if x.strip()]
    allowed_exts = [x if x.startswith(".") else f".{x}" for x in allowed_exts]
    valid_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if allowed_exts:
        invalid = [e for e in allowed_exts if e not in valid_image_exts]
        if invalid:
            print(json.dumps({
                "stored": False,
                "error": "invalid_extensions",
                "message": "Only image extensions are allowed for images dataset",
                "invalid": invalid,
                "allowed": sorted(list(valid_image_exts)),
            }, ensure_ascii=False))
            return

    def _extract_images(html: str, base_url: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
            from bs4.element import Tag  # type: ignore
            for el in soup.find_all("img"):
                if not isinstance(el, Tag):
                    continue
                raw_src = el.get("src")
                src = str(raw_src) if raw_src is not None else ""
                if not src:
                    continue
                if src.startswith("data:"):
                    continue
                from urllib.parse import urljoin
                u = urljoin(base_url, src)
                raw_alt = el.get("alt")
                raw_title = el.get("title")
                alt = (str(raw_alt).strip() if raw_alt is not None else "")
                title = (str(raw_title).strip() if raw_title is not None else "")
                out.append({"url": u, "alt": alt, "title": title})
                # Also parse srcset candidates to be more aggressive
                srcset = el.get("srcset")
                if srcset:
                    parts = [p.strip().split(" ")[0] for p in str(srcset).split(",") if p.strip()]
                    for cand in parts:
                        cu = urljoin(base_url, cand)
                        out.append({"url": cu, "alt": alt, "title": title})
        except Exception:
            return []
        return out

    def _extract_links(html: str, base_url: str) -> list[str]:
        """Extract internal page links for BFS crawling (same host)."""
        try:
            soup = bs4.BeautifulSoup(html, "lxml")
            from bs4.element import Tag  # type: ignore
            from urllib.parse import urljoin, urlparse
            base_host = urlparse(base_url).netloc
            links: list[str] = []
            for a in soup.find_all("a"):
                if not isinstance(a, Tag):
                    continue
                href = a.get("href")
                if not href:
                    continue
                u = urljoin(base_url, str(href))
                pu = urlparse(u)
                if pu.scheme not in ("http", "https"):
                    continue
                if pu.netloc != base_host:
                    continue
                # Heuristic: skip obvious binaries
                if any(pu.path.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf", ".zip")):
                    continue
                links.append(u)
            # dedupe preserve order
            seen = set()
            uniq = []
            for u in links:
                if u in seen:
                    continue
                seen.add(u)
                uniq.append(u)
            return uniq
        except Exception:
            return []

    def _phash_from_bytes(data: bytes) -> Optional[int]:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None
        try:
            with Image.open(BytesIO(data)) as im:
                im = im.convert("L")
                try:
                    # Pillow >=9.1
                    from PIL import Image as _Image
                    im = im.resize((8, 8), resample=_Image.Resampling.LANCZOS)  # type: ignore[attr-defined]
                except Exception:
                    im = im.resize((8, 8))
                pixels = list(im.getdata())
            avg = sum(pixels) / 64.0
            bits = 0
            for i, p in enumerate(pixels):
                if p >= avg:
                    bits |= (1 << i)
            return bits
        except Exception:
            return None

    async def _download_image(session: aiohttp.ClientSession, url: str) -> Optional[tuple[str, bytes, int, str, Optional[int]]]:
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                    if resp.status >= 400:
                        return None
                    ctype = resp.headers.get("Content-Type", "").lower()
                    if not any(x in ctype for x in ("image/jpeg", "image/png", "image/webp", "image/jpg", "image/gif")):
                        # allow based on extension as fallback
                        if not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                            return None
                    data = await resp.read()
                    if len(data) < int(min_bytes):
                        return None
                    # determine extension
                    ext = ".jpg"
                    if ".png" in ctype or url.lower().endswith(".png"):
                        ext = ".png"
                    elif ".webp" in ctype or url.lower().endswith(".webp"):
                        ext = ".webp"
                    elif "/gif" in ctype or url.lower().endswith(".gif"):
                        ext = ".gif"
                    elif "/jpeg" in ctype or url.lower().endswith((".jpg", ".jpeg")):
                        ext = ".jpg"
                    # If filtering by extension, enforce now that we know ext
                    if allowed_exts and ext not in allowed_exts:
                        return None
                    # Optional perceptual hash to filter near-duplicates
                    ph = None
                    if int(near_duplicate_threshold) > 0:
                        ph = _phash_from_bytes(data)
                    # filename is deterministic by content hash to dedupe
                    h = hashlib.sha256(data).hexdigest()[:16]
                    fn = f"{h}{ext}"
                    fp = img_dir / fn
                    # Skip if we've already seen this content or file exists
                    try:
                        if fp.exists():
                            return None
                    except Exception:
                        pass
                    # Defer writing until after near-duplicate check in caller; return data
                    return (fn, data, len(data), h, ph)
        except Exception:
            return None

    async def _run():
        # Search (with optional expansion if target not met)
        current_limit = max(1, int(search_results))
        seen_site_urls: set[str] = set()
        # Crawl sites, expanding search results until we hit the target or a cap
        total_downloaded = 0
        written_bytes = 0
        manifest_lines: list[str] = []
        label = query.strip()
        # determine filename prefix
        prefix = (file_prefix or dataset_name or _slug(query)).strip()
        if prefix:
            # sanitize prefix to safe filename chunk
            import re as _re
            prefix = _re.sub(r"[^a-zA-Z0-9_\-]+", "_", prefix).strip("_-")
        # capacity pre-check (rough estimate 200KB per image); best-effort
        est_gb = (limit * 200_000) / float(1024 ** 3)
        if not can_store_additional_gb(est_gb):
            print(json.dumps({"stored": False, "reason": "cap_exceeded", "cap_gb": float(datasets_storage_cap_gb())}, ensure_ascii=False))
            return
        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {"User-Agent": f"aios-bot/0.1{ua_suffix}", "Accept-Language": "en-US,en;q=0.7"}
        async with aiohttp.ClientSession(headers=headers) as session:
            img_idx = 1
            seen_hashes: set[str] = set()
            seen_img_urls: set[str] = set()
            seen_phashes: list[int] = []
            search_cap = 50
            while total_downloaded < int(limit) and current_limit <= search_cap:
                sites = await ddg_search(query, limit=current_limit)
                # Filter out sites already seen
                new_sites = [s for s in sites if s.url not in seen_site_urls]
                seen_site_urls.update(s.url for s in new_sites)
                if progress:
                    try:
                        print(json.dumps({"event": "search", "query": query, "sites": [s.url for s in new_sites]}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
                for site in new_sites:
                    if total_downloaded >= int(limit):
                        break
                    # BFS crawl within site
                    visited_pages: set[str] = set()
                    queue = deque([site.url])
                    pages_crawled = 0
                    count_this_site = 0
                    while queue and pages_crawled < int(pages_per_site) and total_downloaded < int(limit):
                        page_url = queue.popleft()
                        if page_url in visited_pages:
                            continue
                        visited_pages.add(page_url)
                        html = await _fetch_url(session, page_url)
                        if not html:
                            continue
                        pages_crawled += 1
                        # enqueue new internal links
                        for link in _extract_links(html, page_url):
                            if link not in visited_pages:
                                queue.append(link)
                        # extract images on this page
                        imgs = _extract_images(html, page_url)
                        if progress:
                            try:
                                print(json.dumps({"event": "page", "url": page_url, "images_found": len(imgs)}, ensure_ascii=False), flush=True)
                            except Exception:
                                pass
                        for info in imgs:
                            if total_downloaded >= int(limit):
                                break
                            if count_this_site >= int(per_site):
                                break
                            # Skip if we've already attempted/saved this URL
                            uln = (info["url"] or "").strip().lower()
                            if uln in seen_img_urls:
                                continue
                            res = await _download_image(session, info["url"])
                            img_idx += 1
                            if not res:
                                continue
                            fn, data, nbytes, h, ph = res
                            if prefix:
                                # replace default fn with prefixed form preserving extension
                                ext = Path(fn).suffix
                                fn = f"{prefix}_{h}{ext}"
                            if h in seen_hashes:
                                continue
                            # If near-duplicate filtering is enabled and phash computed, compare
                            if int(near_duplicate_threshold) > 0 and ph is not None:
                                is_near_dup = False
                                for oph in seen_phashes:
                                    d = (ph ^ oph).bit_count()
                                    if d <= int(near_duplicate_threshold):
                                        is_near_dup = True
                                        break
                                if is_near_dup:
                                    continue
                            # Commit file now that it passed checks
                            try:
                                with open(img_dir / fn, "wb") as f:
                                    f.write(data)
                            except Exception:
                                # If writing fails, skip accounting
                                continue
                            seen_hashes.add(h)
                            seen_img_urls.add(uln)
                            if ph is not None:
                                seen_phashes.append(ph)
                            total_downloaded += 1
                            count_this_site += 1
                            written_bytes += nbytes
                            rec = {
                                "path": fn,
                                "label": label,
                                "source_url": info["url"],
                                "page_url": page_url,
                                "title": info.get("title", ""),
                                "alt": info.get("alt", ""),
                            }
                            manifest_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                            if progress:
                                try:
                                    print(json.dumps({"event": "image", "downloaded": total_downloaded, "target": int(limit), "file": fn}, ensure_ascii=False), flush=True)
                                except Exception:
                                    pass
                # If still not enough, expand search results and try again
                if total_downloaded < int(limit):
                    current_limit = min(search_cap, max(current_limit + int(search_results), current_limit * 2))
        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            for ln in manifest_lines:
                f.write(ln)
        out = {
            "stored": True,
            "dataset_path": str(dataset_root),
            "images": int(total_downloaded),
            "wrote_bytes": int(written_bytes),
            "manifest": str(manifest_path),
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)

    asyncio.run(_run())
