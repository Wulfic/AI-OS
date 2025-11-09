from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.memory.store import init_db
from aios.tools.crawler import Crawler
from aios.data.datasets import datasets_storage_cap_gb


def crawl(
    url: str = typer.Argument(..., help="URL to crawl"),
    db: Optional[str] = typer.Option(None, help="Path to SQLite DB file (defaults to ~/.local/share/aios/aios.db)"),
    no_robots: bool = typer.Option(False, help="Ignore robots.txt (for testing only)"),
    ttl_sec: int = typer.Option(0, help="Return cached page if fetched within this many seconds; 0 disables TTL"),
    render: bool = typer.Option(False, help="Use Playwright to render the page before extraction"),
    trafilatura: bool = typer.Option(False, help="Use trafilatura for article-like text extraction"),
    recursive: bool = typer.Option(False, "--recursive/--no-recursive", help="Enable recursive crawl (BFS)"),
    max_pages: int = typer.Option(50, "--max-pages", help="Max pages to fetch when --recursive"),
    max_depth: int = typer.Option(2, "--max-depth", help="Max link depth from the root when --recursive"),
    same_domain: bool = typer.Option(True, "--same-domain/--any-domain", help="Restrict traversal to the root domain when --recursive"),
    store_dataset: Optional[str] = typer.Option(None, "--store-dataset", help="If provided, append crawled pages into this dataset (JSONL) under the configured datasets pool"),
    overwrite: bool = typer.Option(False, "--overwrite", help="When storing dataset, overwrite the dataset file instead of appending"),
    rps: float = typer.Option(0.0, "--rps", help="Requests per second cap for crawling (0 disables; e.g., 2.0 = 2 req/s)"),
    delay_ms: int = typer.Option(0, "--delay-ms", help="Fixed delay in milliseconds between page fetches (overrides --rps if > 0)"),
    progress: bool = typer.Option(False, "--progress", help="Emit per-page JSON progress lines to stdout while crawling"),
):
    cfg = load_config(None)
    setup_logging(cfg)
    conn = (sqlite3.connect(db) if db else sqlite3.connect(str(Path.home() / ".local/share/aios/aios.db")))
    init_db(conn)

    async def _run():
        # Compute throttle: delay_ms wins if provided; otherwise derive from rps
        min_delay_sec: float
        if delay_ms and delay_ms > 0:
            min_delay_sec = max(0.0, float(delay_ms) / 1000.0)
        elif rps and rps > 0:
            try:
                min_delay_sec = max(0.0, 1.0 / float(rps))
            except Exception:
                min_delay_sec = 0.0
        else:
            min_delay_sec = 0.0

        crawler = Crawler(
            conn,
            respect_robots=not no_robots,
            ttl_sec=ttl_sec,
            render=render,
            use_trafilatura=trafilatura,
            min_delay_sec=min_delay_sec,
        )
        pages: list
        if recursive or store_dataset:
            async def _on_page(p, n, mx):
                if progress:
                    try:
                        rec = {"event": "page", "n": int(n), "max": int(mx), "url": p.url, "title": p.title, "chars": len(p.text)}
                        print(json.dumps(rec, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
            pages = await crawler.crawl_site(
                url,
                max_pages=max(1, int(max_pages)),
                max_depth=max(0, int(max_depth)),
                same_domain=bool(same_domain),
                on_page=_on_page,
            )
        else:
            pg = await crawler.fetch_and_parse(url)
            if progress:
                try:
                    rec = {"event": "page", "n": 1, "max": 1, "url": pg.url, "title": pg.title, "chars": len(pg.text)}
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
                except Exception:
                    pass
            pages = [pg]
        stored = False
        dataset_path: Optional[str] = None
        wrote_bytes = 0
        if store_dataset:
            try:
                from aios.data.datasets import datasets_base_dir, can_store_additional_gb
                base = datasets_base_dir()
                ds_dir = base / str(store_dataset)
                ds_dir.mkdir(parents=True, exist_ok=True)
                outp = ds_dir / "data.jsonl"
                encoded_lines = []
                total_bytes = 0
                for p in pages:
                    rec = {"url": p.url, "title": p.title, "fetched_ts": p.fetched_ts, "hash": p.hash, "text": p.text}
                    b = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
                    encoded_lines.append(b)
                    total_bytes += len(b)
                req_gb = total_bytes / float(1024 ** 3)
                if not can_store_additional_gb(req_gb):
                    print(
                        json.dumps(
                            {
                                "stored": False,
                                "reason": "cap_exceeded",
                                "would_write_bytes": total_bytes,
                                "cap_gb": float(datasets_storage_cap_gb()),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    return
                mode = "wb" if overwrite else ("ab" if outp.exists() else "wb")
                with open(outp, mode) as f:
                    for b in encoded_lines:
                        f.write(b)
                        wrote_bytes += len(b)
                stored = True
                dataset_path = str(outp)
            except Exception as e:
                print(json.dumps({"stored": False, "error": str(e)}, ensure_ascii=False), flush=True)
                return
        total_chars = int(sum(len(p.text) for p in pages))
        print(
            json.dumps(
                {
                    "pages": [
                        {
                            "url": p.url,
                            "title": p.title,
                            "chars": len(p.text),
                            "hash": p.hash,
                            "ts": p.fetched_ts,
                        }
                        for p in pages[:50]
                    ],
                    "count": len(pages),
                    "total_chars": total_chars,
                    "stored": stored,
                    "dataset_path": dataset_path,
                    "wrote_bytes": wrote_bytes if stored else 0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        asyncio.run(_run())
    finally:
        conn.close()


def register(app: typer.Typer) -> None:
    app.command()(crawl)
