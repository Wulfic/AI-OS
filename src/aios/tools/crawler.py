from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Iterable, List, Set, Deque, Callable, Awaitable
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque
from urllib.robotparser import RobotFileParser

# Optional trafilatura import at module scope for easier testing/monkeypatching
try:  # pragma: no cover - import guard
    import trafilatura as _TRAFILATURA  # type: ignore
except Exception:  # pragma: no cover - absent in some envs
    _TRAFILATURA = None

import bs4  # type: ignore

# (no re-exports needed)


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_ts: str  # ISO8601 UTC string
    hash: str
    meta: Dict[str, object]


def _utc_now() -> str:
    # Persist UTC timestamps in DB
    return datetime.now(timezone.utc).isoformat()


def parse_html_to_text(html: str, base_url: Optional[str] = None) -> Tuple[str, str]:
    """Extract a simple title and readable text from HTML.

    - Uses BeautifulSoup to get <title>
    - Uses a lightweight extraction strategy for text content suitable for tests
    """
    soup = bs4.BeautifulSoup(html, "lxml")
    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # Heuristic: drop script/style and get visible text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return title, text


def extract_links(html: str, base_url: Optional[str]) -> List[str]:
    """Extract absolute http(s) links from HTML using BeautifulSoup.

    - Resolves relative hrefs against base_url
    - Drops fragments and non-http(s) schemes
    - Deduplicates while preserving order (first occurrence wins)
    """
    if not base_url:
        base = ""
    else:
        base = base_url
    soup = bs4.BeautifulSoup(html, "lxml")
    seen: Set[str] = set()
    out: List[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        absu = urljoin(base, href)
        # Drop fragment
        absu, _frag = urldefrag(absu)
        parsed = urlparse(absu)
        if parsed.scheme not in ("http", "https"):
            continue
        if absu in seen:
            continue
        seen.add(absu)
        out.append(absu)
    return out


class Crawler:
    # Class-level attribute declarations for static analysis
    _robots_cache: Dict[str, RobotFileParser]
    _robots_delay_cache: Dict[str, float]

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        user_agent: str = "aios-bot/0.1",
        respect_robots: bool = True,
        timeout_sec: int = 20,
        ttl_sec: int = 0,
        render: bool = False,
        use_trafilatura: bool = False,
        min_delay_sec: float = 0.0,
    ) -> None:
        # Basic config
        self.conn = conn
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.timeout_sec = timeout_sec
        self.ttl_sec = ttl_sec
        self.render = render
        self.use_trafilatura = use_trafilatura
        # Simple global rate limiter: wait at least this delay between page fetches in crawl_site
        # 0 disables throttling; set to e.g. 0.25 for 4 req/s, 1.0 for 1 req/s
        try:
            self.min_delay_sec = max(0.0, float(min_delay_sec))
        except Exception:
            self.min_delay_sec = 0.0
        # Cache of robots.txt per origin
        self._robots_cache = {}
        # Optional cache for computed crawl-delays per origin (seconds)
        self._robots_delay_cache = {}

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    async def can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = urljoin(base, "/robots.txt")
        rfp = self._robots_cache.get(robots_url)
        if rfp is None:
            # Fetch robots.txt lazily; keep network out of unit tests by allowing caller to disable
            import aiohttp

            rfp = RobotFileParser()
            try:
                async with aiohttp.ClientSession(
                    headers={"User-Agent": self.user_agent}
                ) as session:
                    async with session.get(
                        robots_url, timeout=aiohttp.ClientTimeout(total=self.timeout_sec)
                    ) as resp:
                        if resp.status >= 400:
                            # Treat missing robots as allow by default
                            rfp.parse("")
                        else:
                            body = await resp.text()
                            rfp.parse(body.splitlines())
            except Exception:
                # Network failures → be conservative: disallow
                rfp.parse("User-agent: *\nDisallow: /".splitlines())
            self._robots_cache[robots_url] = rfp
            # Cache crawl-delay if provided
            try:
                cd = rfp.crawl_delay(self.user_agent) or rfp.crawl_delay("*")
                if cd is not None:
                    self._robots_delay_cache[base] = float(cd)
            except Exception:
                pass
        return rfp.can_fetch(self.user_agent, url)

    def robots_crawl_delay(self, url: str) -> float:
        """Return robots.txt Crawl-delay for the URL's origin if known, else 0.

        Only populated if respect_robots is True and robots.txt has been fetched.
        """
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            val = self._robots_delay_cache.get(base)
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    def get_cached(self, url: str) -> Optional[Page]:
        cur = self.conn.execute(
            "SELECT url, title, fetched_ts, text, hash, meta_json FROM pages WHERE url = ?",
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return None
        meta = json.loads(row[5]) if row[5] else {}
        return Page(
            url=row[0],
            title=row[1] or "",
            fetched_ts=row[2] or "",
            text=row[3] or "",
            hash=row[4] or "",
            meta=meta,
        )

    def cache_page(self, page: Page) -> None:
        self.conn.execute(
            """
            INSERT INTO pages (url, title, fetched_ts, text, hash, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                fetched_ts=excluded.fetched_ts,
                text=excluded.text,
                hash=excluded.hash,
                meta_json=excluded.meta_json
            """,
            (
                page.url,
                page.title,
                page.fetched_ts,
                page.text,
                page.hash,
                json.dumps(page.meta),
            ),
        )
        self.conn.commit()

    async def fetch(self, url: str) -> bytes:
        import aiohttp
        import os

        ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX") or "; locale=en-US"
        headers = {
            "User-Agent": f"{self.user_agent}{ua_suffix}",
            "Accept-Language": "en-US,en;q=0.7",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=self.timeout_sec)
            ) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def render_html(self, url: str) -> str:
        """Render a page with Playwright and return HTML.

        Notes:
        - Requires Playwright browsers installed (e.g., `playwright install --with-deps`).
        - Tests monkeypatch this method to avoid launching a browser.
        """
        from playwright.async_api import async_playwright

        # Playwright handles its own timeouts; apply a sensible overall timeout
        timeout_ms = max(1, int(self.timeout_sec * 1000))
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(user_agent=self.user_agent)
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                # Give a little idle time for dynamic content if possible, bounded by timeout
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=min(2000, timeout_ms)
                    )
                except Exception:
                    pass
                content = await page.content()
                await context.close()
                return content
            finally:
                await browser.close()

    async def fetch_and_parse(self, url: str) -> Page:
        # Return cached if fresh
        if self.ttl_sec > 0:
            cached = self.get_cached(url)
            if cached:
                try:
                    ts = datetime.fromisoformat(cached.fetched_ts)
                except Exception:
                    ts = None
                if ts is not None:
                    age = (datetime.now(timezone.utc) - ts).total_seconds()
                    if age >= 0 and age < self.ttl_sec:
                        return cached
        if not await self.can_fetch(url):
            raise PermissionError(f"robots.txt forbids fetching: {url}")
        html: str
        if self.render:
            html = await self.render_html(url)
        else:
            body = await self.fetch(url)
            html = body.decode("utf-8", errors="ignore")

        # Try trafilatura if enabled; otherwise fallback to simple parsing
        title: str
        text: str
        if self.use_trafilatura and _TRAFILATURA is not None:
            try:
                # Trafilatura extract gives article text; we still take <title> via BeautifulSoup
                extracted = _TRAFILATURA.extract(html) or ""
                t, simple_text = parse_html_to_text(html, base_url=url)
                title = t
                text = extracted.strip() or simple_text
            except Exception:
                title, text = parse_html_to_text(html, base_url=url)
        else:
            title, text = parse_html_to_text(html, base_url=url)
        # Extract links for potential recursive crawling; store in meta
        try:
            links = extract_links(html, base_url=url)
        except Exception:
            links = []
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        page = Page(
            url=url, title=title, text=text, fetched_ts=_utc_now(), hash=h, meta={"links": links}
        )
        self.cache_page(page)
        return page

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for de-duplication: strip fragments, keep scheme/host/path/query."""
        u, _frag = urldefrag(url)
        return u

    def _same_domain(self, a: str, b: str) -> bool:
        try:
            return self._domain(a) == self._domain(b)
        except Exception:
            return False

    async def crawl_site(
        self,
        root_url: str,
        *,
        max_pages: int = 50,
        max_depth: int = 2,
        same_domain: bool = True,
        on_page: Optional[Callable[[Page, int, int], Awaitable[None] | None]] = None,
    ) -> List[Page]:
        """Breadth-first crawl starting at root_url.

        - Respects robots.txt for each URL (unless disabled in constructor)
        - Stores each fetched Page in SQLite (pages table)
        - Returns the list of Page objects in crawl order
        - Limits traversal by max_pages and max_depth
        - If same_domain, only follows links with the same netloc as root_url
        """
        start = self._normalize_url(root_url)
        pages: List[Page] = []
        visited: Set[str] = set()
        enqueued: Set[str] = set([start])
        q: Deque[Tuple[str, int]] = deque([(start, 0)])
        # Local import to avoid import cycle; used only when throttling
        import asyncio
        import inspect

        while q and len(pages) < max(1, int(max_pages)):
            url, depth = q.popleft()
            url_n = self._normalize_url(url)
            if url_n in visited:
                continue
            # Enforce domain restriction at visit time as well (root always allowed)
            if depth > 0 and same_domain and not self._same_domain(root_url, url_n):
                continue
            # Polite rate limit BEFORE fetching as well
            effective_delay_pre = max(self.min_delay_sec, self.robots_crawl_delay(url_n))
            if effective_delay_pre > 0:
                try:
                    await asyncio.sleep(effective_delay_pre)
                except Exception:
                    pass
            # Fetch and parse; skip on errors or robots disallow
            try:
                page = await self.fetch_and_parse(url_n)
            except PermissionError:
                visited.add(url_n)
                continue
            except Exception:
                visited.add(url_n)
                continue
            pages.append(page)
            visited.add(url_n)
            # Emit progress to callback if provided
            if on_page is not None:
                try:
                    res = on_page(page, len(pages), max_pages)
                    if inspect.isawaitable(res):
                        await res  # type: ignore[func-returns-value]
                except Exception:
                    pass
            # Polite rate limit AFTER fetching as well
            effective_delay = max(self.min_delay_sec, self.robots_crawl_delay(url_n))
            if effective_delay > 0:
                try:
                    await asyncio.sleep(effective_delay)
                except Exception:
                    pass
            # Queue children if depth allows
            if depth < max_depth:
                links: List[str] = []
                try:
                    raw = page.meta.get("links") if isinstance(page.meta, dict) else []
                    if isinstance(raw, list):
                        links = [str(x) for x in raw]
                    else:
                        links = []
                except Exception:
                    links = []
                for ln in links:
                    ln_n = self._normalize_url(ln)
                    if ln_n in visited or ln_n in enqueued:
                        continue
                    if same_domain and not self._same_domain(root_url, ln_n):
                        continue
                    enqueued.add(ln_n)
                    q.append((ln_n, depth + 1))
            # Respect page cap
            if len(pages) >= max_pages:
                break
        return pages
