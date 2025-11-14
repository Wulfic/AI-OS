from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import logging
import os
import aiohttp
from urllib.parse import urlparse, parse_qs, unquote, urljoin

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str


def _us_params_from_env() -> Dict[str, str]:
    # Allow overrides via environment so CLI/config can inject defaults.
    # Fallback to US region if nothing is set.
    kl = os.environ.get("AIOS_DDG_KL") or "us-en"
    kad = os.environ.get("AIOS_DDG_KAD") or "us-en"
    return {"kl": kl, "kad": kad}


def _normalize_ddg_href(href: str) -> str:
    """DuckDuckGo results often use redirector `/l/?uddg=...`. Extract the target if present."""
    try:
        if not href:
            return href
        u = urlparse(href)
        if not u.netloc:  # relative like /l/?uddg=...
            # assume duckduckgo
            u = urlparse("https://duckduckgo.com" + href)
        if "duckduckgo.com" in (u.netloc or "") and u.path.startswith("/l"):
            qs = parse_qs(u.query)
            target = (qs.get("uddg") or qs.get("u") or [None])[0]
            if target:
                return unquote(target)
        return href
    except Exception as e:
        logger.debug(f"Failed to normalize DDG href {href}: {e}")
        return href


async def ddg_search(query: str, limit: int = 3) -> List[SearchResult]:
    # Use HTML-only endpoint more resilient to JS changes
    logger.debug(f"Starting async DDG search operation: query='{query}', limit={limit}")
    logger.info(f"Starting DuckDuckGo search: query='{query}', limit={limit}")
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, **_us_params_from_env()}
    headers = {
        "User-Agent": os.environ.get("AIOS_WEB_UA")
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.7",
    }
    # Allow suffix override/appending
    ua_suffix = os.environ.get("AIOS_WEB_UA_SUFFIX")
    if ua_suffix:
        headers["User-Agent"] = headers["User-Agent"] + f" {ua_suffix}"
    
    try:
        logger.debug("Creating aiohttp session for DDG search")
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                text = await resp.text(errors="ignore")
                logger.debug(f"DDG search response received: {len(text)} bytes")
    except aiohttp.ClientError as e:
        logger.error(f"DuckDuckGo search request failed for query '{query}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during DuckDuckGo search for query '{query}': {e}")
        return []
    
    # Parse more than needed to allow filtering
    parsed = parse_ddg_html(text, limit=max(10, limit * 3))
    logger.debug(f"Parsed {len(parsed)} raw results from DDG HTML")
    filtered: List[SearchResult] = []
    blocked_hosts = {
        "duckduckgo.com",  # ads/redirects
        "facebook.com", "m.facebook.com", "lm.facebook.com", "web.facebook.com",
        "twitter.com", "x.com",
    }
    for r in parsed:
        try:
            u = urlparse(r.url)
            host = (u.netloc or "").lower()
            # Skip non-http(s)
            if u.scheme not in ("http", "https"):
                continue
            # Handle duckduckgo redirector
            if host.endswith("duckduckgo.com"):
                if u.path.startswith("/l"):
                    nu = _normalize_ddg_href(r.url)
                    u = urlparse(nu)
                    host = (u.netloc or "").lower()
                else:
                    continue
            # Skip blocked hosts
            if any(host == bh or host.endswith("." + bh) for bh in blocked_hosts):
                continue
            # Skip obvious ad links
            q = u.query.lower()
            if any(k in q for k in ("ad_domain=", "ad_provider=", "ad_type=", "msclkid=", "gclid=")):
                continue
            href = _normalize_ddg_href(u.geturl())
            filtered.append(SearchResult(title=r.title, url=href))
            if len(filtered) >= limit:
                break
        except Exception as e:
            logger.debug(f"Failed to process search result: {e}")
            continue
    
    logger.info(f"DuckDuckGo search completed: {len(filtered)} results for query '{query}'")
    logger.debug(f"Async DDG search operation completed: filtered {len(filtered)} results from {len(parsed)} raw results")
    return filtered or parsed[:limit]


def parse_ddg_html(html: str, limit: int = 3) -> List[SearchResult]:
    """Parse DuckDuckGo HTML-only results page and return top results."""
    results: List[SearchResult] = []
    try:
        import bs4  # type: ignore
        from bs4.element import Tag  # type: ignore
        soup = bs4.BeautifulSoup(html, "lxml")
        # Primary selector used by HTML endpoint
        anchors = soup.select("a.result__a")
        if not anchors:
            # Fallback: older lite structure
            anchors = soup.select("a.result__url") or soup.find_all("a")
        for a in anchors:
            try:
                if not isinstance(a, Tag):
                    continue
                href_val = a.get("href")
                href = str(href_val) if href_val is not None else ""
                if not href:
                    continue
                href = urljoin("https://duckduckgo.com", href)
                href = _normalize_ddg_href(href)
                title = a.get_text(" ", strip=True)
                if href and title:
                    results.append(SearchResult(title=title, url=href))
                if len(results) >= limit:
                    break
            except Exception as e:
                logger.debug(f"Failed to parse HTML result element: {e}")
                continue
        if results:
            return results
    except Exception as e:
        logger.warning(f"BeautifulSoup parsing failed, using regex fallback: {e}")
    # Regex fallback if bs4 not available or structure changed
    import re
    pattern = r'<a[^>]+class=["\']result__a["\'][^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
    for m in re.finditer(pattern, html, re.I | re.S):
        href = _normalize_ddg_href(urljoin("https://duckduckgo.com", m.group(1)))
        title = re.sub("<[^<]+?>", "", m.group(2))
        if href and title:
            results.append(SearchResult(title=title.strip(), url=href))
        if len(results) >= limit:
            break
    return results
