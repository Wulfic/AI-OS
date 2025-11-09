"""Utility functions for the Help Panel."""

import os
import re
from pathlib import Path
from typing import Optional


def find_project_root(start: Path) -> Path:
    """Find the project root by looking for pyproject.toml or docs/ directory.
    
    Args:
        start: Starting path (file or directory)
        
    Returns:
        Path to project root
    """
    cur = start if start.is_dir() else start.parent
    try:
        cur = cur.resolve()
    except Exception:
        pass
    
    for node in [cur] + list(cur.parents):
        try:
            if (node / "pyproject.toml").exists() or (node / "docs").is_dir():
                return node
        except Exception:
            pass
    
    return cur


def resolve_docs_root(project_root: Path) -> Path:
    """Find the docs directory, checking the project and parent directories.
    
    Args:
        project_root: Project root path
        
    Returns:
        Path to docs directory
    """
    def has_md(d: Path) -> bool:
        """Check if directory contains markdown files."""
        try:
            if not d.is_dir():
                return False
            for _r, _dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".md", ".mdx")):
                        return True
            return False
        except Exception:
            return False
    
    cand = project_root / "docs"
    if has_md(cand):
        return cand
    
    for node in [project_root.parent] + list(project_root.parents):
        d = node / "docs"
        if has_md(d):
            return d
    
    return cand


def slugify_github_anchor(text: str) -> str:
    """Convert heading text to GitHub-style anchor ID.
    
    Args:
        text: Heading text
        
    Returns:
        Slugified anchor ID (e.g., "API Contract (v1)" -> "api-contract-v1")
    """
    try:
        s = text.strip().lower()
        s = s.replace('`', '')
        s = re.sub(r"\s+", "-", s)
        s = re.sub(r"[^a-z0-9-]", "", s)
        s = re.sub(r"-+", "-", s).strip('-')
        return s
    except Exception:
        return ""


def normalize_path(path: str) -> str:
    """Normalize a path string to use forward slashes and no leading slash.
    
    Args:
        path: Path string to normalize
        
    Returns:
        Normalized path string
    """
    return path.replace('\\', '/').strip('/')


def resolve_relative_path(base_dir: str, href: str) -> str:
    """Resolve a relative href against a base directory.
    
    Args:
        base_dir: Base directory path (relative to docs root)
        href: Relative href path
        
    Returns:
        Resolved path (relative to docs root)
    """
    # Handle ./path
    if href.startswith('./'):
        href = href[2:]
        full_path = f"{base_dir}/{href}" if base_dir else href
    # Handle ../path
    elif href.startswith('../'):
        parent = os.path.dirname(base_dir) if base_dir else ""
        href = href[3:]
        full_path = f"{parent}/{href}" if parent else href
    # Handle absolute path from docs root
    else:
        full_path = f"{base_dir}/{href}" if base_dir else href
    
    return normalize_path(full_path)


def extract_anchor_from_url(url: str) -> tuple[str, Optional[str]]:
    """Extract the path and anchor from a URL.
    
    Args:
        url: URL string (may contain #anchor)
        
    Returns:
        Tuple of (path_without_anchor, anchor_or_none)
    """
    if '#' in url:
        path, anchor = url.split('#', 1)
        return path, anchor
    return url, None


def is_markdown_file(path: str) -> bool:
    """Check if a path is a markdown file.
    
    Args:
        path: File path
        
    Returns:
        True if the file is .md or .mdx
    """
    return path.lower().endswith((".md", ".mdx"))


def is_external_url(url: str) -> bool:
    """Check if a URL is an external HTTP(S) URL.
    
    Args:
        url: URL string
        
    Returns:
        True if the URL starts with http:// or https://
    """
    return url.startswith(('http://', 'https://'))


def is_anchor_link(url: str) -> bool:
    """Check if a URL is an anchor-only link.
    
    Args:
        url: URL string
        
    Returns:
        True if the URL is just an anchor (#section)
    """
    return url.startswith('#')
