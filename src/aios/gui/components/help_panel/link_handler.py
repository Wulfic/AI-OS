"""Link handling and navigation for the Help Panel."""

import os
import re
import webbrowser
from pathlib import Path
from typing import Any, Optional, Callable, List, Tuple
from urllib.parse import urlparse, unquote
from . import utils


class LinkHandler:
    """Handles link interception, transformation, and navigation."""
    
    INTERNAL_PROTOCOL = "http://aios-internal-doc/"
    
    def __init__(
        self, 
        html_view: Any,
        docs_root: Path,
        index: List[Tuple[str, str, List[str], List[Tuple[int, str]]]],
        open_doc_callback: Callable[[str, int], None]
    ):
        """Initialize the link handler.
        
        Args:
            html_view: The HTML widget (tkinterweb HtmlFrame)
            docs_root: Root directory for documentation
            index: Document index for resolving paths
            open_doc_callback: Callback function to open a document (rel_path, heading_line)
        """
        self.html_view = html_view
        self.docs_root = docs_root
        self.index = index
        self.open_doc_callback = open_doc_callback
        self._installed = False
        self._current_doc_rel = ""
        self._pending_anchor: Optional[str] = None
    
    def set_current_doc(self, rel_path: str) -> None:
        """Set the currently displayed document path.
        
        Args:
            rel_path: Relative path of current document
        """
        self._current_doc_rel = rel_path
    
    def get_pending_anchor(self) -> Optional[str]:
        """Get and clear any pending anchor to scroll to.
        
        Returns:
            Anchor ID if pending, None otherwise
        """
        anchor = self._pending_anchor
        self._pending_anchor = None
        return anchor
    
    def transform_html_links(self, html: str) -> str:
        """Transform HTML links to use internal protocol for interception.
        
        Args:
            html: HTML content with <a href="..."> links
            
        Returns:
            HTML with transformed links
        """
        def replace_link(match: re.Match[str]) -> str:
            href = match.group(1)
            rest = match.group(2) if match.lastindex >= 2 else ""
            
            # Keep anchor links as-is
            if href.startswith('#'):
                return f'<a href="{href}"{rest}'
            
            # Keep external links as-is (with target="_blank")
            if utils.is_external_url(href):
                return f'<a href="{href}"{rest} target="_blank"'
            
            # Transform relative/internal links
            return f'<a href="{self.INTERNAL_PROTOCOL}{href}"{rest}'
        
        pattern = r'<a href="([^"]+)"([^>]*)'
        return re.sub(pattern, replace_link, html)
    
    def get_base_url(self) -> str:
        """Get the base URL for the current document.
        
        Returns:
            Base URL string for use with load_html
        """
        base_dir = os.path.dirname(self._current_doc_rel) if self._current_doc_rel else ""
        return f"{self.INTERNAL_PROTOCOL}{base_dir}/" if base_dir else self.INTERNAL_PROTOCOL
    
    def handle_link_click(self, url: str) -> None:
        """Handle a link click from the HTML view.
        
        Args:
            url: The URL that was clicked
        """
        print(f"[LinkHandler] Link clicked: {url}")
        
        # Handle anchor links
        if utils.is_anchor_link(url):
            anchor_id = url[1:]
            print(f"[LinkHandler] Anchor link, scrolling to: {anchor_id}")
            self._scroll_to_anchor(anchor_id)
            return
        
        # Handle internal doc URLs
        if url.startswith(self.INTERNAL_PROTOCOL):
            actual_href = url[len(self.INTERNAL_PROTOCOL):]
            self._handle_internal_link(actual_href)
            return
        
        # Handle external links
        if utils.is_external_url(url):
            print(f"[LinkHandler] External link, opening in browser: {url}")
            try:
                webbrowser.open_new_tab(url)
            except Exception as e:
                print(f"[LinkHandler] Failed to open external link: {e}")
            return
        
        print(f"[LinkHandler] Unknown link type: {url}")
    
    def _handle_internal_link(self, href: str) -> None:
        """Handle an internal documentation link.
        
        Args:
            href: The href path (may include anchor)
        """
        # Extract path and anchor
        doc_path, anchor = utils.extract_anchor_from_url(href)
        
        if anchor:
            print(f"[LinkHandler] Internal link with anchor: {doc_path} -> #{anchor}")
            self._pending_anchor = anchor
        else:
            self._pending_anchor = None
        
        # Resolve path relative to current document's directory
        # Only absolute paths starting with '/' are treated as absolute from docs root
        if doc_path.startswith('/'):
            # Absolute path from docs root
            full_path = utils.normalize_path(doc_path[1:])  # Remove leading /
        else:
            # Relative path (including ./, ../, or just filename/folder)
            base_dir = os.path.dirname(self._current_doc_rel or "")
            full_path = utils.resolve_relative_path(base_dir, doc_path)
        
        print(f"[LinkHandler] Resolved internal link: {href} -> {full_path}")
        
        # Find in index
        found = self._find_in_index(full_path)
        if found:
            print(f"[LinkHandler] Found document in index: {found}")
            
            # If anchor is present, try to find the heading line
            heading_line = -1
            if anchor:
                heading_line = self._find_heading_line(found, anchor)
            
            self.open_doc_callback(found, heading_line)
        else:
            print(f"[LinkHandler] Document not found in index: {full_path}")
    
    def _find_in_index(self, path: str) -> Optional[str]:
        """Find a document in the index (case-insensitive).
        
        Args:
            path: Document path to find
            
        Returns:
            Actual path from index, or None if not found
        """
        path_lower = path.lower()
        for pth, _c, _t, _h in self.index:
            if pth.lower() == path_lower:
                return pth
        return None
    
    def _find_heading_line(self, doc_path: str, anchor: str) -> int:
        """Find the line number for a heading anchor.
        
        Args:
            doc_path: Document path
            anchor: Anchor ID (slugified heading)
            
        Returns:
            Line number, or -1 if not found
        """
        for pth, _c, _t, headings in self.index:
            if pth == doc_path:
                slug = utils.slugify_github_anchor(anchor)
                for ln, htxt in headings:
                    if utils.slugify_github_anchor(htxt) == slug:
                        return ln
                break
        return -1
    
    def _scroll_to_anchor(self, anchor_id: str) -> None:
        """Scroll to an anchor element in the current document.
        
        Args:
            anchor_id: Anchor element ID
        """
        try:
            if self.html_view:
                # Try different methods depending on tkinterweb version
                try:
                    element = self.html_view.document.getElementById(anchor_id)
                    if element:
                        element.scrollIntoView()
                        print(f"[LinkHandler] Scrolled to anchor: {anchor_id}")
                        return
                except Exception:
                    pass
                
                # Fallback: try JavaScript
                try:
                    js = f"document.getElementById('{anchor_id}')?.scrollIntoView({{behavior:'smooth'}});"
                    if hasattr(self.html_view, 'execute_script'):
                        self.html_view.execute_script(js)  # type: ignore
                    elif hasattr(self.html_view, 'run_script'):
                        self.html_view.run_script(js)  # type: ignore
                    print(f"[LinkHandler] Scrolled to anchor via JS: {anchor_id}")
                    return
                except Exception:
                    pass
                
                print(f"[LinkHandler] Could not scroll to anchor: {anchor_id}")
        except Exception as e:
            print(f"[LinkHandler] Error scrolling to anchor: {e}")
    
    def install_handler(self) -> bool:
        """Install the link click handler on the HTML view.
        
        Returns:
            True if handler was installed successfully, False otherwise
        """
        if self._installed:
            print("[LinkHandler] Handler already installed")
            return True
        
        if self.html_view is None:
            print("[LinkHandler] HTML view is None, cannot install handler")
            return False
        
        print("[LinkHandler] Installing link handler")
        
        # The handler is set via the on_link_click parameter in the HtmlFrame constructor
        # So we don't need to do anything here if it's already set
        # This method is kept for compatibility
        
        self._installed = True
        return True
    
    def open_external_in_browser(self, rel_path: str, anchor: Optional[str] = None) -> None:
        """Open a document externally in the system browser.
        
        Args:
            rel_path: Relative path to the document
            anchor: Optional GitHub-style anchor
        """
        try:
            # Try GitHub URL first
            gh_base = "https://github.com/Wulfic/AI-OS/blob/main/docs/"
            url = gh_base + rel_path.replace("\\", "/")
            if anchor:
                url += f"#{anchor}"
            webbrowser.open_new_tab(url)
        except Exception:
            # Fallback to local file
            try:
                abs_path = str((self.docs_root / rel_path).resolve())
                webbrowser.open_new_tab(abs_path)
            except Exception as e:
                print(f"[LinkHandler] Failed to open externally: {e}")
