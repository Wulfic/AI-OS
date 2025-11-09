"""Markdown to HTML renderer for the Help Panel."""

import re
from typing import List, Optional, Dict
from . import utils


class MarkdownRenderer:
    """Converts Markdown to HTML with fallback support."""
    
    def __init__(self):
        """Initialize the renderer."""
        self._md_available = False
        try:
            import markdown as md  # type: ignore
            self._md = md
            self._md_available = True
        except Exception:
            self._md = None
    
    def render(self, md_text: str) -> str:
        """Render markdown text to HTML.
        
        Args:
            md_text: Markdown text to convert
            
        Returns:
            HTML string
        """
        if self._md_available and self._md is not None:
            return self._md.markdown(
                md_text, 
                extensions=["fenced_code", "tables", "toc", "codehilite"]
            )
        else:
            return self._simple_markdown_to_html(md_text)
    
    def _simple_markdown_to_html(self, text: str) -> str:
        """Simple Markdown fallback parser for basic formatting.
        
        Handles:
        - Headings (# ## ###)
        - Lists (unordered and ordered)
        - Code blocks (```fenced```)
        - Inline code (`code`)
        - Links ([text](url) and [text][ref])
        - Horizontal rules
        
        Args:
            text: Markdown text
            
        Returns:
            HTML string with warning banner about limited parsing
        """
        # HTML escape function
        try:
            from html import escape as _escape  # type: ignore
        except Exception:
            def _escape(s: str) -> str:  # type: ignore
                return (
                    s.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&#x27;")
                )
        
        def esc(s: str) -> str:
            return _escape(s)
        
        out: List[str] = []
        in_code = False
        code_buf: List[str] = []
        list_mode: Optional[str] = None  # 'ul' or 'ol'
        para_buf: List[str] = []
        
        # Parse reference-style link definitions
        ref_links: Dict[str, str] = {}
        try:
            for m in re.finditer(
                r"^\s*\[([^\]]+)\]:\s*(\S+)(?:\s+\S.*)?$", 
                text, 
                flags=re.MULTILINE
            ):
                label = m.group(1).strip().lower()
                url = m.group(2).strip()
                if url.startswith("<") and url.endswith(">"):
                    url = url[1:-1]
                ref_links[label] = url
        except Exception:
            ref_links = {}
        
        def _apply_inline_markup(s: str) -> str:
            """Apply inline formatting (code, links)."""
            # Inline code
            s = re.sub(r"`([^`]+)`", lambda m: f"<code>{esc(m.group(1))}</code>", s)
            
            # Reference-style links: [text][label] and shortcut [label]
            def _ref_sub(match: re.Match[str]) -> str:
                txt = match.group(1)
                lbl = match.group(2)
                url = ref_links.get(lbl.strip().lower())
                return f"<a href='{esc(url)}'>{esc(txt)}</a>" if url else esc(match.group(0))
            
            try:
                s = re.sub(r"\[([^\]]+)\]\[([^\]]+)\]", _ref_sub, s)
                # Shortcut style: [label]
                def _shortcut_sub(m: re.Match[str]) -> str:
                    txt = m.group(1)
                    url = ref_links.get(txt.strip().lower())
                    return f"<a href='{esc(url)}'>{esc(txt)}</a>" if url else m.group(0)
                # Negative lookahead to avoid swallowing inline links
                s = re.sub(r"\[([^\]]+)\](?!\()", _shortcut_sub, s)
            except Exception:
                pass
            
            # Inline links [text](url)
            s = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)", 
                lambda m: f"<a href='{esc(m.group(2))}'>{esc(m.group(1))}</a>", 
                s
            )
            
            return s
        
        def flush_para() -> None:
            """Flush accumulated paragraph buffer."""
            nonlocal para_buf
            if para_buf:
                p = " ".join(s.strip() for s in para_buf if s is not None)
                p = _apply_inline_markup(p)
                out.append(f"<p>{p}</p>")
                para_buf = []
        
        def flush_list() -> None:
            """Flush current list."""
            nonlocal list_mode
            if list_mode:
                out.append(f"</{list_mode}>")
                list_mode = None
        
        # Process line by line
        lines = text.splitlines()
        for line in lines:
            line = line.rstrip("\n")
            
            # Fenced code blocks
            if line.strip().startswith("```"):
                if not in_code:
                    flush_para()
                    flush_list()
                    in_code = True
                    code_buf = []
                else:
                    out.append("<pre><code>" + esc("\n".join(code_buf)) + "</code></pre>")
                    code_buf = []
                    in_code = False
                continue
            
            if in_code:
                code_buf.append(line)
                continue
            
            # Headings
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                flush_para()
                flush_list()
                level = len(m.group(1))
                title = m.group(2).strip()
                slug = utils.slugify_github_anchor(title)
                out.append(f"<h{level} id='{slug}'>{esc(title)}</h{level}>")
                continue
            
            # Horizontal rule
            if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", line):
                flush_para()
                flush_list()
                out.append("<hr/>")
                continue
            
            # Unordered lists
            if re.match(r"^\s*[-*]\s+", line):
                flush_para()
                if list_mode not in ("ul",):
                    flush_list()
                    list_mode = "ul"
                    out.append("<ul>")
                item = re.sub(r"^\s*[-*]\s+", "", line)
                item = _apply_inline_markup(item)
                out.append(f"<li>{item}</li>")
                continue
            
            # Ordered lists
            if re.match(r"^\s*\d+\.\s+", line):
                flush_para()
                if list_mode not in ("ol",):
                    flush_list()
                    list_mode = "ol"
                    out.append("<ol>")
                item = re.sub(r"^\s*\d+\.\s+", "", line)
                item = _apply_inline_markup(item)
                out.append(f"<li>{item}</li>")
                continue
            
            # Blank line
            if not line.strip():
                flush_para()
                flush_list()
                continue
            
            # Paragraph text (accumulate)
            para_buf.append(line)
        
        # Final flushes
        if in_code:
            out.append("<pre><code>" + esc("\n".join(code_buf)) + "</code></pre>")
        flush_para()
        flush_list()
        
        # Add banner encouraging python-markdown installation
        banner = (
            "<div class='warn'>Rendering with a minimal built-in Markdown parser. "
            "For better formatting, install the 'markdown' package in your environment.</div>"
        )
        return banner + "\n" + "\n".join(out)
