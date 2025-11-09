"""Search and indexing engine for documentation with improved relevance scoring."""

import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


class SearchEngine:
    """Handles document indexing and search with weighted relevance scoring."""
    
    # Search result limits and thresholds
    MAX_RESULTS = 25  # Maximum number of file results to return
    MIN_SCORE_THRESHOLD = 20.0  # Minimum score to include in results
    MAX_HEADINGS_PER_FILE = 5  # Maximum heading matches to show per file
    
    # Scoring weights (higher = more important)
    WEIGHT_FILENAME_EXACT = 100.0  # Exact match in filename
    WEIGHT_FILENAME_PARTIAL = 60.0  # Partial match in filename
    WEIGHT_HEADING = 40.0  # Match in heading
    WEIGHT_TAGS = 30.0  # Match in tags
    WEIGHT_CONTENT = 10.0  # Match in content
    
    def __init__(self, docs_root: Path):
        """Initialize the search engine.
        
        Args:
            docs_root: Root directory containing documentation files
        """
        self.docs_root = docs_root
        self.index: List[Tuple[str, str, List[str], List[Tuple[int, str]]]] = []
        
    def build_index(self) -> bool:
        """Build or load the search index.
        
        Returns:
            True if index was loaded/built successfully, False otherwise
        """
        # Try to load prebuilt index first
        prebuilt = self.docs_root / "search_index.json"
        if prebuilt.exists():
            try:
                data = json.loads(prebuilt.read_text(encoding="utf-8", errors="ignore"))
                index: List[Tuple[str, str, List[str], List[Tuple[int, str]]]] = []
                for it in data:
                    rel = str(it.get("path", "")).replace("\\", "/").lstrip("/")
                    # Skip maintenance/research
                    top = rel.split("/", 1)[0].lower() if rel else ""
                    if top in ("maintenance", "research"):
                        continue
                    content = it.get("content", "")
                    tags = [str(t).lower() for t in it.get("tags", [])]
                    headings = [
                        (int(h.get("line", 0)), str(h.get("text", ""))) 
                        for h in it.get("headings", [])
                    ]
                    index.append((rel, content, tags, headings))
                self.index = index
                return True
            except Exception as e:
                print(f"[SearchEngine] Failed to load prebuilt index: {e}")

        # Build fresh index
        try:
            doc_files: List[Path] = []
            if self.docs_root.exists():
                for root, _dirs, files in os.walk(self.docs_root):
                    for f in files:
                        if f.lower().endswith((".md", ".mdx")):
                            doc_files.append(Path(root) / f)
            
            index: List[Tuple[str, str, List[str], List[Tuple[int, str]]]] = []
            tag_map = {
                "cli": ["cli"],
                "gui": ["gui"],
                "hrm": ["hrm", "training"],
                "expert": ["experts"],
                "dataset": ["datasets"],
                "train": ["training"],
                "eval": ["evaluation"],
                "mcp": ["mcp"],
            }
            
            for p in doc_files:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    rel = str(p.relative_to(self.docs_root)).replace("\\", "/")
                    
                    # Skip maintenance/research
                    top = rel.split("/", 1)[0].lower() if rel else ""
                    if top in ("maintenance", "research"):
                        continue
                    
                    # Extract tags
                    tags: List[str] = []
                    low = (rel + "\n" + text[:3000]).lower()
                    for k, vals in tag_map.items():
                        if k in low:
                            tags.extend(vals)
                    tags = sorted(list(set(tags)))
                    
                    # Extract headings
                    headings: List[Tuple[int, str]] = []
                    for i, line in enumerate(text.splitlines()):
                        if line.lstrip().startswith("#"):
                            headings.append((i, line.lstrip("# ")))
                    
                    index.append((rel, text, tags, headings))
                except Exception as e:
                    print(f"[SearchEngine] Failed to index {p}: {e}")
            
            self.index = index
            
            # Save prebuilt index for future runs
            try:
                self._save_prebuilt_index()
            except Exception as e:
                print(f"[SearchEngine] Failed to save prebuilt index: {e}")
            
            return True
        except Exception as e:
            print(f"[SearchEngine] Failed to build index: {e}")
            return False
    
    def _save_prebuilt_index(self) -> None:
        """Save the current index to disk for faster loading next time."""
        prebuilt_path = self.docs_root / "search_index.json"
        if prebuilt_path.exists():
            return  # Don't overwrite existing index
        
        payload: List[Dict[str, Any]] = []
        for rel, content, tags, headings in self.index:
            payload.append({
                "path": rel,
                "content": content if len(content) <= 500000 else content[:500000],
                "tags": tags,
                "headings": [{"line": ln, "text": tx} for (ln, tx) in headings],
            })
        prebuilt_path.write_text(json.dumps(payload), encoding="utf-8")
    
    def search(self, query: str) -> List[Tuple[float, str, List[Tuple[float, int, str]]]]:
        """Search the index with improved relevance scoring.
        
        Args:
            query: Search query string
            
        Returns:
            List of (score, file_path, [(heading_score, line_num, heading_text), ...])
            sorted by relevance (highest score first), limited to top results
        """
        if not query or not query.strip():
            return []
        
        q = query.strip()
        ql = q.lower()
        results: List[Tuple[float, str, List[Tuple[float, int, str]]]] = []
        
        for rel, text, tags, headings in self.index:
            # Calculate weighted scores from different sources
            score_breakdown = {
                'filename': self._score_filename(ql, rel),
                'headings': 0.0,
                'tags': self._score_tags(ql, tags),
                'content': self._score_content(ql, text)
            }
            
            # Find matching headings
            h_local: List[Tuple[float, int, str]] = []
            for idx, htxt in headings:
                hs = self._score_heading(ql, htxt)
                if hs > 0:
                    h_local.append((hs, idx, htxt))
                    score_breakdown['headings'] = max(score_breakdown['headings'], hs)
            
            # Calculate overall score as weighted max
            overall_score = max(score_breakdown.values())
            
            # Only include results above threshold
            if overall_score >= self.MIN_SCORE_THRESHOLD:
                # Sort heading matches by score and limit
                h_local.sort(key=lambda x: x[0], reverse=True)
                h_local = h_local[:self.MAX_HEADINGS_PER_FILE]
                results.append((overall_score, rel, h_local))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:self.MAX_RESULTS]
        
        return results
    
    def _score_filename(self, query: str, filepath: str) -> float:
        """Score match in filename with exact and partial matching."""
        filename = os.path.basename(filepath).lower()
        filepath_lower = filepath.lower()
        
        # Exact filename match (highest priority)
        if query == filename.replace('.md', '').replace('.mdx', ''):
            return self.WEIGHT_FILENAME_EXACT
        
        # Filename contains full query
        if query in filename:
            return self.WEIGHT_FILENAME_PARTIAL + 20.0
        
        # Path contains full query
        if query in filepath_lower:
            return self.WEIGHT_FILENAME_PARTIAL
        
        # Word-level matching in filename
        query_words = query.split()
        if len(query_words) > 1:
            matches = sum(1 for word in query_words if word in filename)
            if matches > 0:
                return self.WEIGHT_FILENAME_PARTIAL * (matches / len(query_words))
        
        # Fuzzy matching
        return self._fuzzy_score(query, filepath) * 0.6
    
    def _score_heading(self, query: str, heading: str) -> float:
        """Score match in heading text."""
        heading_lower = heading.lower()
        
        # Exact match
        if query == heading_lower:
            return self.WEIGHT_HEADING + 20.0
        
        # Contains full query
        if query in heading_lower:
            return self.WEIGHT_HEADING + 10.0
        
        # Word-level matching
        query_words = query.split()
        if len(query_words) > 1:
            matches = sum(1 for word in query_words if word in heading_lower)
            if matches > 0:
                return self.WEIGHT_HEADING * (matches / len(query_words))
        
        # Fuzzy matching
        return self._fuzzy_score(query, heading) * 0.4
    
    def _score_tags(self, query: str, tags: List[str]) -> float:
        """Score match in tags."""
        if not tags:
            return 0.0
        
        tags_text = " ".join(tags).lower()
        
        # Exact tag match
        if query in tags:
            return self.WEIGHT_TAGS + 10.0
        
        # Contains in tags text
        if query in tags_text:
            return self.WEIGHT_TAGS
        
        # Fuzzy matching
        return self._fuzzy_score(query, tags_text) * 0.3
    
    def _score_content(self, query: str, content: str) -> float:
        """Score match in document content (lowest priority)."""
        content_lower = content.lower()
        
        # Count occurrences (but cap the score)
        count = content_lower.count(query)
        if count > 0:
            # Logarithmic scale to avoid content spam dominating
            import math
            return min(self.WEIGHT_CONTENT * math.log(count + 1), self.WEIGHT_CONTENT * 2)
        
        # Fuzzy matching (very low weight)
        return self._fuzzy_score(query, content[:5000]) * 0.1
    
    def _fuzzy_score(self, query: str, text: str) -> float:
        """Calculate fuzzy match score using rapidfuzz or fallback."""
        try:
            from rapidfuzz import fuzz
            return float(fuzz.token_set_ratio(query, text))
        except Exception:
            # Fallback: simple word matching
            if not query:
                return 0.0
            q = query.lower().strip()
            t = text.lower()
            score = 0.0
            if q in t:
                score += 50.0
            for w in [w for w in re.split(r"\s+", q) if w]:
                if w in t:
                    score += 10.0
            return score
    
    def get_index_stats(self) -> str:
        """Get a summary of the index statistics."""
        total_docs = len(self.index)
        total_headings = sum(len(headings) for _, _, _, headings in self.index)
        return f"Indexed {total_docs} docs with {total_headings} headings under {self.docs_root}"
