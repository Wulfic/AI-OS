"""Table of Contents tree builder for the Help Panel."""

from typing import Any, Dict, List, Tuple


class TOCBuilder:
    """Builds hierarchical table of contents tree from indexed documents."""
    
    def __init__(self, tree_widget: Any):
        """Initialize the TOC builder.
        
        Args:
            tree_widget: Tkinter Treeview widget to populate
        """
        self.tree_widget = tree_widget
    
    def build_toc(
        self, 
        index: List[Tuple[str, str, List[str], List[Tuple[int, str]]]]
    ) -> List[Any]:
        """Build hierarchical TOC from document index.
        
        Args:
            index: Document index [(path, content, tags, headings), ...]
            
        Returns:
            List of top-level tree item IDs
        """
        # Clear existing tree
        for i in self.tree_widget.get_children():
            self.tree_widget.delete(i)
        
        # Build nested directory structure
        tree: Dict[str, Any] = {}
        for rel, _text, _tags, headings in index:
            parts = rel.split('/')
            node = tree
            
            # Navigate/create directory nodes
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            
            # Add file entry
            node.setdefault('__files__', []).append((parts[-1], headings, rel))
        
        # Insert nodes into tree widget
        def insert_node(
            parent: str, 
            name: str, 
            subtree: Dict[str, Any], 
            prefix_path: str
        ) -> Any:
            """Recursively insert a directory/file node."""
            item = self.tree_widget.insert(
                parent, 
                'end', 
                text=name, 
                values=(prefix_path, '-1')
            )
            
            # Subdirectories
            for k, v in sorted(
                (k, v) for k, v in subtree.items() if k != '__files__'
            ):
                child_path = f"{prefix_path}/{k}" if prefix_path else k
                insert_node(item, k, v, child_path)
            
            # Files in this directory
            for fname, headings, full_rel in sorted(
                subtree.get('__files__', []), 
                key=lambda x: x[0].lower()
            ):
                fitem = self.tree_widget.insert(
                    item, 
                    'end', 
                    text=fname, 
                    values=(full_rel, '-1')
                )
                # Add heading children (collapsed by default)
                for idx, htxt in headings[:20]:  # Limit headings shown
                    self.tree_widget.insert(
                        fitem, 
                        'end', 
                        text=f"# {htxt}", 
                        values=(full_rel, str(idx))
                    )
            
            return item
        
        # Insert top-level directories
        top_items = []
        for k, v in sorted((k, v) for k, v in tree.items() if isinstance(v, dict)):
            top_items.append(insert_node('', k, v, k))
        
        # Insert root-level files
        for fname, headings, full_rel in sorted(
            tree.get('__files__', []), 
            key=lambda x: x[0].lower()
        ):
            fitem = self.tree_widget.insert(
                '', 
                'end', 
                text=fname, 
                values=(full_rel, '-1')
            )
            for idx, htxt in headings[:20]:
                self.tree_widget.insert(
                    fitem, 
                    'end', 
                    text=f"# {htxt}", 
                    values=(full_rel, str(idx))
                )
        
        # Auto-open top-level directories for visibility
        try:
            for it in top_items:
                self.tree_widget.item(it, open=True)
        except Exception:
            pass
        
        return top_items
    
    def populate_search_results(
        self, 
        results: List[Tuple[float, str, List[Tuple[float, int, str]]]]
    ) -> None:
        """Populate tree with search results.
        
        Args:
            results: Search results [(score, path, [(h_score, line, text), ...]), ...]
        """
        # Clear existing tree
        for i in self.tree_widget.get_children():
            self.tree_widget.delete(i)
        
        # Group results by directory for better organization
        by_dir: Dict[str, List[Tuple[float, str, List[Tuple[float, int, str]]]]] = {}
        for score, rel, hlist in results:
            dir_name = rel.split('/')[0] if '/' in rel else 'Root'
            by_dir.setdefault(dir_name, []).append((score, rel, hlist))
        
        # Insert grouped results
        for dir_name in sorted(by_dir.keys()):
            dir_results = by_dir[dir_name]
            
            # Create directory node if there are multiple results in it
            if len(dir_results) > 1 and dir_name != 'Root':
                dir_node = self.tree_widget.insert(
                    '', 
                    'end', 
                    text=f"📁 {dir_name} ({len(dir_results)})", 
                    values=('', '-1')
                )
                # Auto-expand if few results
                if len(results) <= 10:
                    self.tree_widget.item(dir_node, open=True)
            else:
                dir_node = ''  # Insert at root
            
            # Insert files
            for score, rel, hlist in sorted(
                dir_results, 
                key=lambda x: x[0], 
                reverse=True
            ):
                # Show score as visual indicator
                score_indicator = "⭐" * min(3, int(score / 30))
                parent = self.tree_widget.insert(
                    dir_node, 
                    'end', 
                    text=f"{score_indicator} {rel}", 
                    values=(rel, "-1")
                )
                
                # Add matching headings
                for hs, idx, htxt in hlist:
                    self.tree_widget.insert(
                        parent, 
                        'end', 
                        text=f"  → {htxt}", 
                        values=(rel, str(idx))
                    )
