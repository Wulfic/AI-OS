"""Rich message widgets for displaying various content types in chat."""

from __future__ import annotations

from typing import Any, cast, Callable
import os
from pathlib import Path

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk, font as tkfont  # type: ignore
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)
    tkfont = cast(Any, None)
    messagebox = cast(Any, None)


class CodeBlockWidget(ttk.Frame):  # type: ignore[misc]
    """Widget for displaying code with syntax highlighting and copy button."""
    
    # Syntax highlighting colors (basic keywords)
    SYNTAX_COLORS = {
        "python": {
            "keyword": "#CC7832",
            "string": "#6A8759",
            "comment": "#808080",
            "function": "#FFC66D",
            "number": "#6897BB",
        },
        "javascript": {
            "keyword": "#CC7832",
            "string": "#6A8759",
            "comment": "#808080",
            "function": "#FFC66D",
            "number": "#6897BB",
        },
        "default": {
            "keyword": "#0000FF",
            "string": "#008000",
            "comment": "#808080",
            "function": "#795E26",
            "number": "#098658",
        }
    }
    
    KEYWORDS = {
        "python": {
            "keyword": ["def", "class", "if", "else", "elif", "for", "while", "return", 
                       "import", "from", "as", "try", "except", "finally", "with", 
                       "lambda", "yield", "async", "await", "pass", "break", "continue",
                       "True", "False", "None", "and", "or", "not", "in", "is"],
        },
        "javascript": {
            "keyword": ["function", "const", "let", "var", "if", "else", "for", "while",
                       "return", "import", "export", "from", "as", "try", "catch",
                       "finally", "async", "await", "class", "extends", "new",
                       "true", "false", "null", "undefined"],
        }
    }
    
    def __init__(
        self,
        parent: Any,
        code: str,
        language: str = "text",
        **kwargs: Any
    ) -> None:
        """Initialize code block widget.
        
        Args:
            parent: Parent widget
            code: Code content to display
            language: Programming language for syntax highlighting
            **kwargs: Additional frame options
        """
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        super().__init__(parent, **kwargs)
        self.code = code
        self.language = language.lower()
        
        # Header with language label and copy button
        header = ttk.Frame(self)
        header.pack(fill="x", padx=2, pady=(2, 0))
        
        lang_label = ttk.Label(
            header,
            text=f"📝 {language}",
            font=("Segoe UI", 9, "bold")
        )
        lang_label.pack(side="left", padx=5)
        
        copy_btn = ttk.Button(
            header,
            text="📋 Copy",
            command=self._copy_code,
            width=8
        )
        copy_btn.pack(side="right", padx=5)
        
        # Code display area with scrollbar
        code_frame = ttk.Frame(self)
        code_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Use a monospace font
        try:
            code_font = tkfont.Font(family="Consolas", size=10)
        except Exception:
            try:
                code_font = tkfont.Font(family="Courier New", size=10)
            except Exception:
                code_font = tkfont.Font(family="monospace", size=10)
        
        self.text_widget = tk.Text(
            code_frame,
            wrap="none",
            font=code_font,
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#D4D4D4",
            relief="flat",
            padx=10,
            pady=10,
            height=min(code.count('\n') + 1, 20),  # Auto-height up to 20 lines
        )
        
        # Scrollbars
        vsb = ttk.Scrollbar(code_frame, orient="vertical", command=self.text_widget.yview)
        hsb = ttk.Scrollbar(code_frame, orient="horizontal", command=self.text_widget.xview)
        self.text_widget.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.text_widget.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        code_frame.grid_rowconfigure(0, weight=1)
        code_frame.grid_columnconfigure(0, weight=1)
        
        # Insert code with syntax highlighting
        self._insert_highlighted_code()
        
        # Make read-only
        self.text_widget.config(state="disabled")
        
        # Add tooltip
        try:
            from .tooltips import add_tooltip
            add_tooltip(copy_btn, "Copy code to clipboard")
        except Exception:
            pass
    
    def _insert_highlighted_code(self) -> None:
        """Insert code with basic syntax highlighting."""
        colors = self.SYNTAX_COLORS.get(self.language, self.SYNTAX_COLORS["default"])
        
        # Configure tags for syntax highlighting
        for tag_name, color in colors.items():
            self.text_widget.tag_configure(tag_name, foreground=color)
        
        # Insert code
        self.text_widget.insert("1.0", self.code)
        
        # Apply basic syntax highlighting
        if self.language in self.KEYWORDS:
            keywords = self.KEYWORDS[self.language]
            
            # Highlight keywords
            for kw in keywords.get("keyword", []):
                self._highlight_pattern(f"\\b{kw}\\b", "keyword")
            
            # Highlight strings
            self._highlight_pattern(r'"[^"]*"', "string")
            self._highlight_pattern(r"'[^']*'", "string")
            
            # Highlight comments
            if self.language == "python":
                self._highlight_pattern(r'#[^\n]*', "comment")
            elif self.language == "javascript":
                self._highlight_pattern(r'//[^\n]*', "comment")
            
            # Highlight numbers
            self._highlight_pattern(r'\b\d+\.?\d*\b', "number")
    
    def _highlight_pattern(self, pattern: str, tag: str) -> None:
        """Highlight text matching a regex pattern.
        
        Args:
            pattern: Regex pattern to match
            tag: Tag name for highlighting
        """
        import re
        
        content = self.text_widget.get("1.0", "end-1c")
        for match in re.finditer(pattern, content):
            start_idx = f"1.0 + {match.start()} chars"
            end_idx = f"1.0 + {match.end()} chars"
            self.text_widget.tag_add(tag, start_idx, end_idx)
    
    def _copy_code(self) -> None:
        """Copy code to clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.code)
            if messagebox:
                # Show brief confirmation
                self.after(0, lambda: self._show_copied_message())
        except Exception as e:
            if messagebox:
                messagebox.showerror("Copy Error", f"Failed to copy: {e}")
    
    def _show_copied_message(self) -> None:
        """Show a brief 'Copied!' message."""
        # Create a temporary label
        temp_label = tk.Label(
            self,
            text="✓ Copied!",
            bg="#4CAF50",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=5
        )
        temp_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Remove after 1 second
        self.after(1000, temp_label.destroy)


class ImageWidget(ttk.Frame):  # type: ignore[misc]
    """Widget for displaying images inline."""
    
    def __init__(
        self,
        parent: Any,
        image_path: str,
        alt_text: str = "",
        max_width: int = 600,
        max_height: int = 400,
        **kwargs: Any
    ) -> None:
        """Initialize image widget.
        
        Args:
            parent: Parent widget
            image_path: Path or URL to the image
            alt_text: Alternative text for the image
            max_width: Maximum display width
            max_height: Maximum display height
            **kwargs: Additional frame options
        """
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        super().__init__(parent, **kwargs)
        self.image_path = image_path
        self.alt_text = alt_text
        self.max_width = max_width
        self.max_height = max_height
        
        # Try to load and display the image
        self._load_image()
    
    def _load_image(self) -> None:
        """Load and display the image."""
        try:
            # Check if it's a local file
            if os.path.exists(self.image_path):
                self._display_local_image()
            else:
                # URL or invalid path
                self._display_placeholder(f"Image: {self.image_path}")
        except Exception as e:
            self._display_placeholder(f"Failed to load image: {e}")
    
    def _display_local_image(self) -> None:
        """Display a local image file."""
        try:
            from PIL import Image, ImageTk  # type: ignore
            
            # Load image
            img = Image.open(self.image_path)
            
            # Resize if needed
            width, height = img.size
            if width > self.max_width or height > self.max_height:
                ratio = min(self.max_width / width, self.max_height / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display in label
            label = ttk.Label(self, image=photo)  # type: ignore
            label.image = photo  # Keep a reference!  # type: ignore
            label.pack(padx=5, pady=5)
            
            # Add caption if alt text provided
            if self.alt_text:
                caption = ttk.Label(
                    self,
                    text=self.alt_text,
                    font=("Segoe UI", 9, "italic"),
                    foreground="#666666"
                )
                caption.pack()
        except ImportError:
            self._display_placeholder("PIL/Pillow not installed - cannot display images")
        except Exception as e:
            self._display_placeholder(f"Error loading image: {e}")
    
    def _display_placeholder(self, message: str) -> None:
        """Display a placeholder when image can't be loaded.
        
        Args:
            message: Message to display
        """
        frame = ttk.Frame(self, relief="ridge", borderwidth=2)
        frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        icon = ttk.Label(frame, text="🖼️", font=("Segoe UI", 24))
        icon.pack(pady=(10, 5))
        
        label = ttk.Label(
            frame,
            text=message,
            font=("Segoe UI", 9),
            wraplength=self.max_width - 20
        )
        label.pack(pady=(5, 10), padx=10)


class VideoWidget(ttk.Frame):  # type: ignore[misc]
    """Widget for displaying video controls and preview."""
    
    def __init__(
        self,
        parent: Any,
        video_path: str,
        **kwargs: Any
    ) -> None:
        """Initialize video widget.
        
        Args:
            parent: Parent widget
            video_path: Path or URL to the video
            **kwargs: Additional frame options
        """
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        super().__init__(parent, relief="ridge", borderwidth=2, **kwargs)
        self.video_path = video_path
        
        # Video icon and info
        icon = ttk.Label(self, text="🎥", font=("Segoe UI", 32))
        icon.pack(pady=(15, 10))
        
        # Video path/name
        video_name = Path(video_path).name if os.path.exists(video_path) else video_path
        name_label = ttk.Label(
            self,
            text=video_name,
            font=("Segoe UI", 10, "bold")
        )
        name_label.pack(pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        open_btn = ttk.Button(
            btn_frame,
            text="▶ Open Video",
            command=self._open_video
        )
        open_btn.pack(side="left", padx=5)
        
        copy_path_btn = ttk.Button(
            btn_frame,
            text="📋 Copy Path",
            command=self._copy_path
        )
        copy_path_btn.pack(side="left", padx=5)
        
        # Info label
        if os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            info_label = ttk.Label(
                self,
                text=f"Size: {size_mb:.2f} MB",
                font=("Segoe UI", 9),
                foreground="#666666"
            )
            info_label.pack(pady=(0, 10))
    
    def _open_video(self) -> None:
        """Open video in default player."""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(self.video_path)  # type: ignore
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", self.video_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.video_path])
        except Exception as e:
            if messagebox:
                messagebox.showerror("Error", f"Failed to open video: {e}")
    
    def _copy_path(self) -> None:
        """Copy video path to clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.video_path)
            if messagebox:
                messagebox.showinfo("Copied", "Video path copied to clipboard")
        except Exception as e:
            if messagebox:
                messagebox.showerror("Copy Error", f"Failed to copy: {e}")


class LinkWidget(ttk.Frame):  # type: ignore[misc]
    """Widget for displaying clickable links."""
    
    def __init__(
        self,
        parent: Any,
        url: str,
        text: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize link widget.
        
        Args:
            parent: Parent widget
            url: URL to link to
            text: Display text (defaults to URL)
            **kwargs: Additional frame options
        """
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        super().__init__(parent, **kwargs)
        self.url = url
        
        display_text = text or url
        
        link_label = tk.Label(
            self,
            text=f"🔗 {display_text}",
            fg="#0066CC",
            cursor="hand2",
            font=("Segoe UI", 9, "underline")
        )
        link_label.pack(pady=2)
        link_label.bind("<Button-1>", lambda e: self._open_link())
        
        try:
            from .tooltips import add_tooltip
            add_tooltip(link_label, f"Click to open: {url}")
        except Exception:
            pass
    
    def _open_link(self) -> None:
        """Open link in default browser."""
        try:
            import webbrowser
            webbrowser.open(self.url)
        except Exception as e:
            if messagebox:
                messagebox.showerror("Error", f"Failed to open link: {e}")
