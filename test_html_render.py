import tkinter as tk
from tkinterweb import HtmlFrame

root = tk.Tk()
root.title("Test HTML Render")
root.geometry("800x600")

frame = HtmlFrame(root, messages_enabled=False)
frame.pack(fill="both", expand=True)

# Try loading simple HTML
html = """
<html>
<head><meta charset='utf-8'></head>
<body>
<h1>Test Heading</h1>
<p>This is a test paragraph with <b>bold</b> and <i>italic</i> text.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
<li>Item 3</li>
</ul>
<a href="http://example.com">External Link</a>
<a href="#section">Internal Link</a>
</body>
</html>
"""

print("Loading HTML...")
frame.load_html(html, base_url="http://test.local/")
print("HTML loaded")

root.mainloop()
