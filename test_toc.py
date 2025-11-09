import markdown

md_text = """# Main Title

[TOC]

## Section One

Some content here.

## Section Two

More content.
"""

html = markdown.markdown(md_text, extensions=['toc'])
print(html)
