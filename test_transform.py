import re

html = '''
<a href="#section-one">TOC Link</a>
<a href="file.md">Relative Link</a>
<a href="http://example.com">External</a>
'''

def replace_link(match):
    href = match.group(1)
    rest = match.group(2) if match.lastindex >= 2 else ""
    
    if href.startswith('#'):
        print(f"Keeping anchor link as-is: {href}")
        return f'<a href="{href}"{rest}'
    
    if href.startswith('http://') or href.startswith('https://'):
        return f'<a href="{href}"{rest} target="_blank"'
    
    print(f"Transforming link to custom protocol: {href}")
    return f'<a href="aios-doc://{href}"{rest}'

pattern = r'<a href="([^"]+)"([^>]*)'
transformed = re.sub(pattern, replace_link, html)

print("\n=== ORIGINAL ===")
print(html)
print("\n=== TRANSFORMED ===")
print(transformed)
