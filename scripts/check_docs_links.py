import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# very small, local-only link check for markdown files
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`]*`")

failures = []

for md in DOCS.rglob("*.md"):
    raw = md.read_text(encoding="utf-8", errors="ignore")
    # strip fenced code blocks and inline code to avoid false positives like foo[bar](baz)
    text = FENCED_BLOCK_RE.sub("", raw)
    text = INLINE_CODE_RE.sub("", text)
    for m in MD_LINK_RE.finditer(text):
        label, target = m.groups()
        # skip URLs with a scheme or anchors only
        if re.match(r"^[a-z]+://", target) or target.startswith("#"):
            continue
        # strip anchors
        path_part = target.split("#", 1)[0]
        # ignore empty or mailto
        if not path_part or path_part.startswith("mailto:"):
            continue
        # resolve relative to the current md's parent
        link_path = (md.parent / path_part).resolve()
        if not link_path.exists():
            failures.append((str(md.relative_to(ROOT)), target))

if failures:
    print("Broken local links found (relative to docs):")
    for src, tgt in failures:
        print(f" - {src}: {tgt}")
    raise SystemExit(1)
else:
    print("No broken local links detected in docs/.")
