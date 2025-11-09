"""Message parser for detecting and extracting different content types from AI responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class MessageSegment:
    """Represents a segment of a message with a specific type."""
    
    type: Literal["text", "code", "image", "video", "link"]
    content: str
    metadata: dict[str, str] | None = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class MessageParser:
    """Parser for extracting rich content from AI responses.
    
    Supports:
    - Markdown code blocks: ```language\\ncode\\n```
    - Image references: ![alt](url) or [IMAGE: path/url]
    - Video references: [VIDEO: path/url]
    - URLs: http(s)://...
    - Plain text
    """
    
    # Regex patterns
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w+)?\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    IMAGE_MD_PATTERN = re.compile(
        r'!\[(.*?)\]\((.*?)\)'
    )
    
    IMAGE_TAG_PATTERN = re.compile(
        r'\[IMAGE:\s*(.*?)\]',
        re.IGNORECASE
    )
    
    VIDEO_TAG_PATTERN = re.compile(
        r'\[VIDEO:\s*(.*?)\]',
        re.IGNORECASE
    )
    
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, message: str) -> list[MessageSegment]:
        """Parse a message into segments of different types.
        
        Args:
            message: The message text to parse
            
        Returns:
            List of MessageSegment objects representing the parsed content
        """
        segments: list[MessageSegment] = []
        remaining = message
        position = 0
        
        while remaining:
            # Track the earliest match
            earliest_match = None
            earliest_pos = len(remaining)
            match_type = None
            
            # Check for code blocks
            code_match = cls.CODE_BLOCK_PATTERN.search(remaining)
            if code_match and code_match.start() < earliest_pos:
                earliest_match = code_match
                earliest_pos = code_match.start()
                match_type = "code"
            
            # Check for markdown images
            img_md_match = cls.IMAGE_MD_PATTERN.search(remaining)
            if img_md_match and img_md_match.start() < earliest_pos:
                earliest_match = img_md_match
                earliest_pos = img_md_match.start()
                match_type = "image_md"
            
            # Check for image tags
            img_tag_match = cls.IMAGE_TAG_PATTERN.search(remaining)
            if img_tag_match and img_tag_match.start() < earliest_pos:
                earliest_match = img_tag_match
                earliest_pos = img_tag_match.start()
                match_type = "image_tag"
            
            # Check for video tags
            video_match = cls.VIDEO_TAG_PATTERN.search(remaining)
            if video_match and video_match.start() < earliest_pos:
                earliest_match = video_match
                earliest_pos = video_match.start()
                match_type = "video"
            
            # If no special content found, treat rest as text
            if earliest_match is None:
                if remaining.strip():
                    segments.append(MessageSegment(type="text", content=remaining))
                break
            
            # Add text before the match
            if earliest_pos > 0:
                text_before = remaining[:earliest_pos]
                if text_before.strip():
                    segments.append(MessageSegment(type="text", content=text_before))
            
            # Add the matched segment
            if match_type == "code":
                language = earliest_match.group(1) or "text"
                code_content = earliest_match.group(2)
                segments.append(MessageSegment(
                    type="code",
                    content=code_content,
                    metadata={"language": language}
                ))
            elif match_type == "image_md":
                alt_text = earliest_match.group(1)
                url = earliest_match.group(2)
                segments.append(MessageSegment(
                    type="image",
                    content=url,
                    metadata={"alt": alt_text}
                ))
            elif match_type == "image_tag":
                url = earliest_match.group(1).strip()
                segments.append(MessageSegment(
                    type="image",
                    content=url,
                    metadata={}
                ))
            elif match_type == "video":
                url = earliest_match.group(1).strip()
                segments.append(MessageSegment(
                    type="video",
                    content=url,
                    metadata={}
                ))
            
            # Move past the match
            remaining = remaining[earliest_match.end():]
            position = earliest_match.end()
        
        return segments
    
    @classmethod
    def extract_links(cls, text: str) -> list[str]:
        """Extract all URLs from text.
        
        Args:
            text: Text to search for URLs
            
        Returns:
            List of found URLs
        """
        return cls.URL_PATTERN.findall(text)
    
    @classmethod
    def has_code_blocks(cls, text: str) -> bool:
        """Check if text contains code blocks.
        
        Args:
            text: Text to check
            
        Returns:
            True if code blocks are present
        """
        return bool(cls.CODE_BLOCK_PATTERN.search(text))
    
    @classmethod
    def has_media(cls, text: str) -> bool:
        """Check if text contains image or video references.
        
        Args:
            text: Text to check
            
        Returns:
            True if media references are present
        """
        return bool(
            cls.IMAGE_MD_PATTERN.search(text) or
            cls.IMAGE_TAG_PATTERN.search(text) or
            cls.VIDEO_TAG_PATTERN.search(text)
        )
