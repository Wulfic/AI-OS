"""
Unit tests for the Rich Chat System components.

Run with: pytest tests/test_rich_chat.py
"""

import pytest
from aios.gui.components.message_parser import MessageParser, MessageSegment


class TestMessageParser:
    """Tests for the MessageParser class."""
    
    def test_parse_simple_text(self):
        """Test parsing plain text without special content."""
        message = "This is a simple message."
        segments = MessageParser.parse(message)
        
        assert len(segments) == 1
        assert segments[0].type == "text"
        assert segments[0].content == message
    
    def test_parse_code_block(self):
        """Test parsing markdown code blocks."""
        message = """
Here's some code:

```python
def hello():
    print("Hello, World!")
```

That's it!
"""
        segments = MessageParser.parse(message)
        
        # Should have text before, code, and text after
        assert len(segments) >= 2
        
        # Find the code segment
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        
        code_seg = code_segments[0]
        assert code_seg.metadata is not None
        assert code_seg.metadata.get("language") == "python"
        assert "def hello" in code_seg.content
        assert "print" in code_seg.content
    
    def test_parse_code_block_without_language(self):
        """Test parsing code blocks without language specification."""
        message = "```\nsome code\n```"
        segments = MessageParser.parse(message)
        
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        assert code_segments[0].metadata is not None
        assert code_segments[0].metadata.get("language") == "text"
    
    def test_parse_markdown_image(self):
        """Test parsing markdown-style images."""
        message = "Check this: ![Alt text](path/to/image.png)"
        segments = MessageParser.parse(message)
        
        image_segments = [s for s in segments if s.type == "image"]
        assert len(image_segments) == 1
        
        img_seg = image_segments[0]
        assert img_seg.content == "path/to/image.png"
        assert img_seg.metadata is not None
        assert img_seg.metadata.get("alt") == "Alt text"
    
    def test_parse_image_tag(self):
        """Test parsing custom image tags."""
        message = "Look: [IMAGE: /path/to/file.jpg]"
        segments = MessageParser.parse(message)
        
        image_segments = [s for s in segments if s.type == "image"]
        assert len(image_segments) == 1
        assert "/path/to/file.jpg" in image_segments[0].content
    
    def test_parse_video_tag(self):
        """Test parsing video tags."""
        message = "Watch this: [VIDEO: tutorial.mp4]"
        segments = MessageParser.parse(message)
        
        video_segments = [s for s in segments if s.type == "video"]
        assert len(video_segments) == 1
        assert "tutorial.mp4" in video_segments[0].content
    
    def test_parse_mixed_content(self):
        """Test parsing message with multiple content types."""
        message = """
Text before code.

```python
def test():
    pass
```

More text.

![Image](img.png)

Final text.

[VIDEO: vid.mp4]
"""
        segments = MessageParser.parse(message)
        
        # Should have text, code, text, image, text, video
        assert len(segments) >= 5
        
        types = [s.type for s in segments]
        assert "text" in types
        assert "code" in types
        assert "image" in types
        assert "video" in types
    
    def test_extract_links(self):
        """Test URL extraction."""
        text = "Visit https://example.com or http://test.org for more info."
        links = MessageParser.extract_links(text)
        
        assert len(links) == 2
        assert "https://example.com" in links
        assert "http://test.org" in links
    
    def test_has_code_blocks(self):
        """Test code block detection."""
        with_code = "Here: ```python\ncode\n```"
        without_code = "Just text here"
        
        assert MessageParser.has_code_blocks(with_code) is True
        assert MessageParser.has_code_blocks(without_code) is False
    
    def test_has_media(self):
        """Test media detection."""
        with_media = "Check: ![img](file.png)"
        without_media = "Just text"
        
        assert MessageParser.has_media(with_media) is True
        assert MessageParser.has_media(without_media) is False
    
    def test_empty_message(self):
        """Test parsing empty or whitespace-only messages."""
        assert MessageParser.parse("") == []
        assert MessageParser.parse("   ") == []
    
    def test_multiple_code_blocks(self):
        """Test parsing multiple code blocks in one message."""
        message = """
First block:
```python
print("1")
```

Second block:
```javascript
console.log("2");
```
"""
        segments = MessageParser.parse(message)
        code_segments = [s for s in segments if s.type == "code"]
        
        assert len(code_segments) == 2
        assert code_segments[0].metadata is not None
        assert code_segments[0].metadata.get("language") == "python"
        assert code_segments[1].metadata is not None
        assert code_segments[1].metadata.get("language") == "javascript"


class TestMessageSegment:
    """Tests for the MessageSegment dataclass."""
    
    def test_create_text_segment(self):
        """Test creating a text segment."""
        seg = MessageSegment(type="text", content="Hello")
        assert seg.type == "text"
        assert seg.content == "Hello"
        assert seg.metadata == {}
    
    def test_create_code_segment(self):
        """Test creating a code segment with metadata."""
        seg = MessageSegment(
            type="code",
            content="print('hi')",
            metadata={"language": "python"}
        )
        assert seg.type == "code"
        assert seg.metadata is not None
        assert seg.metadata.get("language") == "python"
    
    def test_metadata_initialization(self):
        """Test that metadata is initialized as empty dict if not provided."""
        seg = MessageSegment(type="text", content="test")
        assert seg.metadata is not None
        assert isinstance(seg.metadata, dict)
        assert len(seg.metadata) == 0


class TestRichChatIntegration:
    """Integration tests for the rich chat system."""
    
    def test_format_code_response(self):
        """Test formatting a typical code response."""
        response = """
Here's the solution:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This uses recursion.
"""
        segments = MessageParser.parse(response)
        
        # Should have text, code, text
        assert len(segments) >= 3
        
        # Verify code is parsed correctly
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        assert "factorial" in code_segments[0].content
        assert code_segments[0].metadata is not None
        assert code_segments[0].metadata.get("language") == "python"
    
    def test_format_tutorial_response(self):
        """Test formatting a tutorial with multiple content types."""
        response = """
# Python Tutorial

Learn the basics:

```python
print("Hello, World!")
```

Visual guide:
![Python Logo](logo.png)

Video lesson:
[VIDEO: lesson1.mp4]

More info: https://python.org
"""
        segments = MessageParser.parse(response)
        
        # Should have multiple types
        types = {s.type for s in segments}
        assert "text" in types
        assert "code" in types
        assert "image" in types
        assert "video" in types


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_nested_code_markers(self):
        """Test code blocks that contain code markers in the content."""
        message = "```python\nprint('```')\n```"
        segments = MessageParser.parse(message)
        
        code_segments = [s for s in segments if s.type == "code"]
        # Should parse as one code block
        assert len(code_segments) == 1
    
    def test_special_characters_in_content(self):
        """Test content with special characters."""
        message = "```python\nprint('<>&\"')\n```"
        segments = MessageParser.parse(message)
        
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        assert "<>&\"" in code_segments[0].content
    
    def test_very_long_code_block(self):
        """Test parsing very long code blocks."""
        long_code = "\n".join([f"line_{i}" for i in range(1000)])
        message = f"```python\n{long_code}\n```"
        segments = MessageParser.parse(message)
        
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        assert len(code_segments[0].content) > 5000
    
    def test_unicode_content(self):
        """Test parsing content with unicode characters."""
        message = "```python\nprint('Hello 世界 🌍')\n```"
        segments = MessageParser.parse(message)
        
        code_segments = [s for s in segments if s.type == "code"]
        assert len(code_segments) == 1
        assert "世界" in code_segments[0].content
        assert "🌍" in code_segments[0].content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
