"""
Examples demonstrating the Rich Chat System features.

This script provides examples of:
1. Formatting AI responses for rich display
2. Using the message parser
3. Creating custom content
4. Integrating with existing systems
"""

# Example 1: AI Response with Code Block
def generate_code_response():
    """Generate a response containing a code snippet."""
    return """
Here's a solution to your problem:

```python
def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage
for i in range(10):
    print(f"F({i}) = {calculate_fibonacci(i)}")
```

This implementation uses an iterative approach for better performance.
"""


# Example 2: AI Response with Multiple Content Types
def generate_mixed_response():
    """Generate a response with text, code, and images."""
    return """
Let me explain the concept with both code and visuals:

First, here's the basic structure:

```javascript
class DataProcessor {
    constructor(data) {
        this.data = data;
    }
    
    process() {
        return this.data.map(item => item * 2);
    }
}

// Usage
const processor = new DataProcessor([1, 2, 3, 4, 5]);
console.log(processor.process()); // [2, 4, 6, 8, 10]
```

Here's a visual representation:

![Architecture Diagram](./docs/images/architecture.png)

For more details, check out this video tutorial:

[VIDEO: ./tutorials/data_processing.mp4]

You can also visit the documentation: https://example.com/docs
"""


# Example 3: Using the Message Parser Directly
def parse_message_example():
    """Example of parsing a message manually."""
    from aios.gui.components import MessageParser
    
    message = """
    Here's some text before the code.
    
    ```python
    print("Hello, World!")
    ```
    
    And here's an image:
    ![Example](./image.png)
    """
    
    segments = MessageParser.parse(message)
    
    for segment in segments:
        print(f"Type: {segment.type}")
        print(f"Content: {segment.content[:50]}...")
        print(f"Metadata: {segment.metadata}")
        print("---")


# Example 4: Custom AI Response Formatter
class RichResponseFormatter:
    """Helper class to format AI responses with rich content."""
    
    @staticmethod
    def format_code(code: str, language: str = "python") -> str:
        """Format code as a markdown code block."""
        return f"```{language}\n{code}\n```"
    
    @staticmethod
    def format_image(path: str, alt_text: str = "") -> str:
        """Format an image reference."""
        if alt_text:
            return f"![{alt_text}]({path})"
        return f"[IMAGE: {path}]"
    
    @staticmethod
    def format_video(path: str) -> str:
        """Format a video reference."""
        return f"[VIDEO: {path}]"
    
    @staticmethod
    def format_link(url: str, text: str | None = None) -> str:
        """Format a hyperlink."""
        if text:
            return f"[{text}]({url})"
        return url
    
    @staticmethod
    def create_tutorial_response(
        title: str,
        explanation: str,
        code: str,
        image_path: str | None = None,
        video_path: str | None = None
    ) -> str:
        """Create a comprehensive tutorial response."""
        parts = [f"# {title}", "", explanation, ""]
        
        if code:
            parts.append("**Code Example:**")
            parts.append("")
            parts.append(RichResponseFormatter.format_code(code))
            parts.append("")
        
        if image_path:
            parts.append("**Visual Reference:**")
            parts.append("")
            parts.append(RichResponseFormatter.format_image(image_path))
            parts.append("")
        
        if video_path:
            parts.append("**Video Tutorial:**")
            parts.append("")
            parts.append(RichResponseFormatter.format_video(video_path))
            parts.append("")
        
        return "\n".join(parts)


# Example 5: Using the Rich Response Formatter
def generate_tutorial_response():
    """Generate a complete tutorial response."""
    formatter = RichResponseFormatter()
    
    return formatter.create_tutorial_response(
        title="Getting Started with Python Functions",
        explanation="""
Functions are reusable blocks of code that perform specific tasks.
They help organize code and make it more maintainable.
        """,
        code="""
def greet(name, greeting="Hello"):
    \"\"\"Greet a person with a custom message.\"\"\"
    return f"{greeting}, {name}!"

# Examples
print(greet("Alice"))              # Hello, Alice!
print(greet("Bob", "Hi"))          # Hi, Bob!
print(greet("Charlie", "Hey"))     # Hey, Charlie!
        """,
        image_path="./tutorials/function_diagram.png",
        video_path="./tutorials/functions_explained.mp4"
    )


# Example 6: Testing Rich Content in Chat
def test_rich_chat():
    """Test function to demonstrate rich chat capabilities."""
    test_cases = [
        {
            "name": "Code Only",
            "response": generate_code_response()
        },
        {
            "name": "Mixed Content",
            "response": generate_mixed_response()
        },
        {
            "name": "Tutorial Format",
            "response": generate_tutorial_response()
        }
    ]
    
    print("Rich Chat Test Cases")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 50)
        print(test_case['response'])
        print("\n" + "=" * 50)


# Example 7: Integrating with Existing Brain/Router
def integrate_with_brain_example():
    """Example of integrating rich responses with the brain system."""
    
    def enhanced_brain_handler(user_input: str) -> str:
        """Enhanced handler that formats responses for rich display."""
        
        # Get basic response from brain (existing logic)
        basic_response = "This is the brain's response."
        
        # Enhance with rich content based on user request
        if "code" in user_input.lower() or "example" in user_input.lower():
            # Add code block
            code_example = """
def example_function():
    return "This is an example"
            """
            basic_response += "\n\n" + RichResponseFormatter.format_code(code_example)
        
        if "show" in user_input.lower() and "image" in user_input.lower():
            # Add image reference
            basic_response += "\n\n" + RichResponseFormatter.format_image(
                "./artifacts/example.png",
                "Example visualization"
            )
        
        return basic_response
    
    # Test the enhanced handler
    test_inputs = [
        "Can you show me a code example?",
        "Show me an image of the architecture",
        "Just explain the concept"
    ]
    
    for inp in test_inputs:
        print(f"\nInput: {inp}")
        print(f"Response:\n{enhanced_brain_handler(inp)}")
        print("-" * 50)


# Example 8: Custom Content Type (Future Enhancement)
def custom_table_example():
    """Example of how to add table support (future enhancement)."""
    
    # This would be the desired output format
    table_response = """
Here's a comparison of different algorithms:

| Algorithm | Time Complexity | Space Complexity | Best For |
|-----------|----------------|------------------|----------|
| Quick Sort | O(n log n) | O(log n) | General purpose |
| Merge Sort | O(n log n) | O(n) | Stable sorting |
| Heap Sort | O(n log n) | O(1) | Limited memory |
| Bubble Sort | O(n²) | O(1) | Small datasets |

As you can see, each has its strengths.
    """
    
    return table_response


# Example 9: Batch Message Testing
def batch_test_messages():
    """Test multiple message types in sequence."""
    messages = [
        ("Simple text", "This is a simple text response."),
        ("Code snippet", "Here's a quick example:\n\n```python\nprint('Hello!')```"),
        ("With image", "Check this out:\n\n![Demo](./demo.png)"),
        ("Full tutorial", generate_tutorial_response()),
    ]
    
    print("Batch Message Testing")
    print("=" * 60)
    
    for name, message in messages:
        from aios.gui.components import MessageParser
        
        print(f"\n{name}")
        print("-" * 60)
        segments = MessageParser.parse(message)
        print(f"Segments found: {len(segments)}")
        for i, seg in enumerate(segments, 1):
            print(f"  {i}. Type: {seg.type}, Length: {len(seg.content)} chars")
        print()


# Example 10: Export Format Preview
def preview_export_formats():
    """Preview how messages look in different export formats."""
    
    sample_messages = [
        {"role": "user", "content": "Can you show me a Python example?"},
        {"role": "assistant", "content": generate_code_response()},
        {"role": "user", "content": "That's helpful, thanks!"},
    ]
    
    # HTML preview
    print("HTML Export Preview:")
    print("=" * 60)
    print("<div class='message user'>")
    print("  <strong>You:</strong> Can you show me a Python example?")
    print("</div>")
    print("<div class='message assistant'>")
    print("  <strong>AI:</strong>")
    print("  <pre><code class='language-python'>")
    print("def calculate_fibonacci(n):")
    print("    ...")
    print("  </code></pre>")
    print("</div>")
    print()
    
    # Markdown preview
    print("Markdown Export Preview:")
    print("=" * 60)
    print("## You")
    print("Can you show me a Python example?")
    print()
    print("## AI")
    print("Here's a solution:")
    print("```python")
    print("def calculate_fibonacci(n):")
    print("    ...")
    print("```")
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("Rich Chat System - Usage Examples")
    print("=" * 80)
    
    print("\n[1] Code Response Example")
    print(generate_code_response())
    
    print("\n[2] Mixed Content Example")
    print(generate_mixed_response())
    
    print("\n[3] Message Parsing Example")
    parse_message_example()
    
    print("\n[4] Tutorial Response Example")
    print(generate_tutorial_response())
    
    print("\n[5] Rich Chat Test Cases")
    test_rich_chat()
    
    print("\n[6] Brain Integration Example")
    integrate_with_brain_example()
    
    print("\n[7] Batch Message Testing")
    batch_test_messages()
    
    print("\n[8] Export Format Preview")
    preview_export_formats()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
