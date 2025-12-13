# TOON Format Integration

### Summary

Integrate Token-Oriented Object Notation (TOON) as an optional serialization format for AI-OS to reduce token consumption when passing structured data to LLMs. TOON achieves 30-60% token savings over JSON for uniform tabular data while maintaining human readability and LLM-friendliness. This integration will support encoding/decoding in training pipelines, evaluation outputs, and LLM prompts.

### Why this matters

- **Token cost reduction**: Training and inference with large context windows can save 30-60% tokens on structured data, reducing API costs and improving throughput.
- **LLM-friendly structure**: Explicit length markers `[N]` and field headers `{field1,field2}` help LLMs validate and generate structured output more reliably.
- **Flexible serialization**: Drop-in replacement for JSON in data paths where uniform arrays dominate (logs, metrics, evaluation results, datasets).
- **Python ecosystem alignment**: Multiple Python implementations available; official `toon_format` package in active development.

---

## What ships in PF-007

- **Core utility module**: `src/aios/formats/toon_codec.py` (encoder/decoder with fallback to JSON)
- **CLI integration**: 
  - `aios toon encode` - Convert JSON to TOON
  - `aios toon decode` - Convert TOON to JSON
  - Flags for existing commands: `--output-format toon|json` for metrics/evaluation outputs
- **Training/Eval hooks**: Optional TOON encoding for JSONL metrics and evaluation results
- **Config support**: YAML settings for TOON preferences (delimiter, indent, length markers)
- **Documentation**: Examples, benchmarks, and best practices for when to use TOON vs JSON

---

## Architecture overview

### Data paths affected

1. **Metrics logging** (`artifacts/brains/actv1/metrics.jsonl`)
   - Optional TOON encoding for training metrics with `--metrics-format toon`
   - Particularly beneficial for batch eval results with uniform structure

2. **Evaluation outputs** (`artifacts/evaluation/`)
   - Encode evaluation results in TOON format for token-efficient LLM analysis
   - Support both formats side-by-side for compatibility

3. **Dataset exports** (`training_data/`)
   - Convert curated datasets to TOON format for reduced storage and faster LLM ingestion
   - Command: `aios datasets export --format toon`

4. **LLM prompt payloads** (inline usage)
   - Utility functions to encode context data in TOON before inserting into prompts
   - Automatic format detection when decoding LLM outputs

### Core components

- **ToonCodec** (utility class): Wraps TOON encoder/decoder with graceful fallback
  - `encode(data: Any, *, delimiter: str = ',', indent: int = 2, length_marker: bool = False) -> str`
  - `decode(toon_str: str, *, strict: bool = True) -> Any`
  - `is_toon_available() -> bool` (checks if TOON library installed)
  
- **Format negotiation**: Auto-detect format based on content or extension
  - `.toon` extension for TOON files
  - `.json` or `.jsonl` for JSON files
  - Content sniffing: look for TOON patterns like `[N]{fields}:` headers

- **Config schema** (`config/default.yaml`):
  ```yaml
  toon:
    enabled: false  # Master switch
    default_format: json  # json | toon
    delimiter: ','  # ',' | '\t' | '|'
    indent: 2
    length_marker: false  # Add # prefix to lengths
    metrics: false  # Use TOON for metrics logging
    evaluation: false  # Use TOON for evaluation outputs
  ```

---

## Dependencies and setup

### Python implementation options

**Option 1: Official implementation (recommended)**
```powershell
pip install toon-format  # When officially released
```

**Option 2: Community implementation (current)**
```powershell
pip install python-toon  # https://github.com/xaviviro/python-toon
```

**Option 3: Custom lightweight implementation**
- Implement minimal TOON encoder/decoder following [TOON spec v1.4](https://github.com/toon-format/spec)
- Use conformance tests from spec repo to validate
- Fallback option if no stable Python package exists

### Installation strategy

Make TOON optional dependency:
```toml
# pyproject.toml
[project.optional-dependencies]
toon = [
    "python-toon>=0.1.0",  # Or official package when available
]
```

Graceful degradation: If TOON not installed, log info message and fall back to JSON.

---

## Implementation details

### File: `src/aios/formats/toon_codec.py`

```python
"""TOON (Token-Oriented Object Notation) codec with fallback to JSON."""

from typing import Any, Literal, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ToonCodec:
    """Encode/decode TOON format with graceful JSON fallback."""
    
    def __init__(self):
        self._toon_available = self._check_toon_available()
        if self._toon_available:
            try:
                from toon_format import encode as toon_encode
                from toon_format import decode as toon_decode
                self._encode_fn = toon_encode
                self._decode_fn = toon_decode
            except ImportError:
                # Try alternative package name
                from toon import encode as toon_encode
                from toon import decode as toon_decode
                self._encode_fn = toon_encode
                self._decode_fn = toon_decode
    
    def _check_toon_available(self) -> bool:
        """Check if TOON library is installed."""
        try:
            import toon_format
            return True
        except ImportError:
            try:
                import toon
                return True
            except ImportError:
                return False
    
    def is_available(self) -> bool:
        """Return True if TOON encoding is available."""
        return self._toon_available
    
    def encode(
        self,
        data: Any,
        *,
        delimiter: Literal[',', '\t', '|'] = ',',
        indent: int = 2,
        length_marker: bool = False,
    ) -> str:
        """Encode data to TOON format, fallback to JSON."""
        if not self._toon_available:
            logger.debug("TOON not available, falling back to JSON")
            return json.dumps(data, indent=indent)
        
        try:
            options = {
                'delimiter': delimiter,
                'indent': indent,
            }
            if length_marker:
                options['length_marker'] = '#'
            
            return self._encode_fn(data, **options)
        except Exception as e:
            logger.warning(f"TOON encoding failed: {e}, falling back to JSON")
            return json.dumps(data, indent=indent)
    
    def decode(self, content: str, *, strict: bool = True) -> Any:
        """Decode TOON or JSON format (auto-detect)."""
        # Try TOON first if available and content looks like TOON
        if self._toon_available and self._looks_like_toon(content):
            try:
                return self._decode_fn(content, strict=strict)
            except Exception as e:
                logger.debug(f"TOON decoding failed: {e}, trying JSON")
        
        # Fallback to JSON
        return json.loads(content)
    
    def _looks_like_toon(self, content: str) -> bool:
        """Heuristic check if content is TOON format."""
        # Look for TOON patterns: array headers like [N], [N]{fields}:
        lines = content.strip().split('\n')[:5]  # Check first few lines
        for line in lines:
            if '[' in line and ']:' in line:
                return True
            if '[' in line and ']{' in line and '}:' in line:
                return True
        return False

# Global singleton
_codec = ToonCodec()

def encode_toon(data: Any, **options) -> str:
    """Convenience function for encoding."""
    return _codec.encode(data, **options)

def decode_toon(content: str, **options) -> Any:
    """Convenience function for decoding."""
    return _codec.decode(content, **options)

def is_toon_available() -> bool:
    """Check if TOON encoding is available."""
    return _codec.is_available()
```

### File: `src/aios/cli/toon_cli.py`

```python
"""CLI commands for TOON format conversion."""

import typer
from pathlib import Path
from typing import Optional, Literal
import json

from aios.formats.toon_codec import ToonCodec, is_toon_available

app = typer.Typer(help="Convert between JSON and TOON formats")

@app.command()
def encode(
    input_path: Path = typer.Argument(..., help="Input JSON file or - for stdin"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output TOON file (stdout if omitted)"),
    delimiter: Literal[',', '\t', '|'] = typer.Option(',', "--delimiter", help="Array delimiter"),
    indent: int = typer.Option(2, "--indent", help="Indentation spaces"),
    length_marker: bool = typer.Option(False, "--length-marker", help="Add # prefix to array lengths"),
    stats: bool = typer.Option(False, "--stats", help="Show token count estimates"),
):
    """Convert JSON to TOON format."""
    if not is_toon_available():
        typer.echo("Error: TOON library not installed. Install with: pip install python-toon", err=True)
        raise typer.Exit(1)
    
    # Read input
    if str(input_path) == '-':
        import sys
        data = json.load(sys.stdin)
    else:
        with open(input_path) as f:
            data = json.load(f)
    
    # Encode
    codec = ToonCodec()
    toon_output = codec.encode(
        data,
        delimiter=delimiter,
        indent=indent,
        length_marker=length_marker,
    )
    
    # Show stats if requested
    if stats:
        json_output = json.dumps(data, indent=indent)
        json_tokens = estimate_tokens(json_output)
        toon_tokens = estimate_tokens(toon_output)
        savings = (1 - toon_tokens / json_tokens) * 100
        typer.echo(f"\n📊 Token Comparison:", err=True)
        typer.echo(f"  JSON:    {json_tokens:,} tokens", err=True)
        typer.echo(f"  TOON:    {toon_tokens:,} tokens", err=True)
        typer.echo(f"  Savings: {savings:.1f}%\n", err=True)
    
    # Write output
    if output_path:
        output_path.write_text(toon_output)
        typer.echo(f"✓ Encoded to {output_path}")
    else:
        typer.echo(toon_output)

@app.command()
def decode(
    input_path: Path = typer.Argument(..., help="Input TOON file or - for stdin"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSON file (stdout if omitted)"),
    indent: int = typer.Option(2, "--indent", help="JSON indentation spaces"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Enable strict validation"),
):
    """Convert TOON to JSON format."""
    if not is_toon_available():
        typer.echo("Error: TOON library not installed. Install with: pip install python-toon", err=True)
        raise typer.Exit(1)
    
    # Read input
    if str(input_path) == '-':
        import sys
        content = sys.stdin.read()
    else:
        content = input_path.read_text()
    
    # Decode
    codec = ToonCodec()
    data = codec.decode(content, strict=strict)
    
    # Write output
    json_output = json.dumps(data, indent=indent)
    if output_path:
        output_path.write_text(json_output)
        typer.echo(f"✓ Decoded to {output_path}")
    else:
        typer.echo(json_output)

def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars = 1 token for English)."""
    return len(text) // 4
```

### Integration with training pipeline

**File: `src/aios/cli/hrm_hf/train_actv1.py`**

Add TOON support to metrics logging:

```python
def _write_jsonl_helper(
    log_file: Path,
    payload: dict,
    *,
    format: Literal['json', 'toon'] = 'json',
    toon_options: Optional[dict] = None,
):
    """Write metrics in JSON or TOON format."""
    if format == 'toon':
        from aios.formats.toon_codec import is_toon_available, encode_toon
        if is_toon_available():
            toon_options = toon_options or {}
            line = encode_toon(payload, **toon_options)
        else:
            logger.warning("TOON not available, falling back to JSON")
            line = json.dumps(payload)
    else:
        line = json.dumps(payload)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
```

**CLI flags in `src/aios/cli/hrm_hf_cli.py`**:

```python
@app.command()
def train_actv1(
    # ... existing params ...
    metrics_format: Literal['json', 'toon'] = typer.Option(
        'json',
        '--metrics-format',
        help='Output format for metrics (json or toon)',
    ),
    toon_delimiter: Literal[',', '\t', '|'] = typer.Option(
        ',',
        '--toon-delimiter',
        help='Delimiter for TOON format (comma, tab, or pipe)',
    ),
):
    """Train with optional TOON metrics output."""
    # ... pass through to training function
```

---

## CLI design

### New command group: `aios toon`

```powershell
# Encode JSON to TOON
aios toon encode input.json -o output.toon

# Decode TOON to JSON
aios toon decode data.toon -o output.json

# Pipe operations
cat data.json | aios toon encode --stats

# Tab-delimited for better compression
aios toon encode input.json --delimiter "\t" --stats
```

### Integration with existing commands

```powershell
# Training with TOON metrics
aios hrm-hf train-actv1 \
    --model gpt2 \
    --dataset-file training_data/curated_datasets/test_sample.txt \
    --metrics-format toon \
    --toon-delimiter "\t" \
    --log-file artifacts/brains/actv1/metrics.toon

# Export evaluation in TOON format
aios evaluation export \
    --format toon \
    --output artifacts/evaluation/results.toon
```

---

## Testing and acceptance criteria

### Unit tests: `tests/test_toon_integration.py`

```python
def test_toon_codec_encode_simple():
    """Test encoding simple objects."""
    data = {"id": 123, "name": "Alice", "active": True}
    codec = ToonCodec()
    output = codec.encode(data)
    assert "id:" in output
    assert "Alice" in output

def test_toon_codec_encode_tabular():
    """Test encoding arrays of objects (tabular format)."""
    data = {
        "items": [
            {"sku": "A1", "qty": 2, "price": 9.99},
            {"sku": "B2", "qty": 1, "price": 14.5},
        ]
    }
    codec = ToonCodec()
    output = codec.encode(data)
    assert "items[2]{sku,qty,price}:" in output
    assert "A1,2,9.99" in output

def test_toon_codec_roundtrip():
    """Test encode -> decode roundtrip."""
    original = {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
        ]
    }
    codec = ToonCodec()
    encoded = codec.encode(original)
    decoded = codec.decode(encoded)
    assert decoded == original

def test_toon_fallback_when_unavailable():
    """Test graceful fallback to JSON when TOON not installed."""
    # Mock TOON unavailable
    # Verify JSON output returned instead
```

### Integration tests

```powershell
# Test metrics logging with TOON
aios hrm-hf train-actv1 \
    --model gpt2 \
    --dataset-file training_data/curated_datasets/test_sample.txt \
    --steps 1 --batch-size 2 \
    --metrics-format toon \
    --log-file artifacts/test_toon_metrics.toon

# Verify output is valid TOON
aios toon decode artifacts/test_toon_metrics.toon
```

### Acceptance criteria

- ✅ TOON encoding reduces token count by 30-60% for tabular metrics data
- ✅ CLI can convert between JSON and TOON formats bidirectionally
- ✅ Training pipeline can output metrics in TOON format with flag
- ✅ Graceful fallback to JSON when TOON not installed (no crashes)
- ✅ Documentation includes benchmarks and when to use TOON vs JSON

---

## Use cases and recommendations

### When to use TOON

✅ **Excellent candidates:**
- Training metrics with uniform structure (loss, accuracy, step numbers)
- Evaluation results with consistent fields across samples
- Large tabular datasets for LLM ingestion
- Configuration exports with repeated structures
- Batch prediction outputs

✅ **Benefits:**
- 30-60% token savings on uniform tabular data
- Better than JSON for arrays of objects
- Comparable to CSV but with nested object support

### When to stick with JSON

❌ **Keep using JSON for:**
- Deeply nested configurations with varied structures
- Non-uniform data with inconsistent fields
- Legacy compatibility requirements
- Data consumed by non-TOON-aware tools

### Benchmark expectations

Based on TOON project benchmarks:

| Data Type | TOON vs JSON | TOON vs JSON (compact) |
|-----------|--------------|------------------------|
| Uniform employee records | -60.7% | -36.8% |
| E-commerce orders (mixed) | -33.1% | +5.5% |
| Event logs (semi-uniform) | -15.0% | +19.9% |
| Deeply nested config | -31.3% | +11.9% |

*Note: TOON typically uses more tokens than minified JSON for deeply nested data, but offers better structure for LLMs.*

---

## Risks and mitigations

### Risk: Python package stability
- **Mitigation**: Use optional dependency + graceful fallback; consider custom implementation
- **Status**: Official `toon_format` package in development; community implementations available

### Risk: Format adoption
- **Mitigation**: Keep JSON as default; TOON opt-in via flags; support both formats side-by-side
- **Status**: TOON gaining traction in LLM community; 11.3k GitHub stars

### Risk: Compatibility with existing tools
- **Mitigation**: Provide conversion utilities; document migration path
- **Status**: CLI tools make conversion trivial

### Risk: LLM understanding
- **Mitigation**: Include format examples in prompts; test with multiple models
- **Status**: Benchmarks show 68.7% retrieval accuracy (vs JSON 65.7%) with token savings

---

## Rollout plan

### Phase 1: Foundation (2 days)
- [ ] Implement `ToonCodec` utility with fallback
- [ ] Add optional dependency to `pyproject.toml`
- [ ] Create unit tests for encode/decode
- [ ] Add config schema to `config/default.yaml`

### Phase 2: CLI tools (1 day)
- [ ] Implement `aios toon encode/decode` commands
- [ ] Add `--stats` flag for token comparison
- [ ] Test conversion workflows

### Phase 3: Training integration (1 day)
- [ ] Add `--metrics-format toon` flag to training CLI
- [ ] Implement TOON output in `_write_jsonl_helper`
- [ ] Test training runs with TOON metrics

### Phase 4: Documentation & examples (1 day)
- [ ] Write user guide with examples
- [ ] Add benchmark results
- [ ] Create migration guide from JSON to TOON
- [ ] Document when to use TOON vs JSON

### Phase 5: Optional enhancements (future)
- [ ] TOON support in evaluation exports
- [ ] Dataset conversion utilities
- [ ] VS Code extension for TOON syntax highlighting
- [ ] Custom TOON implementation if official package delays

---

## Documentation outline

### User guide sections

1. **What is TOON?**
   - Overview and benefits
   - Token efficiency benchmarks
   - LLM-friendly structure

2. **Installation**
   ```powershell
   pip install "aios[toon]"
   ```

3. **Quick start**
   - Converting files
   - Using in training
   - Reading TOON output

4. **When to use TOON**
   - Decision matrix
   - Use case examples
   - Performance expectations

5. **Format reference**
   - Syntax examples
   - Object encoding
   - Array encoding (tabular)
   - Delimiter options

6. **Troubleshooting**
   - Package not installed
   - Format detection issues
   - Encoding errors

### Examples

```powershell
# Example 1: Compare token usage
aios toon encode training_metrics.json --stats

# Example 2: Convert existing metrics to TOON
Get-ChildItem artifacts/brains/actv1/*.jsonl | ForEach-Object {
    $output = $_.FullName -replace '\.jsonl$', '.toon'
    aios toon encode $_.FullName -o $output --delimiter "\t" --stats
}

# Example 3: Training with TOON output
aios hrm-hf train-actv1 \
    --model artifacts/hf_implant/gpt2 \
    --dataset-file training_data/curated_datasets/test_sample.txt \
    --steps 100 --batch-size 4 \
    --metrics-format toon \
    --toon-delimiter "\t" \
    --log-file artifacts/brains/actv1/metrics.toon

# Example 4: Analyze TOON metrics with LLM
$metrics = Get-Content artifacts/brains/actv1/metrics.toon -Raw
# Pass to LLM with prompt: "Analyze these training metrics in TOON format..."
```

---

## References

- TOON official spec: https://github.com/toon-format/spec (v1.4)
- TOON TypeScript implementation: https://github.com/toon-format/toon
- TOON website: https://toonformat.dev/
- Community Python implementation: https://github.com/xaviviro/python-toon
- Official Python package (in dev): https://github.com/toon-format/toon-python
- Format playground: https://www.curiouslychase.com/playground/format-tokenization-exploration

### Key insights from TOON project

1. **Token efficiency**: 30-60% savings on uniform data
2. **LLM accuracy**: 68.7% vs JSON's 65.7% on retrieval tasks
3. **Best use case**: Arrays of objects with identical primitive fields
4. **Delimiter impact**: Tab delimiter often more efficient than comma
5. **Length markers**: `[#N]` notation helps LLMs validate structure

---

## Future enhancements

### Post-MVP features

1. **Streaming TOON parser**
   - Process large TOON files line-by-line
   - Memory-efficient for big datasets

2. **TOON compression analysis**
   - Automated tool to analyze JSON → TOON savings potential
   - Recommend TOON for files above threshold

3. **VS Code integration**
   - Syntax highlighting for `.toon` files
   - Hover tooltips with field info
   - Convert commands in context menu

4. **Custom TOON recognizers for AI-OS**
   - Domain-specific patterns (model names, metrics)
   - Optimized for AI-OS data structures

5. **TOON-aware log viewer**
   - GUI tool to browse TOON metrics
   - Side-by-side comparison with JSON

---

## Developer checklist

### Implementation
- [ ] Create `src/aios/formats/toon_codec.py`
- [ ] Create `src/aios/cli/toon_cli.py`
- [ ] Add TOON config to `config/default.yaml`
- [ ] Update `pyproject.toml` with optional dependency
- [ ] Integrate into training pipeline
- [ ] Add CLI flags to `hrm_hf_cli.py`

### Testing
- [ ] Unit tests for `ToonCodec`
- [ ] CLI command tests
- [ ] Integration test with training
- [ ] Roundtrip tests (encode → decode)
- [ ] Fallback behavior tests

### Documentation
- [ ] User guide in `docs/guide/toon_integration.md`
- [ ] Update training docs with TOON flags
- [ ] Add examples to `docs/examples/`
- [ ] Update README with TOON mention

### Validation
- [ ] Benchmark token savings on actual AI-OS metrics
- [ ] Test LLM comprehension with TOON metrics
- [ ] Verify graceful degradation without TOON package
- [ ] Performance testing on large datasets

---

## Operator checklist

### Pre-deployment
- [ ] Install TOON package: `pip install python-toon`
- [ ] Test conversion: `aios toon encode <sample.json> --stats`
- [ ] Verify token savings meet expectations (>20% for tabular data)

### Migration
- [ ] Convert existing metrics: `aios toon encode <metrics.jsonl> -o <metrics.toon>`
- [ ] Update analysis scripts to handle TOON format
- [ ] Train team on TOON syntax and when to use it

### Production use
- [ ] Add `--metrics-format toon` to training scripts for long runs
- [ ] Use tab delimiter for maximum compression: `--toon-delimiter "\t"`
- [ ] Monitor file sizes and token usage
- [ ] Keep JSON format for compatibility where needed

---

## Success metrics

### Quantitative
- ✅ 30-60% token reduction on uniform training metrics
- ✅ No performance regression in training throughput
- ✅ 100% roundtrip accuracy (encode → decode)
- ✅ Graceful fallback when TOON unavailable

### Qualitative
- ✅ Users can easily convert between formats
- ✅ LLM analysis of TOON metrics is as good or better than JSON
- ✅ Documentation is clear and includes decision matrix
- ✅ Integration feels natural (opt-in, not forced)

---

## Quickstart (Windows/PowerShell)

```powershell
# 1) Install TOON support
pip install "aios[toon]"

# 2) Test conversion on existing file
aios toon encode artifacts/brains/actv1/metrics.jsonl --stats

# 3) Run training with TOON output
aios hrm-hf train-actv1 `
    --model gpt2 `
    --dataset-file training_data/curated_datasets/test_sample.txt `
    --steps 100 --batch-size 4 `
    --metrics-format toon `
    --toon-delimiter "\t" `
    --log-file artifacts/brains/actv1/metrics.toon

# 4) Convert back to JSON for analysis
aios toon decode artifacts/brains/actv1/metrics.toon -o metrics_decoded.json

# 5) Compare file sizes
(Get-Item artifacts/brains/actv1/metrics.toon).Length
(Get-Item metrics_decoded.json).Length
```

---

## Summary

TOON integration provides a token-efficient alternative to JSON for AI-OS data serialization, particularly beneficial for training metrics, evaluation outputs, and LLM prompt payloads. The implementation follows AI-OS patterns with optional dependencies, graceful fallbacks, and clear documentation. By adopting TOON for uniform tabular data, operators can reduce token costs by 30-60% while maintaining or improving LLM comprehension.
