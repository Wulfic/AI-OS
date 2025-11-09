"""Check vocab sizes for all available tokenizers."""

from transformers import AutoTokenizer
from pathlib import Path

tokenizers_dir = Path("artifacts/hf_implant/tokenizers")
tokenizers = sorted([d.name for d in tokenizers_dir.iterdir() if d.is_dir()])

print("Tokenizer Vocab Sizes:")
print("=" * 80)

for name in tokenizers:
    try:
        tok_path = f"artifacts/hf_implant/tokenizers/{name}"
        tok = AutoTokenizer.from_pretrained(tok_path)
        vocab_len = len(tok)
        vocab_size = getattr(tok, 'vocab_size', vocab_len)
        print(f"{name:20s}: len={vocab_len:<8d} vocab_size={vocab_size:<8d}")
    except Exception as e:
        print(f"{name:20s}: ERROR - {e}")

print("=" * 80)
