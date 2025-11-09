#!/usr/bin/env python
"""Debug vocab_size handling for different tokenizers."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoTokenizer
from aios.cli.hrm_hf.model import build_actv1_config, build_student

# Test Qwen 2.5
print("=" * 70)
print("Testing Qwen 2.5 (Should FAIL)")
print("=" * 70)
tok = AutoTokenizer.from_pretrained('artifacts/hf_implant/tokenizers/qwen2.5-7b')
print(f'Tokenizer vocab_size: {tok.vocab_size}')

cfg = build_actv1_config(
    batch_size=1,
    max_seq_len=128,
    vocab_size=tok.vocab_size,
    h_cycles=2,
    l_cycles=2,
    h_layers=2,
    l_layers=2,
    hidden_size=512,
    expansion=2.0,
    num_heads=8,
    pos_encodings='rope',
    halt_max_steps=1,
    window_size=None
)
print(f'Config vocab_size: {cfg["vocab_size"]}')

model = build_student(cfg)
print(f'Model embed_tokens.num_embeddings: {model.inner.embed_tokens.embedding_weight.shape[0]}')

# Test encoding
text = "This is a test sentence for tokenizer validation."
ids = tok.encode(text)
print(f'Encoded IDs: {ids}')
print(f'Max ID: {max(ids)}')
print(f'Max ID >= vocab_size? {max(ids) >= tok.vocab_size}')
print(f'Max ID >= model embeddings? {max(ids) >= model.inner.embed_tokens.embedding_weight.shape[0]}')

# Test GPT-2 for comparison
print("\n" + "=" * 70)
print("Testing GPT-2 (Should WORK)")
print("=" * 70)
tok2 = AutoTokenizer.from_pretrained('artifacts/hf_implant/tokenizers/gpt2')
print(f'Tokenizer vocab_size: {tok2.vocab_size}')

cfg2 = build_actv1_config(
    batch_size=1,
    max_seq_len=128,
    vocab_size=tok2.vocab_size,
    h_cycles=2,
    l_cycles=2,
    h_layers=2,
    l_layers=2,
    hidden_size=512,
    expansion=2.0,
    num_heads=8,
    pos_encodings='rope',
    halt_max_steps=1,
    window_size=None
)
print(f'Config vocab_size: {cfg2["vocab_size"]}')

model2 = build_student(cfg2)
print(f'Model embed_tokens.num_embeddings: {model2.inner.embed_tokens.embedding_weight.shape[0]}')

# Test encoding
ids2 = tok2.encode(text)
print(f'Encoded IDs: {ids2}')
print(f'Max ID: {max(ids2)}')
print(f'Max ID >= vocab_size? {max(ids2) >= tok2.vocab_size}')
print(f'Max ID >= model embeddings? {max(ids2) >= model2.inner.embed_tokens.embedding_weight.shape[0]}')
