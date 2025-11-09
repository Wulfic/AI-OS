#!/usr/bin/env python
"""Debug script to investigate why specific tokenizers fail training."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import AutoTokenizer
import json

# Failing tokenizers
failing_tokenizers = [
    ('artifacts/hf_implant/tokenizers/deepseek-coder-v2', 'deepseek-coder-v2'),
    ('artifacts/hf_implant/tokenizers/llava-1.5', 'llava-1.5'),
    ('artifacts/hf_implant/tokenizers/phi3-mini', 'phi3-mini'),
    ('artifacts/hf_implant/tokenizers/qwen2.5-7b', 'qwen2.5-7b'),
]

# Passing tokenizers for comparison
passing_tokenizers = [
    ('artifacts/hf_implant/tokenizers/gpt2', 'gpt2'),
    ('artifacts/hf_implant/tokenizers/mistral-7b', 'mistral-7b'),
]

print("=" * 80)
print("TOKENIZER ANALYSIS: Failing vs Passing")
print("=" * 80)

test_text = "This is a test sentence for tokenizer validation."

def analyze_tokenizer(path, name):
    """Analyze tokenizer properties and encoding behavior."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name}")
    print(f"{'='*80}")
    
    try:
        tok = AutoTokenizer.from_pretrained(path)
        
        # Basic properties
        print(f"\n1. BASIC PROPERTIES:")
        print(f"   Vocab size: {tok.vocab_size}")
        print(f"   Model max length: {tok.model_max_length}")
        print(f"   Padding side: {tok.padding_side}")
        print(f"   Truncation side: {getattr(tok, 'truncation_side', 'N/A')}")
        
        # Special tokens
        print(f"\n2. SPECIAL TOKENS:")
        print(f"   PAD token: {tok.pad_token} (ID: {tok.pad_token_id})")
        print(f"   BOS token: {tok.bos_token} (ID: {tok.bos_token_id})")
        print(f"   EOS token: {tok.eos_token} (ID: {tok.eos_token_id})")
        print(f"   UNK token: {tok.unk_token} (ID: {tok.unk_token_id})")
        
        # Check if special token IDs are within vocab_size
        special_ids = [tok.pad_token_id, tok.bos_token_id, tok.eos_token_id, tok.unk_token_id]
        special_ids = [x for x in special_ids if x is not None]
        
        if special_ids:
            max_special_id = max(special_ids)
            print(f"   Max special token ID: {max_special_id}")
            if max_special_id >= tok.vocab_size:
                print(f"   ⚠️  WARNING: Special token ID {max_special_id} >= vocab_size {tok.vocab_size}!")
        
        # All special tokens
        if hasattr(tok, 'all_special_tokens'):
            print(f"   All special tokens: {tok.all_special_tokens[:5]}...")  # Show first 5
        if hasattr(tok, 'all_special_ids'):
            special_ids_list = tok.all_special_ids
            print(f"   All special IDs: {special_ids_list[:10]}...")  # Show first 10
            if special_ids_list:
                print(f"   Range: {min(special_ids_list)} to {max(special_ids_list)}")
                out_of_bounds = [x for x in special_ids_list if x >= tok.vocab_size]
                if out_of_bounds:
                    print(f"   ⚠️  OUT OF BOUNDS SPECIAL IDs: {out_of_bounds}")
        
        # Encoding test
        print(f"\n3. ENCODING TEST:")
        encoded = tok.encode(test_text)
        print(f"   Text: {test_text}")
        print(f"   Encoded IDs: {encoded}")
        print(f"   Token count: {len(encoded)}")
        print(f"   Min ID: {min(encoded)}, Max ID: {max(encoded)}")
        
        if max(encoded) >= tok.vocab_size:
            print(f"   ⚠️  ERROR: Token ID {max(encoded)} >= vocab_size {tok.vocab_size}!")
        
        # Encoding with padding
        print(f"\n4. ENCODING WITH PADDING:")
        encoded_dict = tok(test_text, padding="max_length", max_length=20, return_tensors="pt")
        input_ids = encoded_dict['input_ids'][0].tolist()
        attention_mask = encoded_dict['attention_mask'][0].tolist()
        
        print(f"   Input IDs: {input_ids}")
        print(f"   Attention mask: {attention_mask}")
        print(f"   Padded positions: {[i for i, x in enumerate(attention_mask) if x == 0]}")
        
        # Check for IDs >= vocab_size
        out_of_bounds_ids = [x for x in input_ids if x >= tok.vocab_size]
        if out_of_bounds_ids:
            print(f"   ⚠️  OUT OF BOUNDS IDs in encoding: {set(out_of_bounds_ids)}")
        
        # Tokenizer config
        print(f"\n5. TOKENIZER CONFIG:")
        if hasattr(tok, 'init_kwargs'):
            print(f"   Init kwargs: {tok.init_kwargs}")
        
        # Check added tokens
        if hasattr(tok, 'added_tokens_encoder'):
            added_count = len(tok.added_tokens_encoder)
            print(f"   Added tokens count: {added_count}")
            if added_count > 0 and added_count < 20:
                print(f"   Added tokens: {dict(list(tok.added_tokens_encoder.items())[:10])}")
        
        # Vocab size vs actual vocab
        if hasattr(tok, 'get_vocab'):
            actual_vocab = tok.get_vocab()
            actual_vocab_size = len(actual_vocab)
            print(f"   Actual vocab dict size: {actual_vocab_size}")
            if actual_vocab_size != tok.vocab_size:
                print(f"   ⚠️  MISMATCH: vocab_size={tok.vocab_size} but vocab dict has {actual_vocab_size} entries")
                
                # Find IDs beyond vocab_size
                max_id = max(actual_vocab.values())
                print(f"   Max ID in vocab dict: {max_id}")
                if max_id >= tok.vocab_size:
                    beyond_vocab = {k: v for k, v in actual_vocab.items() if v >= tok.vocab_size}
                    print(f"   ⚠️  Tokens with ID >= vocab_size: {dict(list(beyond_vocab.items())[:20])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

print("\n" + "="*80)
print("FAILING TOKENIZERS")
print("="*80)

for path, name in failing_tokenizers:
    analyze_tokenizer(path, name)

print("\n" + "="*80)
print("PASSING TOKENIZERS (for comparison)")
print("="*80)

for path, name in passing_tokenizers:
    analyze_tokenizer(path, name)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
