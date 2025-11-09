#!/usr/bin/env python3
    model_path_obj = Path(model_path)
    
    print(f"Loading model from {model_path_obj}...")
    model = GPT2LMHeadModel.from_pretrained(str(model_path_obj))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path_obj))Extend Position Embeddings for Long Context
This script extends GPT-2's position embeddings from 1024 to 5120 tokens.
"""

import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, AutoTokenizer
import shutil

def extend_position_embeddings(
    model_path: str = "artifacts/hf_implant/gpt2",
    new_max_length: int = 5120,
    backup: bool = True
):
    """
    Extend position embeddings for a GPT-2 model.
    
    Args:
        model_path: Path to the model directory
        new_max_length: New maximum context length
        backup: Whether to backup the original model
    """
    model_path_obj = Path(model_path)
    
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Get current position embeddings
    old_pos_emb = model.transformer.wpe.weight.data
    old_max_length = old_pos_emb.shape[0]
    embedding_dim = old_pos_emb.shape[1]
    
    print(f"Current max length: {old_max_length}")
    print(f"Target max length: {new_max_length}")
    print(f"Embedding dimension: {embedding_dim}")
    
    if new_max_length <= old_max_length:
        print("❌ New length must be greater than current length!")
        return
    
    # Backup if requested
    if backup:
        backup_path = model_path_obj.parent / f"{model_path_obj.name}_backup_ctx{old_max_length}"
        if not backup_path.exists():
            print(f"Creating backup at {backup_path}...")
            shutil.copytree(model_path_obj, backup_path)
        else:
            print(f"Backup already exists at {backup_path}")
    
    # Method 1: Linear Interpolation (smoother)
    print(f"\n📐 Extending position embeddings using linear interpolation...")
    
    # Reshape for interpolation: [old_len, dim] -> [1, dim, old_len]
    old_reshaped = old_pos_emb.transpose(0, 1).unsqueeze(0)
    
    # Interpolate to new length
    new_pos_emb = torch.nn.functional.interpolate(
        old_reshaped,
        size=new_max_length,
        mode='linear',
        align_corners=True
    )
    
    # Reshape back: [1, dim, new_len] -> [new_len, dim]
    new_pos_emb = new_pos_emb.squeeze(0).transpose(0, 1)
    
    # Alternative Method 2: Copy + Random Init (uncomment to use instead)
    # print(f"\n🎲 Extending position embeddings with copying + random init...")
    # new_pos_emb = torch.zeros(new_max_length, embedding_dim, dtype=old_pos_emb.dtype)
    # new_pos_emb[:old_max_length] = old_pos_emb  # Copy old embeddings
    # # Initialize new positions with small random values
    # torch.nn.init.normal_(new_pos_emb[old_max_length:], mean=0.0, std=0.02)
    
    # Update model
    print(f"Updating model with new embeddings...")
    model.transformer.wpe = torch.nn.Embedding(new_max_length, embedding_dim)
    model.transformer.wpe.weight.data = new_pos_emb
    
    # Update config
    model.config.n_positions = new_max_length
    model.config.n_ctx = new_max_length
    
    # Save extended model
    print(f"💾 Saving extended model to {model_path_obj}...")
    model.save_pretrained(str(model_path_obj))
    tokenizer.save_pretrained(str(model_path_obj))
    
    print(f"\n✅ SUCCESS! Model now supports {new_max_length} tokens")
    print(f"   Old context: {old_max_length} tokens")
    print(f"   New context: {new_max_length} tokens")
    print(f"   Increase: {(new_max_length/old_max_length - 1)*100:.1f}%")
    
    print(f"\n⚠️  IMPORTANT: You must now fine-tune the model on long sequences!")
    print(f"   The interpolated embeddings need training to work properly.")
    
    return model


def verify_extension(model_path: str = "artifacts/hf_implant/gpt2"):
    """Verify the model can handle long sequences."""
    print(f"\n🔍 Verifying extended model...")
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    max_len = model.config.n_positions
    print(f"Model max context: {max_len}")
    
    # Test with a long sequence
    test_len = min(max_len - 100, 5000)
    test_text = "This is a test. " * (test_len // 4)
    
    print(f"Testing with {test_len} token sequence...")
    tokens = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=test_len)
    
    actual_len = tokens['input_ids'].shape[1]
    print(f"Actual token count: {actual_len}")
    
    try:
        with torch.no_grad():
            output = model(**tokens)
        print(f"✅ Model successfully processed {actual_len} tokens!")
        print(f"   Output shape: {output.logits.shape}")
        return True
    except Exception as e:
        print(f"❌ Error processing long sequence: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extend GPT-2 position embeddings")
    parser.add_argument("--model", default="artifacts/hf_implant/gpt2", help="Model path")
    parser.add_argument("--length", type=int, default=5120, help="New max length")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't extend")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_extension(args.model)
    else:
        model = extend_position_embeddings(
            args.model,
            args.length,
            backup=not args.no_backup
        )
        verify_extension(args.model)
        
        print("\n📚 Next Steps:")
        print("1. Fine-tune the model on long sequences:")
        print(f"   aios hrm-hf train-actv1 --model {args.model} --max-seq-len {args.length} ...")
        print("2. Start with smaller batch sizes (--batch-size 1 or 2)")
        print("3. Monitor memory usage")
        print("4. Gradually increase sequence length during training")
