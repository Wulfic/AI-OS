"""
Simple test for parallel independent training - sequential for debugging
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_sequential_blocks():
    """Test training blocks sequentially first"""
    print("="*60)
    print("Testing Sequential Block Training")
    print("="*60)
    
    # Simple config
    batch_size = 2
    max_seq_len = 128
    num_blocks = 4
    steps_per_block = 5
    checkpoint_dir = Path("artifacts/test_sequential")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 100)
    )
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ Using CPU")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train each block
    checkpoints = []
    for block_id in range(num_blocks):
        print(f"\n🔥 Training Block {block_id}...")
        model.train()
        total_loss = 0.0
        
        for step in range(steps_per_block):
            # Dummy data
            input_ids = torch.randint(0, 100, (batch_size, max_seq_len), device=device).float()
            labels = torch.randint(0, 100, (batch_size,), device=device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # Simple MSE loss for testing
            target = torch.randn_like(outputs.mean(dim=1))
            loss = torch.nn.functional.mse_loss(
                outputs.mean(dim=1),  # Pool sequence dimension
                target
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 2 == 0:
                print(f"  Step {step+1}/{steps_per_block}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / steps_per_block
        print(f"✅ Block {block_id} complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint (safetensors format)
        ckpt_path = checkpoint_dir / f"block{block_id}.safetensors"
        try:
            from safetensors.torch import save_file as save_safetensors
            save_safetensors(model.state_dict(), str(ckpt_path))
        except ImportError:
            ckpt_path = checkpoint_dir / f"block{block_id}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'block_id': block_id,
                'avg_loss': avg_loss
            }, ckpt_path)
        checkpoints.append(str(ckpt_path))
        print(f"   Saved: {ckpt_path}")
    
    print("\n" + "="*60)
    print("Testing Checkpoint Merging")
    print("="*60)
    
    # Test merging
    print(f"\n🔄 Merging {len(checkpoints)} checkpoints...")
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    
    # Load checkpoints (support both formats)
    loaded = []
    for p in checkpoints:
        try:
            state_dict = load_safetensors(p, device='cpu')
            loaded.append({'model_state_dict': state_dict})
        except Exception:
            loaded.append(torch.load(p, map_location='cpu'))
    
    state_dicts = [c['model_state_dict'] for c in loaded]
    
    # Average weights
    merged_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)
        merged_state[key] = stacked.mean(dim=0)
    
    # Save merged
    final_path = checkpoint_dir / "merged_model.safetensors"
    save_safetensors(merged_state, str(final_path))
    print(f"✅ Merged checkpoint saved: {final_path}")
    
    # Verify we can load it
    test_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 100)
    )
    test_model.load_state_dict(merged_state)
    print("✅ Successfully loaded merged weights into model")
    
    print("\n" + "="*60)
    print("✅ All Tests Passed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test with actual multi-GPU (spawn separate processes)")
    print("2. Integrate with ACTV1 training infrastructure")
    print("3. Add to CLI with --parallel-independent flag")
    print("4. Add GUI button in resources page")

if __name__ == "__main__":
    test_sequential_blocks()
