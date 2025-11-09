"""Minimal test to verify chunking gradient fix."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from aios.core.hrm_models.chunked_training.core import chunked_segment_rollout

def create_minimal_model():
    """Create minimal model for testing."""
    class MinimalCarry:
        def __init__(self, B, hidden_size, device):
            self.current_data = None
            self.halted = torch.zeros(B, dtype=torch.bool, device=device)
            self.hidden = torch.zeros(B, 1, hidden_size, device=device)
    
    class MinimalModel(torch.nn.Module):
        def __init__(self, vocab_size=100, hidden_size=64):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Simple embedding and output layers
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size).to(self.device)
            self.output = torch.nn.Linear(hidden_size, vocab_size).to(self.device)
            self.q_halt = torch.nn.Linear(hidden_size, 1).to(self.device)
            self.q_continue = torch.nn.Linear(hidden_size, 1).to(self.device)
        
        def initial_carry(self, batch):
            B = batch["inputs"].shape[0]
            return MinimalCarry(B, self.hidden_size, self.device)
        
        def forward(self, carry, batch):
            inputs = batch["inputs"]  # [B, S]
            B, S = inputs.shape
            
            # Simple forward: embed -> output
            hidden = self.embedding(inputs)  # [B, S, hidden]
            logits = self.output(hidden)  # [B, S, vocab]
            
            # Q-values from last position
            last_hidden = hidden[:, -1, :]  # [B, hidden]
            q_halt = self.q_halt(last_hidden).squeeze(-1)  # [B]
            q_continue = self.q_continue(last_hidden).squeeze(-1)  # [B]
            
            outputs = {
                "logits": logits,
                "q_halt_logits": q_halt,
                "q_continue_logits": q_continue,
            }
            
            # Update carry with current data
            carry.current_data = batch
            carry.hidden = hidden[:, -1:, :]  # [B, 1, hidden]
            
            return carry, outputs
        
        def __call__(self, carry, batch):
            """Make model callable for chunked_segment_rollout."""
            return self.forward(carry, batch)
    
    return MinimalModel()


def test_chunking_with_gradient():
    """Test that chunking returns loss with proper grad_fn."""
    print("Testing context chunking gradient fix...")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create minimal model
    model = create_minimal_model()
    model.train()
    
    # Create test batch
    B, S = 2, 512
    vocab_size = 100
    
    batch = {
        "inputs": torch.randint(0, vocab_size, (B, S), device=device),
        "targets": torch.randint(0, vocab_size, (B, S), device=device),
        "puzzle_identifiers": torch.zeros(B, device=device),
    }
    
    print(f"Batch: {B} samples, {S} tokens")
    print(f"Testing with chunk_size=256")
    print()
    
    # Run chunked segment rollout
    try:
        loss, metrics = chunked_segment_rollout(
            model=model,
            batch=batch,
            max_segments=3,
            chunk_size=256,
            epsilon=0.1,
            gradient_checkpointing=False,
            use_cpu_offload=False,
        )
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn}")
        print()
        
        # Try backward
        print("Attempting backward...")
        loss.backward()
        print("[SUCCESS] Backward pass completed!")
        print()
        
        # Check gradients
        has_grads = False
        for name, param in [("embedding", model.embedding.weight),
                           ("output", model.output.weight)]:
            if param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm().item():.4f}")
                has_grads = True
        
        if has_grads:
            print("\n[SUCCESS] Gradients computed successfully!")
            return True
        else:
            print("\n[FAILED] No gradients computed")
            return False
            
    except RuntimeError as e:
        print(f"[FAILED] RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"[FAILED] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_chunking_with_gradient()
    sys.exit(0 if success else 1)
