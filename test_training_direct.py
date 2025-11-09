"""
Direct test of dual GPU training using environment variables.
This avoids Windows multiprocessing issues by setting up rank manually.
"""
import os
import sys

# Set up environment for Rank 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["AIOS_DDP_RANK"] = "0"
os.environ["AIOS_DDP_WORLD"] = "1"  # Single process to avoid multiprocessing
os.environ["AIOS_DDP_BACKEND"] = "gloo"

# Import after setting environment
from aios.cli.hrm_hf.train_actv1_impl import train_actv1_impl

def test_training():
    """Test training with DDP settings."""
    print("=" * 60)
    print("Testing Single Process with DDP Infrastructure")
    print("=" * 60)
    
    try:
        # Minimal training run
        train_actv1_impl(
            model="gpt2",
            dataset_file=None,
            max_seq_len=128,
            batch_size=2,
            steps=2,
            lr=1e-4,
            device="cuda",
            halt_max_steps=1,
            save_dir="artifacts/brains/actv1/test_fix",
            teacher=None,
            teacher_device="cpu",
            kl=0.0,
            kl_temp=1.0,
            ascii_only=False,
            eval_file=None,
            eval_minutes=60,
            eval_batches=1,
            auto_tune=False,
            cpu_util_target=None,
            sys_mem_cap_pct=None,
            util_target=None,
            util_target_map=None,
            stop_file=None,
            teacher_dataset=True,
            td_num_samples=4,
            td_max_new_tokens=8,
            td_batch=2,
            td_auto_batch=False,
            td_util_target=None,
            td_util_target_map=None,
            td_temperature=0.7,
            td_top_p=0.9,
            td_top_k=50,
            td_prompt=None,
            td_seed=None,
            log_file="artifacts/brains/actv1/test_fix_metrics.jsonl",
            student_init=None,
            brain_name=None,
            bundle_dir="artifacts/brains/actv1",
            h_layers=2,
            l_layers=2,
            hidden_size=512,
            expansion=2.0,
            num_heads=8,
            h_cycles=1,
            l_cycles=1,
            pos_encodings="learned",
            cuda_ids="0,1",
            iterate=False,
            ddp=False,  # Don't spawn, we're testing single process
            world_size=1,
            strict=False,
        )
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        return True
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Training failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)
