"""Test stop_after_epoch feature.

This test validates that the stop_after_epoch parameter properly stops training
after the current epoch completes, setting stopped_early=True.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.core.hrm_training import TrainingConfig
from aios.cli.hrm_hf.finalization import finalize_training


def test_training_config_stop_after_epoch():
    """Test that TrainingConfig includes stop_after_epoch parameter."""
    
    # Test default value (False)
    config = TrainingConfig(
        dataset_file="test.txt"
    )
    assert hasattr(config, "stop_after_epoch"), "TrainingConfig should have stop_after_epoch attribute"
    assert config.stop_after_epoch is False, "stop_after_epoch should default to False"
    
    # Test explicit True value
    config_enabled = TrainingConfig(
        dataset_file="test.txt",
        stop_after_epoch=True
    )
    assert config_enabled.stop_after_epoch is True, "stop_after_epoch should be True when set"
    
    print("✓ TrainingConfig stop_after_epoch parameter works correctly")


def test_to_cli_args_includes_stop_after_epoch():
    """Test that to_cli_args() includes --stop-after-epoch flag when enabled."""
    
    # Test with stop_after_epoch=False (default)
    config_disabled = TrainingConfig(
        dataset_file="test.txt",
        stop_after_epoch=False
    )
    args_disabled = config_disabled.to_cli_args()
    assert "--stop-after-epoch" not in args_disabled, "--stop-after-epoch should not be in args when False"
    
    # Test with stop_after_epoch=True
    config_enabled = TrainingConfig(
        dataset_file="test.txt",
        stop_after_epoch=True
    )
    args_enabled = config_enabled.to_cli_args()
    assert "--stop-after-epoch" in args_enabled, "--stop-after-epoch should be in args when True"
    
    print("✓ to_cli_args() correctly includes stop_after_epoch flag")


def test_to_dict_includes_stop_after_epoch():
    """Test that to_dict() includes stop_after_epoch in serialization."""
    
    config = TrainingConfig(
        dataset_file="test.txt",
        stop_after_epoch=True
    )
    
    config_dict = config.to_dict()
    assert "stop_after_epoch" in config_dict, "to_dict() should include stop_after_epoch"
    assert config_dict["stop_after_epoch"] is True, "stop_after_epoch should be True in dict"
    
    print("✓ to_dict() correctly includes stop_after_epoch")


def test_from_dict_restores_stop_after_epoch():
    """Test that from_dict() restores stop_after_epoch from dictionary."""
    
    config_dict = {
        "dataset_file": "test.txt",
        "stop_after_epoch": True
    }
    
    config = TrainingConfig.from_dict(config_dict)
    assert config.stop_after_epoch is True, "from_dict() should restore stop_after_epoch"
    
    print("✓ from_dict() correctly restores stop_after_epoch")


def test_finalize_training_reports_stop_after_epoch(tmp_path):
    """finalize_training should treat stop_after_epoch as a successful run."""

    torch = pytest.importorskip("torch")

    class DummyModel:
        def state_dict(self):
            return {"weight": torch.zeros(1)}

    class DummyTokenizer:
        vocab_size = 128
        name_or_path = "dummy-tokenizer"

    events = []

    def _capture_event(payload):
        events.append(payload)

    result = finalize_training(
        model_student=DummyModel(),
        save_dir=str(tmp_path),
        stopped_early=True,
        steps_done=5,
        is_distributed=False,
        rank_id=0,
        tok=DummyTokenizer(),
        h_layers=1,
        l_layers=1,
        hidden_size=4,
        num_heads=1,
        expansion=1.0,
        h_cycles=1,
        l_cycles=1,
        pos_encodings="rope",
        log_file=None,
        write_jsonl=_capture_event,
        brain_name="test-brain",
        model="test-model",
        max_seq_len=16,
        halt_max_steps=1,
        default_goal="test-goal",
        dataset_file="dataset.txt",
        use_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_capacity_factor=1.0,
        stop_reason="stop_after_epoch",
    )

    assert result["trained"] is True, "stop_after_epoch should still count as trained"
    assert result.get("stop_reason") == "stop_after_epoch"
    assert "stopped" not in result
    assert any(evt.get("event") == "final_stop_after_epoch" for evt in events)


if __name__ == "__main__":
    print("Testing stop_after_epoch feature...\n")
    
    try:
        test_training_config_stop_after_epoch()
        test_to_cli_args_includes_stop_after_epoch()
        test_to_dict_includes_stop_after_epoch()
        test_from_dict_restores_stop_after_epoch()
        manual_tmp = Path("./tmp_stop_after_epoch")
        manual_tmp.mkdir(parents=True, exist_ok=True)
        test_finalize_training_reports_stop_after_epoch(manual_tmp)

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
