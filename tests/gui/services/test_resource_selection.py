import json

import pytest

from aios.gui.services import (
    DeviceSelectionResult,
    resolve_inference_devices_from_state,
)


def _assert_serialisable(selection: DeviceSelectionResult) -> None:
    """Helper to ensure selection payload serialises cleanly."""

    payload = selection.to_log_payload()
    json.dumps(payload)  # raises if not serialisable


@pytest.mark.parametrize(
    "state,detected,os_name,expected_visible,expected_physical",
    [
        ( {"run_device": "cuda", "run_cuda_selected": [0, 1]}, [0, 1], "Linux", ["cuda:0", "cuda:1"], ["cuda:0", "cuda:1"] ),
        ( {"run_device": "auto", "run_cuda_selected": [2]}, [0, 2], "Linux", ["cuda:0"], ["cuda:2"] ),
    ],
)
def test_linux_multi_gpu_selection(state, detected, os_name, expected_visible, expected_physical):
    result = resolve_inference_devices_from_state(state, os_name, detected_device_ids=detected)

    assert result.device_kind == "cuda"
    assert result.visible_devices == expected_visible
    assert not result.warnings
    assert result.env_overrides["AIOS_INFERENCE_PRIMARY_DEVICE"] == expected_visible[0]
    expected_mask = ",".join(str(idx) for idx in state.get("run_cuda_selected", []))
    assert result.env_overrides["CUDA_VISIBLE_DEVICES"] == expected_mask
    metadata = result.metadata
    assert isinstance(metadata, dict)
    physical = metadata.get("physical_visible_devices")
    assert physical == expected_physical
    assert result.env_overrides["AIOS_INFERENCE_PHYSICAL_DEVICES"] == ",".join(expected_physical)
    alias_map = metadata.get("alias_physical_map")
    assert isinstance(alias_map, dict)
    for alias, phys in zip(expected_visible, expected_physical):
        assert alias_map[alias] == phys
    assert metadata.get("primary_physical_device") == expected_physical[0]
    _assert_serialisable(result)


def test_windows_multi_gpu_supported():
    state = {"run_device": "cuda", "run_cuda_selected": [0, 1]}
    result = resolve_inference_devices_from_state(state, "Windows", detected_device_ids=[0, 1])

    assert result.device_kind == "cuda"
    assert result.visible_devices == ["cuda:0", "cuda:1"]
    assert "windows_single_gpu_fallback" not in result.warnings
    metadata = result.metadata
    assert isinstance(metadata, dict)
    assert metadata.get("physical_visible_devices") == ["cuda:0", "cuda:1"]
    alias_map = metadata.get("alias_physical_map")
    assert isinstance(alias_map, dict)
    assert alias_map["cuda:0"] == "cuda:0"
    assert alias_map["cuda:1"] == "cuda:1"
    _assert_serialisable(result)


def test_cuda_unavailable_falls_back_to_cpu():
    state = {"run_device": "cuda", "run_cuda_selected": []}
    result = resolve_inference_devices_from_state(state, "Linux", detected_device_ids=[])

    assert result.device_kind == "cpu"
    assert result.primary_device == "cpu"
    assert "cuda_devices_unavailable" in result.warnings
    assert result.env_overrides["CUDA_VISIBLE_DEVICES"] == ""
    _assert_serialisable(result)


def test_device_selection_truncated_to_max_visible():
    state = {"run_device": "cuda", "run_cuda_selected": list(range(12))}
    result = resolve_inference_devices_from_state(state, "Linux", detected_device_ids=list(range(12)))

    assert result.device_kind == "cuda"
    assert len(result.visible_devices) == 8
    assert result.visible_devices[0] == "cuda:0"
    assert "device_selection_truncated" in result.warnings
    metadata = result.metadata
    assert isinstance(metadata, dict)
    physical = metadata.get("physical_visible_devices")
    assert physical == [f"cuda:{idx}" for idx in range(8)]
    _assert_serialisable(result)


def test_auto_detect_falls_back_to_detected_gpu():
    state = {"run_device": "auto", "run_cuda_selected": []}
    result = resolve_inference_devices_from_state(state, "Linux", detected_device_ids=[3, 1])

    assert result.device_kind == "cuda"
    assert result.primary_device == "cuda:0"
    assert result.visible_devices == ["cuda:0"]
    assert result.env_overrides["CUDA_VISIBLE_DEVICES"] == "3"
    assert result.env_overrides["AIOS_INFERENCE_PHYSICAL_DEVICES"] == "cuda:3"
    metadata = result.metadata
    assert isinstance(metadata, dict)
    assert metadata.get("physical_visible_devices") == ["cuda:3"]
    assert metadata.get("primary_physical_device") == "cuda:3"
    alias_map = metadata.get("alias_physical_map")
    assert isinstance(alias_map, dict)
    assert alias_map["cuda:0"] == "cuda:3"
    _assert_serialisable(result)
