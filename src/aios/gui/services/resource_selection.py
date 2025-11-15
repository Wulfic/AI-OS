"""Helpers for resolving inference device selections.

This module centralizes the logic that chat and evaluation tabs use to convert
resource panel state into actionable device assignments. Callers can pass the
panel instance directly or provide a persisted state dictionary when running in
headless/background contexts.

Example:
    from aios.gui.services.resource_selection import resolve_inference_devices

    selection = resolve_inference_devices(resources_panel)
    log_payload = selection.to_log_payload()
    message = build_device_message(selection)
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Sequence, cast

logger = logging.getLogger(__name__)

DeviceKind = Literal["cpu", "cuda", "mps", "rocm"]

# Hard cap on GPU indices we expose to CUDA_VISIBLE_DEVICES to avoid runaway envs.
_MAX_VISIBLE_GPUS = 8

_WARNING_MESSAGES: dict[str, str] = {
    "cuda_devices_unavailable": (
        "Requested CUDA devices were unavailable; falling back to CPU execution."
    ),
    "device_selection_truncated": (
        "Requested GPU list exceeded supported limit; truncating to the first 8 devices."
    ),
    "mixed_device_selection": (
        "Mixed CPU/GPU selection detected; prioritizing CUDA devices for inference."
    ),
    "empty_selection_defaulted": (
        "No GPU selected; defaulting to cuda:0 when available."
    ),
}


@dataclass(frozen=True)
class DeviceSelectionResult:
    """Structured representation of the resolved inference device choice."""

    primary_device: str
    visible_devices: list[str]
    requested_devices: list[str]
    device_kind: DeviceKind
    warnings: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_log_payload(self) -> dict[str, Any]:
        """Return a logging-friendly snapshot of the selection state."""

        payload = {
            "primary_device": self.primary_device,
            "visible_devices": list(self.visible_devices),
            "requested_devices": list(self.requested_devices),
            "device_kind": self.device_kind,
            "warnings": list(self.warnings),
            "env_overrides": dict(self.env_overrides),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def warning_message(token: str) -> str:
    """Return a standardised warning string for a selector warning token."""

    return _WARNING_MESSAGES.get(token, f"Device selection warning: {token}")


def resolve_inference_devices(resources_panel: Any | None) -> DeviceSelectionResult:
    """Resolve device selection using a live resources panel when available."""

    state: dict[str, Any] = {}
    detected_ids: Sequence[int] | None = None
    os_name = platform.system()

    if resources_panel is not None:
        state = _pull_panel_state(resources_panel)
        detected_ids = _extract_detected_ids(resources_panel)
        try:
            os_name = getattr(resources_panel, "os_override", os_name)
        except Exception:
            pass

    return resolve_inference_devices_from_state(state, os_name, detected_device_ids=detected_ids)


def resolve_inference_devices_from_state(
    state: dict[str, Any] | None,
    os_name: str,
    *,
    detected_device_ids: Iterable[int] | None = None,
) -> DeviceSelectionResult:
    """Pure helper that converts persisted state into a selection result."""

    state = state or {}
    os_norm = (os_name or "").strip().lower()

    requested_device = str(state.get("run_device") or "auto").lower()
    requested_indices = _normalize_device_ids(state.get("run_cuda_selected"))
    detected_indices = _normalize_device_ids(detected_device_ids)

    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "os": os_norm or "unknown",
        "requested_device": requested_device,
        "requested_indices": list(requested_indices),
    }

    device_kind: DeviceKind
    if requested_device == "cpu" and requested_indices:
        warnings.append("mixed_device_selection")
        device_kind = "cuda"
    elif requested_device in {"cuda", "mps", "rocm"}:
        device_kind = cast(DeviceKind, requested_device)
    elif requested_indices:
        device_kind = "cuda"
    else:
        device_kind = "cpu"

    effective_indices = list(requested_indices)

    if device_kind == "cuda":
        if not effective_indices and detected_indices:
            effective_indices = [detected_indices[0]]
            metadata["auto_selected_index"] = detected_indices[0]
        if not effective_indices:
            warnings.append("cuda_devices_unavailable")
            device_kind = "cpu"
    else:
        effective_indices = []

    if device_kind == "cuda" and len(effective_indices) > _MAX_VISIBLE_GPUS:
        metadata["truncated_from"] = len(effective_indices)
        effective_indices = effective_indices[:_MAX_VISIBLE_GPUS]
        warnings.append("device_selection_truncated")

    visible_indices = list(effective_indices)
    metadata["effective_indices"] = list(visible_indices)

    requested_devices = _format_gpu_aliases(requested_indices) if device_kind == "cuda" else ["cpu"]

    alias_devices: list[str] = []
    alias_map: dict[str, str] = {}
    physical_visible_devices = _format_gpu_aliases(visible_indices) if device_kind == "cuda" else ["cpu"]
    if device_kind == "cuda" and visible_indices:
        alias_devices = [f"cuda:{idx}" for idx in range(len(visible_indices))]
        alias_map = {
            alias: physical for alias, physical in zip(alias_devices, physical_visible_devices)
        }
    elif device_kind != "cuda":
        alias_devices = [device_kind]
        physical_visible_devices = [device_kind]

    if device_kind != "cuda":
        primary_device = "cpu"
        env_overrides: Dict[str, str] = {
            "AIOS_INFERENCE_PRIMARY_DEVICE": primary_device,
            "CUDA_VISIBLE_DEVICES": "",
        }
        visible_devices = [primary_device]
        if device_kind in {"mps", "rocm"}:
            primary_device = device_kind
            visible_devices = [device_kind]
            requested_devices = [device_kind]
            env_overrides["AIOS_INFERENCE_PRIMARY_DEVICE"] = primary_device
        metadata["device_kind"] = device_kind
        return DeviceSelectionResult(
            primary_device=primary_device,
            visible_devices=visible_devices,
            requested_devices=requested_devices,
            device_kind=device_kind,
            warnings=warnings,
            env_overrides=env_overrides,
            metadata=metadata,
        )

    visible_devices = alias_devices or []

    if not visible_devices:
        visible_devices = ["cuda:0"]
        warnings.append("empty_selection_defaulted")
        metadata["fallback_visible_index"] = 0

    primary_device = visible_devices[0]
    env_overrides = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(idx) for idx in visible_indices) if visible_indices else "",
        "AIOS_INFERENCE_PRIMARY_DEVICE": primary_device,
    }
    if physical_visible_devices:
        env_overrides["AIOS_INFERENCE_PHYSICAL_DEVICES"] = ",".join(physical_visible_devices)
        metadata["physical_visible_devices"] = list(physical_visible_devices)
    if alias_map:
        metadata["alias_physical_map"] = dict(alias_map)
        metadata["primary_physical_device"] = alias_map.get(primary_device, physical_visible_devices[0])
    else:
        metadata["primary_physical_device"] = physical_visible_devices[0] if physical_visible_devices else primary_device
    metadata["device_kind"] = device_kind

    return DeviceSelectionResult(
        primary_device=primary_device,
        visible_devices=visible_devices,
        requested_devices=requested_devices,
        device_kind=device_kind,
        warnings=warnings,
        env_overrides=env_overrides,
        metadata=metadata,
    )


def build_device_message(result: DeviceSelectionResult, *, include_warnings: bool = True) -> str:
    """Return a concise user-facing summary string for logs or status labels."""

    alias_map: dict[str, str] = {}
    if isinstance(result.metadata, dict):
        maybe_map = result.metadata.get("alias_physical_map")
        if isinstance(maybe_map, dict):
            alias_map = {str(k): str(v) for k, v in maybe_map.items()}

    primary_label = f"Primary {result.primary_device}"
    if alias_map:
        physical = alias_map.get(result.primary_device)
        if physical and physical != result.primary_device:
            primary_label += f" (phys {physical})"

    parts = [primary_label]
    if result.visible_devices:
        visible_summary = ", ".join(result.visible_devices)
        if alias_map:
            mapped = [alias_map.get(dev, dev) for dev in result.visible_devices]
            if mapped and mapped != result.visible_devices:
                visible_summary += " (phys " + ", ".join(mapped) + ")"
        parts.append("active: " + visible_summary)
    if result.requested_devices:
        parts.append("requested: " + ", ".join(result.requested_devices))
    if include_warnings and result.warnings:
        parts.append("warnings: " + ", ".join(result.warnings))
    return " | ".join(parts)


def _pull_panel_state(panel: Any) -> dict[str, Any]:
    try:
        if hasattr(panel, "get_state"):
            state = panel.get_state()
        elif hasattr(panel, "get_values"):
            state = panel.get_values()
        else:
            state = {}
        return state if isinstance(state, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to read resources panel state: %s", exc)
        return {}


def _extract_detected_ids(panel: Any) -> Sequence[int] | None:
    try:
        snapshot = getattr(panel, "_last_detected_snapshot", None)
        if not snapshot or not isinstance(snapshot, tuple) or len(snapshot) != 2:
            return None
        _, devices = snapshot
        indices = []
        for item in devices or []:
            try:
                indices.append(int(item[0]))
            except Exception:
                continue
        return tuple(sorted(set(indices))) if indices else None
    except Exception:
        return None


def _normalize_device_ids(values: Any) -> list[int]:
    normalized: list[int] = []
    if values is None:
        return normalized
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        for item in values:
            try:
                normalized.append(int(item))
            except Exception:
                continue
    normalized = sorted(set(idx for idx in normalized if idx >= 0))
    return normalized


def _format_gpu_aliases(indices: Sequence[int]) -> list[str]:
    return [f"cuda:{idx}" for idx in indices]
