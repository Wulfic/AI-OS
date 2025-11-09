from __future__ import annotations

from typing import Optional, Any, Dict, Tuple, List


def load_teacher_model_for_generation(
    *,
    teacher_name: str,
    teacher_device: str,
    strict: bool,
    torch,
    AutoModelForCausalLM,
) -> Tuple[Any, Any]:
    """Load teacher model for generation with memory-aware defaults.

    Returns (model, device_for_gen torch.device-like or dml device).
    May raise typer.Exit with specific code when strict GPU requirements fail.
    """
    import typer
    import os

    # Build load kwargs and try optional 8-bit on strict CUDA
    load_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    # Prefer new 'dtype' kwarg; fall back to 'torch_dtype' if transformers is older
    if str(teacher_device).lower() == "cuda":
        load_kwargs["dtype"] = torch.float16

    tried_8bit = False
    if strict and str(teacher_device).lower() == "cuda":
        try:
            import bitsandbytes as _bnb  # noqa: F401
            load_kwargs.update({"load_in_8bit": True})
            tried_8bit = True
        except Exception:
            pass

    def _from_pretrained_with_dtype_fallback(name: str, kwargs: Dict[str, Any]):
        try:
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except TypeError as _e:
            # Backward-compat: older Transformers used 'torch_dtype'
            if "unexpected keyword argument" in str(_e) and "dtype" in str(_e):
                _kw = dict(kwargs)
                _val = _kw.pop("dtype", None)
                if _val is not None:
                    _kw["torch_dtype"] = _val
                return AutoModelForCausalLM.from_pretrained(name, **_kw)
            raise

    try:
        t_model_gen = _from_pretrained_with_dtype_fallback(teacher_name, load_kwargs)
    except Exception as _e_8:
        if tried_8bit:
            try:
                load_kwargs.pop("load_in_8bit", None)
                t_model_gen = _from_pretrained_with_dtype_fallback(teacher_name, load_kwargs)
            except Exception:
                raise _e_8
        else:
            raise _e_8

    # Choose device
    if str(teacher_device).lower() == "auto":
        device_for_gen = torch.device("cpu")
    elif str(teacher_device).lower() == "cuda" and torch.cuda.is_available():
        # Prefer per-rank mapping if exist; caller can override device later
        try:
            device_for_gen = torch.device("cuda")
        except Exception:
            device_for_gen = torch.device("cuda")
    elif str(teacher_device).lower() == "dml":
        try:
            import torch_directml as _dml  # type: ignore
            device_for_gen = _dml.device()
        except Exception:
            device_for_gen = torch.device("cpu")
    else:
        device_for_gen = torch.device("cpu")
    if str(teacher_device).lower() == "cuda" and not torch.cuda.is_available():
        try:
            from rich import print
            print({"teacher_device_request": "cuda", "using": "cpu", "reason": "cuda_unavailable_or_not_compiled"})
        except Exception:
            pass

    # Move to device (prefer fp16 on CUDA)
    try:
        if getattr(device_for_gen, 'type', 'cpu') == 'cuda' and torch.cuda.is_available():
            try:
                t_model_gen.to(device_for_gen, dtype=torch.float16)  # type: ignore[misc]
            except Exception:
                t_model_gen.to(device_for_gen)  # type: ignore[misc]
        else:
            t_model_gen.to(device_for_gen)  # type: ignore[misc]
        t_model_gen.eval()
        try:
            if getattr(device_for_gen, 'type', 'cpu') == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        if strict and str(teacher_device).lower() == "cuda":
            from rich import print
            print({"error": "Failed to move teacher model to CUDA in strict mode"})
            raise typer.Exit(code=3)
        device_for_gen = torch.device("cpu")

    return t_model_gen, device_for_gen


# auto-batch helper removed; generation is fixed-batch with OOM backoff in trainer
