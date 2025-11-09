"""Core Trainer class with initialization and device setup."""

from __future__ import annotations

from typing import Optional, Any
from collections import deque

from .config import TrainConfig
from .numpy_model import NumpyMLP


class TrainerBase:
    """Base Trainer class with initialization and device setup logic."""

    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        self.model_np = NumpyMLP(
            self.cfg.input_dim, self.cfg.hidden, self.cfg.output_dim
        )
        self.total_cost = 0.0
        # moving window for dynamic width
        self._loss_window = deque(maxlen=max(1, int(self.cfg.grow_patience)))
        self._prev_window_avg: Optional[float] = None

        # optional torch backend (not required). Only used if available and explicitly enabled.
        self.torch_available = False
        if self.cfg.use_torch:
            try:  # pragma: no cover - optional path
                import torch  # type: ignore

                self.torch_available = True
                # device is dynamic (torch.device, DirectML device, etc.)
                self.device: Any = None
                self._setup_torch_threads(torch)
                self.torch = torch
                self._setup_device(torch)
                self._setup_model(torch)
                self._setup_optimizer(torch)
                self._setup_amp(torch)
            except Exception:
                self.torch_available = False

    def _setup_torch_threads(self, torch) -> None:  # pragma: no cover
        """Configure torch threading."""
        try:
            if self.cfg.num_threads and self.cfg.num_threads > 0:
                torch.set_num_threads(int(self.cfg.num_threads))
                torch.set_num_interop_threads(min(int(self.cfg.num_threads), 4))
            else:
                # auto: use up to logical cores but cap modestly
                import os as _os  # local import
                th = max(1, min((_os.cpu_count() or 4), 8))
                torch.set_num_threads(th)
                torch.set_num_interop_threads(min(th, 4))
        except Exception:
            pass

    def _setup_device(self, torch) -> None:  # pragma: no cover
        """Setup torch device (CUDA, MPS, XPU, DirectML, CPU)."""
        # Device selection (support aliases like 'rocm' → 'cuda', and DirectML 'dml' on Windows)
        if self.cfg.device in ("cuda", "rocm"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.cfg.device == "mps":
            self.device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        elif self.cfg.device == "xpu":  # Intel GPU via torch.xpu if available
            try:
                has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
            except Exception:
                has_xpu = False
            self.device = torch.device("xpu" if has_xpu else "cpu")
        elif self.cfg.device in ("dml", "directml"):  # Windows DirectML backend
            try:
                import torch_directml as _dml  # type: ignore
                self.device = _dml.device()
                self.torch_directml = _dml  # type: ignore[attr-defined]
            except Exception:
                self.device = torch.device("cpu")
        elif self.cfg.device == "cpu":
            self.device = torch.device("cpu")
        else:
            # auto
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover - optional
                    self.device = torch.device("xpu")
                else:
                    # Try DirectML as a fallback on Windows
                    try:
                        import torch_directml as _dml  # type: ignore
                        self.device = _dml.device()
                        self.torch_directml = _dml  # type: ignore[attr-defined]
                    except Exception:
                        self.device = torch.device("cpu")
            except Exception:
                self.device = torch.device("cpu")

    def _setup_model(self, torch) -> None:  # pragma: no cover
        """Build torch model with optional DataParallel or DDP wrapping."""
        base_model = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.input_dim, self.cfg.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.cfg.hidden, self.cfg.output_dim),
        )

        # Multi-GPU (DataParallel) if requested and available (disabled when DDP)
        self.data_parallel = False
        device_ids: Optional[list[int]] = None
        if (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and self.cfg.data_parallel
            and not getattr(self.cfg, "ddp", False)
        ):
            try:
                if self.cfg.cuda_devices and len(self.cfg.cuda_devices) > 0:
                    device_ids = [int(i) for i in self.cfg.cuda_devices if int(i) >= 0]
                else:
                    device_count = torch.cuda.device_count()
                    if device_count and device_count > 1:
                        device_ids = list(range(device_count))
                if device_ids and len(device_ids) > 1:
                    base_model = torch.nn.DataParallel(base_model, device_ids=device_ids)
                    self.data_parallel = True
            except Exception:
                self.data_parallel = False

        # DDP wrapping
        if getattr(self.cfg, "ddp", False):  # pragma: no cover - optional
            base_model = self._setup_ddp(torch, base_model)

        self.mlp_t = base_model.to(self.device)

    def _setup_ddp(self, torch, base_model):  # pragma: no cover
        """Setup DistributedDataParallel if requested."""
        try:
            import os as _os
            import torch.distributed as dist  # type: ignore
            from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore
            env = _os.environ
            has_rank_env = (
                ("LOCAL_RANK" in env) or ("RANK" in env and "WORLD_SIZE" in env)
            )
            if not has_rank_env:
                raise RuntimeError("DDP requested but no torch.distributed rank env present; skip DDP")
            # Initialize process group if not already
            try:
                if not dist.is_initialized():
                    import platform as _plat
                    if _plat.system().lower().startswith("windows"):
                        backend = "gloo"
                    else:
                        backend = self.cfg.ddp_backend or (
                            "nccl" if (self.device.type == "cuda" and torch.cuda.is_available()) else "gloo"
                        )
                    dist.init_process_group(backend=backend, init_method="env://")
            except Exception:
                pass
            local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
            # If cuda, set correct device
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
                base_model = base_model.to(self.device)
                base_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)
            else:
                # CPU or other backends
                base_model = base_model.to(self.device)
                base_model = DDP(base_model)
            self.data_parallel = False
        except Exception:
            # Fall back to non-DDP
            pass
        return base_model

    def _setup_optimizer(self, torch) -> None:  # pragma: no cover
        """Setup Adam optimizer."""
        # construct optimizer via getattr to satisfy some type checkers/stubs
        self.opt = getattr(torch.optim, "Adam")(self._torch_params(), lr=self.cfg.lr)
        self.loss_fn = torch.nn.MSELoss()

    def _setup_amp(self, torch) -> None:  # pragma: no cover
        """Setup automatic mixed precision if requested."""
        # AMP (CUDA only)
        self._use_amp = bool(self.cfg.amp and self.device.type == "cuda" and torch.cuda.is_available())
        if self._use_amp:
            # Prefer new torch.amp API; fallback to torch.cuda.amp for older versions
            try:
                self.scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler()

    def _torch_base(self):  # pragma: no cover - optional
        """Return the underlying nn.Sequential even if wrapped in DataParallel."""
        if not self.torch_available:
            return None
        m = self.mlp_t
        t = self.torch
        # Unwrap DataParallel or DistributedDataParallel if present
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore
        except Exception:  # pragma: no cover - optional
            DDP = None  # type: ignore
        if isinstance(m, t.nn.DataParallel):
            return m.module
        if DDP is not None and isinstance(m, DDP):  # type: ignore[arg-type]
            return m.module
        return m

    def _torch_params(self):  # pragma: no cover - optional
        """Get torch model parameters."""
        if not self.torch_available:
            return []
        m = self.mlp_t
        return m.parameters()
