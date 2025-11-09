"""Dynamic width adjustment methods for Trainer."""

from __future__ import annotations


class WidthManagementMixin:
    """Mixin providing dynamic width adjustment for Trainer."""

    def _maybe_adjust_width(self, loss: float) -> None:
        """Heuristic dynamic width: grow on plateau, optionally shrink on regressions."""
        if not self.cfg.dynamic_width:
            return
        self._loss_window.append(float(loss))
        win_cap = int(self._loss_window.maxlen or 0)
        if len(self._loss_window) < win_cap:
            return
        curr_avg = float(sum(self._loss_window) / len(self._loss_window))
        if self._prev_window_avg is None:
            self._prev_window_avg = curr_avg
            self._loss_window.clear()
            return
        improvement = max(0.0, self._prev_window_avg - curr_avg)
        # Grow if improvement is tiny (plateau)
        if improvement < self.cfg.grow_threshold and self.cfg.hidden < self.cfg.width_max:
            new_hidden = int(min(self.cfg.width_max, max(self.cfg.hidden + 1, int(self.cfg.hidden * self.cfg.grow_factor))))
            # Optional storage guard: ensure new_hidden fits within width_storage_limit_mb if provided
            lim_mb = self.cfg.width_storage_limit_mb
            if lim_mb is not None and lim_mb > 0:
                in_d = int(self.cfg.input_dim)
                out_d = int(self.cfg.output_dim)
                def params_for(h: int) -> int:
                    return in_d * h + h + h * out_d + out_d
                bytes_per = 4  # FP32 upper bound (numpy path)
                max_params = int((lim_mb * 1024 * 1024) // bytes_per)
                # Bound h so params_for(h) <= max_params
                denom = max(1, in_d + 1 + out_d)
                h_cap = max(1, min(self.cfg.width_max, int((max_params - out_d) // denom)))
                if new_hidden > h_cap:
                    new_hidden = int(max(self.cfg.hidden, h_cap))
            self._rebuild_models(new_hidden)
        # Shrink gently if loss spikes and width is large
        elif curr_avg > (self._prev_window_avg * 1.05) and self.cfg.hidden > self.cfg.width_min:
            new_hidden = int(max(self.cfg.width_min, int(self.cfg.hidden / self.cfg.shrink_factor)))
            if new_hidden < self.cfg.hidden:
                self._rebuild_models(new_hidden)
        self._prev_window_avg = curr_avg
        self._loss_window.clear()

    def _rebuild_models(self, new_hidden: int) -> None:
        """Resize numpy and torch models, preserving weights where possible."""
        from .numpy_model import NumpyMLP
        
        old_h = int(self.cfg.hidden)
        if new_hidden == old_h:
            return
        self.cfg.hidden = int(new_hidden)
        # numpy
        np_new = NumpyMLP(self.cfg.input_dim, self.cfg.hidden, self.cfg.output_dim)
        # copy overlap
        h_min = min(old_h, self.cfg.hidden)
        np_new.W1[:, :h_min] = self.model_np.W1[:, :h_min]
        np_new.b1[:h_min] = self.model_np.b1[:h_min]
        np_new.W2[:h_min, :] = self.model_np.W2[:h_min, :]
        self.model_np = np_new
        # torch
        if self.torch_available:  # pragma: no cover - optional
            self._rebuild_torch_model(old_h)

    def _rebuild_torch_model(self, old_h: int) -> None:  # pragma: no cover
        """Rebuild torch model with new hidden size."""
        t = self.torch
        base = self._torch_base()
        if base is None:
            return
        layers = list(base.children())
        if len(layers) < 3:
            return
        old_l1: t.nn.Linear = layers[0]  # type: ignore[assignment]
        old_act = layers[1]
        old_l2: t.nn.Linear = layers[2]  # type: ignore[assignment]
        new_l1 = t.nn.Linear(self.cfg.input_dim, self.cfg.hidden)
        new_l2 = t.nn.Linear(self.cfg.hidden, self.cfg.output_dim)
        h_min_t = min(old_h, self.cfg.hidden)
        with t.no_grad():
            # copy l1
            new_l1.weight[:h_min_t, :old_l1.in_features].copy_(old_l1.weight[:h_min_t, :old_l1.in_features])
            new_l1.bias[:h_min_t].copy_(old_l1.bias[:h_min_t])
            # copy l2
            new_l2.weight[:, :h_min_t].copy_(old_l2.weight[:, :h_min_t])
            new_l2.bias.copy_(old_l2.bias)
        rebuilt = t.nn.Sequential(new_l1, old_act, new_l2)
        # re-wrap DataParallel if needed
        if self.data_parallel and t.cuda.is_available():
            try:
                device_ids = None
                if self.cfg.cuda_devices and len(self.cfg.cuda_devices) > 1:
                    device_ids = [int(i) for i in self.cfg.cuda_devices]
                else:
                    dc = t.cuda.device_count()
                    if dc and dc > 1:
                        device_ids = list(range(dc))
                if device_ids and len(device_ids) > 1:
                    rebuilt = t.nn.DataParallel(rebuilt, device_ids=device_ids)
            except Exception:
                pass
        self.mlp_t = rebuilt.to(self.device)
        # reset optimizer for new params (use getattr for type checkers)
        self.opt = getattr(t.optim, "Adam")(self._torch_params(), lr=self.cfg.lr)
