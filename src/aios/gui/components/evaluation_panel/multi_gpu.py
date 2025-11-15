"""Multi-GPU evaluation orchestration helpers."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from aios.core.evaluation import EvaluationResult, HarnessWrapper


logger = logging.getLogger(__name__)


@dataclass
class _ShardSpec:
    device: str
    effective_device: str
    env_overrides: Dict[str, str]
    output_path: str
    harness: HarnessWrapper | None
    completed_tasks: int = 0
    current_task: Optional[str] = None
    results: List[EvaluationResult] = field(default_factory=list)


class MultiGpuEvaluationRunner:
    """Coordinate parallel lm-eval runs across multiple GPUs."""

    def __init__(
        self,
        *,
        panel: Any,
        tasks: List[str],
        devices: List[str],
        base_env: Dict[str, str],
        model_kwargs: Dict[str, Any],
        on_complete: Callable[[EvaluationResult], None],
    ) -> None:
        self._panel = panel
        self._tasks = list(tasks)
        self._devices = list(devices)
        self._base_env = dict(base_env)
        self._model_kwargs = dict(model_kwargs)
        self._model_name = self._model_kwargs.pop("model_name", None)
        if not self._model_name:
            raise ValueError("model_name is required for MultiGpuEvaluationRunner")
        self._output_path = model_kwargs.get("output_path") or "artifacts/evaluation"
        self._model_kwargs.pop("env_overrides", None)
        self._model_kwargs.pop("output_path", None)
        self._on_complete = on_complete

        self._executor: ThreadPoolExecutor | None = None
        self._shards: List[_ShardSpec] = []
        self._cancel_event = threading.Event()
        self._progress_lock = threading.Lock()
        self._shard_progress: Dict[str, float] = {}
        self._task_queue: Deque[str] = deque(self._tasks)
        self._total_task_weight: float = float(max(1, len(self._tasks)))
        self._done_event = threading.Event()
        self._executor_lock = threading.Lock()
        self._orchestrator_thread: threading.Thread | None = None

    def start(self) -> None:
        self._done_event.clear()
        orchestrator = threading.Thread(target=self._run, name="eval-mgpu-orchestrator", daemon=True)
        self._orchestrator_thread = orchestrator
        orchestrator.start()

    def cancel(self, wait: bool = True, timeout: float | None = None) -> bool:
        self._cancel_event.set()
        for shard in self._shards:
            try:
                shard.harness.cancel()
            except Exception:
                continue
        self._shutdown_executor(wait=False)
        if wait:
            return self.wait_for_exit(timeout)
        return True

    def wait_for_exit(self, timeout: float | None = None) -> bool:
        finished = self._done_event.wait(timeout)
        thread = self._orchestrator_thread
        if finished and thread is not None:
            try:
                thread.join(timeout=0)
            except Exception:
                pass
        return finished

    # --- internal helpers -------------------------------------------------

    def _shutdown_executor(self, wait: bool = False, timeout: float | None = None) -> None:
        with self._executor_lock:
            executor = self._executor
            if executor is None:
                return
            self._executor = None
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to shutdown eval shard executor (wait=%s): %s", wait, exc, exc_info=True)

    def _run(self) -> None:
        try:
            if not self._tasks or not self._devices:
                result = EvaluationResult(status="failed", error_message="No tasks or devices available")
                self._dispatch_result(result)
                return

            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self._devices)),
                thread_name_prefix="eval-shard",
            )

            futures = []
            for device in self._devices:
                shard_env, effective_device = self._build_env_for_device(device)
                shard_output = self._allocate_output_path(device)
                spec = _ShardSpec(
                    device=device,
                    effective_device=effective_device,
                    env_overrides=shard_env,
                    output_path=shard_output,
                    harness=None,
                )
                harness = HarnessWrapper(
                    log_callback=lambda msg, dev=device: self._panel._log_threadsafe(
                        f"[eval][{dev}] {msg.replace('[eval] ', '')}"
                    ),
                    progress_callback=lambda pct, msg, s=spec: self._on_worker_progress(s, pct, msg),
                )
                spec.harness = harness
                self._shards.append(spec)
                self._shard_progress[device] = 0.0
                future = self._executor.submit(self._worker_loop, spec)
                futures.append(future)

            shard_results: Dict[str, EvaluationResult] = {}
            failed_reason: Optional[str] = None
            for future in as_completed(futures):
                if self._cancel_event.is_set():
                    failed_reason = "cancelled"
                    break
                try:
                    device, device_results, error_message = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    failed_reason = str(exc)
                    break

                if device_results:
                    shard_results[device] = self._merge_result_sequence(device_results)

                if error_message and failed_reason is None:
                    failed_reason = error_message
                    if error_message != "cancelled":
                        self._cancel_event.set()

            if self._cancel_event.is_set():
                for shard in self._shards:
                    if shard.harness:
                        shard.harness.cancel()
                merged = EvaluationResult(status="cancelled", error_message="Evaluation cancelled by user")
                self._dispatch_result(merged)
                return

            if failed_reason is not None and failed_reason != "cancelled":
                merged = EvaluationResult(status="failed", error_message=failed_reason or "Multi-GPU evaluation failed")
                merged.raw_results = {
                    "shards": {dev: res.raw_results for dev, res in shard_results.items()},
                }
                self._dispatch_result(merged)
                return

            merged = self._merge_results(shard_results)
            try:
                self._panel._on_progress_update_threadsafe(1.0, "Multi-GPU evaluation complete")
            except Exception:
                pass
            self._dispatch_result(merged)
        finally:
            self._shutdown_executor(wait=False)
            self._done_event.set()

    def _build_env_for_device(self, device: str) -> tuple[Dict[str, str], str]:
        overrides = dict(self._base_env)
        if "," in overrides.get("CUDA_VISIBLE_DEVICES", ""):
            overrides.pop("CUDA_VISIBLE_DEVICES", None)
        effective_device = device
        if device.startswith("cuda:"):
            cuda_index = device.split(":", 1)[1]
            overrides["CUDA_VISIBLE_DEVICES"] = cuda_index
            effective_device = "cuda:0"
        elif device == "cpu":
            overrides["CUDA_VISIBLE_DEVICES"] = ""
            effective_device = "cpu"
        overrides["AIOS_VISIBLE_DEVICES"] = device
        overrides["AIOS_INFERENCE_PRIMARY_DEVICE"] = effective_device
        return overrides, effective_device

    def _allocate_output_path(self, device: str) -> str:
        base_out = Path(self._output_path)
        shard_dir = base_out / f"shard_{device.replace(':', '_')}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        return str(shard_dir)

    def _allocate_task_output_path(self, shard_output: str, tasks: Sequence[str]) -> str:
        if len(tasks) == 1:
            suffix = self._sanitize_task_name(tasks[0])
        else:
            suffix = self._sanitize_task_name("_".join(tasks))
        out_dir = Path(shard_output) / suffix
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    def _sanitize_task_name(self, task: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", task.strip())
        return cleaned or "task"

    def _next_task(self) -> Optional[str]:
        with self._progress_lock:
            if not self._task_queue:
                return None
            return self._task_queue.popleft()

    def _worker_loop(
        self,
        spec: _ShardSpec,
    ) -> tuple[str, List[EvaluationResult], Optional[str]]:
        if spec.harness is None:
            raise RuntimeError("Shard harness not initialized")

        local_results: List[EvaluationResult] = []
        failure_message: Optional[str] = None

        while not self._cancel_event.is_set():
            next_task = self._next_task()
            if next_task is None:
                break

            spec.current_task = next_task
            result = self._run_task_batch(spec, [next_task])
            spec.results.append(result)
            local_results.append(result)

            if self._cancel_event.is_set():
                failure_message = "cancelled"
                break

            status = result.status.lower()
            if status == "cancelled":
                self._cancel_event.set()
                failure_message = "cancelled"
                break
            if status != "completed":
                self._cancel_event.set()
                failure_message = result.error_message or f"Task {next_task} failed"
                break

            with self._progress_lock:
                spec.completed_tasks += 1
                self._shard_progress[spec.device] = float(spec.completed_tasks)
                total_progress = sum(self._shard_progress.values()) / self._total_task_weight
            self._panel._on_progress_update_threadsafe(
                total_progress,
                f"{spec.device}: completed {next_task}",
            )
            spec.current_task = None

        spec.current_task = None

        with self._progress_lock:
            self._shard_progress[spec.device] = float(spec.completed_tasks)
            total_progress = sum(self._shard_progress.values()) / self._total_task_weight
        self._panel._on_progress_update_threadsafe(
            total_progress,
            f"{spec.device}: worker idle",
        )

        return spec.device, local_results, failure_message

    def _run_task_batch(self, spec: _ShardSpec, tasks: List[str]) -> EvaluationResult:
        if spec.harness is None:
            raise RuntimeError("Shard harness not initialized")

        label = ", ".join(tasks)
        self._panel._log_threadsafe(
            f"[eval] Worker {spec.device} starting {label}"
        )
        kwargs = dict(self._model_kwargs)
        kwargs["output_path"] = self._allocate_task_output_path(spec.output_path, tasks)
        kwargs["env_overrides"] = spec.env_overrides
        kwargs["device"] = spec.effective_device
        result = spec.harness.run_evaluation(
            self._model_name,
            tasks,
            **kwargs,
        )
        self._panel._log_threadsafe(
            f"[eval] Worker {spec.device} completed {label} status={result.status}"
        )
        return result

    def _on_worker_progress(self, spec: _ShardSpec, pct: float, status_msg: str) -> None:
        device = spec.device
        pct = max(0.0, min(1.0, pct))
        with self._progress_lock:
            base = float(spec.completed_tasks)
            self._shard_progress[device] = base + pct
            total_progress = sum(self._shard_progress.values()) / self._total_task_weight
        current_task = spec.current_task or "task"
        message = f"{device}: {current_task} - {status_msg}"
        self._panel._on_progress_update_threadsafe(total_progress, message)

    def _merge_result_sequence(self, results: List[EvaluationResult]) -> EvaluationResult:
        if not results:
            return EvaluationResult(status="completed", raw_results={"results": {}}, benchmark_scores={})
        if len(results) == 1:
            return results[0]

        merged = EvaluationResult(status="completed")
        merged.raw_results = {"results": {}, "runs": []}
        merged.benchmark_scores = {}

        merged.start_time = min((res.start_time for res in results if res.start_time), default=0.0)
        merged.end_time = max((res.end_time for res in results if res.end_time), default=0.0)
        merged.output_path = self._output_path

        total_score = 0.0
        score_count = 0

        for result in results:
            if result.status != "completed" and merged.status == "completed":
                merged.status = result.status
                merged.error_message = result.error_message

            for task_name, data in result.benchmark_scores.items():
                merged.benchmark_scores[task_name] = data
                scores = data.get("scores", {}) if isinstance(data, dict) else {}
                for metric_name, value in scores.items():
                    if metric_name in {"acc", "accuracy", "acc_norm", "exact_match"} and isinstance(value, (int, float)):
                        total_score += float(value)
                        score_count += 1

            raw = result.raw_results or {}
            if isinstance(raw, dict):
                maybe_results = raw.get("results")
                if isinstance(maybe_results, dict):
                    merged.raw_results["results"].update(maybe_results)
                merged.raw_results.setdefault("runs", []).append(raw)

        if score_count:
            merged.overall_score = total_score / score_count

        return merged

    def _merge_results(self, shard_results: Dict[str, EvaluationResult]) -> EvaluationResult:
        merged = EvaluationResult(status="completed")
        if not shard_results:
            return merged

        merged.start_time = min((res.start_time for res in shard_results.values() if res.start_time), default=0.0)
        merged.end_time = max((res.end_time for res in shard_results.values() if res.end_time), default=0.0)
        merged.output_path = self._output_path

        merged.raw_results = {
            "results": {},
            "shards": {},
        }
        merged.benchmark_scores = {}

        total_score = 0.0
        score_count = 0

        for device, result in shard_results.items():
            merged.raw_results["shards"][device] = result.raw_results
            for task_name, data in result.benchmark_scores.items():
                merged.benchmark_scores[task_name] = data
                scores = data.get("scores", {}) if isinstance(data, dict) else {}
                for metric_name, value in scores.items():
                    if metric_name in {"acc", "accuracy", "acc_norm", "exact_match"} and isinstance(value, (int, float)):
                        total_score += float(value)
                        score_count += 1

            raw = result.raw_results or {}
            shard_results_dict = raw.get("results") if isinstance(raw, dict) else None
            if isinstance(shard_results_dict, dict):
                merged.raw_results["results"].update(shard_results_dict)

        if score_count:
            merged.overall_score = total_score / score_count

        self._write_combined_results(merged.raw_results)
        return merged

    def _write_combined_results(self, payload: Dict[str, Any]) -> None:
        try:
            output_path = Path(self._output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            combined_path = output_path / "combined_results.json"
            combined_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # Best effort: log on GUI thread to avoid breaking result reporting
            self._panel._log_threadsafe("[eval] Warning: failed to write combined_results.json")

    def _dispatch_result(self, result: EvaluationResult) -> None:
        def _callback() -> None:
            try:
                self._on_complete(result)
            finally:
                pass
        self._panel.after(0, _callback)