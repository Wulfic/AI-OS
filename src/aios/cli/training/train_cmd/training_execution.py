"""
Training execution loops.

Contains the main training loops with various execution modes.
"""

import time
from pathlib import Path
from typing import Optional, Dict, List

from aios.core.replay import ReplayBuffer
from aios.core.train import Trainer
from aios.memory.store import get_db, init_db, save_artifact


def run_training_with_checkpointing(
    trainer: Trainer,
    replay_buffer: ReplayBuffer,
    steps: int,
    checkpoint_every: int,
    checkpoint_dir: Optional[str],
    tag: Optional[str],
    sleep_every: int,
    cost_budget: Optional[float],
    progress: bool,
    emit_metrics: bool,
    english_corpus_metrics: Optional[Dict],
    task: Optional[str],
    adherence_metrics: Optional[Dict],
) -> Dict:
    """Run training with periodic checkpointing.
    
    Returns:
        Summary dictionary with training results
    """
    losses: List[float] = []
    costs: List[float] = []
    ck_dir = Path(checkpoint_dir or (Path.home() / ".local/share/aios/checkpoints"))
    ck_dir.mkdir(parents=True, exist_ok=True)
    prefix = tag or "ckpt"
    steps_run = 0
    over = False
    last_ck: Optional[str] = None
    
    for i in range(steps):
        if cost_budget is None:
            loss = trainer.train_step(replay_buffer)
            cost = 0.0
        else:
            loss, cost = trainer.train_step_with_cost(replay_buffer)
            if trainer.total_cost > trainer.cfg.cost_budget:
                over = True
        
        steps_run = i + 1
        losses.append(float(loss))
        costs.append(float(cost))
        
        if progress:
            try:
                every = max(1, int(steps // 100) or 1)
                if (i < 5) or (steps_run % every == 0) or (steps - steps_run) < 5:
                    print(f"step {steps_run} / {steps}", flush=True)
            except Exception:
                pass
        
        if sleep_every and sleep_every > 0 and ((i + 1) % sleep_every == 0):
            try:
                trainer.sleep_cycle(replay_buffer)
            except Exception:
                pass
        
        if steps_run % checkpoint_every == 0 or (i + 1) == steps:
            ts = int(time.time())
            ck_path = ck_dir / f"{prefix}-step{steps_run}-{ts}.npz"
            try:
                trainer.save_checkpoint(str(ck_path), {"steps": steps_run})
                last_ck = str(ck_path)
                try:
                    conn = get_db()
                    init_db(conn)
                    save_artifact(conn, kind="training_checkpoint", label=prefix, data={"path": str(ck_path), "steps": steps_run})
                finally:
                    try:
                        conn.close()  # type: ignore[name-defined]
                    except Exception:
                        pass
            except Exception:
                pass
        
        if over:
            break
    
    if emit_metrics:
        try:
            conn = get_db()
            init_db(conn)
            data_dict: Dict = {
                "steps": steps_run,
                "losses": losses[-min(len(losses), 200):],
                "costs": costs[-min(len(costs), 200):],
                "over_budget": over,
            }
            if english_corpus_metrics:
                data_dict["english_corpus"] = english_corpus_metrics
            if task:
                data_dict["english_task"] = {"spec": task}
            if adherence_metrics:
                data_dict["english_adherence"] = adherence_metrics
            save_artifact(conn, kind="training_metrics", label=prefix, data=data_dict)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
    
    return {
        "torch": trainer.torch_available,
        "steps_run": steps_run,
        "last_loss": float(losses[-1]) if losses else None,
        "over_budget": over,
        "checkpoint_dir": str(ck_dir),
        "tag": prefix,
        "last_checkpoint": last_ck,
    }


def run_training_simple(
    trainer: Trainer,
    replay_buffer: ReplayBuffer,
    steps: int,
    sleep_every: int,
    progress: bool,
    save_ckpt: Optional[str],
    emit_metrics: bool,
    tag: Optional[str],
    english_corpus_metrics: Optional[Dict],
    task: Optional[str],
    adherence_metrics: Optional[Dict],
) -> Dict:
    """Run simple training without periodic checkpointing.
    
    Returns:
        Summary dictionary with training results
    """
    if progress:
        loss = 0.0
        steps_run = int(steps)
        for i in range(steps_run):
            loss = trainer.train_step(replay_buffer)
            if sleep_every and sleep_every > 0 and ((i + 1) % sleep_every == 0):
                try:
                    trainer.sleep_cycle(replay_buffer)
                except Exception:
                    pass
            try:
                every = max(1, int(steps_run // 100) or 1)
                if (i < 5) or ((i + 1) % every == 0) or (steps_run - (i + 1)) < 5:
                    print(f"step {i+1} / {steps_run}", flush=True)
            except Exception:
                pass
    else:
        loss = trainer.train(replay_buffer, steps=steps)
    
    out: Dict = {"loss": loss, "torch": trainer.torch_available}
    
    if save_ckpt:
        try:
            trainer.save_checkpoint(save_ckpt, {"steps": steps})
            out["checkpoint_saved"] = save_ckpt
            try:
                conn = get_db()
                init_db(conn)
                save_artifact(conn, kind="training_checkpoint", label="train", data={"path": save_ckpt, "steps": steps})
            except Exception:
                pass
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
        except Exception:
            out["checkpoint_saved"] = False
    
    if emit_metrics:
        try:
            conn = get_db()
            init_db(conn)
            data_dict: Dict = {
                "steps": int(steps),
                "last_loss": float(loss) if loss is not None else None,
            }
            if english_corpus_metrics:
                data_dict["english_corpus"] = english_corpus_metrics
            if task:
                data_dict["english_task"] = {"spec": task}
            if adherence_metrics:
                data_dict["english_adherence"] = adherence_metrics
            save_artifact(conn, kind="training_metrics", label=(tag or "train"), data=data_dict)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
    
    return out


def run_training_with_budget(
    trainer: Trainer,
    replay_buffer: ReplayBuffer,
    steps: int,
    progress: bool,
    save_ckpt: Optional[str],
    emit_metrics: bool,
    tag: Optional[str],
    english_corpus_metrics: Optional[Dict],
    task: Optional[str],
    adherence_metrics: Optional[Dict],
) -> Dict:
    """Run training with cost budget enforcement.
    
    Returns:
        Summary dictionary with training results
    """
    if progress:
        n = int(steps)
        trainer.total_cost = 0.0
        last_loss = 0.0
        over = False
        ran = 0
        for i in range(n):
            last_loss, _ = trainer.train_step_with_cost(replay_buffer)
            ran = i + 1
            try:
                every = max(1, int(n // 100) or 1)
                if (i < 5) or (ran % every == 0) or (n - ran) < 5:
                    print(f"step {ran} / {n}", flush=True)
            except Exception:
                pass
            if trainer.total_cost > trainer.cfg.cost_budget:
                over = True
                break
        summ = {
            "loss": float(last_loss),
            "total_cost": float(trainer.total_cost),
            "over_budget": over,
            "steps_run": ran,
        }
    else:
        summ = trainer.train_with_budgets(replay_buffer, steps=steps)
    
    summ["torch"] = trainer.torch_available
    
    if save_ckpt:
        try:
            trainer.save_checkpoint(save_ckpt, {"steps": summ.get("steps_run", steps)})
            summ["checkpoint_saved"] = save_ckpt
            try:
                conn = get_db()
                init_db(conn)
                save_artifact(conn, kind="training_checkpoint", label="train", data={"path": save_ckpt, "steps": int(summ.get("steps_run", steps))})
            except Exception:
                pass
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
        except Exception:
            summ["checkpoint_saved"] = False
    
    if emit_metrics:
        try:
            conn = get_db()
            init_db(conn)
            data_dict: Dict = {
                "steps": int(summ.get("steps_run", steps) or steps),
                "over_budget": bool(summ.get("over_budget", False)),
            }
            if "last_loss" in summ:
                try:
                    data_dict["last_loss"] = float(summ["last_loss"])  # type: ignore[arg-type]
                except Exception:
                    pass
            if english_corpus_metrics:
                data_dict["english_corpus"] = english_corpus_metrics
            if task:
                data_dict["english_task"] = {"spec": task}
            if adherence_metrics:
                data_dict["english_adherence"] = adherence_metrics
            save_artifact(conn, kind="training_metrics", label=(tag or "train"), data=data_dict)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
    
    return summ
