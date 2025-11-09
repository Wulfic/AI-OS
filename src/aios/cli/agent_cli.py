from __future__ import annotations

import asyncio
import json as _json
from typing import Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.core.hrm import build_default_registry, Manager
from aios.core.inference import run_inference
from aios.core.orchestrator import Orchestrator
from aios.core.brains import BrainRegistry, Router
from aios.core.communicator import format_status_summary
from aios.core.directives import list_directives
from aios.memory.store import get_db, init_db


def chat(
    message: str = typer.Argument(..., help="User message to the trained model"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Optional explicit checkpoint path (.npz)"),
):
    # Prefer Router/HF brain when enabled in config; fallback to classic inference otherwise
    cfg = load_config()
    brains_cfg = (cfg.get("brains") or {}) if isinstance(cfg, dict) else {}
    use_router = bool(brains_cfg.get("enabled", True))
    if use_router:
        try:
            # Build a lightweight registry/router using config overrides
            storage_limit_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0) or None
            reg = BrainRegistry(total_storage_limit_mb=storage_limit_mb)
            reg.store_dir = str(brains_cfg.get("store_dir", "artifacts/brains"))
            # Load persisted pins/masters for consistent behavior across sessions
            try:
                reg.load_pinned()
                reg.load_masters()
            except Exception:
                pass
            create_cfg = dict(brains_cfg.get("trainer_overrides", {}))
            # Wire optional generation/system prompt overrides
            gen_cfg = dict(brains_cfg.get("generation", {}) or {})
            if gen_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["generation"] = gen_cfg
            if "system_prompt" in brains_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["system_prompt"] = brains_cfg.get("system_prompt")
            if "history_max_turns" in brains_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["history_max_turns"] = int(brains_cfg.get("history_max_turns") or 0)
            router = Router(
                registry=reg,
                default_modalities=list(brains_cfg.get("default_modalities", ["text"])),
                brain_prefix=str(brains_cfg.get("prefix", "brain")),
                create_cfg=create_cfg,
                strategy=str(brains_cfg.get("strategy", "hash")),
                modality_overrides=dict(brains_cfg.get("modality_overrides", {})),
            )
            # Send payload as a string so the router generates a distinct brain name
            # (avoids collisions with prior dict-typed idle probes).
            res = router.handle({"modalities": ["text"], "payload": message})
            if isinstance(res, dict) and res.get("ok") and res.get("text"):
                print(_json.dumps({"ok": True, "text": res.get("text")}, ensure_ascii=False))
                return
        except Exception:
            # Continue to fallback path
            pass

    # Fallback: classic numpy-MLP inference
    try:
        inf = run_inference(message, ckpt_path=checkpoint)
        if inf.ok:
            out = {"ok": True, "text": inf.text, "score": inf.score, "checkpoint": inf.checkpoint}
            print(_json.dumps(out, ensure_ascii=False))
            return
    except Exception:
        pass

    # Final fallback: system summary
    orch = Orchestrator(config=cfg or {})
    status = orch.status()
    conn = None
    try:
        conn = get_db()
        init_db(conn)
        ds = list_directives(conn, active_only=True)
        summary = format_status_summary(status, ds)
        resp = {
            "ok": False,
            "text": "Model not ready. Showing system summary instead.",
            "summary": {"headline": summary.headline, "details": summary.details},
        }
        print(_json.dumps(resp, ensure_ascii=False))
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def eval(
    operator: Optional[str] = typer.Option(None, "--operator", "-o", help="Specific operator to run"),
):
    async def _run():
        reg = build_default_registry()
        mgr = Manager(reg)
        pick = await mgr.act({}, [operator] if operator else None)
        print({"picked": pick})
    asyncio.run(_run())


def op_run(
    operator: str = typer.Argument(..., help="Operator name to run"),
    unit: Optional[str] = typer.Option(None, "--unit", help="systemd unit name"),
    html: Optional[str] = typer.Option(None, "--html", help="raw HTML to parse"),
    code: Optional[str] = typer.Option(None, "--code", help="python code to check"),
    lines: int = typer.Option(20, "--lines", help="journal lines for service triage"),
    journal_text: Optional[str] = typer.Option(None, "--journal-text", help="Raw journal text for journal_summary_from_text"),
    label: Optional[str] = typer.Option(None, "--label", help="Optional label key for outputs when using journal-text"),
    buckets: Optional[int] = typer.Option(None, "--buckets", help="Optional number of buckets for journal_trend_from_text"),
):
    async def _run():
        from aios.memory.store import get_db, init_db, save_artifact
        reg = build_default_registry()
        op = reg.get(operator)
        if op is None:
            print({"error": f"operator not found: {operator}"})
            return
        ctx: dict = {}
        if unit:
            ctx["unit"] = unit
        if html:
            ctx["html"] = html
        if code:
            ctx["code"] = code
        if lines:
            ctx["lines"] = lines
        if journal_text is not None:
            ctx["journal_text"] = journal_text
        if label is not None:
            ctx["label"] = label
        if buckets is not None:
            ctx["buckets"] = int(buckets)
        ok = await op.run(ctx)
        out: dict = {"operator": operator, "success": bool(ok)}
        if ctx.get("outputs"):
            out["outputs"] = ctx["outputs"]
            conn = None
            try:
                conn = get_db()
                init_db(conn)
                def _canonical_kind(op_name: str, out_key: str) -> str:
                    mapping = {
                        "journal_summary_from_text": "journal_summary",
                        "journal_trend_from_text": "journal_trend",
                        "journal_top_words_from_text": "journal_top_words",
                        "journal_top_severities_from_text": "journal_top_severities",
                        "journal_severity_ratio_from_text": "journal_severity_ratio",
                    }
                    return mapping.get(op_name, out_key)
                for k, v in ctx["outputs"].items():
                    kind = _canonical_kind(operator, str(k))
                    label_val = str(k)
                    payload = v if isinstance(v, dict) else {"value": v}
                    save_artifact(conn, kind=kind, label=label_val, data=payload)
            except Exception:
                pass
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        print(out)
    asyncio.run(_run())


def register(app: typer.Typer) -> None:
    app.command()(chat)
    app.command()(eval)
    app.command("op-run")(op_run)
