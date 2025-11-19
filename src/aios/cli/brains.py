from __future__ import annotations

import json
from typing import Optional

import typer

from aios.core.brains import BrainRegistry, Router
import os
from pathlib import Path
import yaml

app = typer.Typer(help="Manage and probe AI-OS sub-brains")


def _mk_router(storage_limit_mb: Optional[float], prefix: str, default_modalities: list[str], create_cfg: dict, strategy: str = "hash", modality_overrides: Optional[dict] = None, store_dir: Optional[str] = None, modality_caps_gb: Optional[dict] = None) -> tuple[BrainRegistry, Router]:
    reg = BrainRegistry(total_storage_limit_mb=storage_limit_mb)
    if store_dir:
        reg.store_dir = store_dir
        # Load any persisted pinned brains state
    reg.load_pinned()
    reg.load_masters()
    if modality_caps_gb:
        # convert GB caps to MB
        reg.storage_limit_mb_by_modality = {str(k).strip().lower(): float(v) * 1024.0 for k, v in modality_caps_gb.items()}
    router = Router(
        registry=reg,
        default_modalities=list(default_modalities or ["text"]),
        brain_prefix=prefix or "brain",
    create_cfg=dict(create_cfg or {}),
    strategy=strategy,
    modality_overrides=dict(modality_overrides or {}),
    )
    return reg, router


@app.command()
def list_brains(
    storage_limit_mb: Optional[float] = typer.Option(None, help="Global storage cap in MB"),
    storage_limit_gb: Optional[float] = typer.Option(None, help="Global storage cap in GB (overrides MB if set)"),
    prefix: str = typer.Option("brain", help="Brain name prefix"),
    default_modalities: list[str] = typer.Option(["text"], help="Default modalities"),
    store_dir: Optional[str] = typer.Option(None, help="Directory to offload/load brains"),
    modality_caps_gb: Optional[str] = typer.Option(None, help="JSON map of modality->GB cap (e.g., {\"text\":0.25})"),
):
    mb = storage_limit_mb
    if storage_limit_gb is not None:
        mb = float(storage_limit_gb) * 1024.0
    caps = json.loads(modality_caps_gb) if modality_caps_gb else None
    reg, _ = _mk_router(mb, prefix, default_modalities, {}, store_dir=store_dir, modality_caps_gb=caps)
    typer.echo(json.dumps({"brains": reg.list()}))


@app.command()
def route(
    payload: str = typer.Option("{}", help="JSON payload string"),
    modalities: list[str] = typer.Option(["text"], help="Modalities for the task"),
    storage_limit_mb: Optional[float] = typer.Option(None, help="Global storage cap in MB"),
    storage_limit_gb: Optional[float] = typer.Option(None, help="Global storage cap in GB (overrides MB if set)"),
    prefix: str = typer.Option("brain", help="Brain name prefix"),
    create_cfg: str = typer.Option("{}", help="JSON with trainer overrides"),
    strategy: str = typer.Option("hash", help="Routing strategy: hash|round_robin"),
    modality_overrides: str = typer.Option("{}", help="JSON mapping modality->trainer overrides"),
    per_brain_limit_gb: Optional[float] = typer.Option(None, help="Per-brain width_storage_limit in GB (overrides create_cfg if set)"),
    store_dir: Optional[str] = typer.Option(None, help="Directory to offload/load brains"),
    modality_caps_gb: Optional[str] = typer.Option(None, help="JSON map of modality->GB cap (e.g., {\"text\":0.25})"),
):
    mb = storage_limit_mb
    if storage_limit_gb is not None:
        mb = float(storage_limit_gb) * 1024.0
    ccfg = json.loads(create_cfg)
    if per_brain_limit_gb is not None:
        ccfg = dict(ccfg or {})
        ccfg["width_storage_limit_mb"] = float(per_brain_limit_gb) * 1024.0
        ccfg.setdefault("dynamic_width", True)
    caps = json.loads(modality_caps_gb) if modality_caps_gb else None
    reg, router = _mk_router(mb, prefix, modalities, ccfg, strategy=strategy, modality_overrides=json.loads(modality_overrides), store_dir=store_dir, modality_caps_gb=caps)
    before = reg.list()
    res = router.handle({"modalities": modalities, "payload": json.loads(payload or "{}")})
    after = reg.list()
    typer.echo(json.dumps({"before": before, "after": after, "result": res}))


@app.command()
def stats(
    storage_limit_mb: Optional[float] = typer.Option(None, help="Global storage cap in MB"),
    storage_limit_gb: Optional[float] = typer.Option(None, help="Global storage cap in GB (overrides MB if set)"),
    prefix: str = typer.Option("brain", help="Brain name prefix"),
    default_modalities: list[str] = typer.Option(["text"], help="Default modalities"),
    store_dir: Optional[str] = typer.Option(None, help="Directory to offload/load brains"),
    modality_caps_gb: Optional[str] = typer.Option(None, help="JSON map of modality->GB cap"),
):
    mb = storage_limit_mb
    if storage_limit_gb is not None:
        mb = float(storage_limit_gb) * 1024.0
    caps = json.loads(modality_caps_gb) if modality_caps_gb else None
    reg, _ = _mk_router(mb, prefix, default_modalities, {}, store_dir=store_dir, modality_caps_gb=caps)
    typer.echo(json.dumps(reg.stats()))


@app.command()
def pin(
    name: str = typer.Argument(..., help="Brain name to pin (prevent eviction)"),
    store_dir: Optional[str] = typer.Option(None, help="Directory to persist pinned list (recommended)"),
):
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    # Ensure we load existing pins first (done in _mk_router), then add
    reg.pin(name)
    typer.echo(json.dumps({"ok": True, "pinned": sorted(list(reg.pinned))}))


@app.command()
def unpin(
    name: str = typer.Argument(..., help="Brain name to unpin"),
    store_dir: Optional[str] = typer.Option(None, help="Directory containing pinned.json"),
):
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    # Prevent unpin if this is a master
    if name in reg.masters:
        typer.echo(json.dumps({"ok": False, "error": "cannot unpin master brain", "name": name}))
        raise typer.Exit(code=1)
    reg.unpin(name)
    typer.echo(json.dumps({"ok": True, "pinned": sorted(list(reg.pinned))}))


@app.command(name="list-pinned")
def list_pinned(
    store_dir: Optional[str] = typer.Option(None, help="Directory containing pinned.json"),
):
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    typer.echo(json.dumps({"pinned": sorted(list(reg.pinned))}))


@app.command()
def prune(
    storage_limit_mb: Optional[float] = typer.Option(None, help="Global storage cap in MB"),
    storage_limit_gb: Optional[float] = typer.Option(None, help="Global storage cap in GB (overrides MB if set)"),
    prefix: str = typer.Option("brain", help="Brain name prefix"),
    default_modalities: list[str] = typer.Option(["text"], help="Default modalities"),
    target_mb: Optional[float] = typer.Option(None, help="Target memory after prune (MB); default uses global cap"),
    offload: bool = typer.Option(False, help="Offload brains to store_dir before eviction"),
    store_dir: Optional[str] = typer.Option(None, help="Directory to offload brains"),
    modality_caps_gb: Optional[str] = typer.Option(None, help="JSON map of modality->GB cap"),
):
    mb = storage_limit_mb
    if storage_limit_gb is not None:
        mb = float(storage_limit_gb) * 1024.0
    caps = json.loads(modality_caps_gb) if modality_caps_gb else None
    reg, router = _mk_router(mb, prefix, default_modalities, {}, store_dir=store_dir, modality_caps_gb=caps)
    # warm up by creating one brain so stats are non-empty in demo
    router.handle({"modalities": default_modalities, "payload": {}})
    evicted = reg.prune(target_mb, offload=offload)
    typer.echo(json.dumps({"evicted": evicted, "remaining": reg.list(), "used_bytes": reg.stats().get("used_bytes")}))


@app.command()
def list_offloaded(
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
):
    import glob
    from pathlib import Path
    p = Path(store_dir)
    names = []
    if p.exists():
        for m in p.glob("*.json"):
            names.append(m.stem)
    typer.echo(json.dumps({"offloaded": names}))


@app.command()
def load(
    name: str = typer.Argument(..., help="Brain name to load into memory"),
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
    set_master: bool = typer.Option(True, help="Mark as master brain (default: true)"),
):
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    b = reg.get(name)
    ok = b is not None
    
    # If loaded successfully and set_master is true, mark as master for routing
    if ok and set_master:
        # Clear any other masters for the same modalities to prevent conflicts
        brain_modalities = b.modalities if hasattr(b, 'modalities') else ["text"]
        masters_to_remove = []
        for master_name in list(reg.masters):
            if master_name != name:
                # Check if this master has the same modalities
                master_mods = reg.usage.get(master_name, {}).get("modalities", [])
                if master_mods == brain_modalities:
                    masters_to_remove.append(master_name)
        
        # Remove conflicting masters
        for master_name in masters_to_remove:
            reg.unmark_master(master_name)
        
        # Mark the loaded brain as master
        reg.mark_master(name)
        # Also ensure it's pinned to prevent eviction
        reg.pin(name)
    
    typer.echo(json.dumps({"ok": ok, "name": name, "master": bool(ok and set_master)}))


@app.command()
def delete(
    name: str = typer.Argument(..., help="Brain name to delete (from memory and offloaded store)"),
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
):
    """Delete a brain: remove from registry memory and delete offloaded files if present."""
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    # Remove from memory
    ok = False
    error_details: Optional[str] = None
    try:
        if name in reg.brains:
            del reg.brains[name]
        # Clear relations and pins
        reg.unpin(name)
        reg.unmark_master(name)
        try:
            if name in reg.parent:
                reg.clear_parent(name)
            # Remove from any other parent's children list
            for p, kids in list(reg.children.items()):
                if name in kids:
                    reg.children[p] = [k for k in kids if k != name]
            # Remove its own children map
            if name in reg.children:
                del reg.children[name]
        except Exception:
            pass
        # Remove usage metadata
        try:
            if name in reg.usage:
                del reg.usage[name]
        except Exception:
            pass
        # Delete offloaded files and actv1 bundle dir
        try:
            import os, shutil
            from aios.core.brains.registry_storage import get_store_paths
            
            npz, meta = get_store_paths(reg, name)
            if os.path.exists(npz):
                os.remove(npz)
            if os.path.exists(meta):
                os.remove(meta)
            bundle_dir = os.path.join(store_dir, "actv1", name)
            if os.path.isdir(bundle_dir):
                # Remove directory with proper error handling
                try:
                    shutil.rmtree(bundle_dir)
                except PermissionError as e:
                    error_details = f"Permission denied: {e}"
                    # Try to remove files individually
                    for root, dirs, files in os.walk(bundle_dir, topdown=False):
                        for file in files:
                            try:
                                os.chmod(os.path.join(root, file), 0o777)
                                os.remove(os.path.join(root, file))
                            except Exception:
                                pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except Exception:
                                pass
                    try:
                        os.rmdir(bundle_dir)
                    except Exception:
                        pass
                except Exception as e:
                    error_details = f"Delete failed: {e}"
        except Exception as e:
            error_details = f"Cleanup error: {e}"
        ok = True
    except Exception as e:
        ok = False
        error_details = str(e)

    result = {"ok": ok, "deleted": name}
    if error_details:
        key = "error" if not ok else "warning"
        result[key] = error_details
    typer.echo(json.dumps(result))
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def rename(
    old: str = typer.Argument(..., help="Existing brain name"),
    new: str = typer.Argument(..., help="New brain name"),
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
):
    """Rename a brain in memory and best-effort on disk (offloaded files)."""
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    ok = reg.rename(old, new)
    # Also rename ACTV1 brain bundle directory and update brain.json name
    try:
        import os, json as _json
        ob = os.path.join(store_dir, "actv1", old)
        nb = os.path.join(store_dir, "actv1", new)
        if os.path.isdir(ob) and not os.path.exists(nb):
            os.rename(ob, nb)
            meta = os.path.join(nb, "brain.json")
            if os.path.exists(meta):
                try:
                    with open(meta, "r", encoding="utf-8") as f:
                        data = _json.load(f) or {}
                    data["name"] = new
                    with open(meta, "w", encoding="utf-8") as f:
                        _json.dump(data, f, indent=2)
                except Exception:
                    pass
    except Exception:
        pass
    typer.echo(json.dumps({"ok": bool(ok), "old": old, "new": new}))


@app.command(name="set-master")
def set_master(
    name: str = typer.Argument(..., help="Brain name"),
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="Mark/unmark as master"),
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
):
    """Mark or unmark a brain as master (masters are always pinned)."""
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    if enabled:
        reg.mark_master(name)
        ok = True
    else:
        reg.unmark_master(name)
        ok = True
    typer.echo(json.dumps({"ok": ok, "name": name, "master": bool(enabled)}))


@app.command(name="set-parent")
def set_parent(
    child: str = typer.Argument(..., help="Child brain name"),
    parent: Optional[str] = typer.Argument(None, help="Parent brain name (omit to clear)"),
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing offloaded brains"),
):
    """Set or clear a brain's parent (master) relationship."""
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    reg.set_parent(child, parent)
    typer.echo(json.dumps({"ok": True, "child": child, "parent": parent}))


@app.command()
def cleanup(
    store_dir: str = typer.Option("artifacts/brains", help="Directory containing brains"),
):
    """Remove phantom brains (masters/pins with no actual brain files) and fix registry state."""
    reg, _ = _mk_router(None, "brain", ["text"], {}, store_dir=store_dir)
    
    removed = []
    
    # Check each pinned brain
    for name in list(reg.pinned):
        # Try to load the brain
        brain = reg.get(name)
        if brain is None:
            # Brain doesn't exist - remove from pinned
            reg.unpin(name)
            removed.append(name)
    
    # Check each master brain
    for name in list(reg.masters):
        # Try to load the brain
        brain = reg.get(name)
        if brain is None:
            # Brain doesn't exist - remove from masters
            reg.unmark_master(name)
            if name not in removed:
                removed.append(name)
    
    typer.echo(json.dumps({"ok": True, "removed_phantoms": removed, "remaining_pins": list(reg.pinned), "remaining_masters": list(reg.masters)}))


@app.command()
def config_show(
    config: str = typer.Option("config/default.yaml", help="Path to config YAML"),
):
    """Show brains config (global + trainer overrides)."""
    path = Path(config)
    if not path.exists():
        typer.echo(json.dumps({"error": f"config not found: {str(path)}"}))
        raise typer.Exit(code=1)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    brains = data.get("brains", {}) if isinstance(data, dict) else {}
    typer.echo(json.dumps({"brains": brains}))


@app.command()
def config_set(
    config: str = typer.Option("config/default.yaml", help="Path to config YAML"),
    enabled: Optional[bool] = typer.Option(None, help="Enable/disable brains orchestrator features"),
    storage_limit_mb: Optional[float] = typer.Option(None, help="Global brains storage cap (MB)"),
    storage_limit_gb: Optional[float] = typer.Option(None, help="Global brains storage cap (GB), overrides MB if set"),
    per_brain_limit_mb: Optional[float] = typer.Option(
        None, help="Per-brain width_storage_limit_mb (MB) for trainer overrides"
    ),
    per_brain_limit_gb: Optional[float] = typer.Option(
        None, help="Per-brain width limit (GB), overrides MB if set"
    ),
    brain_prefix: Optional[str] = typer.Option(None, help="Brain name prefix"),
    default_modalities: Optional[list[str]] = typer.Option(None, help="Default modalities list"),
):
    """Update brains config values in YAML (creates section if missing)."""
    path = Path(config)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    brains = dict(cfg.get("brains") or {})
    if enabled is not None:
        brains["enabled"] = bool(enabled)
    if storage_limit_mb is not None:
        brains["storage_limit_mb"] = float(storage_limit_mb)
    if storage_limit_gb is not None:
        brains["storage_limit_mb"] = float(storage_limit_gb) * 1024.0
        brains["storage_limit_gb"] = float(storage_limit_gb)
    if brain_prefix is not None:
        brains["prefix"] = str(brain_prefix)
    if default_modalities is not None:
        brains["default_modalities"] = list(default_modalities)
    # trainer overrides
    tovr = dict(brains.get("trainer_overrides") or {})
    if per_brain_limit_mb is not None:
        tovr["width_storage_limit_mb"] = float(per_brain_limit_mb)
        # Ensure dynamic width enabled when setting a width limit
        if "dynamic_width" not in tovr:
            tovr["dynamic_width"] = True
    if per_brain_limit_gb is not None:
        tovr["width_storage_limit_mb"] = float(per_brain_limit_gb) * 1024.0
        tovr["width_storage_limit_gb"] = float(per_brain_limit_gb)
        if "dynamic_width" not in tovr:
            tovr["dynamic_width"] = True
    if tovr:
        brains["trainer_overrides"] = tovr
    cfg["brains"] = brains
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    typer.echo(json.dumps({"ok": True, "brains": brains, "path": str(path)}))

if __name__ == "__main__":
    app()
