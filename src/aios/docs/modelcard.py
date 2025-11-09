from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json


def scaffold_config(cfg: Dict[str, Any], output: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        p = Path(output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write YAML if available, else JSON
        try:
            import yaml  # type: ignore

            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            return True, {"path": str(p), "format": "yaml"}
        except Exception:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            return True, {"path": str(p), "format": "json"}
    except Exception as e:
        return False, {"error": str(e)}


def generate_modelcard(config: str, output: str, version: str = "1.0") -> Tuple[bool, Dict[str, Any]]:
    """Generate a model card HTML. Tries dreams_mc if installed; falls back to a minimal HTML template.

    Returns: (ok, details)
    details includes 'path' on success or 'error' on failure.
    """
    cfg_path = Path(config).expanduser()
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try dreams_mc
    try:
        from dreams_mc.make_model_card import generate_modelcard as _gen  # type: ignore
    except Exception:
        _gen = None  # type: ignore

    if _gen is not None:  # type: ignore
        try:
            _gen(str(cfg_path), str(out_path), str(version))
            return True, {"path": str(out_path), "vendor": "dreams_mc"}
        except Exception:
            # Fall through to fallback on error
            pass

    # Fallback: very small HTML reading minimal fields
    try:
        import yaml  # type: ignore
        cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    title = "AI-OS Model Card"
    desc = str(cfg.get("describe_overview", "Auto-generated model card."))
    logo = cfg.get("logo_path")
    body = f"""
<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{title}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:24px}} .img{{max-width:640px}}</style></head>
<body>
<h1>{title}</h1>
<p><em>Spec version {version}</em></p>
{f'<img class="img" src="{logo}" alt="logo" />' if logo else ''}
<h2>Overview</h2>
<p>{desc}</p>
<h2>Artifacts</h2>
<ul>
"""
    for key in [
        "data_figpath",
        "result_table_figpath",
        "cm_figpath",
        "acc_figpath",
        "loss_figpath",
        "uncertainty_figpath",
    ]:
        val = cfg.get(key)
        if val:
            body += f"<li>{key}: <img class=\"img\" src=\"{val}\" alt=\"{key}\" /></li>\n"
    body += "</ul>\n</body></html>\n"
    out_path.write_text(body, encoding="utf-8")
    return True, {"path": str(out_path), "vendor": None}
