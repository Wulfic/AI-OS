from __future__ import annotations

import asyncio
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_config(path: Optional[str] = None) -> dict:
    env_path = os.environ.get("AIOS_CONFIG")
    p = Path(path or env_path or Path.home() / ".config/aios/config.yaml")
    if not p.exists():
        # fall back to repo default if available
        repo_default = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
        if repo_default.exists():
            with open(repo_default, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(
    cfg: dict,
    *,
    level: Optional[str] = None,
    json_preferred: Optional[bool] = None,
    stream: Optional[str] = None,
    logfile: Optional[str] = None,
) -> None:
    """Initialize logging with sensible defaults and VS Code-friendly overrides.

    Honors environment variables:
      - AIOS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default DEBUG in VS Code debug sessions)
      - AIOS_LOG_JSON: 1/0 to force json vs text formatter (default json when logging.yaml specifies it)
      - AIOS_LOG_STREAM: stdout|stderr (default stderr for debug sessions)
      - AIOS_LOG_FILE: path to log file (overrides configured filename if provided)
      - AIOS_DEBUG: any non-empty value enables DEBUG level and stderr stream
    """
    cfg_path = cfg.get("logging", {}).get("config_path") if isinstance(cfg, dict) else None

    # Detect if we're running under a VS Code debug session and prefer DEBUG to stderr
    is_vscode = bool(os.environ.get("VSCODE_PID"))
    is_debug = bool(os.environ.get("AIOS_DEBUG")) or is_vscode

    # Gather env overrides
    env_level = (level or os.environ.get("AIOS_LOG_LEVEL") or ("DEBUG" if is_debug else "")).upper()
    env_json = str(json_preferred) if json_preferred is not None else os.environ.get("AIOS_LOG_JSON")
    env_stream = (stream or os.environ.get("AIOS_LOG_STREAM") or ("stderr" if is_debug else "")).lower()
    env_file = logfile or os.environ.get("AIOS_LOG_FILE")

    # Load config dict from YAML if available; otherwise craft a minimal default
    conf: dict
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
    else:
        conf = {
            "version": 1,
            "formatters": {
                "json": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
                "verbose": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "aios": {"level": "INFO", "handlers": ["console"], "propagate": False},
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }

    # Ensure keys exist
    conf.setdefault("formatters", {})
    conf.setdefault("handlers", {})
    conf.setdefault("loggers", {})
    conf.setdefault("root", {})

    # Add/ensure a good text formatter for debug sessions
    conf["formatters"].setdefault(
        "verbose",
        {"format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"},
    )

    # Determine desired console formatter (json vs verbose text)
    use_json = None
    if env_json is not None:
        use_json = env_json.strip() not in ("0", "false", "False")
    # If not explicitly set, keep whatever logging.yaml configured
    console = conf["handlers"].get("console", {})
    current_formatter = console.get("formatter") or "json"
    desired_formatter = ("json" if use_json else "verbose") if use_json is not None else current_formatter

    # Determine desired stream for console
    desired_stream = console.get("stream") or "ext://sys.stdout"
    if env_stream in ("stdout", "stderr"):
        desired_stream = f"ext://sys.{env_stream}"
    elif is_debug:
        desired_stream = "ext://sys.stderr"

    # Determine desired logging level
    def _normalize_level(val: str | int | None) -> str:
        if val is None:
            return ""
        if isinstance(val, int):
            return logging.getLevelName(val)
        return val.upper()

    existing_root_level = _normalize_level(conf.get("root", {}).get("level")) or "WARNING"
    desired_level = env_level or existing_root_level
    if not desired_level:
        desired_level = "DEBUG" if is_debug else "INFO"

    # Apply overrides to console handler
    conf["handlers"].setdefault("console", {"class": "logging.StreamHandler"})
    conf["handlers"]["console"]["level"] = desired_level
    conf["handlers"]["console"]["formatter"] = desired_formatter
    conf["handlers"]["console"]["stream"] = desired_stream

    # If a file handler exists or an override is provided, ensure it uses at least INFO unless explicitly set
    if "file" in conf["handlers"] or env_file:
        fh = conf["handlers"].setdefault("file", {"class": "logging.handlers.RotatingFileHandler"})
        fh.setdefault("maxBytes", 1048576)
        fh.setdefault("backupCount", 3)
        fh.setdefault("formatter", desired_formatter)
        if env_file:
            fh["filename"] = env_file
        fh.setdefault("level", "INFO")

    # Raise 'aios' logger and root to desired level for debug visibility
    conf["loggers"].setdefault("aios", {"handlers": ["console"], "propagate": False})
    conf["loggers"]["aios"]["level"] = desired_level
    conf["root"]["level"] = desired_level
    conf["root"].setdefault("handlers", ["console"])

    # Install configuration
    logging.config.dictConfig(conf)
    # Capture warnings module messages into logging
    try:
        logging.captureWarnings(True)
    except Exception:
        pass

    # Global exception hooks to surface unexpected errors into logs (and thus Debug Console)
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("aios").exception("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = _excepthook

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and not loop.is_closed():
        def _handle_asyncio(loop, context):  # type: ignore[no-redef]
            msg = context.get("message") or "Unhandled asyncio exception"
            err = context.get("exception")
            if err:
                logging.getLogger("aios").exception(msg, exc_info=err)
            else:
                logging.getLogger("aios").error(msg)
        try:
            loop.set_exception_handler(_handle_asyncio)
        except Exception:
            pass


def dml_cfg_path() -> Path:
    return Path.home() / ".config/aios/dml_python.txt"
