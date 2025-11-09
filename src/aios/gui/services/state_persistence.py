from __future__ import annotations

import json
import configparser
from pathlib import Path
from typing import Any


def save_app_state(
    path: Path,
    *,
    core_toggles: dict[str, Any],
    dataset_path: str,
    builder_state: dict[str, Any],
    resources_values: dict[str, Any],
    hrm_training_state: dict[str, Any] | None,
    evaluation_state: dict[str, Any] | None = None,
    settings_state: dict[str, Any] | None = None,
) -> None:
    """Persist app state to both JSON and INI files (best effort)."""
    # JSON format - main state file
    d = {
        # core toggles
        "cpu": bool(core_toggles.get("cpu", False)),
        "cuda": bool(core_toggles.get("cuda", False)),
        "xpu": bool(core_toggles.get("xpu", False)),
        "dml": bool(core_toggles.get("dml", False)),
        "mps": bool(core_toggles.get("mps", False)),
        "dml_python": core_toggles.get("dml_python", ""),
        # dataset path
        "dataset_path": dataset_path or "",
        # dataset builder
        "builder_type": builder_state.get("type"),
        "builder_query": builder_state.get("query"),
        "builder_max_images": builder_state.get("max_images"),
        "builder_per_site": builder_state.get("per_site"),
        "builder_search_results": builder_state.get("search_results"),
        "builder_dataset_name": builder_state.get("dataset_name"),
        "builder_overwrite": bool(builder_state.get("overwrite", False)),
        # resources
        **{
            "cpu_threads": int(resources_values.get("cpu_threads", 0) or 0),
            "gpu_mem_pct": int(resources_values.get("gpu_mem_pct", 0) or 0),
            "cpu_util_pct": int(resources_values.get("cpu_util_pct", 0) or 0),
            "gpu_util_pct": int(resources_values.get("gpu_util_pct", 0) or 0),
            "train_device": resources_values.get("train_device"),
            "run_device": resources_values.get("run_device"),
            "train_cuda_selected": resources_values.get("train_cuda_selected"),
            "train_cuda_mem_pct": resources_values.get("train_cuda_mem_pct"),
            "train_cuda_util_pct": resources_values.get("train_cuda_util_pct"),
            "run_cuda_selected": resources_values.get("run_cuda_selected"),
            "run_cuda_mem_pct": resources_values.get("run_cuda_mem_pct"),
            "run_cuda_util_pct": resources_values.get("run_cuda_util_pct"),
            "dataset_cap": resources_values.get("dataset_cap", ""),
        },
    }
    if isinstance(hrm_training_state, dict):
        d["hrm_training"] = hrm_training_state
    if isinstance(evaluation_state, dict):
        d["evaluation"] = evaluation_state
    if isinstance(settings_state, dict):
        d["settings"] = settings_state
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass
    
    # INI format (new - more user-friendly and cross-session persistent)
    try:
        ini_path = path.parent / "settings.ini"
        _save_settings_ini(
            ini_path,
            core_toggles=core_toggles,
            dataset_path=dataset_path,
            builder_state=builder_state,
            resources_values=resources_values,
            hrm_training_state=hrm_training_state,
            evaluation_state=evaluation_state,
            settings_state=settings_state,
        )
    except Exception:
        pass


def _save_settings_ini(
    path: Path,
    *,
    core_toggles: dict[str, Any],
    dataset_path: str,
    builder_state: dict[str, Any],
    resources_values: dict[str, Any],
    hrm_training_state: dict[str, Any] | None,
    evaluation_state: dict[str, Any] | None = None,
    settings_state: dict[str, Any] | None = None,
) -> None:
    """Save settings to INI format."""
    config = configparser.ConfigParser()
    
    # [Core] section
    config["Core"] = {
        "cpu": str(core_toggles.get("cpu", False)),
        "cuda": str(core_toggles.get("cuda", False)),
        "xpu": str(core_toggles.get("xpu", False)),
        "dml": str(core_toggles.get("dml", False)),
        "mps": str(core_toggles.get("mps", False)),
        "dml_python": core_toggles.get("dml_python", ""),
        "dataset_path": dataset_path or "",
    }
    
    # [DatasetBuilder] section
    config["DatasetBuilder"] = {
        "type": str(builder_state.get("type", "")),
        "query": str(builder_state.get("query", "")),
        "max_images": str(builder_state.get("max_images", "")),
        "per_site": str(builder_state.get("per_site", "")),
        "search_results": str(builder_state.get("search_results", "")),
        "dataset_name": str(builder_state.get("dataset_name", "")),
        "overwrite": str(builder_state.get("overwrite", False)),
    }
    
    # [Resources] section
    config["Resources"] = {
        "cpu_threads": str(resources_values.get("cpu_threads", 0)),
        "gpu_mem_pct": str(resources_values.get("gpu_mem_pct", 0)),
        "cpu_util_pct": str(resources_values.get("cpu_util_pct", 0)),
        "gpu_util_pct": str(resources_values.get("gpu_util_pct", 0)),
        "train_device": str(resources_values.get("train_device", "auto")),
        "run_device": str(resources_values.get("run_device", "auto")),
        "dataset_cap": str(resources_values.get("dataset_cap", "")),
    }
    
    # Store list values as JSON strings in INI
    if resources_values.get("train_cuda_selected"):
        config["Resources"]["train_cuda_selected"] = json.dumps(resources_values["train_cuda_selected"])
    if resources_values.get("train_cuda_mem_pct"):
        config["Resources"]["train_cuda_mem_pct"] = json.dumps(resources_values["train_cuda_mem_pct"])
    if resources_values.get("train_cuda_util_pct"):
        config["Resources"]["train_cuda_util_pct"] = json.dumps(resources_values["train_cuda_util_pct"])
    if resources_values.get("run_cuda_selected"):
        config["Resources"]["run_cuda_selected"] = json.dumps(resources_values["run_cuda_selected"])
    if resources_values.get("run_cuda_mem_pct"):
        config["Resources"]["run_cuda_mem_pct"] = json.dumps(resources_values["run_cuda_mem_pct"])
    if resources_values.get("run_cuda_util_pct"):
        config["Resources"]["run_cuda_util_pct"] = json.dumps(resources_values["run_cuda_util_pct"])
    
    # [HRMTraining] section
    if isinstance(hrm_training_state, dict):
        config["HRMTraining"] = {}
        for key, value in hrm_training_state.items():
            # Convert all values to strings for INI format
            if isinstance(value, (list, dict)):
                config["HRMTraining"][key] = json.dumps(value)
            else:
                config["HRMTraining"][key] = str(value)
    
    # [Evaluation] section
    if isinstance(evaluation_state, dict):
        config["Evaluation"] = {}
        for key, value in evaluation_state.items():
            if isinstance(value, (list, dict)):
                config["Evaluation"][key] = json.dumps(value)
            else:
                config["Evaluation"][key] = str(value)
    
    # [Settings] section (theme, etc.)
    if isinstance(settings_state, dict):
        config["Settings"] = {}
        for key, value in settings_state.items():
            if isinstance(value, (list, dict)):
                config["Settings"][key] = json.dumps(value)
            else:
                config["Settings"][key] = str(value)
    
    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        config.write(f)


def load_app_state(path: Path) -> dict[str, Any]:
    """Load app state from INI (preferred) or JSON (fallback); return dict or {} if missing/invalid."""
    # Try INI format first
    ini_path = path.parent / "settings.ini"
    if ini_path.exists():
        try:
            return _load_settings_ini(ini_path)
        except Exception:
            pass  # Fall back to JSON
    
    # Fallback to JSON format
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_settings_ini(path: Path) -> dict[str, Any]:
    """Load settings from INI format."""
    config = configparser.ConfigParser()
    config.read(path, encoding='utf-8')
    
    result: dict[str, Any] = {}
    
    # [Core] section
    if "Core" in config:
        result["cpu"] = config["Core"].getboolean("cpu", fallback=False)
        result["cuda"] = config["Core"].getboolean("cuda", fallback=False)
        result["xpu"] = config["Core"].getboolean("xpu", fallback=False)
        result["dml"] = config["Core"].getboolean("dml", fallback=False)
        result["mps"] = config["Core"].getboolean("mps", fallback=False)
        result["dml_python"] = config["Core"].get("dml_python", fallback="")
        result["dataset_path"] = config["Core"].get("dataset_path", fallback="")
    
    # [DatasetBuilder] section
    if "DatasetBuilder" in config:
        result["builder_type"] = config["DatasetBuilder"].get("type", fallback="")
        result["builder_query"] = config["DatasetBuilder"].get("query", fallback="")
        result["builder_max_images"] = config["DatasetBuilder"].get("max_images", fallback="")
        result["builder_per_site"] = config["DatasetBuilder"].get("per_site", fallback="")
        result["builder_search_results"] = config["DatasetBuilder"].get("search_results", fallback="")
        result["builder_dataset_name"] = config["DatasetBuilder"].get("dataset_name", fallback="")
        result["builder_overwrite"] = config["DatasetBuilder"].getboolean("overwrite", fallback=False)
    
    # [Resources] section
    if "Resources" in config:
        result["cpu_threads"] = config["Resources"].getint("cpu_threads", fallback=0)
        result["gpu_mem_pct"] = config["Resources"].getint("gpu_mem_pct", fallback=0)
        result["cpu_util_pct"] = config["Resources"].getint("cpu_util_pct", fallback=0)
        result["gpu_util_pct"] = config["Resources"].getint("gpu_util_pct", fallback=0)
        result["train_device"] = config["Resources"].get("train_device", fallback="auto")
        result["run_device"] = config["Resources"].get("run_device", fallback="auto")
        result["dataset_cap"] = config["Resources"].get("dataset_cap", fallback="")
        
        # Parse JSON-encoded lists
        try:
            train_cuda_str = config["Resources"].get("train_cuda_selected", fallback=None)
            if train_cuda_str:
                result["train_cuda_selected"] = json.loads(train_cuda_str)
        except Exception:
            pass
        
        try:
            train_mem_str = config["Resources"].get("train_cuda_mem_pct", fallback=None)
            if train_mem_str:
                result["train_cuda_mem_pct"] = json.loads(train_mem_str)
        except Exception:
            pass
        
        try:
            train_util_str = config["Resources"].get("train_cuda_util_pct", fallback=None)
            if train_util_str:
                result["train_cuda_util_pct"] = json.loads(train_util_str)
        except Exception:
            pass
        
        try:
            run_cuda_str = config["Resources"].get("run_cuda_selected", fallback=None)
            if run_cuda_str:
                result["run_cuda_selected"] = json.loads(run_cuda_str)
        except Exception:
            pass
        
        try:
            run_mem_str = config["Resources"].get("run_cuda_mem_pct", fallback=None)
            if run_mem_str:
                result["run_cuda_mem_pct"] = json.loads(run_mem_str)
        except Exception:
            pass
        
        try:
            run_util_str = config["Resources"].get("run_cuda_util_pct", fallback=None)
            if run_util_str:
                result["run_cuda_util_pct"] = json.loads(run_util_str)
        except Exception:
            pass
    
    # [HRMTraining] section
    if "HRMTraining" in config:
        hrm_state: dict[str, Any] = {}
        for key, value in config["HRMTraining"].items():
            # Try to parse as JSON first (for lists/dicts), otherwise use raw string
            try:
                hrm_state[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Try to parse as boolean
                if value.lower() in ("true", "false"):
                    hrm_state[key] = value.lower() == "true"
                # Try to parse as int
                elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                    hrm_state[key] = int(value)
                # Try to parse as float
                elif "." in value:
                    try:
                        hrm_state[key] = float(value)
                    except ValueError:
                        hrm_state[key] = value
                else:
                    hrm_state[key] = value
        result["hrm_training"] = hrm_state
    
    # [Evaluation] section
    if "Evaluation" in config:
        eval_state: dict[str, Any] = {}
        for key, value in config["Evaluation"].items():
            try:
                eval_state[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Try to parse as boolean
                if value.lower() in ("true", "false"):
                    eval_state[key] = value.lower() == "true"
                # Try to parse as int
                elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                    eval_state[key] = int(value)
                # Try to parse as float
                elif "." in value:
                    try:
                        eval_state[key] = float(value)
                    except ValueError:
                        eval_state[key] = value
                else:
                    eval_state[key] = value
        result["evaluation"] = eval_state
    
    # [Settings] section
    if "Settings" in config:
        settings_state: dict[str, Any] = {}
        for key, value in config["Settings"].items():
            try:
                settings_state[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Try to parse as boolean
                if value.lower() in ("true", "false"):
                    settings_state[key] = value.lower() == "true"
                else:
                    settings_state[key] = value
        result["settings"] = settings_state
    
    return result

