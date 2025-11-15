"""Chat operations module for AI-OS GUI.

This module handles:
- Chat routing and execution
- Brain loading/unloading
- Brain listing
- Chat message processing
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable, Optional
import logging
import json
import os
import threading
import time
from pathlib import Path
import platform

if TYPE_CHECKING:
    pass

from ..services import (
    LogCategory,
    DeviceSelectionResult,
    build_device_message,
    resolve_inference_devices,
    resolve_inference_devices_from_state,
    warning_message,
)
from ..utils.resource_management import submit_background
from ..services.brain_registry_service import list_brains as list_available_brains

logger = logging.getLogger(__name__)

# Global persistent registry and router to keep models loaded across chat sessions
_persistent_registry: Optional[Any] = None
_persistent_router: Optional[Any] = None
_router_init_lock = threading.Lock()
_router_warmup_started = False

# Global MCP tool executor for handling tool calls
_tool_executor: Optional[Any] = None

# Shared device selection state for chat routing (guarded by _router_init_lock)
_chat_device_selection: DeviceSelectionResult | None = None
_chat_device_signature: tuple[str, tuple[str, ...]] | None = None
_chat_warning_tokens: set[str] = set()
_chat_env_snapshot: dict[str, str] = {}


def prepare_tensor_parallel(devices: list[str]) -> None:
    """Placeholder hook for future tensor-parallel inference plumbing."""

    if len(devices) < 2:
        return

    if platform.system().lower() != "linux":
        return

    logger.debug("Tensor-parallel preparation stub invoked for devices: %s", devices)


def setup_chat_operations(app: Any) -> None:
    """
    Set up chat operation handlers.
    
    Args:
        app: AiosTkApp instance with _run_cli available
    """
    global _persistent_registry, _persistent_router, _tool_executor
    
    def _resolve_brain_store_dir() -> str:
        """Determine the absolute path to the configured brain store directory."""
        configured: str | None = None

        try:
            from aios.cli.utils import load_config

            cfg = load_config()
            if isinstance(cfg, dict):
                brains_cfg = cfg.get("brains") or {}
                value = brains_cfg.get("store_dir")
                if isinstance(value, str) and value.strip():
                    configured = value.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Falling back to default brain store dir (config load failed): {exc}")

        if not configured:
            configured = "artifacts/brains"

        candidate_path = Path(configured)
        if candidate_path.is_absolute():
            logger.debug(f"Resolved brain store directory (absolute): {candidate_path}")
            return str(candidate_path)

        bases: list[Path] = []
        seen: set[Path] = set()

        def _add_base(path: Path) -> None:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved not in seen:
                seen.add(resolved)
                bases.append(resolved)

        try:
            root_hint = Path(getattr(app, "_project_root", Path.cwd()))
        except Exception:
            root_hint = Path.cwd()

        if root_hint.name.lower() == "src" and root_hint.parent != root_hint:
            _add_base(root_hint.parent)
        _add_base(root_hint)
        _add_base(Path.cwd())

        fallback: Path | None = None
        for base in bases:
            try:
                resolved = (base / candidate_path).resolve()
            except Exception:
                resolved = base / candidate_path

            has_metadata = any(
                (resolved / marker).exists() for marker in ("pinned.json", "masters.json")
            ) or (resolved / "actv1").is_dir()

            if has_metadata:
                logger.debug(f"Resolved brain store directory via metadata: {resolved}")
                return str(resolved)

            if fallback is None and resolved.exists():
                fallback = resolved

        if fallback is not None:
            logger.debug(f"Using existing brain store directory: {fallback}")
            return str(fallback)

        default_path = (bases[0] if bases else Path.cwd()) / candidate_path
        try:
            default_path = default_path.resolve()
        except Exception:
            pass
        logger.debug(f"Brain store directory not found; defaulting to {default_path}")
        return str(default_path)

    _brain_store_dir = _resolve_brain_store_dir()

    resources_panel = getattr(app, "resources_panel", None)

    def _apply_env_overrides(selection: DeviceSelectionResult) -> None:
        global _chat_env_snapshot
        overrides = dict(selection.env_overrides)
        if selection.requested_devices:
            overrides.setdefault("AIOS_REQUESTED_DEVICES", ",".join(selection.requested_devices))
        metadata = selection.metadata if isinstance(selection.metadata, dict) else {}
        physical_devices = []
        if isinstance(metadata, dict):
            maybe_phys = metadata.get("physical_visible_devices")
            if isinstance(maybe_phys, list):
                physical_devices = [str(item) for item in maybe_phys]

        if selection.visible_devices:
            overrides.setdefault("AIOS_VISIBLE_ALIAS_DEVICES", ",".join(selection.visible_devices))
        if physical_devices:
            overrides.setdefault("AIOS_VISIBLE_DEVICES", ",".join(physical_devices))
        elif selection.visible_devices:
            overrides.setdefault("AIOS_VISIBLE_DEVICES", ",".join(selection.visible_devices))

        primary_physical = metadata.get("primary_physical_device") if isinstance(metadata, dict) else None
        if isinstance(primary_physical, str) and primary_physical:
            overrides.setdefault("AIOS_CHAT_PRIMARY_PHYSICAL_DEVICE", primary_physical)
            if selection.device_kind == "cuda" and len(physical_devices) == 1:
                overrides.setdefault("AIOS_CHAT_PINNED_DEVICE", primary_physical)

        updated: dict[str, str] = {}
        for key, value in overrides.items():
            current = os.environ.get(key)
            if current != value:
                os.environ[key] = value
            updated[key] = value

        # Remove environment keys no longer present
        for stale_key in list(_chat_env_snapshot.keys()):
            if stale_key not in overrides and stale_key.startswith("AIOS"):
                os.environ.pop(stale_key, None)

        _chat_env_snapshot = updated

    def _refresh_chat_device_selection() -> DeviceSelectionResult:
        global _chat_device_selection, _chat_device_signature, _chat_warning_tokens
        selection: DeviceSelectionResult
        try:
            selection = resolve_inference_devices(resources_panel)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Chat device resolution failed: %s", exc, exc_info=True)
            selection = resolve_inference_devices_from_state({}, platform.system())

        _chat_device_selection = selection
        app._chat_device_selection = selection  # type: ignore[attr-defined]

        signature = (selection.primary_device, tuple(selection.visible_devices))
        if signature != _chat_device_signature:
            message = f"Chat device selection: {build_device_message(selection)}"
            logger.info(message)
            if hasattr(app, "_log_router") and app._log_router:  # type: ignore[attr-defined]
                app._log_router.log(message, LogCategory.CHAT, "INFO")
            _chat_device_signature = signature

        for token in selection.warnings:
            if token not in _chat_warning_tokens:
                warning_text = warning_message(token)
                logger.warning(warning_text)
                if hasattr(app, "_log_router") and app._log_router:  # type: ignore[attr-defined]
                    app._log_router.log(warning_text, LogCategory.CHAT, "WARNING")
                _chat_warning_tokens.add(token)

        _apply_env_overrides(selection)
        if selection.device_kind == "cuda" and len(selection.visible_devices) > 1:
            prepare_tensor_parallel(selection.visible_devices)
        return selection

    def _ensure_persistent_router() -> tuple[Any, Any]:
        """Ensure persistent registry and router are initialized.
        
        Returns:
            Tuple of (registry, router)
        """
        global _persistent_registry, _persistent_router, _tool_executor

        # Prevent concurrent initialisation when multiple callers race on startup
        with _router_init_lock:
            selection = _refresh_chat_device_selection()

            # Initialize MCP tool executor if not done yet
            if _tool_executor is None:
                try:
                    from aios.core.mcp import ToolExecutor
                    _tool_executor = ToolExecutor()
                    if _tool_executor.enabled:
                        logger.info(
                            "MCP tool executor initialized with %d tools",
                            len(_tool_executor.mcp_client.available_tools),
                        )
                    else:
                        logger.info("MCP tool executor initialized but no tools available")
                except Exception as e:
                    logger.warning("Failed to initialize MCP tool executor: %s", e)
                    _tool_executor = None

            if _persistent_registry is None or _persistent_router is None:
                from aios.cli.utils import load_config
                from aios.core.brains import BrainRegistry, Router

                cfg = load_config()
                brains_cfg = (cfg.get("brains") or {}) if isinstance(cfg, dict) else {}

                storage_limit_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0) or None
                _persistent_registry = BrainRegistry(total_storage_limit_mb=storage_limit_mb)
                _persistent_registry.store_dir = str(brains_cfg.get("store_dir", "artifacts/brains"))

                try:
                    _persistent_registry.load_pinned()
                    _persistent_registry.load_masters()
                except Exception:
                    pass

                create_cfg = dict(brains_cfg.get("trainer_overrides", {}))
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

                _persistent_router = Router(
                    registry=_persistent_registry,
                    default_modalities=list(brains_cfg.get("default_modalities", ["text"])),
                    brain_prefix=str(brains_cfg.get("prefix", "brain")),
                    create_cfg=create_cfg,
                    strategy=str(brains_cfg.get("strategy", "hash")),
                    modality_overrides=dict(brains_cfg.get("modality_overrides", {})),
                )

                logger.info("Initialized persistent chat registry and router")
            else:
                try:
                    _persistent_registry.load_pinned()
                    _persistent_registry.load_masters()
                except Exception:
                    pass

            if _persistent_router is not None:
                _persistent_router.inference_device_getter = (
                    lambda: _chat_device_selection.primary_device if _chat_device_selection else None
                )
                if selection and selection.device_kind == "cuda":
                    logger.debug(
                        "Chat router primary device set to %s (visible=%s)",
                        selection.primary_device,
                        selection.visible_devices,
                    )

            return _persistent_registry, _persistent_router
    
    def _on_chat_route_and_run(
        prompt: str,
        on_token: Callable[[str], None],
        on_done: Callable[[], None],
        on_error: Callable[[str], None],
        model_override: str | None = None,
        max_response_chars: int = 0,
        sampling_params: dict[str, float] | None = None,
    ) -> None:
        """
        Route chat prompt using persistent router (keeps model loaded).
        
        This runs in a background thread to prevent UI freezing during model loading.
        Supports MCP tool calling if enabled.
        
        Args:
            prompt: User message
            on_token: Callback for each token streamed
            on_done: Callback when generation complete
            on_error: Callback on error
            model_override: Optional model name override
            max_response_chars: Max response length in chars (0 = unlimited, use brain's max)
            sampling_params: Optional dict with temperature, top_p, top_k
        """
        def _background_work():
            """Execute generation in background thread to prevent UI freezing."""
            try:
                # Log to chat category
                app._log_router.log(f"Chat: {prompt[:100]}...", LogCategory.CHAT)
                
                # Enhance prompt with tool information if MCP is enabled
                enhanced_prompt = prompt
                if _tool_executor and _tool_executor.enabled:
                    tool_prompt_addition = _tool_executor.get_system_prompt_addition()
                    if tool_prompt_addition:
                        # Prepend tool instructions to user message
                        enhanced_prompt = f"{tool_prompt_addition}\n\nUser: {prompt}"
                        logger.info("Enhanced prompt with MCP tool information")
                
                # Use persistent router to keep model loaded
                registry, router = _ensure_persistent_router()
                
                # Get the brain that will handle this request to configure response limit
                # We need to set max_response_chars BEFORE generation
                task = {"modalities": ["text"], "payload": enhanced_prompt}
                
                # Apply max_response_chars to brain if specified
                # Note: This configures response length but doesn't reload the model
                if max_response_chars > 0:
                    # User set an explicit limit - apply it
                    for brain in registry.brains.values():
                        if hasattr(brain, 'max_response_chars'):
                            brain.max_response_chars = max_response_chars
                            # Recalculate gen_max_new_tokens based on char limit
                            brain.gen_max_new_tokens = min(
                                brain.gen_max_new_tokens,
                                max(256, max_response_chars // 4)
                            )
                else:
                    # 0 = unlimited - use brain's FULL trained max context (not just 75%)
                    for brain in registry.brains.values():
                        if hasattr(brain, 'max_response_chars') and hasattr(brain, '_loaded_max_seq_len'):
                            # When user sets 0, allow using the FULL context window for generation
                            if brain._loaded_max_seq_len > 0:
                                # Use full context minus a small buffer for prompt
                                brain.gen_max_new_tokens = max(256, brain._loaded_max_seq_len - 256)
                                brain.max_response_chars = brain.gen_max_new_tokens * 4
                                logger.info(f"[Chat] Unlimited mode: allowing up to {brain.gen_max_new_tokens} tokens (~{brain.max_response_chars} chars)")
                            else:
                                # Fallback if context not loaded yet - use large default
                                brain.gen_max_new_tokens = 2048
                                brain.max_response_chars = 8192
                
                # Apply sampling parameters if provided
                if sampling_params:
                    for brain in registry.brains.values():
                        if hasattr(brain, 'set_sampling_params'):
                            brain.set_sampling_params(**sampling_params)
                        # Fallback: set individual attributes
                        for key, value in sampling_params.items():
                            if hasattr(brain, key):
                                setattr(brain, key, value)
                
                # Send payload as a string so the router generates a distinct brain name
                # This may trigger brain loading on first call (happens in background)
                res = router.handle(task)
                
                # Process response
                if isinstance(res, dict) and res.get("ok") and res.get("text"):
                    response_text = res.get("text", "")
                    
                    # Check if response contains a tool call (MCP integration)
                    if _tool_executor and _tool_executor.enabled:
                        import asyncio
                        tool_result = asyncio.run(_tool_executor.execute_if_tool_call(response_text))
                        
                        if tool_result:
                            # Model requested a tool - show the tool call info
                            tool_name = tool_result.get("tool_name", "unknown")
                            on_token(f"🔧 Using tool: {tool_name}\n\n")
                            
                            # Format and send tool result
                            formatted_result = _tool_executor.format_tool_result_for_model(tool_result)
                            on_token(formatted_result)
                            on_token("\n\n")
                            
                            # Ask model to synthesize final response based on tool result
                            # Create a follow-up task with tool context
                            followup_prompt = f"{prompt}\n\nTool result:\n{formatted_result}\n\nBased on this tool result, provide a clear answer to the user's question."
                            followup_task = {"modalities": ["text"], "payload": followup_prompt}
                            
                            followup_res = router.handle(followup_task)
                            if isinstance(followup_res, dict) and followup_res.get("ok") and followup_res.get("text"):
                                on_token(followup_res.get("text", ""))
                            
                            on_done()
                            app._log_router.log(f"Chat response with tool call ({tool_name}) completed", LogCategory.CHAT)
                            return
                    
                    # No tool call - send normal response
                    on_token(response_text)
                    on_done()
                    app._log_router.log("Chat response generated", LogCategory.CHAT)
                else:
                    # Handle error response
                    error_msg = res.get("error", "Unknown error") if isinstance(res, dict) else "Invalid response format"
                    on_error(error_msg)
                    logger.error(f"Chat generation failed: {error_msg}")
                
            except Exception as e:
                logger.error(f"Chat routing failed: {e}")
                on_error(str(e))
        
        # Run asynchronously via dispatcher to prevent UI freezing
        try:
            submit_background(
                "chat-ops-dispatch",
                _background_work,
                pool=getattr(app, "_worker_pool", None),
            )
        except RuntimeError as exc:
            logger.error("Failed to queue chat operation: %s", exc)
            fallback_thread = threading.Thread(target=_background_work, name="chat-ops-fallback", daemon=True)
            fallback_thread.start()
    
    def _on_load_brain(brain_name: str) -> str:
        """
        Load a brain model into the persistent registry and set as master.
        
        Args:
            brain_name: Name of brain to load
            
        Returns:
            Result message string
        """
        global _persistent_registry, _persistent_router
        
        try:
            logger.info(f"User action: Loading brain '{brain_name}' for chat")
            app._log_router.log(f"Loading brain: {brain_name}", LogCategory.CHAT, "INFO")
            
            # Ensure persistent registry exists
            registry, router = _ensure_persistent_router()
            
            # Load the brain using registry.get() which auto-loads from disk
            try:
                import time
                start_time = time.time()
                
                logger.debug(f"Attempting to load brain '{brain_name}' from registry")
                brain = registry.get(brain_name)
                
                if brain:
                    load_time = time.time() - start_time
                    # Mark as master so router uses it for chat
                    registry.mark_master(brain_name)
                    
                    logger.info(f"Successfully loaded brain '{brain_name}' in {load_time:.2f}s and set as master")
                    app._log_router.log(f"Brain {brain_name} loaded and set as master for chat ({load_time:.2f}s)", LogCategory.CHAT, "INFO")
                    return f"✓ Brain '{brain_name}' loaded successfully and set as active for chat."
                else:
                    error_msg = f"Failed to load brain '{brain_name}' - brain not found or failed to load. Check that the brain exists in artifacts/brains/actv1/{brain_name}/"
                    logger.warning(f"Brain '{brain_name}' not found in registry")
                    app._log_router.log(error_msg, LogCategory.CHAT, "WARNING")
                    return error_msg
            except Exception as load_error:
                import traceback
                error_details = traceback.format_exc()
                error_msg = f"Failed to load brain '{brain_name}': {load_error}\n{error_details}"
                logger.error(f"Exception while loading brain '{brain_name}': {load_error}", exc_info=True)
                app._log_router.log(error_msg, LogCategory.CHAT, "ERROR")
                return f"Failed to load brain '{brain_name}': {load_error}"
                
        except Exception as e:
            logger.error(f"Failed to load brain '{brain_name}': {e}", exc_info=True)
            error_msg = f"Failed to load brain: {e}"
            app._set_error(error_msg)
            return error_msg
    
    def _on_unload_model() -> str:
        """
        Unload current brain model by clearing the persistent registry.
        
        Returns:
            Result message string
        """
        global _persistent_registry, _persistent_router
        
        try:
            logger.info("User action: Unloading current brain model")
            app._log_router.log("Unloading model", LogCategory.CHAT, "INFO")
            
            # Clear the persistent registry to unload models
            if _persistent_registry is not None:
                # Clear loaded brains
                brain_count = len(_persistent_registry.brains)
                _persistent_registry.brains.clear()
                # Also clear history on all loaded brains
                result = "✓ Model unloaded successfully. GPU memory freed."
                logger.info(f"Successfully unloaded {brain_count} brain(s) from memory")
                app._log_router.log(f"Model unloaded successfully ({brain_count} brain(s) freed)", LogCategory.CHAT, "INFO")
            else:
                result = "No model loaded or registry not initialized."
                logger.debug("No active model to unload - registry not initialized")
                app._log_router.log("No active model to unload", LogCategory.CHAT, "INFO")
            
            return result
        except Exception as e:
            logger.error(f"Failed to unload model: {e}", exc_info=True)
            error_msg = f"Failed to unload model: {e}"
            app._set_error(error_msg)
            return error_msg
    
    def _on_list_brains() -> list[str]:
        """Return available brains without invoking the CLI."""
        try:
            brains = list_available_brains(_brain_store_dir)
            if brains:
                return brains

            # Fallback candidates: process CWD default and project root parent
            alt_candidates: list[str] = []
            alt_candidates.append("artifacts/brains")

            try:
                alt_parent = Path(_brain_store_dir).resolve().parents[2]
                alt_candidates.append(str((alt_parent / "artifacts" / "brains").resolve()))
            except Exception:
                pass

            for alt in alt_candidates:
                if not alt or str(alt) == _brain_store_dir:
                    continue
                try:
                    alt_brains = list_available_brains(alt)
                except Exception:
                    continue
                if alt_brains:
                    logger.debug(
                        "Primary brain store empty; using alternate path %s with %d brain(s)",
                        alt,
                        len(alt_brains),
                    )
                    return alt_brains

            return brains
        except Exception as exc:
            logger.error(f"Failed to list brains via registry service: {exc}", exc_info=True)
            return []
    
    # Attach handlers to app
    app._on_chat_route_and_run = _on_chat_route_and_run
    app._on_load_brain = _on_load_brain
    app._on_unload_model = _on_unload_model
    app._on_list_brains = _on_list_brains

    # Opportunistically warm the router during startup so the first user prompt doesn't stall
    def _warm_router() -> None:
        global _router_warmup_started
        with _router_init_lock:
            if _router_warmup_started:
                return
            _router_warmup_started = True

        start = time.time()
        try:
            registry, _ = _ensure_persistent_router()
            brain_count = len(registry.brains) if getattr(registry, "brains", None) else 0
            logger.info("Chat router warm-up complete in %.3fs (%d cached brain(s))", time.time() - start, brain_count)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Chat router warm-up failed: %s", exc, exc_info=True)

    try:
        submit_background(
            "chat-router-warmup",
            _warm_router,
            pool=getattr(app, "_worker_pool", None),
        )
    except RuntimeError as exc:  # pragma: no cover - defensive logging
        logger.debug("Unable to schedule chat router warm-up: %s", exc, exc_info=True)
        _warm_router()

