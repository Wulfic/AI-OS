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

if TYPE_CHECKING:
    pass

from ..services import LogCategory

logger = logging.getLogger(__name__)

# Global persistent registry and router to keep models loaded across chat sessions
_persistent_registry: Optional[Any] = None
_persistent_router: Optional[Any] = None

# Global MCP tool executor for handling tool calls
_tool_executor: Optional[Any] = None


def setup_chat_operations(app: Any) -> None:
    """
    Set up chat operation handlers.
    
    Args:
        app: AiosTkApp instance with _run_cli available
    """
    global _persistent_registry, _persistent_router, _tool_executor
    
    def _ensure_persistent_router() -> tuple[Any, Any]:
        """Ensure persistent registry and router are initialized.
        
        Returns:
            Tuple of (registry, router)
        """
        global _persistent_registry, _persistent_router, _tool_executor
        
        # Initialize MCP tool executor if not done yet
        if _tool_executor is None:
            try:
                from aios.core.mcp import ToolExecutor
                _tool_executor = ToolExecutor()
                if _tool_executor.enabled:
                    logger.info(f"MCP tool executor initialized with {len(_tool_executor.mcp_client.available_tools)} tools")
                else:
                    logger.info("MCP tool executor initialized but no tools available")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP tool executor: {e}")
                _tool_executor = None
        
        if _persistent_registry is None or _persistent_router is None:
            # Load config and initialize persistent registry/router
            from aios.cli.utils import load_config
            from aios.core.brains import BrainRegistry, Router
            
            cfg = load_config()
            brains_cfg = (cfg.get("brains") or {}) if isinstance(cfg, dict) else {}
            
            # Create persistent registry
            storage_limit_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0) or None
            _persistent_registry = BrainRegistry(total_storage_limit_mb=storage_limit_mb)
            _persistent_registry.store_dir = str(brains_cfg.get("store_dir", "artifacts/brains"))
            
            # Load persisted pins/masters
            try:
                _persistent_registry.load_pinned()
                _persistent_registry.load_masters()
            except Exception:
                pass
            
            # Build router config
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
            
            # Create persistent router
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
            # Reload masters/pins in case they changed (from "Load Brain" button)
            try:
                _persistent_registry.load_pinned()
                _persistent_registry.load_masters()
            except Exception:
                pass
        
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
        
        # Run in background thread to prevent UI freezing
        import threading
        thread = threading.Thread(target=_background_work, daemon=True)
        thread.start()
    
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
            app._log_router.log(f"Loading brain: {brain_name}", LogCategory.CHAT)
            
            # Ensure persistent registry exists
            registry, router = _ensure_persistent_router()
            
            # Load the brain using registry.get() which auto-loads from disk
            try:
                brain = registry.get(brain_name)
                if brain:
                    # Mark as master so router uses it for chat
                    registry.mark_master(brain_name)
                    
                    app._log_router.log(f"Brain {brain_name} loaded and set as master for chat", LogCategory.CHAT)
                    return f"✓ Brain '{brain_name}' loaded successfully and set as active for chat."
                else:
                    error_msg = f"Failed to load brain '{brain_name}' - brain not found or failed to load. Check that the brain exists in artifacts/brains/actv1/{brain_name}/"
                    app._log_router.log(error_msg, LogCategory.CHAT)
                    return error_msg
            except Exception as load_error:
                import traceback
                error_details = traceback.format_exc()
                error_msg = f"Failed to load brain '{brain_name}': {load_error}\n{error_details}"
                logger.error(error_msg)
                app._log_router.log(error_msg, LogCategory.CHAT)
                return f"Failed to load brain '{brain_name}': {load_error}"
                
        except Exception as e:
            logger.error(f"Failed to load brain: {e}")
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
            app._log_router.log("Unloading model", LogCategory.CHAT)
            
            # Clear the persistent registry to unload models
            if _persistent_registry is not None:
                # Clear loaded brains
                _persistent_registry.brains.clear()
                # Also clear history on all loaded brains
                result = "✓ Model unloaded successfully. GPU memory freed."
                app._log_router.log("Model unloaded successfully", LogCategory.CHAT)
            else:
                result = "No model loaded or registry not initialized."
                app._log_router.log("No active model to unload", LogCategory.CHAT)
            
            return result
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            error_msg = f"Failed to unload model: {e}"
            app._set_error(error_msg)
            return error_msg
    
    def _on_list_brains() -> list[str]:
        """
        List available brains.
        
        Returns:
            List of brain names (filtered to exclude temporary brains)
        """
        try:
            import os
            import re
            
            # Use absolute path to artifacts/brains directory
            store_dir = os.path.join(os.getcwd(), "artifacts", "brains")
            result = app._run_cli(["brains", "list-brains", "--store-dir", store_dir])
            data = app._parse_cli_dict(result or "{}")
            
            # Handle {"brains": [list]} format
            brain_names = []
            if isinstance(data, dict) and "brains" in data:
                brains_data = data["brains"]
                if isinstance(brains_data, list):
                    brain_names = [str(b) for b in brains_data]
                elif isinstance(brains_data, dict):
                    brain_names = list(brains_data.keys())
            elif isinstance(data, list):
                brain_names = [str(b) for b in data]
            elif isinstance(data, dict):
                brain_names = list(data.keys())
            
            # Filter out temporary/internal brains
            def _is_temporary(name: str) -> bool:
                if name.startswith('_'):
                    return True
                # Check for router-generated temporary brains: brain-text-de5aae40, brain-image-abc123, etc.
                if re.match(r'^brain-[a-z]+-[0-9a-f]{8}$', name):
                    return True
                return False
            
            return [name for name in brain_names if not _is_temporary(name)]
        except Exception as e:
            logger.error(f"Failed to list brains: {e}")
            return []
    
    # Attach handlers to app
    app._on_chat_route_and_run = _on_chat_route_and_run
    app._on_load_brain = _on_load_brain
    app._on_unload_model = _on_unload_model
    app._on_list_brains = _on_list_brains

