"""Centralized logging router for GUI components.

Routes log messages to appropriate tabs based on category.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Any


class LogCategory(Enum):
    """Categories for log messages."""
    DATASET = "dataset"
    CHAT = "chat"
    TRAINING = "training"
    DEBUG = "debug"
    ERROR = "error"
    SYSTEM = "system"
    THOUGHT = "thought"


class LogRouter:
    """Routes log messages to appropriate output handlers based on category."""
    
    def __init__(self):
        self._handlers: dict[LogCategory, list[Callable[[str, str | None], None]]] = {
            cat: [] for cat in LogCategory
        }
        self._auto_detect = True
        
    def register_handler(self, category: LogCategory, handler: Callable[[str, str | None], None]) -> None:
        """Register a handler for a specific log category.
        
        Args:
            category: The log category
            handler: Callable that accepts (message: str, level: str | None)
        """
        if category not in self._handlers:
            self._handlers[category] = []
        self._handlers[category].append(handler)
    
    def unregister_handler(self, category: LogCategory, handler: Callable[[str, str | None], None]) -> None:
        """Unregister a handler for a specific log category."""
        if category in self._handlers and handler in self._handlers[category]:
            self._handlers[category].remove(handler)
    
    def log(self, message: str, category: LogCategory | None = None, level: str | None = None) -> None:
        """Log a message to appropriate handlers.
        
        Args:
            message: The message to log
            category: Optional explicit category. If None, will auto-detect.
            level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if not message:
            return
            
        # Auto-detect category if not specified
        if category is None and self._auto_detect:
            category = self._detect_category(message)
        
        # Default to DEBUG if still None
        if category is None:
            category = LogCategory.DEBUG
        
        # Route to registered handlers
        handlers = self._handlers.get(category, [])
        for handler in handlers:
            try:
                # Call handler with message and level
                handler(message, level)
            except Exception:
                # Fail silently to avoid cascading errors
                pass
    
    def _detect_category(self, message: str) -> LogCategory:
        """Auto-detect message category based on content."""
        msg_lower = message.lower()
        
        # Check for explicit tags first (e.g., "[datasets]", "[chat]")
        tag_match = re.match(r'^\[([^\]]+)\]', message)
        if tag_match:
            tag = tag_match.group(1).lower()
            if 'dataset' in tag or 'download' in tag:
                return LogCategory.DATASET
            elif 'chat' in tag:
                return LogCategory.CHAT
            elif 'train' in tag or 'hrm' in tag or 'optimization' in tag:
                return LogCategory.TRAINING
            elif 'error' in tag or 'exception' in tag:
                return LogCategory.ERROR
            elif 'system' in tag or 'status' in tag:
                return LogCategory.SYSTEM
            elif 'thought' in tag or 'thinking' in tag:
                return LogCategory.THOUGHT
        
        # Check for error indicators
        if any(x in msg_lower for x in ['error', 'exception', 'traceback', 'failed', 'failure']):
            return LogCategory.ERROR
        
        # Check for dataset-related keywords
        if any(x in msg_lower for x in ['dataset', 'download', 'downloading', 'fetching', 'url', 'sha256']):
            return LogCategory.DATASET
        
        # Check for chat-related keywords
        if any(x in msg_lower for x in ['chat', 'message', 'response', 'conversation']):
            return LogCategory.CHAT
        
        # Check for training-related keywords
        if any(x in msg_lower for x in ['training', 'epoch', 'loss', 'accuracy', 'batch', 'optimizer', 'gradient']):
            return LogCategory.TRAINING
        
        # Check for thought process keywords
        if any(x in msg_lower for x in ['thinking', 'reasoning', 'analyzing', 'considering', 'evaluating']):
            return LogCategory.THOUGHT
        
        # Check for system-related keywords
        if any(x in msg_lower for x in ['system', 'status', 'memory', 'cpu', 'gpu', 'device']):
            return LogCategory.SYSTEM
        
        # Default to debug
        return LogCategory.DEBUG
    
    def set_auto_detect(self, enabled: bool) -> None:
        """Enable or disable automatic category detection."""
        self._auto_detect = enabled
