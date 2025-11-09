"""
Async CLI Bridge

Provides async versions of CLI operations for non-blocking subprocess execution.

Author: AI-OS Development Team
Date: October 12, 2025
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, List, Optional


class AsyncCLIBridge:
    """Async version of CLI operations."""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize async CLI bridge.
        
        Args:
            project_root: Project root directory (auto-detected if None)
        """
        self._project_root = project_root or self._detect_project_root()
    
    @staticmethod
    def _detect_project_root() -> str:
        """Detect project root by finding pyproject.toml."""
        import os
        cur = os.path.abspath(os.getcwd())
        for _ in range(8):
            if os.path.exists(os.path.join(cur, "pyproject.toml")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return os.path.abspath(os.getcwd())
    
    async def run_cli(
        self,
        args: List[str],
        timeout: Optional[float] = 30.0
    ) -> str:
        """
        Run CLI command asynchronously.
        
        Args:
            args: CLI arguments (e.g., ["brains", "stats"])
            timeout: Command timeout in seconds
        
        Returns:
            Command output (stdout)
        
        Raises:
            asyncio.TimeoutError: If command times out
            RuntimeError: If command fails
        
        Example:
            bridge = AsyncCLIBridge()
            result = await bridge.run_cli(["brains", "list"])
        """
        cmd = [sys.executable, "-m", "aios.cli.aios"] + args
        
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._project_root
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            if proc.returncode != 0:
                error = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"CLI command failed: {error}")
            
            return stdout.decode() if stdout else ""
            
        except asyncio.TimeoutError:
            # Kill the process if it times out
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise asyncio.TimeoutError(f"CLI command timed out after {timeout}s")
    
    async def run_cli_json(
        self,
        args: List[str],
        timeout: Optional[float] = 30.0
    ) -> Any:
        """
        Run CLI command and parse JSON output.
        
        Args:
            args: CLI arguments
            timeout: Command timeout in seconds
        
        Returns:
            Parsed JSON result
        
        Example:
            data = await bridge.run_cli_json(["brains", "stats"])
            print(data["brains"])
        """
        output = await self.run_cli(args, timeout)
        return self._parse_cli_json(output)
    
    @staticmethod
    def _parse_cli_json(text: str) -> Any:
        """
        Parse CLI JSON output, handling [cli] headers.
        
        Args:
            text: Raw CLI output
        
        Returns:
            Parsed JSON object
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Find first '{' after header
        try:
            idx = text.find('{')
            if idx != -1:
                return json.loads(text[idx:])
        except json.JSONDecodeError:
            pass
        
        # Try ast.literal_eval as fallback
        try:
            import ast
            idx = text.find('{')
            if idx != -1:
                return ast.literal_eval(text[idx:])
        except Exception:
            pass
        
        return {}
    
    async def run_cli_batch(
        self,
        commands: List[List[str]],
        timeout: Optional[float] = 30.0
    ) -> List[str]:
        """
        Run multiple CLI commands in parallel.
        
        Args:
            commands: List of command argument lists
            timeout: Per-command timeout in seconds
        
        Returns:
            List of command outputs (in same order as input)
        
        Example:
            results = await bridge.run_cli_batch([
                ["brains", "stats"],
                ["datasets-stats"],
                ["torch-info"]
            ])
        """
        tasks = [self.run_cli(cmd, timeout) for cmd in commands]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def run_cli_batch_json(
        self,
        commands: List[List[str]],
        timeout: Optional[float] = 30.0
    ) -> List[Any]:
        """
        Run multiple CLI commands in parallel and parse JSON.
        
        Args:
            commands: List of command argument lists
            timeout: Per-command timeout in seconds
        
        Returns:
            List of parsed JSON results
        
        Example:
            brain_stats, ds_stats, torch = await bridge.run_cli_batch_json([
                ["brains", "stats"],
                ["datasets-stats"],
                ["torch-info"]
            ])
        """
        tasks = [self.run_cli_json(cmd, timeout) for cmd in commands]
        return await asyncio.gather(*tasks, return_exceptions=False)
