"""Brain protocol definition for AI-OS brain interface."""

from typing import Any, Dict, List, Protocol


class Brain(Protocol):
    """Protocol for a sub-brain. Implement minimal run() and size() for storage checks."""

    name: str
    modalities: List[str]

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def size_bytes(self) -> int:
        ...
