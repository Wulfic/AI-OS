from __future__ import annotations
import platform
from dataclasses import dataclass


@dataclass
class SystemInfo:
    os: str
    release: str
    python: str


def get_system_info() -> SystemInfo:
    return SystemInfo(
        os=platform.system(),
        release=platform.release(),
        python=platform.python_version(),
    )
