from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, source: Path) -> Path:
        ...

    def resolve(self, key: str) -> Path:
        ...
