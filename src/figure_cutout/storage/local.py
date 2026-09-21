from __future__ import annotations

import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError("Storage key escapes the configured root.")
        return candidate

    def put(self, key: str, source: Path) -> Path:
        destination = self.resolve(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
