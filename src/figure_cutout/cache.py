from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

CHUNK_SIZE = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class ResultCache:
    """Processed results keyed by pipeline id, pipeline fingerprint and input content hash.

    An entry directory exists only once it is complete: entries are written to a temporary
    directory and renamed into place, so interrupted runs leave no partial entry.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_dir(self, pipeline_id: str, fingerprint: str) -> Path:
        return self.root / pipeline_id / fingerprint

    def entry(self, pipeline_id: str, fingerprint: str, sha256: str) -> Path:
        return self.run_dir(pipeline_id, fingerprint) / sha256

    def lookup(self, pipeline_id: str, fingerprint: str, sha256: str) -> Path | None:
        path = self.entry(pipeline_id, fingerprint, sha256)
        return path if path.is_dir() else None

    def store(
        self,
        pipeline_id: str,
        fingerprint: str,
        sha256: str,
        write: Callable[[Path], None],
    ) -> Path:
        final = self.entry(pipeline_id, fingerprint, sha256)
        staging = final.with_name(f"{final.name}.tmp-{uuid.uuid4().hex[:8]}")
        staging.mkdir(parents=True)
        try:
            write(staging)
            if final.is_dir():  # stored concurrently by another run
                shutil.rmtree(staging)
            else:
                staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return final
