from pathlib import Path

import pytest

from figure_cutout.storage.local import LocalStorage


def test_local_storage_rejects_parent_escape(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.resolve("../outside.txt")
