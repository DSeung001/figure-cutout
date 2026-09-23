from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from figure_cutout.dataset import collect_samples, init_export_dataset
from figure_cutout.export_format import read_export_index


@pytest.fixture
def contract(tmp_path, monkeypatch):
    monkeypatch.setenv("FIGURE_PROJECT_DIR", str(tmp_path))
    fixture = json.loads((Path(__file__).parent / "fixtures/image-export-v4.json").read_text())
    root = tmp_path / "exports/new"
    root.mkdir(parents=True)
    record = fixture["entry"]["files"][0]
    image = tmp_path / "images" / record["path"]
    image.parent.mkdir(parents=True)
    image.write_bytes(base64.b64decode(fixture["image_base64"]))
    (root / "export.json").write_text(json.dumps({"formatVersion": 4}))
    (root / "index.json").write_text(json.dumps([fixture["entry"]]))
    return root, fixture["entry"], image


def test_shared_contract_and_missing_frozen_sample(contract):
    root, _entry, image = contract
    original = (root / "index.json").read_bytes()
    assert init_export_dataset(root)["accepted_count"] == 1
    assert collect_samples(root)[0].path == image
    assert not (root / "masks").exists()
    assert (root / "index.json").read_bytes() == original
    assert json.loads((root / "manifest.json").read_text())["export_format_version"] == 4
    image.unlink()
    assert init_export_dataset(root)["rejected_count"] == {"missing_file": 1}
    with pytest.raises(FileNotFoundError, match="missing samples"):
        collect_samples(root)


@pytest.mark.parametrize("status", ["ok", "skipped"])
def test_v4_export_local_paths(contract, status):
    root, entry, image = contract
    record = entry["files"][0]
    image.rename(root / "original.png")
    record.update(path="original.png", storage="export", status=status)
    (root / "index.json").write_text(json.dumps([entry]))
    assert init_export_dataset(root)["accepted_count"] == 1
    assert collect_samples(root)[0].id == record["key"]
    assert json.loads((root / "manifest.json").read_text())["export_format_version"] == 4


@pytest.mark.parametrize("version", [None, 1, 2, 3, 99])
def test_non_v4_exports_are_rejected(contract, version):
    root, _entry, _image = contract
    if version is None:
        (root / "export.json").unlink()
    else:
        (root / "export.json").write_text(json.dumps({"formatVersion": version}))
    with pytest.raises(ValueError, match="expected 4"):
        init_export_dataset(root)
    assert not (root / "metadata").exists()


@pytest.mark.parametrize("path", ["../../outside.png", "/tmp/outside.png", ""])
def test_reject_escaping_references(contract, path):
    root, entry, _image = contract
    entry["files"][0]["path"] = path
    (root / "index.json").write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="path"):
        init_export_dataset(root)


def test_reject_symlink_escape(contract, tmp_path):
    root, _entry, image = contract
    outside = tmp_path / "outside.png"
    image.rename(outside)
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        init_export_dataset(root)


def test_empty_export_creates_no_placeholder_directories(contract):
    root, entry, _image = contract
    entry["files"] = []
    (root / "index.json").write_text(json.dumps([entry]))
    assert init_export_dataset(root)["accepted_count"] == 0
    assert not any(p.is_dir() for p in root.iterdir())


def test_reject_unknown_version_and_storage(contract):
    root, entry, _image = contract
    entry["files"][0]["storage"] = "unknown"
    (root / "index.json").write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="storage"):
        init_export_dataset(root)
    (root / "export.json").write_text('{"formatVersion": 99}')
    with pytest.raises(ValueError, match="formatVersion"):
        read_export_index(root)
