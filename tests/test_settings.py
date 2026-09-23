from __future__ import annotations

import json
from pathlib import Path

import pytest

from figure_cutout.cli import main
from figure_cutout.settings import export_dir, figure_project_dir, latest_export


def _export(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "export.json").write_text(json.dumps({"formatVersion": 4}), encoding="utf-8")
    (root / "index.json").write_text("[]", encoding="utf-8")
    return root


def test_figure_project_dir_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIGURE_PROJECT_DIR", raising=False)
    assert figure_project_dir() == Path.home() / "figure_project"
    monkeypatch.setenv("FIGURE_PROJECT_DIR", "  ")
    assert figure_project_dir() == Path.home() / "figure_project"


def test_latest_export_picks_newest_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_PROJECT_DIR", str(tmp_path))
    assert export_dir() == tmp_path / "exports"
    with pytest.raises(FileNotFoundError, match="FIGURE_PROJECT_DIR"):
        latest_export()

    _export(tmp_path / "exports" / "20260101T000000000000Z")
    newest = _export(tmp_path / "exports" / "20260201T000000000000Z")
    (tmp_path / "exports" / "20260301T000000000000Z").mkdir()  # incomplete: no index.json
    (tmp_path / "exports" / "20260401T000000000000Z.zip").write_bytes(b"PK")
    assert latest_export() == newest


def test_cli_defaults_to_latest_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FIGURE_PROJECT_DIR", str(tmp_path))
    newest = _export(tmp_path / "exports" / "20260201T000000000000Z")
    assert main(["init-dataset"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset"] == str(newest)
    assert (newest / "manifest.json").is_file()
