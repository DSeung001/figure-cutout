from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from figure_cutout.benchmark import benchmark_pipeline
from figure_cutout.compare import compare_runs
from figure_cutout.dataset import collect_samples, init_export_dataset
from figure_cutout.image_io import load_rgba
from figure_cutout.ml.factory import build_pipeline


def _encode(image: Image.Image, fmt: str, **params: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **params)
    return buffer.getvalue()


def _noise(size: tuple[int, int], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


class ExportBuilder:
    """Writes a formatVersion 2 export (see docs/image-export-format.md)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index: list[dict[str, Any]] = []

    def item(self, item_id: str, category: str, files: list[tuple[str, bytes | None]]) -> None:
        folder = item_id.replace(":", "_")
        records = []
        for position, (role, body) in enumerate(files):
            record: dict[str, Any] = {
                "key": f"{folder}-{position:02d}",
                "role": role,
                "url": f"https://cdn.example.com/{folder}/{position}",
                "path": None,
                "status": "error",
                "error": "404",
                "format": None,
                "sha256": None,
                "bytes": None,
            }
            if body is not None:
                fmt = "png" if body.startswith(b"\x89PNG") else "jpeg"
                path = f"items/{folder}/{position:02d}_{role}.{'png' if fmt == 'png' else 'jpg'}"
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).write_bytes(body)
                record.update(
                    path=path,
                    status="ok",
                    error=None,
                    format=fmt,
                    sha256=hashlib.sha256(body).hexdigest(),
                    bytes=len(body),
                )
            records.append(record)
        self.index.append(
            {
                "id": item_id,
                "title": f"title {item_id}",
                "titleKo": "",
                "url": f"https://shop.example.com/{folder}",
                "shop": "shop",
                "category": category,
                "source": "source",
                "imageUrl": "",
                "detailImageUrls": [],
                "files": records,
            }
        )

    def missing(self, item_id: str) -> None:
        self.index.append({"id": item_id, "error": "not_in_library", "files": []})

    def write(self, version: int | None = 2) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        if version is not None:
            (self.root / "export.json").write_text(
                json.dumps({"formatVersion": version}), encoding="utf-8"
            )
        (self.root / "index.json").write_text(json.dumps(self.index), encoding="utf-8")
        return self.root


@pytest.fixture
def export(tmp_path: Path) -> Path:
    main_a = _encode(_noise((400, 600), 1), "JPEG")
    transparent = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    transparent.paste((200, 50, 50, 255), (100, 100, 200, 200))

    builder = ExportBuilder(tmp_path / "figure-shop-v1")
    builder.item(
        "FIGURE:a",
        "FIGURE",
        [
            ("main", main_a),
            ("detail", None),
            ("detail", _encode(_noise((300, 1200), 2), "PNG")),
            ("detail", _encode(_noise((100, 100), 3), "JPEG")),
        ],
    )
    builder.item(
        "FIGURE:b",
        "FIGURE",
        [("main", main_a), ("detail", _encode(transparent, "PNG"))],
    )
    builder.item("GOODS:c", "GOODS", [("main", _encode(_noise((400, 400), 4), "JPEG"))])
    # Stored under GOODS but re-categorized by the user: the `category` field wins.
    builder.item("GOODS:d", "FIGURE", [("main", _encode(_noise((400, 400), 5), "JPEG"))])
    # Empty `category` falls back to the storage category.
    builder.item("FIGURE:e", "", [("main", _encode(_noise((400, 400), 6), "JPEG"))])
    builder.missing("FIGURE:gone")
    return builder.write()


def test_init_export_dataset_judges_files(export: Path) -> None:
    summary = init_export_dataset(export)
    assert summary["accepted_count"] == 4
    assert summary["rejected"] == {
        "category_excluded": ["GOODS_c-00"],
        "download_error": ["FIGURE_a-01"],
        "duplicate": ["FIGURE_b-00"],
        "extreme_aspect": ["FIGURE_a-02"],
        "not_in_library": ["FIGURE:gone"],
        "too_small": ["FIGURE_a-03"],
    }
    assert (export / "splits" / "val.txt").read_text(encoding="utf-8").split() == [
        "FIGURE_a-00",
        "FIGURE_b-01",
        "GOODS_d-00",
        "FIGURE_e-00",
    ]
    meta = json.loads((export / "metadata" / "FIGURE_b-01.json").read_text(encoding="utf-8"))
    assert meta["item_id"] == "FIGURE:b" and meta["role"] == "detail"
    assert meta["auto_tags"] == ["has_alpha", "plain_border"]
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "figure-shop-v1"
    assert manifest["filters"]["categories"] == ["FIGURE"]


def test_init_export_dataset_filters_roles(export: Path) -> None:
    summary = init_export_dataset(export, roles=["main"])
    assert summary["rejected"]["role_excluded"] == ["FIGURE_a-02", "FIGURE_a-03", "FIGURE_b-01"]
    assert summary["accepted_count"] == 3


def test_init_export_dataset_skips_empty_split(export: Path) -> None:
    summary = init_export_dataset(export, categories=["NONE"])
    assert summary["accepted_count"] == 0 and summary["val_split_created"] is False
    assert not (export / "splits" / "val.txt").exists()
    assert init_export_dataset(export)["val_split_created"] is True


def test_init_export_dataset_is_idempotent(export: Path) -> None:
    init_export_dataset(export)
    meta_path = export / "metadata" / "FIGURE_a-00.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tags"] = ["display_base"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    val = export / "splits" / "val.txt"
    val.write_text("FIGURE_a-00\n", encoding="utf-8")

    again = init_export_dataset(export)
    assert again["val_split_created"] is False
    assert again["unlisted_in_val"] == ["FIGURE_b-01", "GOODS_d-00", "FIGURE_e-00"]
    assert again["metadata_created"] == 0
    assert json.loads(meta_path.read_text(encoding="utf-8"))["tags"] == ["display_base"]
    assert val.read_text(encoding="utf-8") == "FIGURE_a-00\n"


def test_export_v1_is_rejected(tmp_path: Path) -> None:
    root = ExportBuilder(tmp_path / "old").write(version=None)
    with pytest.raises(ValueError, match="Re-export"):
        init_export_dataset(root)


def test_incomplete_export_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="index.json"):
        init_export_dataset(tmp_path)


def test_collect_samples_resolves_export_keys(export: Path) -> None:
    init_export_dataset(export)
    samples = collect_samples(export)
    assert [s.id for s in samples] == ["FIGURE_a-00", "FIGURE_b-01", "GOODS_d-00", "FIGURE_e-00"]
    assert samples[0].path == export / "items" / "FIGURE_a" / "00_main.jpg"

    (export / "splits" / "val.txt").write_text("FIGURE_a-01\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="FIGURE_a-01"):
        collect_samples(export)


def test_benchmark_and_compare_on_export(
    export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    init_export_dataset(export)
    report = benchmark_pipeline("placeholder", export)
    assert (report["image_count"], report["success_count"]) == (4, 4)
    # Two items share the file name 00_main; outputs must still be distinct.
    assert sorted(p.name for p in Path(report["result_dir"]).iterdir()) == [
        "000000-FIGURE_a-00.png",
        "000001-FIGURE_b-01.png",
        "000002-GOODS_d-00.png",
        "000003-FIGURE_e-00.png",
    ]
    summary = compare_runs(export, ["placeholder"])
    assert [row["sample_id"] for row in summary["samples"]] == [
        "FIGURE_a-00",
        "FIGURE_b-01",
        "GOODS_d-00",
        "FIGURE_e-00",
    ]


def test_load_rgba_applies_exif_orientation(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise on display
    path = tmp_path / "rotated.jpg"
    Image.new("RGB", (40, 20)).save(path, exif=exif)
    assert load_rgba(path).size == (20, 40)


def test_pipeline_keeps_input_transparency(tmp_path: Path) -> None:
    source = tmp_path / "cutout.png"
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 5, 10))
    image.save(source)
    result = build_pipeline("placeholder").run(source, tmp_path / "out.png")
    alpha = np.asarray(Image.open(result.output).split()[-1])
    assert alpha[:, :5].min() == 255 and alpha[:, 5:].max() == 0
