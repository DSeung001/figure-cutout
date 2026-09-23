from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from figure_cutout.image_io import load_rgba

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPORT_FORMAT_VERSION = 2
PLAIN_BORDER_STD = 8.0


@dataclass(frozen=True, slots=True)
class Sample:
    """One evaluation image. `id` names benchmark outputs, masks and metadata."""

    id: str
    path: Path

# Representative difficulty tags from the ML roadmap evaluation set.
SAMPLE_TAG_SETS: list[list[str]] = [
    ["simple_background"],
    ["cluttered_background"],
    ["dark_background"],
    ["bright_background"],
    ["thin_hair"],
    ["sword"],
    ["spear"],
    ["wings"],
    ["display_base"],
    ["detached_accessory"],
    ["transparent_effect"],
    ["reflective_surface"],
    ["multi_object"],
    ["sword", "display_base"],
    ["wings", "cluttered_background"],
    ["thin_hair", "bright_background"],
    ["display_base", "detached_accessory"],
    ["transparent_effect", "reflective_surface"],
    ["spear", "multi_object"],
    ["sword", "wings", "display_base"],
]


HARD_TAGS = frozenset(
    {
        "thin_hair",
        "transparent_effect",
        "reflective_surface",
        "multi_object",
        "cluttered_background",
        "detached_accessory",
    }
)


def _difficulty_for_tags(tags: list[str]) -> str:
    if HARD_TAGS.intersection(tags):
        return "hard"
    if len(tags) >= 2:
        return "medium"
    return "easy"


def _draw_synthetic_figure(draw: ImageDraw.ImageDraw, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed)
    body_color = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220), 255)
    cx = width // 2 + rng.randint(-40, 40)
    cy = height // 2 + rng.randint(-30, 30)
    body_w = rng.randint(60, 120)
    body_h = rng.randint(120, 220)
    draw.ellipse(
        (cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2),
        fill=body_color,
    )
    # Thin structure (weapon / hair strand)
    tip = (cx + body_w // 2 + rng.randint(40, 90), cy - body_h // 2)
    draw.line((cx + body_w // 2, cy - body_h // 3, *tip), fill=(230, 230, 230, 255), width=3)
    # Display base
    base_top = cy + body_h // 2 - 10
    draw.rectangle(
        (cx - body_w // 2 - 10, base_top, cx + body_w // 2 + 10, base_top + 28),
        fill=(90, 90, 100, 255),
    )


def create_synthetic_image(path: Path, index: int, tags: list[str]) -> None:
    width, height = 512, 768
    rng = random.Random(index)
    if "dark_background" in tags:
        bg = (20, 20, 28, 255)
    elif "bright_background" in tags:
        bg = (240, 240, 245, 255)
    elif "cluttered_background" in tags:
        bg = (rng.randint(60, 180), rng.randint(60, 180), rng.randint(60, 180), 255)
    else:
        bg = (180, 190, 200, 255)

    image = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(image)

    if "cluttered_background" in tags or "multi_object" in tags:
        for _ in range(12):
            x0 = rng.randint(0, width - 40)
            y0 = rng.randint(0, height - 40)
            draw.rectangle(
                (x0, y0, x0 + rng.randint(20, 80), y0 + rng.randint(20, 80)),
                fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 180),
            )

    _draw_synthetic_figure(draw, width, height, seed=index)

    if "multi_object" in tags:
        # Second blob to simulate multi-object confusion cases.
        draw.ellipse((80, 120, 180, 280), fill=(200, 80, 80, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=92)


def prepare_eval_set(
    root: Path = Path("datasets/figure-v1"),
    count: int = 50,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    if count < 1:
        raise ValueError("count must be >= 1")
    if (root / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {root}. Use overwrite to regenerate.")

    images_dir = root / "images"
    metadata_dir = root / "metadata"
    masks_dir = root / "masks"
    splits_dir = root / "splits"
    for path in (images_dir, metadata_dir, masks_dir, splits_dir):
        path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    split_names: list[str] = []

    for index in range(1, count + 1):
        sample_id = f"sample-{index:03d}"
        tags = list(rng.choice(SAMPLE_TAG_SETS))
        image_name = f"{sample_id}.jpg"
        image_path = images_dir / image_name
        create_synthetic_image(image_path, index=index, tags=tags)

        meta = {
            "id": sample_id,
            "tags": tags,
            "difficulty": _difficulty_for_tags(tags),
            "source": "synthetic",
        }
        (metadata_dir / f"{sample_id}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        split_names.append(image_name)

    (splits_dir / "val.txt").write_text("\n".join(split_names) + "\n", encoding="utf-8")
    (splits_dir / "train.txt").write_text("", encoding="utf-8")
    (splits_dir / "test.txt").write_text("", encoding="utf-8")

    manifest = {
        "dataset": root.name,
        "image_count": count,
        "seed": seed,
        "split": "val",
        "note": (
            "Synthetic bootstrap set for harness validation. Tags are coverage labels; only "
            "background and multi_object tags are rendered. Replace with real figures."
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def dataset_name(dataset: Path) -> str:
    manifest = dataset / "manifest.json"
    if manifest.is_file():
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("dataset", dataset.name))
    return dataset.name


def resolve_image_dir(dataset: Path) -> Path:
    images = dataset / "images"
    if images.is_dir():
        return images
    if dataset.is_dir():
        return dataset
    raise FileNotFoundError(f"Dataset path not found: {dataset}")


def _read_split(dataset: Path, split: str) -> list[str] | None:
    split_file = dataset / "splits" / f"{split}.txt"
    if not split_file.is_file():
        return None
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def is_export_dataset(dataset: Path) -> bool:
    return (dataset / "index.json").is_file()


def read_export_index(root: Path) -> list[dict[str, Any]]:
    """index.json of an image export; only formatVersion 2 is accepted."""
    if not (root / "index.json").is_file():
        raise FileNotFoundError(f"No index.json under {root}: export is missing or incomplete.")
    meta_path = root / "export.json"
    version = (
        json.loads(meta_path.read_text(encoding="utf-8")).get("formatVersion")
        if meta_path.is_file()
        else 1
    )
    if version != EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Export formatVersion {version} at {root}; expected {EXPORT_FORMAT_VERSION}. "
            "Re-export the images (see docs/image-export-format.md)."
        )
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def _export_paths(root: Path) -> dict[str, Path]:
    return {
        record["key"]: root / record["path"]
        for entry in read_export_index(root)
        for record in entry["files"]
        if record["status"] == "ok"
    }


def collect_samples(dataset: Path, split: str = "val") -> list[Sample]:
    """Return samples of a split. A split file must resolve completely to keep runs comparable."""
    names = _read_split(dataset, split)

    if is_export_dataset(dataset):
        if names is None:
            raise FileNotFoundError(
                f"No splits/{split}.txt under {dataset}. Run: figure-cutout init-dataset"
            )
        paths = _export_paths(dataset)
        missing = [key for key in names if key not in paths or not paths[key].is_file()]
        if missing:
            raise FileNotFoundError(f"Split '{split}' lists missing samples: {missing[:5]}")
        samples = [Sample(key, paths[key]) for key in names]
    else:
        image_dir = resolve_image_dir(dataset)
        if names is not None:
            missing = [name for name in names if not (image_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Split '{split}' lists missing images: {missing[:5]}")
            image_paths = sorted(image_dir / name for name in names)
        else:
            image_paths = sorted(
                path for path in image_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        samples = [Sample(path.stem, path) for path in image_paths]

    if not samples:
        raise FileNotFoundError(f"No images for split '{split}' under {dataset}")
    return samples


def _auto_tags(image: Image.Image) -> list[str]:
    """Cheap background hints for product shots; not a replacement for manual tags."""
    rgba = np.asarray(image.convert("RGBA"))
    tags: list[str] = []
    if int(rgba[..., 3].min()) < 255:
        tags.append("has_alpha")
    border = np.concatenate(
        [rgba[0, :, :3], rgba[-1, :, :3], rgba[:, 0, :3], rgba[:, -1, :3]]
    ).astype(np.float32)
    if float(border.std(axis=0).max()) < PLAIN_BORDER_STD:
        tags.append("plain_border")
    return tags


def item_category(entry: dict[str, Any]) -> str:
    """User-edited `category`; the storage category (id prefix) when it is empty."""
    return entry.get("category") or entry["id"].partition(":")[0]


def _judge_file(
    entry: dict[str, Any],
    record: dict[str, Any],
    root: Path,
    *,
    categories: set[str],
    roles: set[str],
    seen_hashes: dict[str, str],
    min_side: int,
    max_aspect: float,
) -> tuple[str | None, Image.Image | None]:
    """Reject reason (None when accepted) and the loaded image for accepted files."""
    if record["status"] != "ok":
        return "download_error", None
    if item_category(entry) not in categories:
        return "category_excluded", None
    if record["role"] not in roles:
        return "role_excluded", None
    if record["sha256"] in seen_hashes:
        return "duplicate", None
    path = root / record["path"]
    if not path.is_file():
        return "missing_file", None
    try:
        image = load_rgba(path)
    except Exception:  # noqa: BLE001 - truncated/corrupt files are rejected, not fatal
        return "decode_error", None
    seen_hashes[record["sha256"]] = record["key"]
    short, long = sorted(image.size)
    if short < min_side:
        return "too_small", None
    if long / short > max_aspect:
        return "extreme_aspect", None
    return None, image


def init_export_dataset(
    root: Path,
    *,
    categories: Iterable[str] = ("FIGURE",),
    roles: Iterable[str] = ("main", "detail"),
    min_side: int = 256,
    max_aspect: float = 3.0,
) -> dict[str, Any]:
    """Index an image export in place: metadata stubs, val split and manifest beside it.

    Export files are never modified. Existing metadata and splits are never rewritten, so
    manual tags survive and earlier benchmark runs stay comparable.
    """
    index = read_export_index(root)
    category_set, role_set = set(categories), set(roles)
    for path in (root / "masks", root / "metadata", root / "splits"):
        path.mkdir(parents=True, exist_ok=True)

    accepted: list[str] = []
    rejected: dict[str, list[str]] = {}
    created_metadata: list[str] = []
    seen_hashes: dict[str, str] = {}
    for entry in index:
        if entry.get("error") == "not_in_library":
            rejected.setdefault("not_in_library", []).append(entry["id"])
            continue
        for record in entry["files"]:
            reason, image = _judge_file(
                entry,
                record,
                root,
                categories=category_set,
                roles=role_set,
                seen_hashes=seen_hashes,
                min_side=min_side,
                max_aspect=max_aspect,
            )
            if reason is not None:
                rejected.setdefault(reason, []).append(record["key"])
                continue
            assert image is not None
            key = record["key"]
            accepted.append(key)
            meta_path = root / "metadata" / f"{key}.json"
            if meta_path.exists():
                continue
            meta = {
                "id": key,
                "tags": [],
                "auto_tags": _auto_tags(image),
                "difficulty": "unknown",
                "source": "export",
                "item_id": entry["id"],
                "role": record["role"],
                "category": item_category(entry),
                "title": entry["title"],
                "title_ko": entry["titleKo"],
                "shop": entry["shop"],
                "product_url": entry["url"],
                "image_url": record["url"],
                "width": image.width,
                "height": image.height,
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            created_metadata.append(key)

    val_path = root / "splits" / "val.txt"
    split_created = False
    unlisted: list[str] = []
    if val_path.exists():
        listed = set(_read_split(root, "val") or [])
        unlisted = [key for key in accepted if key not in listed]
    elif accepted:  # an empty split would be frozen forever; wait for accepted samples
        val_path.write_text("".join(f"{key}\n" for key in accepted), encoding="utf-8")
        split_created = True
    for split in ("train", "test"):
        (root / "splits" / f"{split}.txt").touch()

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "dataset": root.name,
            "source": "export",
            "export_format_version": EXPORT_FORMAT_VERSION,
            "split": "val",
            "filters": {
                "categories": sorted(category_set),
                "roles": sorted(role_set),
                "min_side": min_side,
                "max_aspect": max_aspect,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "dataset": str(root),
        "accepted_count": len(accepted),
        "rejected_count": {reason: len(keys) for reason, keys in sorted(rejected.items())},
        "rejected": dict(sorted(rejected.items())),
        "metadata_created": len(created_metadata),
        "val_split_created": split_created,
        "unlisted_in_val": unlisted,
    }
