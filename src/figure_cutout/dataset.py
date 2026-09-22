from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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


def collect_images(dataset: Path, split: str = "val") -> list[Path]:
    """Return images of a split. A split file must resolve completely to keep runs comparable."""
    image_dir = resolve_image_dir(dataset)
    split_file = dataset / "splits" / f"{split}.txt"

    if split_file.is_file():
        names = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        missing = [name for name in names if not (image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Split '{split}' lists missing images: {missing[:5]}")
        images = sorted(image_dir / name for name in names)
    else:
        images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not images:
        raise FileNotFoundError(f"No images for split '{split}' under {dataset}")
    return images
