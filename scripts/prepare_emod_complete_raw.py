"""Build a complete EMOd raw-image folder for saliency runs.

The local EMOd folder is split into author-collected images and IAPS images.
Three IAPS raw files are missing from EMOd but are present in the Kim IAPS
copy. This script creates a derived folder with one image for every EMOd
ground-truth fixation map without modifying the source data folders.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parents[0] / "data"
DEFAULT_EMOD_ROOT = DATA_ROOT / "EMOd"
DEFAULT_KIM_ROOT = DATA_ROOT / "Kim"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceImage:
    image_id: str
    path: Path
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emod-root", type=Path, default=DEFAULT_EMOD_ROOT)
    parser.add_argument("--kim-root", type=Path, default=DEFAULT_KIM_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "emod_raw_complete",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "emod_complete.tsv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_id(path: Path) -> str:
    return path.stem


def sort_key(value: str) -> tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (float("inf"), value)


def collect_images(directory: Path, source: str) -> dict[str, SourceImage]:
    if not directory.exists():
        return {}
    found: dict[str, SourceImage] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            found.setdefault(image_id(path), SourceImage(image_id(path), path, source))
    return found


def expected_ids(emod_root: Path) -> list[str]:
    gt_dir = emod_root / "gt_fixation_con_EMOd_1019"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    ids = [
        image_id(path)
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(ids, key=sort_key)


def build_source_index(emod_root: Path, kim_root: Path) -> dict[str, SourceImage]:
    index: dict[str, SourceImage] = {}
    source_dirs = [
        (emod_root / "EMOdImages1019_RawImage", "EMOdImages1019_RawImage"),
        (emod_root / "raw_image", "EMOd_raw_image"),
        (kim_root / "Kim_all", "Kim_all"),
        (kim_root / "Kim_test", "Kim_test"),
    ]
    for directory, source in source_dirs:
        for key, item in collect_images(directory, source).items():
            index.setdefault(key, item)
    return index


def main() -> int:
    args = parse_args()
    ids = expected_ids(args.emod_root)
    source_index = build_source_index(args.emod_root, args.kim_root)
    missing = [item_id for item_id in ids if item_id not in source_index]
    if missing:
        joined = ", ".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} raw images: {joined}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for item_id in ids:
        source = source_index[item_id]
        target = args.output_dir / f"{item_id}.jpg"
        if args.overwrite or not target.exists():
            shutil.copy2(source.path, target)
        rows.append(
            {
                "item_id": item_id,
                "image_path": str(target),
                "source_path": str(source.path),
                "source": source.source,
            }
        )

    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "image_path", "source_path", "source"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    recovered = [row for row in rows if row["source"].startswith("Kim")]
    print(f"Wrote {len(rows)} images to {args.output_dir}")
    print(f"Wrote manifest: {args.manifest}")
    if recovered:
        print("Recovered from Kim: " + ", ".join(row["item_id"] for row in recovered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
