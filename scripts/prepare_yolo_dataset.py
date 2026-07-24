"""Prepare the official Phase 4 YOLO dataset from WBF COCO labels and Phase 3 splits.

The script links PNG images instead of copying them. It writes one YOLO label
file per image, including empty files for No Finding images, so Ultralytics can
scan the dataset deterministically.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class_names import CLASS_NAMES  # noqa: E402


SPLITS = ("train", "val", "test")


def read_split_ids(splits_dir: Path):
    split_ids = {}
    seen = {}
    for split in SPLITS:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        ids = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "image_id" not in (reader.fieldnames or []):
                raise ValueError(f"{path} must contain an image_id column")
            for row in reader:
                image_id = row["image_id"]
                if image_id in seen:
                    raise ValueError(
                        f"image_id={image_id} appears in both {seen[image_id]} and {split}"
                    )
                seen[image_id] = split
                ids.append(image_id)
        split_ids[split] = ids

    return split_ids


def load_coco(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {path}")
    with path.open(encoding="utf-8") as f:
        coco = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{path} is missing COCO key: {key}")

    images = {image["id"]: image for image in coco["images"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)

    return coco, images, annotations


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def symlink_image(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def find_image(images_dir: Path, image_id: str, image_ext: str):
    image_path = images_dir / f"{image_id}.{image_ext}"
    if not image_path.exists():
        return None
    return image_path


def coco_bbox_to_yolo_line(ann, image):
    img_w = float(image["width"])
    img_h = float(image["height"])
    x, y, w, h = [float(value) for value in ann["bbox"]]
    class_id = int(ann["category_id"])

    if class_id < 0 or class_id >= 14:
        raise ValueError(f"Unexpected category_id={class_id} in annotation id={ann.get('id')}")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size for image_id={image['id']}: {img_w}x{img_h}")
    if w < 0 or h < 0:
        raise ValueError(f"Negative bbox size in annotation id={ann.get('id')}: {ann['bbox']}")

    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    values = (cx, cy, w_norm, h_norm)
    if any(value < -1e-6 or value > 1.0 + 1e-6 for value in values):
        raise ValueError(
            f"YOLO bbox outside [0, 1] for image_id={image['id']}, "
            f"ann_id={ann.get('id')}: {values}"
        )

    cx, cy, w_norm, h_norm = [min(max(value, 0.0), 1.0) for value in values]
    return f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"


def write_label_file(path: Path, image, anns):
    lines = [coco_bbox_to_yolo_line(ann, image) for ann in anns]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(coco_json: Path, images_dir: Path, splits_dir: Path, output_dir: Path, image_ext: str):
    split_ids = read_split_ids(splits_dir)
    _, coco_images, coco_annotations = load_coco(coco_json)
    source_images = images_dir.resolve()
    if not source_images.exists():
        raise FileNotFoundError(f"Image directory does not exist: {source_images}")

    all_split_ids = {image_id for ids in split_ids.values() for image_id in ids}
    missing_from_coco = sorted(all_split_ids - set(coco_images))
    extra_in_coco = sorted(set(coco_images) - all_split_ids)
    if missing_from_coco:
        raise RuntimeError(
            f"{len(missing_from_coco)} split image_ids are missing from COCO. "
            f"Examples: {missing_from_coco[:5]}"
        )
    if extra_in_coco:
        raise RuntimeError(
            f"{len(extra_in_coco)} COCO image_ids are not present in Phase 3 splits. "
            f"Examples: {extra_in_coco[:5]}"
        )

    summary = []
    class_counts = {split: Counter() for split in SPLITS}
    missing_images = []
    for split, image_ids in split_ids.items():
        image_out = output_dir / "images" / split
        label_out = output_dir / "labels" / split
        linked = 0
        labels = 0
        boxes = 0

        for image_id in image_ids:
            src = find_image(source_images, image_id, image_ext)
            if src is None:
                missing_images.append(image_id)
                continue

            symlink_image(src, image_out / src.name)
            linked += 1

            image = coco_images[image_id]
            anns = coco_annotations.get(image_id, [])
            write_label_file(label_out / f"{image_id}.txt", image, anns)
            labels += 1
            boxes += len(anns)
            for ann in anns:
                class_counts[split][int(ann["category_id"])] += 1

        summary.append(
            {
                "split": split,
                "images": len(image_ids),
                "linked_images": linked,
                "label_files": labels,
                "boxes": boxes,
            }
        )

    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} split images are missing from {source_images}. "
            f"Examples: {missing_images[:5]}"
        )

    return summary, class_counts


def write_data_yaml(path: Path, output_dir: Path):
    names = [CLASS_NAMES[i] for i in range(14)]
    path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 14",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_summary_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "images", "linked_images", "label_files", "boxes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_output(output_dir: Path, summary):
    for row in summary:
        split = row["split"]
        image_count = len(list((output_dir / "images" / split).glob("*.png")))
        label_count = len(list((output_dir / "labels" / split).glob("*.txt")))
        if image_count != row["images"] or label_count != row["images"]:
            raise RuntimeError(
                f"{split} output mismatch: expected={row['images']}, "
                f"images={image_count}, labels={label_count}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco_json",
        default="data/processed/labels_coco/wbf/annotations.json",
        help="COCO annotations.json from the selected fusion mode.",
    )
    parser.add_argument(
        "--images_dir",
        default="data/processed/images_png",
        help="Directory that directly contains <image_id>.png files.",
    )
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--output_dir", default="data/processed/yolo_wbf_phase3")
    parser.add_argument("--image_ext", default="png")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    summary, _ = build_dataset(
        coco_json=Path(args.coco_json),
        images_dir=Path(args.images_dir),
        splits_dir=Path(args.splits_dir),
        output_dir=output_dir,
        image_ext=args.image_ext,
    )
    validate_output(output_dir, summary)
    write_data_yaml(output_dir / "data.yaml", output_dir)
    write_summary_csv(output_dir / "dataset_summary.csv", summary)

    print(f"Wrote YOLO dataset to {output_dir}")
    for row in summary:
        print(
            f"{row['split']}: images={row['images']}, linked={row['linked_images']}, "
            f"labels={row['label_files']}, boxes={row['boxes']}"
        )


if __name__ == "__main__":
    main()
