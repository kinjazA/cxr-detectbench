"""Render deterministic GT/prediction overlays for a detection split.

The renderer consumes the shared COCO ground-truth JSON and prediction JSON.
Ground-truth boxes are green; predictions are red and include their score.
Only a small, explicitly selected image set is written to avoid large Kaggle
outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_coco_records(path: str | Path) -> tuple[dict[str, dict], dict[str, list], dict[int, str]]:
    """Load image, annotation, and category records using string image IDs."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"COCO ground truth must be an object: {path}")

    images: dict[str, dict] = {}
    for image in data.get("images", []):
        image_id = str(image["id"])
        if image_id in images:
            raise ValueError(f"Duplicate image ID in ground truth: {image_id}")
        images[image_id] = image

    annotations: dict[str, list] = {image_id: [] for image_id in images}
    for annotation in data.get("annotations", []):
        image_id = str(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"Annotation references unknown image ID: {image_id}")
        annotations[image_id].append(annotation)

    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in data.get("categories", [])
    }
    return images, annotations, categories


def load_predictions(path: str | Path) -> dict[str, list]:
    """Load prediction records grouped by string image ID."""
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Prediction JSON must be a list: {path}")
    predictions: dict[str, list] = {}
    for prediction in data:
        image_id = str(prediction["image_id"])
        predictions.setdefault(image_id, []).append(prediction)
    return predictions


def select_image_ids(
    available_image_ids: Iterable[str],
    requested_image_ids: Iterable[str] | None,
    max_images: int,
) -> list[str]:
    """Select a deterministic image subset and reject unknown explicit IDs."""
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    available = set(available_image_ids)
    if requested_image_ids is None:
        return sorted(available)[:max_images]

    selected = [str(image_id) for image_id in requested_image_ids]
    if len(selected) != len(set(selected)):
        raise ValueError("requested image IDs must be unique")
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"Requested image IDs are not in ground truth: {unknown[:5]}")
    return selected[:max_images]


def resolve_image_path(images_dir: str | Path, file_name: str) -> Path:
    """Resolve a COCO file name without allowing it to escape images_dir."""
    root = Path(images_dir).resolve()
    candidate = (root / file_name).resolve()
    if candidate.is_file() and root in candidate.parents:
        return candidate
    basename_candidate = (root / Path(file_name).name).resolve()
    if basename_candidate.is_file() and root in basename_candidate.parents:
        return basename_candidate
    raise FileNotFoundError(f"Image file not found under {root}: {file_name}")


def _validate_box(box: list | tuple) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("COCO bbox must contain [x, y, width, height]")
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("COCO bbox must have positive width and height")
    return x, y, width, height


def render_visualizations(
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    image_ids: Iterable[str] | None = None,
    max_images: int = 12,
    score_threshold: float = 0.0,
) -> dict:
    """Write overlays and a compact manifest for the selected images."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")

    images, annotations, categories = load_coco_records(ground_truth_path)
    predictions = load_predictions(predictions_path)
    unknown_prediction_ids = sorted(set(predictions) - set(images))
    if unknown_prediction_ids:
        raise ValueError(
            "Predictions reference unknown image IDs: " f"{unknown_prediction_ids[:5]}"
        )
    selected = select_image_ids(images, image_ids, max_images)
    if not selected:
        raise ValueError("Ground truth contains no images to visualize")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to render detection visualizations") from error

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_id in selected:
        image_record = images[image_id]
        source = resolve_image_path(images_dir, str(image_record["file_name"]))
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"OpenCV could not read image: {source}")

        gt_count = 0
        pred_count = 0
        for annotation in annotations[image_id]:
            x, y, width, height = _validate_box(annotation["bbox"])
            _draw_box(cv2, image, x, y, width, height, (0, 200, 0), categories, annotation, "GT")
            gt_count += 1
        for prediction in predictions.get(image_id, []):
            score = float(prediction["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction score must be finite and in [0, 1]")
            if score < score_threshold:
                continue
            x, y, width, height = _validate_box(prediction["bbox"])
            _draw_box(cv2, image, x, y, width, height, (0, 0, 220), categories, prediction, "P")
            pred_count += 1

        destination = output_path / f"{image_id}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"OpenCV could not write image: {destination}")
        manifest.append(
            {
                "image_id": image_id,
                "source": str(source),
                "output": str(destination),
                "ground_truth_boxes": gt_count,
                "prediction_boxes": pred_count,
            }
        )

    result = {
        "ground_truth": str(Path(ground_truth_path).resolve()),
        "predictions": str(Path(predictions_path).resolve()),
        "images_dir": str(Path(images_dir).resolve()),
        "score_threshold": score_threshold,
        "images": manifest,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _draw_box(cv2, image, x, y, width, height, color, categories, record, prefix):
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width - 1, round(x)))
    y1 = max(0, min(image_height - 1, round(y)))
    x2 = max(0, min(image_width - 1, round(x + width)))
    y2 = max(0, min(image_height - 1, round(y + height)))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    category_id = int(record["category_id"])
    label = f"{prefix}:{categories.get(category_id, str(category_id))}"
    if "score" in record:
        label += f" {float(record['score']):.2f}"
    text_y = max(15, y1 - 5)
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-id", action="append", dest="image_ids")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_visualizations(
        args.ground_truth,
        args.predictions,
        args.images_dir,
        args.output_dir,
        image_ids=args.image_ids,
        max_images=args.max_images,
        score_threshold=args.score_threshold,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
