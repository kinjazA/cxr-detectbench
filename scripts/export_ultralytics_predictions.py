"""Export Ultralytics detector predictions to the shared COCO JSON contract."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .evaluate_detection import load_split_image_ids, validate_predictions
except ImportError:
    from evaluate_detection import load_split_image_ids, validate_predictions


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    if len(box) != 4:
        raise ValueError("xyxy box must contain exactly four values")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("xyxy box must have positive width and height")
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def result_to_coco(result) -> list[dict]:
    image_id = Path(result.path).stem
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    categories = result.boxes.cls.detach().cpu().tolist()
    if not len(xyxy) == len(scores) == len(categories):
        raise RuntimeError("Ultralytics returned inconsistent box, score, and class counts")
    return [
        {
            "image_id": image_id,
            "category_id": int(category_id),
            "bbox": xyxy_to_xywh(box),
            "score": float(score),
        }
        for box, score, category_id in zip(xyxy, scores, categories)
    ]


def export_predictions(
    model_path: str | Path,
    images_dir: str | Path,
    output_path: str | Path,
    *,
    expected_image_ids: Iterable[str] | None,
    imgsz: int,
    batch: int,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    device: str | None,
) -> dict:
    from ultralytics import YOLO

    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_path}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    if not 0 < nms_iou <= 1:
        raise ValueError("nms_iou must be in (0, 1]")

    expected = set(expected_image_ids) if expected_image_ids is not None else None
    model = YOLO(str(model_path))
    start = time.perf_counter()
    results = model.predict(
        source=str(images_path),
        imgsz=imgsz,
        batch=batch,
        conf=confidence,
        iou=nms_iou,
        max_det=max_detections,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )

    predictions: list[dict] = []
    seen_image_ids: set[str] = set()
    for result in results:
        image_id = Path(result.path).stem
        if image_id in seen_image_ids:
            raise RuntimeError(f"Ultralytics returned duplicate image ID: {image_id}")
        seen_image_ids.add(image_id)
        predictions.extend(result_to_coco(result))
    elapsed_seconds = time.perf_counter() - start

    if expected is not None and seen_image_ids != expected:
        missing = sorted(expected.difference(seen_image_ids))[:5]
        unexpected = sorted(seen_image_ids.difference(expected))[:5]
        raise RuntimeError(
            "Predicted image set does not match the requested split: "
            f"missing={missing}, unexpected={unexpected}"
        )
    validate_predictions(predictions)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)

    return {
        "model": str(model_path),
        "images_dir": str(images_path),
        "image_count": len(seen_image_ids),
        "prediction_count": len(predictions),
        "imgsz": imgsz,
        "batch": batch,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "max_detections": max_detections,
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_image": (
            elapsed_seconds * 1000 / len(seen_image_ids) if seen_image_ids else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--split-csv", help="Optional expected image_id set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", help="Optional JSON runtime summary")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_image_ids = load_split_image_ids(args.split_csv) if args.split_csv else None
    summary = export_predictions(
        args.model,
        args.images_dir,
        args.output,
        expected_image_ids=expected_image_ids,
        imgsz=args.imgsz,
        batch=args.batch,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        max_detections=args.max_detections,
        device=args.device,
    )
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
