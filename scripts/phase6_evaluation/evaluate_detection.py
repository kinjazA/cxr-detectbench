"""Evaluate model-agnostic COCO detection predictions.

Every detector in CXR-DetectBench must export a JSON list with COCO detection
records: ``image_id``, ``category_id``, ``bbox`` in ``[x, y, w, h]`` format,
and ``score``. This entry point computes metrics with one shared protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Hashable, Iterable, Mapping

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from .eval_froc import compute_froc
except ImportError:
    from eval_froc import compute_froc


IOU_THRESHOLDS = np.array([0.4, *np.arange(0.5, 0.96, 0.05)], dtype=np.float64)


def load_predictions(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    if not isinstance(predictions, list):
        raise ValueError("Predictions JSON must be a list of COCO detection records")
    validate_predictions(predictions)
    return predictions


def validate_predictions(predictions: Iterable[Mapping]) -> None:
    required = {"image_id", "category_id", "bbox", "score"}
    for index, prediction in enumerate(predictions):
        missing = required.difference(prediction)
        if missing:
            raise ValueError(f"Prediction {index} is missing fields: {sorted(missing)}")
        bbox = prediction["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Prediction {index} bbox must be a four-value list")
        if not all(np.isfinite(float(value)) for value in bbox):
            raise ValueError(f"Prediction {index} bbox contains a non-finite value")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"Prediction {index} bbox width and height must be positive")
        score = float(prediction["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index} score must be in [0, 1]")


def load_split_image_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError("Split CSV must contain an image_id column")
        image_ids = [row["image_id"] for row in reader]
    if not image_ids or any(not image_id for image_id in image_ids):
        raise ValueError("Split CSV contains no image IDs or contains an empty image ID")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Split CSV contains duplicate image IDs")
    return image_ids


def _empty_results(coco_gt: COCO) -> COCO:
    coco_dt = COCO()
    coco_dt.dataset = {
        "images": list(coco_gt.dataset["images"]),
        "categories": list(coco_gt.dataset["categories"]),
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else 0.0


def _ap_at_iou(evaluator: COCOeval, iou: float, category_index: int | None = None) -> float:
    iou_indices = np.flatnonzero(np.isclose(evaluator.params.iouThrs, iou))
    if iou_indices.size != 1:
        raise RuntimeError(f"Expected exactly one IoU={iou} evaluation slice")
    precision = evaluator.eval["precision"][iou_indices[0], :, :, 0, -1]
    if category_index is not None:
        precision = precision[:, category_index]
    return _mean_valid(precision)


def _ap_range(
    evaluator: COCOeval,
    minimum_iou: float,
    category_index: int | None = None,
) -> float:
    iou_indices = np.flatnonzero(evaluator.params.iouThrs >= minimum_iou - 1e-9)
    precision = evaluator.eval["precision"][iou_indices, :, :, 0, -1]
    if category_index is not None:
        precision = precision[:, :, category_index]
    return _mean_valid(precision)


def evaluate_coco(
    ground_truth_path: str | Path,
    predictions: list[dict],
    *,
    image_ids: Iterable[Hashable] | None = None,
    max_detections: int = 100,
    froc_iou: float = 0.5,
) -> tuple[dict, list[dict]]:
    if max_detections < 10:
        raise ValueError("max_detections must be at least 10")

    coco_gt = COCO(str(ground_truth_path))
    # pycocotools 2.0.10 treats this optional COCO field as mandatory in loadRes.
    coco_gt.dataset.setdefault("info", {})
    known_image_ids = set(coco_gt.getImgIds())
    evaluation_image_ids = list(image_ids) if image_ids is not None else list(known_image_ids)
    if len(evaluation_image_ids) != len(set(evaluation_image_ids)):
        raise ValueError("Evaluation image IDs must be unique")
    unknown_image_ids = set(evaluation_image_ids).difference(known_image_ids)
    if unknown_image_ids:
        examples = sorted(str(image_id) for image_id in unknown_image_ids)[:5]
        raise ValueError(f"Evaluation split contains image IDs absent from GT: {examples}")
    if not evaluation_image_ids:
        raise ValueError("Evaluation split contains no images")

    unknown_prediction_ids = {
        prediction["image_id"] for prediction in predictions
    }.difference(known_image_ids)
    if unknown_prediction_ids:
        examples = sorted(str(image_id) for image_id in unknown_prediction_ids)[:5]
        raise ValueError(f"Predictions contain image IDs absent from GT: {examples}")
    known_category_ids = set(coco_gt.getCatIds())
    unknown_category_ids = {
        int(prediction["category_id"]) for prediction in predictions
    }.difference(known_category_ids)
    if unknown_category_ids:
        raise ValueError(
            f"Predictions contain category IDs absent from GT: {sorted(unknown_category_ids)}"
        )

    coco_dt = coco_gt.loadRes(predictions) if predictions else _empty_results(coco_gt)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = evaluation_image_ids
    evaluator.params.catIds = sorted(coco_gt.getCatIds())
    evaluator.params.iouThrs = IOU_THRESHOLDS.copy()
    evaluator.params.maxDets = [1, 10, max_detections]
    evaluator.evaluate()
    evaluator.accumulate()

    metrics = {
        "protocol": {
            "prediction_format": "COCO detection JSON [x, y, width, height]",
            "image_count": len(evaluation_image_ids),
            "category_count": len(evaluator.params.catIds),
            "max_detections_per_image": max_detections,
            "coco_map_iou_range": "0.50:0.05:0.95",
            "froc_iou": froc_iou,
        },
        "coco": {
            "map50_95": _ap_range(evaluator, 0.5),
            "map40": _ap_at_iou(evaluator, 0.4),
            "map50": _ap_at_iou(evaluator, 0.5),
            "map75": _ap_at_iou(evaluator, 0.75),
        },
    }

    category_names = {
        category["id"]: category["name"] for category in coco_gt.dataset["categories"]
    }
    per_class = []
    for category_index, category_id in enumerate(evaluator.params.catIds):
        per_class.append(
            {
                "category_id": int(category_id),
                "category_name": category_names[category_id],
                "ap50_95": _ap_range(evaluator, 0.5, category_index),
                "ap40": _ap_at_iou(evaluator, 0.4, category_index),
                "ap50": _ap_at_iou(evaluator, 0.5, category_index),
            }
        )

    froc = compute_froc(
        predictions,
        coco_gt.dataset["annotations"],
        image_ids=evaluation_image_ids,
        iou_threshold=froc_iou,
    )
    metrics["froc"] = froc.to_dict()
    return metrics, per_class


def write_outputs(output_dir: str | Path, metrics: Mapping, per_class: list[Mapping]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    compact_metrics = dict(metrics)
    compact_metrics["froc"] = {
        key: value for key, value in metrics["froc"].items() if key != "points"
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(compact_metrics, file, indent=2, ensure_ascii=False)
    with (output_path / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fieldnames = ["category_id", "category_name", "ap50_95", "ap40", "ap50"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)
    with (output_path / "froc_curve.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "score_threshold",
            "false_positives_per_image",
            "sensitivity",
            "true_positives",
            "false_positives",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["froc"]["points"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, help="COCO ground-truth JSON")
    parser.add_argument("--predictions", required=True, help="COCO detection JSON list")
    parser.add_argument("--split-csv", help="Optional CSV containing the evaluation image_id set")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--froc-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    image_ids = load_split_image_ids(args.split_csv) if args.split_csv else None
    metrics, per_class = evaluate_coco(
        args.ground_truth,
        predictions,
        image_ids=image_ids,
        max_detections=args.max_detections,
        froc_iou=args.froc_iou,
    )
    write_outputs(args.output_dir, metrics, per_class)
    print(json.dumps({"coco": metrics["coco"], "froc": metrics["froc"]["sensitivity_at_fp"]}, indent=2))


if __name__ == "__main__":
    main()
