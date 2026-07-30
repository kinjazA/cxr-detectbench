"""Class-aware lesion-level FROC evaluation for COCO detections.

Predictions are processed from highest to lowest confidence. A prediction is a
true positive when it matches an as-yet-unmatched ground-truth box from the
same image and category at or above the configured IoU threshold. All other
predictions are false positives, including duplicate detections.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Hashable, Iterable, Mapping, Sequence


DEFAULT_FP_LIMITS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class FrocPoint:
    score_threshold: float
    false_positives_per_image: float
    sensitivity: float
    true_positives: int
    false_positives: int


@dataclass(frozen=True)
class FrocResult:
    iou_threshold: float
    num_images: int
    num_ground_truths: int
    points: tuple[FrocPoint, ...]
    sensitivity_at_fp: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "num_images": self.num_images,
            "num_ground_truths": self.num_ground_truths,
            "points": [asdict(point) for point in self.points],
            "sensitivity_at_fp": self.sensitivity_at_fp,
        }


def bbox_iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return IoU for two COCO-format ``[x, y, width, height]`` boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Bounding boxes must contain exactly four values")
    if box_a[2] <= 0 or box_a[3] <= 0 or box_b[2] <= 0 or box_b[3] <= 0:
        return 0.0

    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]
    intersection_width = max(0.0, min(ax2, bx2) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(ay2, by2) - max(box_a[1], box_b[1]))
    intersection = intersection_width * intersection_height
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - intersection
    return intersection / union if union > 0 else 0.0


def _group_ground_truths(
    ground_truths: Iterable[Mapping],
    image_ids: set[Hashable],
) -> dict[tuple[Hashable, int], list[Mapping]]:
    grouped: dict[tuple[Hashable, int], list[Mapping]] = defaultdict(list)
    for annotation in ground_truths:
        image_id = annotation["image_id"]
        if image_id in image_ids and not annotation.get("iscrowd", 0):
            grouped[(image_id, int(annotation["category_id"]))].append(annotation)
    return grouped


def compute_froc(
    predictions: Iterable[Mapping],
    ground_truths: Iterable[Mapping],
    *,
    image_ids: Iterable[Hashable],
    iou_threshold: float = 0.5,
    fp_limits: Sequence[float] = DEFAULT_FP_LIMITS,
) -> FrocResult:
    """Compute a global, class-aware lesion-level FROC curve.

    ``image_ids`` must include normal images so that FP/image uses the complete
    evaluation population rather than only images containing lesions.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    evaluation_image_ids = set(image_ids)
    if not evaluation_image_ids:
        raise ValueError("image_ids must contain at least one image")
    if any(limit < 0 for limit in fp_limits):
        raise ValueError("fp_limits cannot contain negative values")

    grouped_ground_truths = _group_ground_truths(ground_truths, evaluation_image_ids)
    num_ground_truths = sum(len(items) for items in grouped_ground_truths.values())
    matched = {key: [False] * len(items) for key, items in grouped_ground_truths.items()}

    indexed_predictions = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if prediction["image_id"] in evaluation_image_ids
    ]
    indexed_predictions.sort(key=lambda item: (-float(item[1]["score"]), item[0]))

    true_positives = 0
    false_positives = 0
    points: list[FrocPoint] = []

    for _, prediction in indexed_predictions:
        key = (prediction["image_id"], int(prediction["category_id"]))
        candidates = grouped_ground_truths.get(key, [])
        best_index = -1
        best_iou = -1.0
        for index, ground_truth in enumerate(candidates):
            if matched[key][index]:
                continue
            iou = bbox_iou_xywh(prediction["bbox"], ground_truth["bbox"])
            if iou > best_iou:
                best_index = index
                best_iou = iou

        if best_index >= 0 and best_iou >= iou_threshold:
            matched[key][best_index] = True
            true_positives += 1
        else:
            false_positives += 1

        points.append(
            FrocPoint(
                score_threshold=float(prediction["score"]),
                false_positives_per_image=false_positives / len(evaluation_image_ids),
                sensitivity=(true_positives / num_ground_truths if num_ground_truths else 0.0),
                true_positives=true_positives,
                false_positives=false_positives,
            )
        )

    sensitivity_at_fp = {
        f"{limit:g}": max(
            (point.sensitivity for point in points if point.false_positives_per_image <= limit),
            default=0.0,
        )
        for limit in fp_limits
    }
    return FrocResult(
        iou_threshold=iou_threshold,
        num_images=len(evaluation_image_ids),
        num_ground_truths=num_ground_truths,
        points=tuple(points),
        sensitivity_at_fp=sensitivity_at_fp,
    )
