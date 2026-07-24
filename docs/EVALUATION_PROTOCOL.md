# Unified Detection Evaluation Protocol

This protocol is the comparison boundary for every detector in CXR-DetectBench.
Framework-native metrics may be logged for debugging, but the final comparison
table must be produced from the shared COCO prediction format below.

## Evaluation population

- Development comparisons use the fixed Phase 3 validation split in
  `data/splits/val.csv` (2,250 images).
- The Phase 3 test split remains untouched until a model or project stage is
  frozen. It must not be used for hyperparameter selection.
- Normal images are part of the evaluation population. They have no detection
  annotations and are included in the FP/image denominator.

## Ground truth and predictions

Ground truth uses COCO detection JSON. Every model adapter must export one JSON
list containing records with exactly these shared semantics:

```json
{
  "image_id": "00150343289f317a0ad5629d5b7d9ef9",
  "category_id": 3,
  "bbox": [100.0, 120.0, 80.0, 60.0],
  "score": 0.91
}
```

- `bbox` is `[x, y, width, height]` in original-image pixel coordinates.
- `category_id` uses the frozen 0-13 mapping in `scripts/class_names.py`.
- `score` is a finite number in `[0, 1]`.
- Prediction export should use a low confidence floor so the evaluator, rather
  than a framework-specific display threshold, determines the PR/FROC curve.

The Ultralytics adapter is `scripts/export_ultralytics_predictions.py`. Future
MMDetection adapters must produce the same JSON contract.

## Metrics

`scripts/evaluate_detection.py` computes:

- COCO-style mAP at IoU 0.50:0.05:0.95 using 101 recall points.
- AP@0.50 and AP@0.75 from the same evaluator.
- AP@0.40 using the same COCO 101-point interpolation. This is a domain-oriented
  localization tolerance, not an attempt to reproduce the Kaggle leaderboard.
- Per-class AP@0.50:0.95, AP@0.40, and AP@0.50.
- A global class-aware lesion-level FROC curve at a declared IoU threshold.

COCO evaluation uses area=`all` and at most 100 detections per image. The
maximum is fixed across frameworks. Any change requires a protocol revision
and a rerun of all compared models.

## FROC definition

Predictions are sorted by confidence. A prediction is a true positive only if
it matches an unmatched ground-truth box from the same image and category at
or above the configured IoU. Duplicate detections, wrong-class detections, and
detections on normal images are false positives.

The reported curve is micro-averaged across all lesions:

- x-axis: cumulative false positives / all evaluated images
- y-axis: cumulative matched lesions / all ground-truth lesions

Operating points are reported at 0.125, 0.25, 0.5, 1, 2, and 4 FP/image. A
future per-class FROC extension must be reported separately rather than mixed
into this global definition.

## Outputs

Each evaluation run writes:

- `metrics.json`: compact protocol, aggregate COCO metrics, and FROC operating
  points
- `per_class_metrics.csv`: AP@0.50:0.95, AP@0.40, and AP@0.50 by category
- `froc_curve.csv`: full threshold curve for plotting

The prediction JSON and framework-native logs remain reproducibility artifacts,
not additional metric definitions.
