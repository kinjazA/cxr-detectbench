from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_detection import evaluate_coco, validate_predictions


class DetectionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ground_truth_path = Path(self.temp_dir.name) / "ground_truth.json"
        ground_truth = {
            "images": [
                {"id": "image-a", "width": 100, "height": 100},
                {"id": "image-normal", "width": 100, "height": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": "image-a",
                    "category_id": 0,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 0, "name": "finding"}],
        }
        self.ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_perfect_prediction_with_string_image_id(self):
        predictions = [
            {
                "image_id": "image-a",
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
                "score": 0.9,
            }
        ]
        metrics, per_class = evaluate_coco(
            self.ground_truth_path,
            predictions,
            image_ids=["image-a", "image-normal"],
        )

        self.assertAlmostEqual(metrics["coco"]["map50_95"], 1.0)
        self.assertAlmostEqual(metrics["coco"]["map40"], 1.0)
        self.assertAlmostEqual(metrics["coco"]["map50"], 1.0)
        self.assertAlmostEqual(per_class[0]["ap50_95"], 1.0)
        self.assertAlmostEqual(per_class[0]["ap50"], 1.0)
        self.assertEqual(metrics["froc"]["num_images"], 2)

    def test_empty_predictions_return_zero_metrics(self):
        metrics, _ = evaluate_coco(
            self.ground_truth_path,
            [],
            image_ids=["image-a", "image-normal"],
        )
        self.assertEqual(metrics["coco"]["map50_95"], 0.0)
        self.assertEqual(metrics["coco"]["map50"], 0.0)

    def test_prediction_validation_rejects_invalid_box_and_score(self):
        with self.assertRaisesRegex(ValueError, "width and height"):
            validate_predictions(
                [{"image_id": "a", "category_id": 0, "bbox": [0, 0, 0, 1], "score": 0.5}]
            )
        with self.assertRaisesRegex(ValueError, "score"):
            validate_predictions(
                [{"image_id": "a", "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1.1}]
            )

    def test_evaluation_rejects_duplicate_images_and_unknown_categories(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_coco(
                self.ground_truth_path,
                [],
                image_ids=["image-a", "image-a"],
            )
        with self.assertRaisesRegex(ValueError, "category IDs"):
            evaluate_coco(
                self.ground_truth_path,
                [
                    {
                        "image_id": "image-a",
                        "category_id": 99,
                        "bbox": [10, 10, 20, 20],
                        "score": 0.5,
                    }
                ],
                image_ids=["image-a"],
            )


if __name__ == "__main__":
    unittest.main()
