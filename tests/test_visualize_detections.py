from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - exercised by minimal test environments
    cv2 = None
    np = None

from scripts.phase7_analysis.visualize_detections import (
    load_coco_records,
    load_predictions,
    render_visualizations,
    select_image_ids,
)


class VisualizationTests(unittest.TestCase):
    def test_select_image_ids_is_deterministic_and_validates_requests(self):
        self.assertEqual(
            select_image_ids(["image-b", "image-a"], None, 1),
            ["image-a"],
        )
        with self.assertRaisesRegex(ValueError, "unknown|not in ground truth"):
            select_image_ids(["image-a"], ["missing"], 1)

    def test_loaders_normalize_image_ids_and_group_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 7, "file_name": "seven.png"}],
                        "annotations": [
                            {"image_id": 7, "category_id": 2, "bbox": [1, 2, 3, 4]}
                        ],
                        "categories": [{"id": 2, "name": "Calcification"}],
                    }
                ),
                encoding="utf-8",
            )
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    [{"image_id": "7", "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.9}]
                ),
                encoding="utf-8",
            )

            images, annotations, categories = load_coco_records(ground_truth_path)
            predictions = load_predictions(predictions_path)

            self.assertEqual(list(images), ["7"])
            self.assertEqual(len(annotations["7"]), 1)
            self.assertEqual(categories[2], "Calcification")
            self.assertEqual(len(predictions["7"]), 1)

    @unittest.skipUnless(cv2 is not None and np is not None, "OpenCV and NumPy are required")
    def test_render_visualizations_writes_overlay_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            output_dir = root / "output"
            images_dir.mkdir()
            image_path = images_dir / "image-a.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((100, 120, 3), dtype=np.uint8)))

            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": "image-a", "file_name": "image-a.png"}],
                        "annotations": [
                            {"image_id": "image-a", "category_id": 0, "bbox": [10, 20, 30, 25]}
                        ],
                        "categories": [{"id": 0, "name": "finding"}],
                    }
                ),
                encoding="utf-8",
            )
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    [{"image_id": "image-a", "category_id": 0, "bbox": [12, 22, 28, 24], "score": 0.9}]
                ),
                encoding="utf-8",
            )

            result = render_visualizations(
                ground_truth_path,
                predictions_path,
                images_dir,
                output_dir,
                max_images=1,
            )

            self.assertEqual(result["images"][0]["ground_truth_boxes"], 1)
            self.assertEqual(result["images"][0]["prediction_boxes"], 1)
            self.assertTrue((output_dir / "image-a.png").is_file())
            self.assertTrue((output_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
