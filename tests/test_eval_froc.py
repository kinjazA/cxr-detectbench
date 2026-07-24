from __future__ import annotations

import unittest

from scripts.eval_froc import bbox_iou_xywh, compute_froc


class FrocTests(unittest.TestCase):
    def test_iou_for_identical_and_disjoint_boxes(self):
        self.assertEqual(bbox_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_duplicate_wrong_class_and_normal_image_false_positives(self):
        ground_truths = [
            {"image_id": "abnormal-1", "category_id": 0, "bbox": [0, 0, 10, 10]},
            {"image_id": "abnormal-2", "category_id": 1, "bbox": [20, 20, 10, 10]},
        ]
        predictions = [
            {"image_id": "abnormal-1", "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": "abnormal-1", "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": "abnormal-2", "category_id": 0, "bbox": [20, 20, 10, 10], "score": 0.7},
            {"image_id": "normal", "category_id": 1, "bbox": [5, 5, 3, 3], "score": 0.65},
            {"image_id": "abnormal-2", "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.6},
        ]

        result = compute_froc(
            predictions,
            ground_truths,
            image_ids=["abnormal-1", "abnormal-2", "normal"],
            iou_threshold=0.5,
            fp_limits=[0, 1],
        )

        self.assertEqual(result.num_images, 3)
        self.assertEqual(result.num_ground_truths, 2)
        self.assertEqual(result.points[-1].true_positives, 2)
        self.assertEqual(result.points[-1].false_positives, 3)
        self.assertEqual(result.points[-1].false_positives_per_image, 1.0)
        self.assertEqual(result.points[-1].sensitivity, 1.0)
        self.assertEqual(result.sensitivity_at_fp["0"], 0.5)
        self.assertEqual(result.sensitivity_at_fp["1"], 1.0)

    def test_requires_complete_nonempty_image_population(self):
        with self.assertRaisesRegex(ValueError, "at least one image"):
            compute_froc([], [], image_ids=[])


if __name__ == "__main__":
    unittest.main()
