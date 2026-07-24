from __future__ import annotations

import unittest

from scripts.export_ultralytics_predictions import result_to_coco, xyxy_to_xywh


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10, 20, 35, 50], [12, 20, 12, 55]])
        self.conf = FakeTensor([0.9, 0.8])
        self.cls = FakeTensor([1, 2])


class FakeResult:
    path = "/tmp/example_image.png"
    boxes = FakeBoxes()


class UltralyticsExportTests(unittest.TestCase):
    def test_xyxy_to_xywh(self):
        self.assertEqual(xyxy_to_xywh([10, 20, 35, 50]), [10.0, 20.0, 25.0, 30.0])

    def test_xyxy_to_xywh_rejects_degenerate_box(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            xyxy_to_xywh([10, 20, 10, 50])

    def test_result_to_coco_skips_degenerate_boxes_with_audit_example(self):
        invalid_examples = []
        predictions, skipped = result_to_coco(
            FakeResult(),
            invalid_box_examples=invalid_examples,
        )

        self.assertEqual(skipped, 1)
        self.assertEqual(
            predictions,
            [
                {
                    "image_id": "example_image",
                    "category_id": 1,
                    "bbox": [10.0, 20.0, 25.0, 30.0],
                    "score": 0.9,
                }
            ],
        )
        self.assertEqual(invalid_examples[0]["image_id"], "example_image")
        self.assertEqual(invalid_examples[0]["xyxy"], [12.0, 20.0, 12.0, 55.0])


if __name__ == "__main__":
    unittest.main()
