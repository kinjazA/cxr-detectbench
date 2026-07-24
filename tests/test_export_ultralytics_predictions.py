from __future__ import annotations

import unittest

from scripts.export_ultralytics_predictions import xyxy_to_xywh


class UltralyticsExportTests(unittest.TestCase):
    def test_xyxy_to_xywh(self):
        self.assertEqual(xyxy_to_xywh([10, 20, 35, 50]), [10.0, 20.0, 25.0, 30.0])

    def test_xyxy_to_xywh_rejects_degenerate_box(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            xyxy_to_xywh([10, 20, 10, 50])


if __name__ == "__main__":
    unittest.main()
