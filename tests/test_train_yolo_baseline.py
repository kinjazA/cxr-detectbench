from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.phase4_yolo.train_yolo_baseline import resolve_resume_checkpoint


class TrainBaselineTests(unittest.TestCase):
    def test_resume_checkpoint_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            checkpoint.write_bytes(b"checkpoint")

            self.assertEqual(resolve_resume_checkpoint(str(checkpoint)), checkpoint.resolve())

    def test_resume_checkpoint_must_exist(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            resolve_resume_checkpoint("missing.pt")

    def test_resume_checkpoint_must_be_pt(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.bin"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(ValueError, "must be a .pt file"):
                resolve_resume_checkpoint(str(checkpoint))


if __name__ == "__main__":
    unittest.main()
