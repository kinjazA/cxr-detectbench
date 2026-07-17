"""COCO <-> YOLO 格式互转（Phase 2.6）。

从最终选定的融合标注版本，生成：
  - 完整 COCO json（给 MMDetection）
  - YOLO txt（给 Ultralytics，归一化的 <cls> <cx> <cy> <w> <h>）

TODO（Phase 2.6）：实现 + 保证两种格式共享同一张图像集与同一划分。
"""
from __future__ import annotations


def coco_to_yolo(coco_json, out_dir):
    raise NotImplementedError("Phase 2.6 实现")


def main():
    raise NotImplementedError("Phase 2.6 实现")


if __name__ == "__main__":
    main()