"""DICOM -> PNG 预处理（Phase 2.1）。

输入：data/raw 下的 .dicom 原图 + train.csv
输出：data/processed/images_png/<image_id>.png

流程：
  1. pydicom 读取，取 RescaleSlope/RescaleIntercept，做 HU/值还原
  2. 窗宽窗位：优先 VOI LUT / WindowCenter+WindowWidth；
     头信息缺失时用全局像素分布 min-max 兜底（见 Phase 2.1）
  3. CLAHE 增强对比度
  4. 灰度转 3 通道（方案见脚本内选择）
  5. 保存 PNG

TODO（Phase 2 实现时填）：
  - 兜底归一化分支
  - 灰度转3通道方案二选一：复制通道 / 伪彩色映射，选定后在注释记理由
  - 处理前后对比图导出（Phase 2.2，供 README）
"""
from __future__ import annotations

# import pydicom
# import numpy as np
# import cv2
# from pathlib import Path


def dicom_to_png(dcm_path, out_dir, *, apply_clahe=True, to_3ch="replicate"):
    """单张 DICOM 转 PNG。to_3ch: 'replicate' | 'pseudo_color'."""
    raise NotImplementedError("Phase 2.1 实现")


def main():
    raise NotImplementedError("Phase 2.1 实现")


if __name__ == "__main__":
    main()