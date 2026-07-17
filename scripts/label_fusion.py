"""多标注融合（Phase 2.4 / 2.5）。

VinDr-CXR 每张图由 3 名放射科医生独立标注，同一区域最多 3 行。
本脚本产出三版 COCO 标注，用于消融实验：

  - labels_coco/raw/   3 医生框全保留
  - labels_coco/wbf/   ensemble-boxes 的 weighted_boxes_fusion（IoU 阈值默认 0.5）
  - labels_coco/nms/   NMS 去重

Phase 2.5 消融：YOLOv8n 少量 epoch 分别在三版上训练，对比 val mAP，
锁定最终融合策略；IoU 阈值可再做一次消融。

TODO（Phase 2）：实现 + 产出 "融合方式 vs mAP" 表。
"""
from __future__ import annotations


def fuse_wbf(boxes_list, scores_list, labels_list, iou_thr=0.5):
    """Weighted Boxes Fusion."""
    raise NotImplementedError("Phase 2.4 实现")


def fuse_nms(boxes, scores, labels, iou_thr=0.5):
    raise NotImplementedError("Phase 2.4 实现")


def build_coco(images, annotations, out_path, *, fusion="wbf"):
    raise NotImplementedError("Phase 2.4 实现")


if __name__ == "__main__":
    pass