"""Faster R-CNN (R50-FPN) CXR 配置（MMDetection，Phase 5.1）。

基线参考: mmdet configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
针对性优化（Phase 5.1）：
  - k-means 重新聚类 anchor 尺寸（针对小病灶）
  - 调整 RPN 正负样本采样比例
  - (可选) Cascade R-CNN 对比

Phase 5 实现时：用 _base_ 继承 coco 配置，覆盖 num_classes / data / img_scale /
anchor / batch / lr / epoch / checkpoint 保存频率。
"""
# TODO(Phase 5.1): from mmengine.config import Config; 写完整 config