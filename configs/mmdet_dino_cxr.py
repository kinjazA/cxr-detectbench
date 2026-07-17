"""DINO CXR 配置（MMDetection，Phase 5.5）。

基线参考: mmdet configs/dino/dino-4scale_r50_8xb2-12e_coco.py
关键约束（Phase 5.5）：
  - **加载 COCO 预训练权重微调，不从零训练**（15k 图对 DINO 偏小，从零易过拟合）
  - 10-20 epoch + early stopping
  - 冻结 backbone 前几层，只微调 neck + head
  - 控制训练时长防超预算
"""
# TODO(Phase 5.5): 完整 config + load_from COCO 预训练