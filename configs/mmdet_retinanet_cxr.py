"""RetinaNet (R50-FPN) CXR 配置（MMDetection，Phase 5.2）。

基线参考: mmdet configs/retinanet/retinanet_r50_fpn_1x_coco.py
**重点优化（Phase 5.2）**：Focal Loss α/γ 消融实验，
对比不同 γ 对稀有类别（Pneumothorax 等）召回率的影响。
"""
# TODO(Phase 5.2): 完整 config，含一组 focal_loss 的 alpha/gamma sweep