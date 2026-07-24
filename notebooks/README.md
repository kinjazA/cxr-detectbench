# Notebooks（Kaggle 上执行）

本目录只保存已经在仓库中存在的 Kaggle Notebook 源码和 metadata。训练日志、图表、checkpoint 与预测 JSON 不入 Git；它们应作为 Kaggle 输出或 Dataset artifact 保存，并在阶段结果文档中记录。

| 路径 | 阶段 | Kaggle kernel | 状态 | 说明 |
|---|---|---|---|---|
| `phase2_preprocessing.ipynb` | Phase 2 | `kinjaza/phase2-preprocessing` | 已完成 | original PNG 路径核验、WBF/NMS/raw 融合、YOLO 格式准备与融合消融 |
| `phase4_yolo_smoke/phase4_yolo_smoke.ipynb` | Phase 4 | `kinjaza/phase4-yolo-smoke` | 已完成 | 用 Phase 3 固定 split 构建 YOLO 数据集并运行 3 epoch smoke test |
| `phase4_yolo_baseline/phase4_yolo_baseline.ipynb` | Phase 4 | `kinjaza/phase4-yolo-baseline` | 已完成 | YOLOv8n / WBF / 640 / 50 epoch 的 P100 baseline |
| `phase4_yolo_unified_eval/phase4_yolo_unified_eval.ipynb` | Phase 4/6 | `kinjaza/phase4-yolo-unified-eval` | 已完成 | 对冻结 `best.pt` 导出完整 val 预测并执行统一 COCO AP/FROC 评估 |

Phase 5-8 的 RT-DETR、MMDetection、统一多模型汇总、错误分析和 ONNX/Demo Notebook 仍是计划项，尚未在此目录创建。不要把旧计划中的 `01_*` 到 `11_*` 文件名当作已存在的源码入口。

所有 kernel metadata 当前均为 private。Kaggle notebook 通过 clone GitHub 获取脚本，因此在 Kaggle 运行前必须先将相关代码推送到 GitHub；Kaggle 路径和输入名称以 Notebook 内的显式检查为准，不在本地文档中猜测。

Phase 4 的指标和运行参数见 [../docs/RESULTS_PHASE4_YOLO.md](../docs/RESULTS_PHASE4_YOLO.md)。
