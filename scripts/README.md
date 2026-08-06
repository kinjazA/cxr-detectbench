# Scripts by Phase

本目录按项目阶段组织可执行脚本。Kaggle Notebook 从仓库根目录执行，因此文档和 notebook 中的命令都应使用下表里的完整路径。

| 阶段 | 目录 | 入口 | 状态 |
|---|---|---|---|
| Shared | `scripts/shared/` | `class_names.py` | 14 类映射已冻结 |
| Phase 2 | `scripts/phase2_preprocessing/` | `label_fusion.py`, `run_fusion_ablation.py`, `convert_coco_yolo.py`, `apply_clahe.py`, `check_fusion_output.py` | 已完成 WBF/NMS/raw 融合与消融 |
| Phase 3 | `scripts/phase3_splits/` | `prepare_phase3_splits.py` | 已生成 image-level 70/15/15 split |
| Phase 4 | `scripts/phase4_yolo/` | `prepare_yolo_dataset.py`, `train_yolo_baseline.py` | YOLOv8n baseline；训练入口支持 `--resume` |
| Phase 6 | `scripts/phase6_evaluation/` | `export_ultralytics_predictions.py`, `evaluate_detection.py`, `eval_froc.py` | 已完成统一 COCO/FROC 评估 |
| Phase 7 | `scripts/phase7_analysis/` | `visualize_detections.py`, `error_analysis.py` | 可视化入口已完成；结构化错误归因计划中 |

保留在根目录的 `requirements.txt` 是环境说明入口。后续新增脚本时优先放入对应阶段目录；跨阶段常量或轻量工具放入 `scripts/shared/`。

## 常用入口

```bash
python scripts/phase2_preprocessing/label_fusion.py --help
python scripts/phase3_splits/prepare_phase3_splits.py --help
python scripts/phase4_yolo/prepare_yolo_dataset.py --help
python scripts/phase4_yolo/train_yolo_baseline.py --help
python scripts/phase6_evaluation/export_ultralytics_predictions.py --help
python scripts/phase6_evaluation/evaluate_detection.py --help
python scripts/phase7_analysis/visualize_detections.py --help
```

续训时，`--epochs` 表示恢复后要达到的总 epoch 数，不是额外 epoch 数：

```bash
python scripts/phase4_yolo/train_yolo_baseline.py \
    --resume /kaggle/input/<checkpoint-dataset>/last.pt \
    --epochs 100 \
    --data data/processed/yolo_wbf_phase3/data.yaml \
    --project /kaggle/working/yolo_baseline_runs \
    --name yolov8n_wbf_phase3_img896_ep100
```

已核实的 YOLO/P100 精确依赖另列在
`scripts/requirements-kaggle-yolo-p100.txt`；MMDetection 依赖仍需 Kaggle 实测后再冻结。

不要再使用旧的根目录脚本路径，例如 `scripts/label_fusion.py` 或 `scripts/evaluate_detection.py`；这些文件已经按阶段移动。
