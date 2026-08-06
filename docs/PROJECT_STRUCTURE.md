# Project Structure

本文件记录当前真实仓库结构，用来避免 README、Notebook 和脚本文档继续引用早期规划里的虚构路径。

## Top Level

```text
cxr-detectbench/
├── configs/        # Phase 5 模型配置草案：YOLO / MMDetection
├── data/splits/    # 已入库的小型 split 索引和诊断图表
├── demo/           # Phase 8 Demo 草案
├── docs/           # 协议、任务、日志、阶段结果
├── notebooks/      # 已创建并推送 Kaggle 的 notebook 源码与 metadata
├── outputs/        # 只保留 .gitkeep；大输出不入 Git
├── scripts/        # 按 Phase 分组的可执行脚本
├── tests/          # 当前覆盖统一评估、预测导出、训练入口和可视化
└── CXR-DetectBench-Project-Plan.md
```

## Script Layout

```text
scripts/
├── requirements.txt
├── shared/
│   └── class_names.py
├── phase2_preprocessing/
│   ├── apply_clahe.py
│   ├── check_fusion_output.py
│   ├── convert_coco_yolo.py
│   ├── label_fusion.py
│   └── run_fusion_ablation.py
├── phase3_splits/
│   └── prepare_phase3_splits.py
├── phase4_yolo/
│   ├── prepare_yolo_dataset.py
│   └── train_yolo_baseline.py
├── phase6_evaluation/
│   ├── eval_froc.py
│   ├── evaluate_detection.py
│   └── export_ultralytics_predictions.py
└── phase7_analysis/
    ├── visualize_detections.py
    └── error_analysis.py
```

## Notebook Layout

```text
notebooks/
├── phase2_preprocessing.ipynb
├── phase4_yolo_smoke/
│   └── phase4_yolo_smoke.ipynb
├── phase4_yolo_baseline/
│   └── phase4_yolo_baseline.ipynb
└── phase4_yolo_unified_eval/
    └── phase4_yolo_unified_eval.ipynb
```

Phase 5-8 的 RT-DETR、MMDetection、错误分析和 ONNX/Demo notebook 尚未创建。早期计划中的 `01_*` 到 `11_*` 文件名不是当前仓库入口。
