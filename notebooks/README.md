# Notebooks（Kaggle 上执行）

每个 Notebook 对应计划第 2 节。在 Kaggle 新建对应 Notebook，源码提交到本目录。

| 文件 | 阶段 | 说明 | GPU |
|---|---|---|---|
| 01_data_preprocessing.ipynb | Phase 2 | DICOM->PNG + 融合 + 格式转换 | 否 |
| 02_eda.ipynb | Phase 3 | 类别/尺寸分布 + 划分 | 否 |
| 03_label_fusion_ablation.ipynb | Phase 2.5 | 三版标注融合消融（YOLOv8n 少量 epoch） | 是(短) |
| 04_train_yolo.ipynb | Phase 5.3 | YOLO 训练调优 | 是 |
| 05_train_rtdetr.ipynb | Phase 5.4 | RT-DETR 训练调优 | 是 |
| 06_train_faster_rcnn.ipynb | Phase 5.1 | Faster R-CNN 训练调优 | 是 |
| 07_train_retinanet.ipynb | Phase 5.2 | RetinaNet + Focal Loss 消融 | 是 |
| 08_train_dino.ipynb | Phase 5.5 | DINO 微调 | 是 |
| 09_unified_evaluation.ipynb | Phase 6 | 五模型统一评估 + 总表 | 否 |
| 10_error_analysis.ipynb | Phase 7 | 错误分析与 bad case | 否 |
| 11_onnx_export_demo.ipynb | Phase 8 | ONNX 导出 + 延迟对比 | 短 |

> Notebooks 内的输出 cell（图表、checkpoint）不入库；源码入库，大产物存 Kaggle Dataset。