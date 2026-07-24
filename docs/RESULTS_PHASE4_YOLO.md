# Phase 4 YOLO 结果归档

本文件保存 Phase 4 的历史结果，不随之后的实验覆盖。训练框架原生验证结果与统一评估结果分别记录，因为两者的评估实现和检测数量上限不同，不能直接当作同一个指标。

## 1. 运行记录

- 日期：2026-07-24
- Kaggle kernels：`kinjaza/phase4-yolo-smoke`、`kinjaza/phase4-yolo-baseline`、`kinjaza/phase4-yolo-unified-eval`
- GPU：Tesla P100-PCIE-16GB
- 数据：`corochann/vinbigdata-chest-xray-original-png` 的 original PNG
- 标注：WBF 融合（IoU 阈值 0.5）
- 划分：Phase 3 固定 image-level stratified split，val 为 2,250 张图
- 模型：YOLOv8n
- 正式训练：`imgsz=640`、`epochs=50`、`batch=16`、`workers=4`、`cache=False`、`seed=42`
- 正式训练耗时：6.874 小时
- 统一推理：`imgsz=640`、`batch=16`、`confidence=0.001`、`nms_iou=0.7`、最多导出 300 框/图
- 统一 COCO 评估：最多计入 100 框/图，FROC IoU=0.5

## 2. 数据与工程链路

正式 YOLO 数据目录构建结果：

| split | images | linked images | label files | boxes |
|---|---:|---:|---:|---:|
| train | 10,500 | 10,500 | 10,500 | 16,941 |
| val | 2,250 | 2,250 | 2,250 | 3,483 |
| test | 2,250 | 2,250 | 2,250 | 3,510 |

smoke test 用于确认 split、WBF 标签、PNG symlink、YOLO 标签和训练入口可用，3 epoch 结果为 `mAP@0.5=0.1998`、`mAP@0.5:0.95=0.1002`。这个数字不能与 20 epoch 融合消融或 50 epoch baseline 直接比较。

统一评估导出结果：

- 评估图像：2,250
- GT 病灶框：3,483
- 导出预测：101,132，约 45.0 个/图；低 confidence floor 是为了保留 PR/FROC 曲线，不代表最终部署阈值
- 退化框：跳过 1 个，并保留审计样例
- 推理耗时：282.742 秒，125.663 ms/图

这个退化框只占导出结果中的极小部分，不能解释整体指标偏低；它说明适配器需要容忍模型输出的异常框，同时继续让共享 evaluator 对最终预测契约保持严格校验。

## 3. 统一评估结果

统一协议下的总体结果：

| mAP50-95 | AP40 | AP50 | AP75 |
|---:|---:|---:|---:|
| 0.1812 | 0.3806 | 0.3499 | 0.1694 |

FROC 使用类别感知、病灶级 micro 匹配，正常图像也计入 FP/image 分母：

| FP/image | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 |
|---:|---:|---:|---:|---:|---:|---:|
| sensitivity | 0.2759 | 0.3299 | 0.3876 | 0.4502 | 0.5177 | 0.5777 |

按类别的统一评估结果：

| category | AP50-95 | AP40 | AP50 |
|---|---:|---:|---:|
| Aortic enlargement | 0.5325 | 0.8600 | 0.8474 |
| Atelectasis | 0.1416 | 0.3033 | 0.3033 |
| Calcification | 0.0722 | 0.1779 | 0.1595 |
| Cardiomegaly | 0.6058 | 0.8889 | 0.8879 |
| Consolidation | 0.1441 | 0.3376 | 0.3080 |
| ILD | 0.1750 | 0.3797 | 0.3298 |
| Infiltration | 0.1336 | 0.3080 | 0.2793 |
| Lung Opacity | 0.0637 | 0.2453 | 0.1772 |
| Nodule/Mass | 0.1229 | 0.2444 | 0.2336 |
| Other lesion | 0.0325 | 0.1089 | 0.0931 |
| Pleural effusion | 0.1722 | 0.4640 | 0.4051 |
| Pleural thickening | 0.0736 | 0.2682 | 0.2176 |
| Pneumothorax | 0.1573 | 0.3722 | 0.3550 |
| Pulmonary fibrosis | 0.1098 | 0.3702 | 0.3024 |

## 4. 阶段结论

第一，baseline 已经从“训练能跑”升级为“可以被统一协议复核”的结果。框架原生 val 为 `mAP50=0.3692`、`mAP50-95=0.1931`；统一 evaluator 为 `AP50=0.3499`、`mAP50-95=0.1812`。这两个结果的 split 和 checkpoint 相同，但统一协议固定了 COCO evaluator、101 点插值和 100 框/图上限，因此应分别保存，不能用其中一个替换另一个。

第二，主要问题不是完全检不出病灶，而是定位质量和难类稳定性不足。`AP50=0.3499` 明显高于 `mAP50-95=0.1812`，`AP75=0.1694` 也进一步下降，说明框大致落在目标附近时仍有一定能力，但高 IoU 下框的位置、大小或边界不够稳定。最弱类别是 `Other lesion`、`Lung Opacity`、`Calcification` 和 `Pleural thickening`；这些类别同时受到类别定义宽泛、边界模糊或目标较小的影响。

第三，类别差异与 Phase 3 的尺寸诊断一致。`Nodule/Mass`、`Calcification`、`Pleural thickening` 的 median normalized area 分别为 0.0014、0.0038、0.0044，正是需要优先验证高分辨率收益的类别。`Cardiomegaly` 和 `Aortic enlargement` 的 AP50-95 分别为 0.6058 和 0.5325，较大的、形态更稳定的目标明显更容易。

第四，FROC 表明在每图 0.5 个假阳性时 sensitivity 为 0.3876，每图 1 个假阳性时为 0.4502；这比单看 AP 更能体现当前系统在低误报约束下的实际召回能力。现阶段没有图像级 AUC、bad case 可视化或 test split 结果，因此不能进一步声称临床工作流性能或泛化性能。

## 5. 当前限制与解释边界

- 这是单个 YOLOv8n、单个 WBF 标注策略、单一 validation split 的 baseline，不是五范式横评结论。
- 采用 image-level split；由于当前 metadata 没有可靠的 patient/study id，不能宣称 patient-level 防泄漏。
- test split 尚未使用，必须继续保留给冻结模型或最终阶段评估。
- 统一评估只覆盖 COCO AP、AP40/AP50/AP75 和 FROC；图像级 AUC、参数量/FLOPs、FPS 和错误可视化尚未完成。
- 原始训练脚本目前没有 checkpoint resume 参数；在继续消耗 Kaggle GPU 前必须补齐或明确每次从头训练的风险。
- `scripts/requirements.txt` 尚未完全按 Kaggle 实际版本冻结；当前已验证的关键版本是 Python 3.12.13、PyTorch 2.4.0+cu121、torchvision 0.19.0+cu121、Ultralytics 8.4.104 和 pycocotools 2.0.10。

## 6. 下一阶段实验候选

本周 Kaggle GPU 配额已耗尽，因此当前只做本地准备，不启动新实验。恢复配额后，优先执行一个资源受限的分辨率实验：保持模型、split、WBF、epoch、seed 和评估协议不变，尝试提高 `imgsz`（候选 896）；如果 P100 显存不允许，才把 batch 降到 8，并在结果中明确这不是严格的单变量比较。重点观察 `mAP75`、`mAP50-95`、`Nodule/Mass`、`Calcification`、`Pleural thickening`、`Lung Opacity` 和 FROC@0.25/0.5/1。

在该实验前，先完成本地工程准备：补齐 YOLO checkpoint resume、冻结实际依赖版本、把预测和评估输出保存为可追溯的阶段 artifact，并准备预测框/GT 可视化。只有高分辨率实验明确改善定位或小目标后，才进入训练增强、后期关闭 mosaic 或其他变量；否则应转向 RT-DETR/Faster R-CNN 的跨范式比较，而不是继续堆 YOLO 超参。
