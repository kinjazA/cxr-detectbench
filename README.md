# CXR-DetectBench
## 胸部 X 光多范式目标检测基准系统

项目状态：在建。当前已完成 Phase 1 数据核查、Phase 2 多标注融合消融与 Phase 3 image-level 正式划分，后续进入基线训练链路。详细任务拆解见 [docs/TASK_BREAKDOWN.md](docs/TASK_BREAKDOWN.md)，执行日志见 [docs/PLAN_PROGRESS.md](docs/PLAN_PROGRESS.md)。

在真实临床胸部 X 光数据集 **VinDr-CXR** 上，系统化对比五种主流目标检测范式，完成从数据处理、多标注融合、多模型训练调优、统一评估、错误分析到 ONNX 导出与 Demo 部署的完整闭环。

| 范式 | 模型 | 框架 |
|---|---|---|
| 两阶段 anchor-based | Faster R-CNN (R50-FPN) | MMDetection |
| 一阶段 anchor-based | RetinaNet (R50-FPN) | MMDetection |
| 一阶段 anchor-free | YOLOv8 / 最新稳定版 | Ultralytics |
| 端到端 Transformer | RT-DETR | Ultralytics |
| 端到端 Transformer | DINO | MMDetection |

## 核心能力

1. **医疗影像数据管道**：DICOM/PNG 数据接入、图像增强、多标注融合、COCO/YOLO 双格式标注。
2. **多范式检测横评**：覆盖 two-stage、one-stage、anchor-free 和 Transformer detection，而不是单模型调参。
3. **领域适配评估体系**：除 COCO mAP 外，规划 mAP@0.4、FROC、图像级 AUC 等更贴近胸片 CAD 的指标。
4. **结构化错误分析**：区分标注噪声、真实漏检、正常片误报、长尾类别召回不足等问题来源。
5. **工程闭环**：Kaggle 训练、checkpoint 持久化、统一评估、ONNX 导出与 Demo 部署。

## 数据集

[Kaggle: vinbigdata-chest-xray-abnormalities-detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

- 15,000 张训练胸片，其中 10,606 张 No finding，4,394 张含异常标注。
- 由 17 位放射科医生组成标注池，每张异常图恰好由其中 3 位完成独立标注。
- 14 类胸部异常 + No finding。No finding 是图像级标签，不参与 box 回归。
- 本项目当前 Kaggle 成功链路使用 `corochann/vinbigdata-chest-xray-original-png` 的 original PNG；是否引入 CLAHE 增强留到 Phase 3/4 通过实验决定。

数据使用协议：VinDr-CXR 数据仅限学术 / 非商业研究用途。本仓库不包含任何原始 DICOM、PNG 或标注数据，仅包含处理脚本、配置与文档；实际训练在 Kaggle Notebook 中挂载数据完成。

## 项目结构

```text
cxr-detectbench/
├── data/                # 原始/处理数据不入库，仅保留 split 索引等小文件
├── notebooks/           # Kaggle Notebook 与 kernel metadata
├── configs/             # YOLO / MMDetection 模型配置
├── scripts/             # 数据处理、融合、格式转换、评估、错误分析脚本
├── outputs/             # checkpoints / eval results / bad cases，不入库
├── demo/                # Demo 应用
├── docs/                # 任务拆解与执行记录
└── CXR-DetectBench-Project-Plan.md
```

## 环境与运行

训练主要在 Kaggle Notebook 上完成，当前验证过的 GPU 为 Tesla P100-PCIE-16GB。P100 环境下 Kaggle 默认 PyTorch 版本不兼容 sm_60，因此 notebook 会安装 `torch==2.4.0` / `torchvision==0.19.0`。完整依赖见 [scripts/requirements.txt](scripts/requirements.txt)。

Kaggle kernel：

- Phase 2 preprocessing / fusion ablation：`kinjaza/phase2-preprocessing`
- Phase 4 YOLO smoke：`kinjaza/phase4-yolo-smoke`
- Phase 4 YOLO baseline：`kinjaza/phase4-yolo-baseline`

## 阶段进展

### Phase 0：环境与工程基建

状态：基本完成，仍需补一份更干净的冻结依赖与 checkpoint 持久化规范。

已经确认本地仓库、GitHub remote、Kaggle CLI、Kaggle competition access 和 kernel 运行链路可用。Kaggle kernel 已改为 private 配置，但实际代码仍通过 GitHub clone 获取，所以关键脚本改动必须先推到 GitHub 再触发 Kaggle run。

阶段分析：工程链路已经跑通，但还不够“可长期复现”。下一步应把 P100 特例、PyTorch 版本、Ultralytics 版本、Kaggle Dataset 持久化路径写进固定 requirements / runbook，避免每次换 session 都重新猜环境。

### Phase 1：数据核查

状态：完成。

核查结论：

| 项目 | 结果 |
|---|---:|
| train image_id | 15,000 |
| No finding 图像 | 10,606 |
| 异常图像 | 4,394 |
| 标注医生池 | 17 位 |
| 每张异常图标注者 | 3 位 |

DICOM metadata 核查显示 WindowCenter/WindowWidth 可用率约 97.18%，RescaleSlope/RescaleIntercept 可用率约 81.88%。结合算力和已有 Kaggle PNG 数据集，本项目当前优先采用 original PNG 路线。

阶段分析：最重要的修正是“不是固定 3 名医生，而是 17 位医生池、每张异常图由其中 3 位标注”。这增强了多标注融合实验的意义，也解释了为什么 raw/WBF/NMS 消融是项目核心卖点之一。

### Phase 2：数据预处理与多标注融合

状态：完成主链路，最终融合策略锁定为 `wbf`。

Kaggle `kinjaza/phase2-preprocessing` 于 2026-07-24 完整跑通，状态 `COMPLETE`。关键数据链路如下：

- PNG 路径：`/kaggle/input/datasets/corochann/vinbigdata-chest-xray-original-png/train`
- `data/processed/images_png` 成功 symlink 到上述目录
- 识别 PNG：15,000 张
- split：12,000 train / 3,000 val
- YOLO image links：12,000 train / 3,000 val / 0 skipped
- YOLO labels：12,000 train txt / 3,000 val txt
- Ultralytics scan：0 corrupt

三版 COCO 标注产出：

| fusion_mode | images | annotations |
|---|---:|---:|
| raw | 15,000 | 36,096 |
| wbf | 15,000 | 23,934 |
| nms | 15,000 | 24,034 |

Phase 2.4 融合消融结果（YOLOv8n，20 epochs）：

| fusion_mode | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| raw | 0.2812 | 0.1385 |
| wbf | **0.3210** | **0.1662** |
| nms | 0.3043 | 0.1403 |

阶段分析：之前 mAP 全 0 的根因不是模型无效，而是数据路径和训练准备逻辑问题。已修复 Kaggle mount 路径、`dim0/dim1` 宽高映射、异常吞错、标签 symlink 和复制 PNG 爆盘问题。WBF 在 mAP@0.5 和 mAP@0.5:0.95 上都领先，后续默认使用 `wbf` 标注。需要注意，本轮 full-data 三组消融耗时接近 10 小时，后续调试应优先使用小样本 smoke test。

### Phase 3：EDA 与正式数据划分

状态：完成第一版正式 image-level split；patient/study-level split 暂不声称完成。

本地 `data/raw/images.csv` 暂未发现可确认用于同一病人或同一 study 分组的字段，只能看到 `PatientSex`、`PatientAge`、`Rows`、`Columns`、`fname` 等 metadata。因此当前正式划分采用 image-level stratified split，不做 patient/study-level 防泄漏承诺；如果后续提供可靠 patient/study id 字段，需要重做划分。

固定划分产出见 `data/splits/`：

| split | images | abnormal_images | normal_images | abnormal_rate |
|---|---:|---:|---:|---:|
| train | 10,500 | 3,076 | 7,424 | 0.2930 |
| val | 2,250 | 659 | 1,591 | 0.2929 |
| test | 2,250 | 659 | 1,591 | 0.2929 |

质量检查：

- 每个异常类别的图片级 split 比例接近 70/15/15，最大偏差为 0.0083（class 12 / Pneumothorax，由 96 张阳性图像的取整造成）。
- annotation 级别最大偏差为 0.0385（class 2 / Calcification 的 val split），作为诊断指标记录；由于一张图可能有多个框，当前接受标准以 image-level balance 为准。
- bbox median normalized area 最小的是 Nodule/Mass（class 8，0.0014），其次是 Calcification（class 2，0.0038）和 Pleural thickening（class 11，0.0044），后续训练需要重点关注小目标召回。

阶段分析：正式 split 已从 Phase 2 消融用的 12k/3k 随机划分升级为 70/15/15 固定划分，并加入 split 完整性与类别比例质量闸门。下一步不应继续消耗 Kaggle GPU 做融合消融，而应基于 `wbf` 标注和该 split 先跑 YOLO baseline smoke test，确认正式训练、评估、checkpoint 保存和推理输出全链路。

### Phase 4：Baseline 跑通全链路

状态：smoke test 已通过，正式 baseline 待跑。

目标是用 `wbf` 标注和 Phase 3 固定 split，跑通一次 YOLO baseline 的训练、验证、推理、mAP 计算和可视化。这里不追求最终分数，重点是验证正式训练代码、数据格式、checkpoint、评估脚本和结果保存规范。

2026-07-24，Kaggle `kinjaza/phase4-yolo-smoke` 用 `wbf` 标注和 Phase 3 split 跑通 3 epoch YOLOv8n smoke test：train/val/test YOLO 数据目录构建通过，val mAP@0.5 = 0.1998，mAP@0.5:0.95 = 0.1002。

正式 baseline 入口已准备：`scripts/train_yolo_baseline.py` + Kaggle `kinjaza/phase4-yolo-baseline`。第一版参数按 P100 保守设置为 YOLOv8n、imgsz=640、epochs=50、batch=16、workers=4、cache=False，只在 val 上评估并保留 compact summary + best/last checkpoints。

阶段分析：Phase 4 的关键风险已经从“路径/标签是否能训”转移到“正式 baseline 分数是否合理”。第一版 baseline 先不拉到 1024，也不启用 image cache，避免 P100 显存和 Kaggle disk 同时吃紧；如果 50 epoch 仍明显低于 Phase 2 的 20 epoch WBF 消融结果，再优先排查训练配置、Ultralytics 版本差异、数据划分和增强策略。

### Phase 5：五模型训练矩阵

状态：未开始。

计划横评：

| 模型 | 框架 | 重点 |
|---|---|---|
| Faster R-CNN (R50-FPN) | MMDetection | two-stage baseline，anchor 与 RPN 策略可调 |
| RetinaNet (R50-FPN) | MMDetection | Focal Loss 对长尾类别召回的影响 |
| YOLOv8 / 最新稳定版 | Ultralytics | 高分辨率、小目标增强、训练后期关闭 mosaic |
| RT-DETR | Ultralytics | 端到端 Transformer baseline |
| DINO | MMDetection | COCO 预训练微调，控制过拟合 |

阶段分析：不建议一开始就五个模型全量训练。更稳的顺序是 YOLO baseline -> RT-DETR -> Faster R-CNN，再根据算力决定 RetinaNet/DINO 的训练规模。每个模型至少保留一轮“默认配置”和一轮“针对胸片问题的改进配置”对比，否则项目会变成单纯跑模型列表。

### Phase 6：统一评估体系

状态：脚本占位/规划阶段。

规划指标：

- COCO mAP@0.5 / mAP@0.5:0.95 / per-class AP
- 领域 mAP@0.4
- FROC 曲线
- 图像级异常检测 AUC / sensitivity / specificity
- 参数量、FLOPs、FPS、精度-速度 Pareto

阶段分析：这个阶段是项目能否从“训练几个模型”升级成“检测基准系统”的关键。建议尽早把评估输入输出格式固定下来，不要等五个模型全部训练完才写统一评估，否则很容易出现不同模型结果不可比的问题。

### Phase 7：错误分析

状态：未开始。

计划从 FP、FN、正常片误报、长尾类别漏检、标注分歧区域等角度整理 bad cases。输出应包括定量统计和可视化图，而不是只挑几张图展示。

阶段分析：VinDr-CXR 的多医生标注天然存在分歧，错误分析不能简单把所有 FP 都视作模型错误。后续需要保留 raw 多标注信息，用来判断某些“误报”是否其实落在医生分歧区域。

### Phase 8：ONNX 导出与 Demo

状态：未开始。

计划导出 YOLO / RT-DETR ONNX，并尝试 MMDetection 模型导出。Demo 使用上传胸片 -> 预处理 -> 推理 -> 可视化检测框流程，并明确非临床诊断用途。

阶段分析：Demo 不应过早做。建议等至少一个正式 baseline 和统一评估稳定后再做，否则 UI 会掩盖模型和数据链路还没定型的问题。ONNX 导出也要记录失败原因，尤其是 MMDetection / Transformer 模型可能遇到动态 shape 或自定义算子问题。

### Phase 9：交付物整理

状态：持续维护。

README 会随每个阶段更新，不再只保留“最新进展”。最终交付应包括阶段结论、五模型总表、关键曲线、bad case 图、可复现实验入口、环境说明和 Demo 链接。

阶段分析：从现在开始 README 要作为项目叙事的骨架维护，`docs/PLAN_PROGRESS.md` 保存细节，`docs/TASK_BREAKDOWN.md` 保存任务状态。这样后面新增结果时，不会覆盖前面阶段的判断依据。

## 当前关键决策

- 多标注融合策略：默认采用 `wbf`。
- 图像输入：当前成功链路使用 original PNG；CLAHE 是否进入正式训练需在 Phase 3/4 做小规模对比。
- 正式划分：当前采用 image-level stratified 70/15/15；patient/study-level 分组字段尚未确认，不能宣称 patient-level split。
- 训练环境：Kaggle P100 可用，但 PyTorch 需固定到 2.4 系列。
- 数据组织：Kaggle 中大图像只做 symlink，不复制到 `/kaggle/working`。

## 下一步

1. 跑 `kinjaza/phase4-yolo-baseline`，获得 YOLOv8n 50 epoch 正式 baseline。
2. 分析 baseline 的 per-class AP、训练曲线和小目标/长尾类别表现。
3. 根据 baseline 结果决定是否先做 1024 分辨率、小目标增强或 Mosaic 调整。
4. 在 YOLO baseline 稳定后，再扩展 RT-DETR / Faster R-CNN / RetinaNet / DINO。

## Demo

待 Phase 8 完成后部署。Demo 仅供技术演示与学习交流，不能用于临床诊断。

## License

[MIT](LICENSE) - 代码部分遵循 MIT 许可；VinDr-CXR 数据集遵循其原始使用协议，本项目不重新授权数据。
