# CXR-DetectBench：胸部 X 光多范式目标检测基准系统 —— 执行手册

> **给 Claude Code 的说明**：本文档是本项目的完整执行计划，按 Phase 0-9 顺序推进。每个 Phase 下有 `- [ ]` 格式的任务清单和验收标准（Acceptance Criteria）。请按顺序执行，完成一项就把对应 checkbox 改成 `- [x]`，并在每个 Phase 末尾的 `### Notes` 小节里记录实际情况（用的分辨率、遇到的报错和解决方式、实际训练耗时等），这些记录后续会直接用于撰写 README 和简历成果总结，所以要如实、具体地写。如果某一步因为 Kaggle 环境限制（网络/显存/时长）无法按计划执行，请在 Notes 里说明替代方案，不要跳过不记录。

> **状态同步（2026-07-24）**：Phase 1 数据核查、Phase 2 WBF 融合消融、Phase 3 image-level 固定划分、Phase 4 YOLOv8n baseline，以及该 baseline 的统一 COCO/FROC 评估均已完成。完整事实日志见 `docs/PLAN_PROGRESS.md`，不可覆盖的 Phase 4 指标见 `docs/RESULTS_PHASE4_YOLO.md`。patient/study-level split、GT/预测可视化、跨模型比较、图像级 AUC、速度指标和部署仍未完成。本周 Kaggle GPU 配额耗尽，不启动新实验。

---

## 0. 项目概览

- **项目名称（建议简历/仓库用名）**：CXR-DetectBench —— 胸部 X 光多范式目标检测基准系统
- **一句话定位**：在真实临床胸部 X 光数据集 VinDr-CXR 上，系统化对比两阶段 / 一阶段 anchor-based / 一阶段 anchor-free / 端到端 Transformer 五种目标检测范式，完成从数据处理、多标注融合、模型训练调优、统一评估、错误分析到部署 Demo 的完整闭环。
- **核心能力体现点**（写 README 和简历时始终围绕这几点展开）：
  1. 自主构建医疗影像数据管道（DICOM 解码、窗宽窗位、CLAHE、多标注者融合）
  2. 覆盖检测技术演进全谱系的多模型对比能力，而非单一模型调参
  3. 领域适配的评估体系设计（mAP@0.4、FROC，而不是照搬通用 CV 指标）
  4. 结构化错误分析方法论（区分标注噪声导致的 FP vs 真实漏检）
  5. 工程闭环能力（ONNX 导出、Demo 部署、算力受限下的取舍与说明）
- **运行环境**：Kaggle Notebook。GPU 配额：每周 30 小时（P100 16GB 或 T4×2 二选一，配额消耗相同），单次 session 上限约 9-12 小时。**所有阶段设计都要考虑这个约束，训练任务要能断点续训。**

---

## 1. 技术栈与依赖

```bash
# 目标检测框架
pip install ultralytics              # YOLO 系列 + RT-DETR
pip install -U openmim
mim install mmengine mmcv mmdet       # Faster R-CNN / RetinaNet / DINO

# 医疗影像与数据处理
pip install pydicom                   # DICOM 解码
pip install opencv-python-headless
pip install albumentations            # 数据增强，含 CLAHE
pip install ensemble-boxes            # WBF (Weighted Boxes Fusion) 实现，用于多标注融合
pip install pycocotools               # COCO 格式评估

# 评估与可视化
pip install scikit-learn              # 分层划分 StratifiedKFold
pip install pandas numpy matplotlib seaborn

# 部署
pip install onnx onnxruntime
pip install gradio
```

> Claude Code 执行时请先运行 `pip list | grep -E "ultralytics|mmdet|mmcv"` 确认实际安装版本，本文档里涉及的模型名/配置文件名（如 `yolov8m.pt`、`dino-4scale_r50_8xb2-12e_coco.py`）可能随版本迭代变化，如与本地实际不符，以 `mim search`、`yolo checks`、或对应框架的 model zoo 为准，并在 Notes 里记录实际使用的版本号。

---

## 2. 目录结构规划

```
cxr-detectbench/
├── data/
│   ├── raw/                      # 原始 DICOM + train.csv
│   ├── processed/
│   │   ├── images_png/           # 窗宽窗位+CLAHE处理后的PNG图
│   │   ├── labels_coco/          # COCO格式标注（多个版本：raw/wbf/nms融合）
│   │   └── labels_yolo/          # YOLO格式标注
│   └── splits/                   # train/val/test 划分文件（按patient分层）
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_label_fusion_ablation.ipynb
│   ├── 04_train_yolo.ipynb
│   ├── 05_train_rtdetr.ipynb
│   ├── 06_train_faster_rcnn.ipynb
│   ├── 07_train_retinanet.ipynb
│   ├── 08_train_dino.ipynb
│   ├── 09_unified_evaluation.ipynb
│   ├── 10_error_analysis.ipynb
│   └── 11_onnx_export_demo.ipynb
├── configs/
│   ├── yolo_cxr.yaml
│   ├── mmdet_faster_rcnn_cxr.py
│   ├── mmdet_retinanet_cxr.py
│   └── mmdet_dino_cxr.py
├── scripts/
│   ├── dicom_to_png.py
│   ├── label_fusion.py           # WBF融合逻辑
│   ├── convert_coco_yolo.py
│   ├── eval_froc.py              # FROC曲线计算
│   └── error_analysis.py
├── outputs/
│   ├── checkpoints/               # 定期存为 Kaggle Dataset 输出
│   ├── eval_results/
│   └── bad_cases/                 # 可视化的错误样本
├── demo/
│   └── app.py                    # Gradio demo
└── README.md
```

---

## 3. 数据集信息（供 Claude Code 直接引用，避免重复查证）

- **来源**：Kaggle 竞赛 `vinbigdata-chest-xray-abnormalities-detection`，在 Kaggle Notebook 中用 "Add Input" 直接挂载，无需下载到本地。
- **规模**：训练集 15,000 张（10,606 张 "No finding" 正常片 + 4,394 张含异常标注的片），DICOM 格式。官方测试集 3,000 张标签隐藏（本项目不使用，自行在训练集内划分 train/val/test）。
- **标注方式**：由 **17 位放射科医生（R1–R17）**独立标注，每张异常图由其中恰好 3 位完成（这是多标注融合环节的数据基础；Phase 1 核查后已从初稿"固定 3 名医生"修正为此说法）。
- **类别列表（14类异常 + No finding，class_id 以实际下载的 `train.csv`/`classes.csv` 表头为准，下表为已知的标准映射，Claude Code 首次读取数据后请核对并在 Notes 中确认）**：

| class_id | 类别英文名 | 备注 |
|---|---|---|
| 0 | Aortic enlargement | 样本相对较多，目标较大，预期容易检测 |
| 1 | Atelectasis | |
| 2 | Calcification | |
| 3 | Cardiomegaly | 样本较多，边界清晰 |
| 4 | Consolidation | |
| 5 | ILD (Interstitial lung disease) | 边界模糊，弥漫性病变，预期较难 |
| 6 | Infiltration | |
| 7 | Lung Opacity | |
| 8 | Nodule/Mass | 小目标，预期较难 |
| 9 | Other lesion | 类别定义宽泛 |
| 10 | Pleural effusion | |
| 11 | Pleural thickening | |
| 12 | Pneumothorax | 样本稀少，长尾类别 |
| 13 | Pulmonary fibrosis | |
| 14 | No finding | 图像级标签，代表该图无异常，不作为检测框类别参与box回归 |

- **train.csv 关键字段（预期结构，以实际文件为准）**：`image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max`。同一 `image_id` + 同一病灶区域会出现最多3行（来自3位医生），这是多标注融合要处理的核心对象。

---

## 4. Phase 任务清单

### Phase 0：环境与工程基建

- [x] 在 Kaggle 创建项目 Notebook，挂载竞赛数据与所需 PNG / metadata 输入
- [ ] 安装第1节列出的全部依赖，运行 `yolo checks` 和 `mim list` 确认环境可用（YOLO 环境已实测；MMDetection 尚未实测）
- [x] 建立第2节目录结构
- [ ] 设计 checkpoint 持久化方案：训练中间产物定期 `Save Version` 为 Kaggle Dataset，供后续 session 挂载续训
- [x] 数据处理（CPU密集）与模型训练（GPU密集）拆分成不同 Notebook，避免浪费GPU配额

**验收标准**：能在一个新开的 Notebook session 里，通过 Add Data 直接拿到 Phase 2 处理好的数据，无需重新跑预处理。

### Notes
（Claude Code 在此记录实际环境版本、遇到的安装问题等）

---

### Phase 1：数据获取与初步核查

- [x] 挂载数据集，读取 `train.csv`，核对实际字段名与第3节表格一致
- [x] 统计实际的 image 数量、正常/异常图比例、各类别标注框数量，与 10,606/4,394 先验交叉验证
- [x] 核查 DICOM metadata 中 `WindowCenter`/`WindowWidth`/`RescaleSlope`/`RescaleIntercept` 的可用率

**验收标准**：产出一份数据核查小结（数量、字段、异常样本），确认与文档记录一致或记录差异。

### Notes

---

### Phase 2：数据预处理

- [ ] **DICOM → PNG**：按窗宽窗位（VOI LUT / WindowCenter+WindowWidth）转换到可视范围，再做 CLAHE 增强，保存为 `data/processed/images_png/`；保留一份处理前后对比图（用于 README 展示）
- [x] **多标注融合消融实验**（`scripts/label_fusion.py`）：分别产出三个版本的标注
  - `labels_coco/raw/`：3位医生的框全部保留
  - `labels_coco/wbf/`：用 `ensemble-boxes` 库的 `weighted_boxes_fusion` 按 IoU 阈值融合成共识框（IoU阈值建议先尝试 0.5，可再做一次阈值消融）
  - `labels_coco/nms/`：简单 NMS 去重
  - 用一个轻量模型（YOLOv8n，少量 epoch）分别在三版标注上快速训练，对比验证集 mAP，确定最终采用哪种融合策略，**把这组对比结果记录下来，这是本项目最核心的实验之一**
- [x] **格式转换**：从最终选定的融合标注版本，生成完整的 COCO json（给 MMDetection 用）和 YOLO txt（给 Ultralytics 用），`scripts/convert_coco_yolo.py`
- [ ] 灰度图转3通道处理（复制通道或伪彩色映射二选一，记录选择理由）

**验收标准**：`data/processed/` 下有完整的 PNG 图像 + 两种格式标注；有一份"融合策略消融实验"的结果记录（表格：融合方式 vs mAP）。

### Notes

---

### Phase 3：EDA 与数据集划分

- [x] 类别分布柱状图（识别长尾类别，如 Pneumothorax）
- [x] 病灶框尺寸分布（识别哪些类别是小目标，指导后续输入分辨率选择）
- [x] 正常/异常图比例可视化
- [ ] **按病人/study 级别分层划分** train/val/test（比如 70/15/15），保证：(a) 同一病人不跨集合出现；(b) 各类别在三个集合里的比例基本一致
- [x] 划分结果存成 `data/splits/` 下的 csv 索引文件，供所有后续训练脚本统一读取（保证五个模型用的是完全相同的数据划分，这是公平对比的前提）

**验收标准**：`data/splits/train.csv / val.csv / test.csv`，附一份类别分布在三个集合里基本一致的验证图。

### Notes

---

### Phase 4：Baseline 打通全流程

- [x] 用 YOLOv8n 跑通"训练 → 验证 → 推理 → mAP计算"完整链路；50 epoch baseline 的框架原生 val mAP50=0.3692、mAP50-95=0.1931
- [x] 确认 COCO 格式标注可以被 pycocotools 正确加载评估，并在真实 `best.pt` 上完成统一评估：mAP50-95=0.1812、AP40=0.3806、AP50=0.3499、AP75=0.1694
- [ ] 确认可视化脚本（画预测框+GT框对比图）可用

**验收标准**：产出一次完整的、指标可信的 baseline 结果（不追求分数，只验证流程正确）。

### Notes

---

### Phase 5：五模型训练矩阵

> 五个模型统一用 Phase 3 产出的同一份 train/val/test 划分，统一用 Phase 6 的评估脚本，确保公平对比。
>
> 下表的起始配置是待验证候选，不是已经在 Kaggle P100 上确认可运行的事实。每个新框架必须先完成版本、数据路径、显存和短 smoke test 核验；不得因为计划表中写了某个 batch、分辨率或模型名就假定它可用。

| 模型 | 框架/入口 | 建议起始配置 | 本项目的针对性优化方向 |
|---|---|---|---|
| Faster R-CNN (R50-FPN) | MMDetection，参考 `configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py` | batch=8, lr=0.01, 12 epoch起步 | 针对小病灶用k-means重新聚类anchor尺寸；调整RPN正负样本采样比例；可加做Cascade R-CNN对比 |
| RetinaNet (R50-FPN) | MMDetection，参考 `configs/retinanet/retinanet_r50_fpn_1x_coco.py` | 同上 | **重点做 Focal Loss 的 α/γ 消融实验**，对比不同γ对稀有类别（Pneumothorax等）召回率的影响 |
| YOLO (v8/v11/最新稳定版) | Ultralytics，`yolo detect train` | 已冻结 YOLOv8n baseline：imgsz=640, epochs=50, batch=16；下一候选 imgsz=896，需 P100 显存 smoke 确认 | 先验证高分辨率对小目标与高 IoU 的收益，再单独测试 copy-paste 或训练后期关闭 mosaic |
| RT-DETR | Ultralytics，`model=rtdetr-l.pt` | imgsz=1024, epochs=72起步 | 调整decoder query数量（病灶数通常不多，可减少query降低计算量）；调整去噪训练比例 |
| DINO | MMDetection，参考 `configs/dino/dino-4scale_r50_8xb2-12e_coco.py` | **加载COCO预训练权重微调，不从零训练**，10-20 epoch + early stopping | 小数据集上容易过拟合，尝试冻结backbone前几层only微调neck+head；控制训练时长 |

- [ ] 每个模型：训练 → 记录训练曲线（loss/mAP随epoch变化）→ 保存最优checkpoint为Kaggle Dataset
- [ ] 每个模型：至少完成1轮"针对性优化"对比实验（不是只跑默认参数），把优化前后的指标变化记录进结果表
- [ ] 汇总五个模型的训练日志、超参、实际训练耗时到统一表格，供后续排期分析和README使用

**验收标准**：五个模型都有可复现的训练脚本/config、最优checkpoint、训练曲线图，以及至少一项针对性优化的对比数据。

### Notes
（记录每个模型的实际训练耗时，用于跟第7节的GPU时间预算做校对）

---

### Phase 6：统一评估体系

- [x] 标准 COCO 指标：mAP@0.5、mAP@0.5:0.95、per-class AP；已在 YOLO baseline 上实测
- [x] **领域标准 mAP@IoU>0.4**（PASCAL VOC风格，对齐 VinDr-CXR 相关文献的评估惯例）；YOLO baseline AP40=0.3806
- [x] **FROC 曲线**（`scripts/eval_froc.py`）：类别感知病灶级 micro 定义已实现并在 YOLO baseline 上实测
- [ ] 图像级二分类指标：把检测结果聚合成"该图是否有异常"的判断，计算 AUC/敏感度/特异度，对应医生"看一眼判断有无问题"的临床工作流
- [ ] 参数量/FLOPs/推理FPS对比，画一张"精度-速度" Pareto图
- [ ] 汇总成一张总表：模型 × (mAP@0.5, mAP@0.5:0.95, mAP@0.4, FROC敏感度@某FP率, FPS, 参数量)

**验收标准**：一张完整的五模型对比总表 + FROC曲线图 + Pareto图，是README的核心图表。

### Notes

---

### Phase 7：错误分析

- [ ] 按类别拆分AP，找出最难/最容易的类别，给出解释（形态是否规则、是否弥漫性、样本量是否充足）
- [ ] **FP溯源**：区分"模型真的检测错了"vs"该区域只有1/3医生标注、被判定为FP但可能是合理检测"——需要回查原始三份标注做判断
- [ ] **FN溯源**：小尺寸病灶漏检 vs 与解剖结构（肋骨/心影）重叠导致漏检，各挑3-5张代表性图叠加GT/预测框对比
- [ ] **图像级误报分析**：统计模型在完全正常胸片上误报"有异常"的比例（对应临床"过度诊断"风险）
- [ ] 产出"四模型在同一张难例上的对比图"，直观展示模型能力差异

**验收标准**：`outputs/bad_cases/` 下有分类整理好的错误样本可视化图集 + 一份结构化的错误归因文字总结。

### Notes

---

### Phase 8：ONNX 导出与 Demo

- [ ] YOLO / RT-DETR 用 Ultralytics 自带 export 接口导出 ONNX
- [ ] 尝试 Faster R-CNN / DINO 的 ONNX 导出，如实记录遇到的动态shape/自定义算子问题，无法完全导出的部分说明处理方式（如仅导出backbone+neck，或固定输入分辨率绕过动态shape）
- [ ] 用 Gradio 搭建 Demo（`demo/app.py`）：上传胸片（PNG，可选DICOM）→ 自动窗宽窗位/CLAHE预处理 → 推理 → 可视化检测框+类别+置信度
- [ ] **Demo界面必须包含明确的免责声明**："本Demo仅供技术演示与学习交流，不能用于临床诊断"
- [ ] 记录 PyTorch GPU 推理 vs ONNXRuntime CPU 推理的延迟对比

**验收标准**：可运行的 Gradio Demo + ONNX 导出踩坑记录 + 延迟对比数据。

### Notes

---

### Phase 9：README 与交付物整理

- [ ] 按以下结构撰写 README.md：项目背景 → 数据集与预处理（含多标注融合实验）→ 五模型方法对比 → 训练细节 → 实验结果（总表+Pareto图+FROC曲线）→ 错误分析（bad case图集）→ ONNX导出与Demo → 局限性与后续方向
- [ ] 整理简历用的成果总结（把 Phase 6 的总表数字填进简历话术模板）
- [ ] 检查所有图表、代码、checkpoint链接在README里都能正常访问

**验收标准**：完整的 README.md + 可复现的代码仓库结构 + 一段可直接使用的简历成果总结（含具体数字，不留占位符）。

### Notes

---

## 5. GPU 时间预算与排期（按 Kaggle 30小时/周配额）

| 周次 | 内容 | 预估GPU小时 |
|---|---|---|
| Week 1 | Phase 0-3（数据处理+EDA+划分，几乎不耗GPU；含融合消融实验的小规模训练） | ~3小时 |
| Week 2 | Phase 5：YOLO + RT-DETR 训练调优 | ~20小时 |
| Week 3 | Phase 5：Faster R-CNN + RetinaNet 训练调优 | ~20小时 |
| Week 4 | Phase 5：DINO微调 + Phase 6统一评估 + Phase 7错误分析 | ~15小时 |
| Week 5 | Phase 8-9：ONNX导出、Demo、README | ~5小时 |

> 实际执行中如某模型训练超出预算，优先保证 YOLO / RT-DETR / Faster R-CNN 三个跑满优化，RetinaNet / DINO 可以适当降低epoch数并在Notes里说明是算力权衡下的取舍——**诚实记录比强行跑满更重要**，这也是简历里"具备资源规划意识"的实证。

---

## 6. 简历成果产出 Checklist

项目完成后，确认以下数字/材料都已就绪，用于填入简历话术：

- [ ] 五模型 mAP@0.5 / mAP@0.5:0.95 / mAP@0.4 对比数字
- [ ] FROC 曲线关键点（如"每图1个假阳性时的敏感度"）
- [ ] Focal Loss消融实验：稀有类别召回率提升的具体百分点
- [ ] 多标注融合消融实验：不同融合策略的mAP差异
- [ ] 五模型推理速度(FPS)/参数量对比
- [ ] ONNX导出成功率及PyTorch vs ONNXRuntime延迟对比数字
- [ ] GitHub仓库链接 + Demo可访问链接（部署了HuggingFace Spaces）

---

## 7. 风险与注意事项

1. **数据使用协议**：VinDr-CXR 数据仅限学术/非商业研究用途，README中需注明。
2. **DICOM头信息缺失**：部分图像窗宽窗位信息可能不完整，需要设计默认兜底逻辑（如用全局像素分布做min-max归一化代替）。
3. **DINO 在小数据集上过拟合风险**：15,000张图对DINO这种大模型规模偏小，务必用预训练权重微调+early stopping，不要从零训练。
4. **Kaggle Session中断风险**：所有训练脚本要支持从checkpoint恢复，避免9-12小时上限导致进度丢失。
5. **多标注融合的主观性**：WBF的IoU阈值选择本身是一个需要论证的超参数，不要默认选一个值就不做验证。

---

*本文档与 `docs/TASK_BREAKDOWN.md` 共同维护计划状态；`docs/PLAN_PROGRESS.md` 保存事实日志，`docs/RESULTS_PHASE4_YOLO.md` 等阶段结果文档保存不可覆盖的指标与分析。执行过程中如有计划调整，请同步更新这些对应文档。*
