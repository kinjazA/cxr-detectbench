# CXR-DetectBench — 任务执行细分

> 本文件把 `CXR-DetectBench-Project-Plan.md` 的每个 Phase 拆成可执行的最小步骤。每完成一步把 `[ ]` 改成 `[x]`。实际执行的报错、版本、耗时、取舍记录在 `PLAN_PROGRESS.md`（执行日志），不在这里堆细节——这里只放结论性产出索引。

执行总原则：
- 训练在 **Kaggle** 跑；本仓库存脚本 / config / notebook 源码。
- CPU 数据处理与 GPU 训练拆成不同 Notebook。
- 所有训练脚本支持从 checkpoint 续训。
- 每次本地改动后提交推送；Kaggle 上产出的中间产物 Save Version 成 Kaggle Dataset 供下个 session 挂载。

---

## Phase 0：环境与工程基建

- [ ] 0.1 本地仓库骨架（README / LICENSE / .gitignore / 目录）✅ 已完成
- [ ] 0.2 在 Kaggle 新建 Notebook，Add Input 挂载 `vinbigdata-chest-xray-abnormalities-detection` 竞赛数据集
- [ ] 0.3 在 Kaggle Notebook 安装依赖：`pip install ultralytics pydicom opencv-python-headless albumentations ensemble-boxes pycocotools onnx onnxruntime gradio` + `mim install mmengine mmcv mmdet`
- [ ] 0.4 跑 `yolo checks` 与 `mim list`，确认 ultralytics / mmdet 版本，把版本号记入 PLAN_PROGRESS.md
- [ ] 0.5 在本仓库的 `scripts/requirements.txt` 冻结一版依赖（标明实际版本）
- [ ] 0.6 设计 checkpoint 持久化流程：每阶段训练结束 `Save Version` 成 Kaggle Dataset `cxr-detectbench-ckpt-<phase>`，下个 session Add Data 挂载续训；在 README/PLAN_PROGRESS 里登记每个 Dataset 的引用路径

**验收**：新开一个 Kaggle session，Add Data 能直接拿到 Phase 2 处理好的数据，无需重跑预处理。

---

## Phase 1：数据获取与初步核查

- [x] 1.1 挂载数据集，读取 `train.csv`，核对字段名与计划第 3 节表格 ✅ 字段完全一致
- [x] 1.2 统计实际 image 数量、正常/异常比例、各类别标注框数 ✅ 15000/10606/4394 与先验一致；标注者实为 17 位医生池、每图 3 人
- [x] 1.3 DICOM 头信息缺失率统计 ✅ 用 `sunhwan/.../dicom-metadata`，WindowCenter/Width 97.18%可用，RescaleSlope/Intercept 81.88%可用
- [x] 1.4 核对 14 类映射表与实际 `train.csv` ✅ 一致，`scripts/class_names.py` 已冻结

**验收**：✅ Phase 1 完成。数据核查小结已记入 PLAN_PROGRESS.md。

---

## Phase 2：数据预处理

**数据源调整（基于 Phase 1 与算力约束）：**
- 采用 `corochann/vinbigdata-chest-xray-original-png`（第三方已从 DICOM 转 PNG）
- 本项目重点：CLAHE 增强 + 多标注融合管道（WBF/NMS 消融是核心卖点）
- README 诚实说明数据来源

- [x] 2.1 写 `scripts/apply_clahe.py`：读 PNG → CLAHE 增强 → 灰度转 3 通道 → 存 `images_png/`；保留处理前后对比图（供 README）
- [x] 2.2 灰度转 3 通道方案选定：复制通道 / 伪彩色二选一，在脚本注释记理由
- [x] 2.3 写 `scripts/label_fusion.py`，产出三版 COCO 标注：
      - `labels_coco/raw/`：3 医生框全保留
      - `labels_coco/wbf/`：`ensemble-boxes` 的 `weighted_boxes_fusion`，IoU 阈值先 0.5
      - `labels_coco/nms/`：简单 NMS 去重
- [x] 2.4 **融合策略消融实验**：YOLOv8n 少量 epoch（如 20ep），分别在三版标注上训练，对比 val mAP，产出"融合方式 vs mAP"表 → 锁定最终融合策略（IoU 阈值可再做一次消融）
      - 2026-07-24 Kaggle 完成：raw 0.2812 / wbf 0.3210 / nms 0.3043（mAP@0.5），最终策略：`wbf`
- [x] 2.5 写 `scripts/convert_coco_yolo.py`：从选定融合标注生成完整 COCO json（MMDet 用）+ YOLO txt（Ultralytics 用）

**验收**：Phase 2 融合消融已完成并记录在 `docs/PLAN_PROGRESS.md`；本轮成功链路直接使用 original PNG，CLAHE 是否纳入正式训练留到 Phase 3/4 决策。

---

## Phase 3：EDA 与数据集划分

- [x] 3.1 类别分布柱状图（识别长尾，如 Pneumothorax）✅ `data/splits/class_image_distribution_by_split.svg`
- [x] 3.2 病灶框尺寸分布（识别小目标类别，指导输入分辨率）✅ `data/splits/bbox_median_area_by_class.svg` + `bbox_size_summary_by_class.csv`
- [x] 3.3 正常/异常图比例可视化 ✅ `data/splits/split_report.md` + `split_summary.csv`
- [ ] 3.4 按 patient/study 级别分层划分 70/15/15：(a) 同病人不跨集合；(b) 各类别三集合比例一致
      - 当前本地 metadata 未发现可确认 patient_id/study_id 的字段，因此不能声称完成 patient/study-level split。
      - 已完成替代方案：image-level stratified 70/15/15，且 14 个异常类图片级比例最大偏差 0.0083。
- [x] 3.5 划分结果存 `data/splits/{train,val,test}.csv`，后续所有脚本统一读取（五模型公平对比前提）

**验收**：Phase 3 第一版正式 image-level split 完成。patient/study-level split 需等待可靠分组字段确认后再做。

---

## Phase 4：Baseline 打通全流程

- [x] 4.0 写正式 YOLO 数据集准备脚本：读取 Phase 3 split + WBF COCO，生成 `images/{train,val,test}` symlink、`labels/{train,val,test}` txt 和 `data.yaml` ✅ `scripts/prepare_yolo_dataset.py`
- [x] 4.1 YOLOv8n 跑通"训练 → 验证 → 推理 → mAP"完整链路（不追求分数，只验流程）✅ 2026-07-24 Kaggle 3 epoch smoke test：mAP@0.5=0.1998 / mAP@0.5:0.95=0.1002
- [x] 4.1b YOLOv8n 正式 baseline：P100 配置 `imgsz=640, epochs=50, batch=16, workers=4, cache=False`，6.874 小时完成；best val mAP@0.5=0.3692 / mAP@0.5:0.95=0.1931
- [x] 4.2 确认 COCO 标注可被 pycocotools 2.0.10 正确加载评估；已用实际 WBF JSON + 2,250 张 val split 验证
- [ ] 4.3 确认可视化脚本可用：预测框 + GT 框对比图

**验收**：训练和框架原生 val 已完成；待统一预测导出/评估和 GT 对比可视化后关闭 Phase 4。

---

## Phase 5：五模型训练矩阵

> 五模型统一用 Phase 3 划分、统一用 Phase 6 评估脚本。

- [ ] 5.1 **Faster R-CNN (R50-FPN)** — `configs/mmdet_faster_rcnn_cxr.py`
      - baseline: batch=8, lr=0.01, 12ep
      - 针对性优化：k-means 重聚类 anchor / 调 RPN 正负样本采样比 / (可选)Cascade 对比
- [ ] 5.2 **RetinaNet (R50-FPN)** — `configs/mmdet_retinanet_cxr.py`
      - **重点**：Focal Loss α/γ 消融，对比不同 γ 对稀有类别召回率影响
- [ ] 5.3 **YOLO (v8/v11/最新稳定)** — `configs/yolo_cxr.yaml`
      - imgsz=1024, epochs=100；稀有类别 copy-paste 增强；最后 10% epoch 关 mosaic
- [ ] 5.4 **RT-DETR** — 用 `rtdetr-l.pt`
      - imgsz=1024, epochs=72 起；调 decoder query 数；调去噪训练比例
- [ ] 5.5 **DINO** — `configs/mmdet_dino_cxr.py`
      - **加载 COCO 预训练权重微调，不从零训**；10-20ep + early stopping；冻结 backbone 前几层只微调 neck+head
- [ ] 5.6 每模型：保存训练曲线（loss/mAP vs epoch）+ 最优 checkpoint 存为 Kaggle Dataset
- [ ] 5.7 每模型：至少 1 轮针对性优化前后对比，记入结果表
- [ ] 5.8 汇总五模型日志 / 超参 / 实际耗时到统一表格（供排期校对与 README）

**验收**：五模型各有可复现脚本+config、最优 checkpoint、训练曲线、≥1 项针对性优化对比数据。

> 算力兜底：若某模型超预算，优先保证 YOLO / RT-DETR / Faster R-CNN 跑满；RetinaNet / DINO 可降 epoch 并在 PLAN_PROGRESS 说明是算力取舍。

---

## Phase 6：统一评估体系

- [x] 6.0 冻结跨框架预测契约与评估协议：`docs/EVALUATION_PROTOCOL.md`
- [x] 6.1 实现标准 COCO 指标：mAP@0.5 / mAP@0.5:0.95 / per-class AP；待真实 baseline predictions 回填结果
- [x] 6.2 实现同一 COCO 101 点插值协议下的 AP@0.4，明确其为领域补充指标而非比赛指标
- [ ] 6.3 **FROC 曲线** `scripts/eval_froc.py`：类别感知病灶级计算和固定 FP/image operating points 已实现并测试；待真实预测与多模型叠加图
- [ ] 6.4 图像级二分类指标：检测结果聚合为"该图是否有异常"，算 AUC / 敏感度 / 特异度
- [ ] 6.5 参数量 / FLOPs / 推理 FPS 对比，画精度-速度 Pareto 图
- [ ] 6.6 汇总总表：模型 × (mAP@0.5, mAP@0.5:0.95, mAP@0.4, FROC敏感度@某FP率, FPS, 参数量)

**验收**：五模型对比总表 + FROC 曲线 + Pareto 图（README 核心图表）。

---

## Phase 7：错误分析

- [ ] 7.1 按类别拆分 AP，找最难/最易类别并解释（形态/弥漫性/样本量）
- [ ] 7.2 **FP 溯源**：区分"模型真的错" vs "该区域仅 1/3 医生标注被判 FP 但可能合理"，回查三份原始标注判断
- [ ] 7.3 **FN 溯源**：小尺寸漏检 vs 与解剖结构（肋骨/心影）重叠漏检，各挑 3-5 张 GT/预测对比图
- [ ] 7.4 图像级误报分析：统计模型在完全正常胸片上误报"有异常"的比例（过度诊断风险）
- [ ] 7.5 "四模型在同一张难例上的对比图"，直观展示能力差异

**验收**：`outputs/bad_cases/` 下分类整理的错误样本图集 + 结构化错误归因文字总结。

---

## Phase 8：ONNX 导出与 Demo

- [ ] 8.1 YOLO / RT-DETR 用 Ultralytics export 接口导出 ONNX
- [ ] 8.2 尝试 Faster R-CNN / DINO 的 ONNX 导出，如实记录动态 shape / 自定义算子问题；无法完全导出的说明处理方式（固定输入分辨率 / 仅导 backbone+neck）
- [ ] 8.3 写 `demo/app.py`：上传胸片（PNG，可选 DICOM）→ 自动窗宽窗位/CLAHE → 推理 → 可视化框+类别+置信度
- [ ] 8.4 Demo 界面**必须含免责声明**："本 Demo 仅供技术演示与学习交流，不能用于临床诊断"
- [ ] 8.5 记录 PyTorch GPU 推理 vs ONNXRuntime CPU 推理延迟对比
- [ ] 8.6 部署到 Hugging Face Spaces（Gradio SDK）

**验收**：可运行的 Gradio Demo（HF Spaces）+ ONNX 导出踩坑记录 + 延迟对比数据。

---

## Phase 9：README 与交付物整理

- [ ] 9.1 按 README 框架回填实际数字：项目背景 → 数据集与预处理（含融合实验）→ 五模型方法对比 → 训练细节 → 结果（总表+Pareto+FROC）→ 错误分析 → ONNX/Demo → 局限与后续方向
- [ ] 9.2 整理简历用成果总结（把 Phase 6 总表数字填进话术模板）
- [ ] 9.3 检查所有图表 / 代码 / checkpoint 链接在 README 里可访问
- [ ] 9.4 简历成果 Checklist 全部就绪（见计划第 6 节）

**验收**：完整 README.md + 可复现仓库结构 + 一段含具体数字的简历成果总结。
