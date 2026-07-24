# CXR-DetectBench — 执行日志

> 执行过程中"真实发生的事"写在这里：版本号、报错与解决、实际训练耗时、算力取舍、与计划的差异。
> 不写空话，只写事实和数字。这部分是 README 与简历成果总结的事实源。

---

## 环境基线

- 开发机：Windows 11
- Git：2.53.0.windows.2
- SSH：ed25519 密钥，已认证为 GitHub 用户 `kinjazA`
- 远程仓库：git@github.com:kinjazA/cxr-detectbench.git
- Kaggle CLI：2.2.3（`python -m kaggle`，`kaggle.exe` 未入 PATH，用 module 方式调用）
- Kaggle 账号：`kinjaza`（auth_method=ACCESS_TOKEN，token 存 `~/.kaggle/access_token`）
- 训练环境：Kaggle Notebook（P100 16GB 或 T4×2，每周 30h GPU）

### Kaggle 连通性验证（2026-07-17）
- `kaggle competitions list -s vinbigdata` 成功返回，`userHasEntered=True`
- `kaggle competitions download ... -f train.csv` 成功下载 1.79MB
- 认证 ✅ + 网络 ✅ + 竞赛规则已接受 ✅ 三者一次验证通过

### 依赖实际版本
> 待 Phase 0.4 在 Kaggle 上 `pip list` 后回填。

---

## Phase 1（数据核查）

train.csv 已下载到 `data/raw/train.csv`（67,914 行 / 15,000 unique image_id）。

**核查结论（与计划第 3 节交叉验证）：**

| 核查项 | 计划预期 | 实际 | 结论 |
|---|---|---|---|
| 总 image_id | 15,000 | 15,000 | ✅ |
| No finding 图 | 10,606 | 10,606 | ✅ |
| 含异常标注图 | 4,394 | 4,394 | ✅ |
| train.csv 字段 | 8 列 | image_id,class_name,class_id,rad_id,x_min,y_min,x_max,y_max | ✅ 一致 |

**⚠️ 与计划描述不一致（已修正 README/任务文档叙述）：**
- 计划第 3 节"每张训练图由 3 名放射科医生独立标注" → **实际是 17 位放射科医生池（R1–R17），每张异常图由其中恰好 3 位标注**。
- 4,394 张异常图全部正好 3 位 rad_id 标注，已验证。
- 影响：对融合脚本逻辑无影响（仍按 image_id 聚合多框做 WBF/NMS），但 README/简历叙述应改为"17 位放射科医生、每图 3 人独立标注"，标注者多样性更高反而更好。

**类别长尾分布（标注框数，不含 No finding）：**
- 最多：Aortic enlargement 7162、Pleural thickening 4842、Cardiomegaly 5427、Pulmonary fibrosis 4655
- 最少（长尾）：Pneumothorax 226、Atelectasis 279、Consolidation 556、Calcification 960
- 印证计划判断："Pneumothorax 样本稀少"，预告 RetinaNet Focal Loss 消融价值。

**No finding 行的 x_min/y_min/x_max/y_max 为空**：符合"图像级标签不参与 box 回归"设定，处理时需跳过这些行的框坐标。

**Phase 1.3 DICOM 头信息核查（已完成，用 `sunhwan/vinbigdata-chest-xray-dicom-metadata`）：**

| 字段 | 缺失数 / 总数 | 缺失率 | 可用率 |
|---|---|---|---|
| WindowCenter | 423 / 15000 | 2.82% | 97.18% |
| WindowWidth | 423 / 15000 | 2.82% | 97.18% |
| RescaleSlope | 2718 / 15000 | 18.12% | 81.88% |
| RescaleIntercept | 2718 / 15000 | 18.12% | 81.88% |

- WindowCenter/WindowWidth 同时具备的占 97.18%，成对缺失 2.82%
- RescaleSlope/RescaleIntercept 同时具备的占 81.88%，成对缺失 18.12%
- 原始 DICOM 分辨率不统一（主要有 2430×1994 / 3072×3072 / 2880×2304 等 10+ 种尺寸）

**对 Phase 2 的影响（用现成 PNG 数据集路线）：**
- 采用 `corochann/vinbigdata-chest-xray-original-png`（第三方已从 DICOM 转 PNG）
- Phase 2 脚本只需做 CLAHE + 灰度转 3 通道，不再需要 pydicom 解码与窗宽窗位变换
- README 诚实说明："采用第三方预转换 PNG，本项目重点在 CLAHE 增强与多标注融合管道"

---

## Phase 2（数据预处理脚本实现）

**脚本实现完成（2026-07-17）：**

- `scripts/apply_clahe.py` ✅ 
  - 输入：PNG（来自 corochann/...-original-png）
  - 流程：CLAHE (clip_limit=2.0, tile_size=8) → 灰度转 3 通道（复制通道方案）
  - 输出：处理后 PNG + 对比图（Phase 2.2）
  - 灰度转 3 通道理由已在脚本注释说明：保留原始灰度语义、适配 ImageNet 预训练、无额外计算开销

- `scripts/label_fusion.py` ✅
  - 产出三版 COCO 标注：raw（全保留）/ wbf（ensemble-boxes weighted_boxes_fusion）/ nms（简单 NMS）
  - 支持 IoU 阈值调整（默认 0.5，Phase 2.4 消融时可做阈值扫描）
  - 按 class_id 分组融合（per-class WBF/NMS）
  - 过滤 No finding（class_id=14）

- `scripts/convert_coco_yolo.py` ✅
  - COCO bbox [x, y, w, h] → YOLO <cls> <cx> <cy> <w> <h> (normalized)
  - 每张图一个 .txt 文件

**Phase 2 Kaggle 实际执行完成（2026-07-24）：**

- Kaggle kernel：`kinjaza/phase2-preprocessing`
- 最新状态：`COMPLETE`
- 运行环境：Kaggle Notebook，Tesla P100-PCIE-16GB；P100 下安装 `torch==2.4.0` / `torchvision==0.19.0`
- 数据源：直接使用 `corochann/vinbigdata-chest-xray-original-png` 的 original PNG；本轮没有额外跑 CLAHE 增强
- 路径确认：
  - PNG train dir：`/kaggle/input/datasets/corochann/vinbigdata-chest-xray-original-png/train`
  - `data/processed/images_png` 成功 symlink 到上述目录
  - 识别 PNG：15,000 张
- 修复过的关键问题：
  - Kaggle mount 实际路径是 `/kaggle/input/datasets/...`，不是早期脚本假设的 `/kaggle/input/<dataset>/...`
  - `train_meta.csv` 中 `dim0=height`、`dim1=width`，已在 `label_fusion.py` 显式处理
  - 消融脚本不能复制 15,000 张 PNG 到 `/kaggle/working`，否则触发 `No space left on device`；已改为在 YOLO split 目录下逐文件 symlink
  - 训练异常不再静默写成 `mAP=0.0`，数据/标签计数不一致会直接报错
- 三版 COCO 标注产出：
  - raw：15,000 images / 36,096 annotations
  - wbf：15,000 images / 23,934 annotations
  - nms：15,000 images / 24,034 annotations
- YOLO 消融数据校验：
  - split：12,000 train / 3,000 val
  - image links：12,000 train / 3,000 val / 0 skipped
  - labels：12,000 train txt / 3,000 val txt
  - Ultralytics scan：0 corrupt
- Phase 2.4 融合策略消融结果（YOLOv8n，20 epochs）：

| fusion_mode | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| raw | 0.2812 | 0.1385 |
| wbf | **0.3210** | **0.1662** |
| nms | 0.3043 | 0.1403 |

**结论：最终融合策略锁定为 `wbf`。**

**剩余注意事项：**
- 本轮 full-data 三组消融耗时很长（约 9.9 小时量级），后续不应频繁全量重跑；调试优先加小样本/低 epoch smoke test。
- val 中背景图比例高（日志示例：3,000 val images / 2,121 backgrounds），符合 VinBigData 大量 No Finding 的特征，但 Phase 3 正式 split 时仍需做类别/正负比例校验。
- P100 下 `torchaudio` 与 PyTorch 版本有依赖警告，但不影响当前检测训练；后续可考虑卸载 `torchaudio` 或固定更干净的 requirements。

---

## Phase 0

- 0.1 本地仓库骨架完成（README / LICENSE / .gitignore / 目录结构）。2026-07-17
- Kaggle 环境版本待回填。
