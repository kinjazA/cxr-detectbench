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

> Phase 1 后续：DICOM 头信息（WindowCenter/Width/RescaleSlope/Intercept）缺失率统计
> 需要下载 DICOM 原图才能做，移至 Phase 2 开始时一并完成（避免重复下载）。

---

## Phase 0

- 0.1 本地仓库骨架完成（README / LICENSE / .gitignore / 目录结构）。2026-07-17
- Kaggle 环境版本待回填。