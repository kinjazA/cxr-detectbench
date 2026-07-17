# CXR-DetectBench
## 胸部 X 光多范式目标检测基准系统

> 🚧 **项目状态：在建** — 正在按 [docs/TASK_BREAKDOWN.md](docs/TASK_BREAKDOWN.md) 推进。README 当前为框架版本，数字与图表将在各阶段产出后回填。

在真实临床胸部 X 光数据集 **VinDr-CXR** 上，系统化对比五种主流目标检测范式，完成从数据处理 → 多标注融合 → 多模型训练调优 → 统一评估 → 错误分析 → ONNX 导出与 Demo 部署的完整闭环。

| 范式 | 模型 | 框架 |
|---|---|---|
| 两阶段 anchor-based | Faster R-CNN (R50-FPN) | MMDetection |
| 一阶段 anchor-based | RetinaNet (R50-FPN) | MMDetection |
| 一阶段 anchor-free | YOLOv8 / 最新 | Ultralytics |
| 端到端 Transformer | RT-DETR | Ultralytics |
| 端到端 Transformer | DINO | MMDetection |

## 核心能力

1. **医疗影像数据管道**：DICOM 解码、窗宽窗位 (VOI LUT)、CLAHE 增强、多标注者 (3 名放射科医生) 融合
2. **全谱系检测范式对比**：覆盖检测技术演进全谱系的五模型横评，而非单模型调参
3. **领域适配评估体系**：mAP@0.4、FROC 曲线、图像级 AUC，对齐 VinDr-CXR 临床评估惯例，而非照搬通用 CV 指标
4. **结构化错误分析方法论**：区分"标注噪声导致的 FP"与"真实漏检"
5. **工程闭环**：ONNX 导出、Gradio Demo 部署、算力受限下的取舍与诚实说明

## 数据集

[Kaggle: vinbigdata-chest-xray-abnormalities-detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

- 15,000 张训练胸片（10,606 张 No finding + 4,394 张含异常标注），DICOM 格式
- 每张图由 3 名放射科医生独立标注（多标注融合的数据基础）
- 14 类胸部异常 + No finding

> ⚠️ **数据使用协议**：VinDr-CXR 数据仅限学术 / 非商业研究用途。本仓库不包含任何原始 DICOM 或标注文件，仅含处理脚本与配置；实际训练在 Kaggle Notebook 中挂载官方数据集完成。

## 项目结构

```
cxr-detectbench/
├── data/                # 原始/处理数据（不入库，仅留 split 索引）
├── notebooks/           # 各阶段 Kaggle Notebook
├── configs/             # 五模型训练配置（YOLO yaml / MMDet py）
├── scripts/             # 数据处理、融合、评估、错误分析脚本
├── outputs/             # checkpoints / 评估结果 / bad case（不入库）
├── demo/                # Gradio Demo（部署到 HF Spaces）
├── docs/                # 任务细分与执行记录
└── CXR-DetectBench-Project-Plan.md  # 完整执行手册
```

详见 [docs/TASK_BREAKDOWN.md](docs/TASK_BREAKDOWN.md)。

## 环境与运行

训练在 Kaggle Notebook（P100 16GB / T4×2，每周 30h GPU 配额）中完成，按阶段拆分 CPU 数据处理 / GPU 训练 Notebook 以节省配额。完整依赖见 [scripts/requirements.txt](scripts/requirements.txt)。

## 结果

> 待回填（Phase 6 完成后）。预期产出：五模型 mAP@0.5 / mAP@0.5:0.95 / mAP@0.4 / FROC / FPS / 参数量对比总表 + 精度-速度 Pareto 图 + FROC 曲线。

## Demo

> 待部署（Phase 8 完成后）。将上线至 Hugging Face Spaces，上传胸片 → 自动预处理 → 推理 → 可视化检测框。

**免责声明**：本 Demo 仅供技术演示与学习交流，不能用于临床诊断。

## License

[MIT](LICENSE) — 代码部分遵循 MIT 许可；VinDr-CXR 数据集遵循其原始使用协议，本项目不重新授权数据。
