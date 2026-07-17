"""CXR-DetectBench Gradio Demo（Phase 8.3）。

功能：上传胸片 (PNG，可选 DICOM) -> 自动窗宽窗位/CLAHE 预处理
      -> 推理 -> 可视化检测框 + 类别 + 置信度

⚠️ 免责声明必须显示在界面："本 Demo 仅供技术演示与学习交流，不能用于临床诊断"

部署：Hugging Face Spaces (Gradio SDK)

TODO（Phase 8）：接 ONNX Runtime 推理 + 绘框 + spec/责任声明 block。
"""
from __future__ import annotations

# import gradio as gr


DISCLAIMER = (
    "⚠️ 免责声明：本 Demo 仅供技术演示与学习交流，不能用于临床诊断。"
)


def preprocess(image):
    """窗宽窗位 + CLAHE + 转三通道。复用 scripts/dicom_to_png 逻辑。"""
    raise NotImplementedError("Phase 8.3 实现")


def predict(image):
    """ONNXRuntime 推理，返回带框图。"""
    raise NotImplementedError("Phase 8.3 实现")


def build_app():
    raise NotImplementedError("Phase 8.3 实现")


if __name__ == "__main__":
    build_app()