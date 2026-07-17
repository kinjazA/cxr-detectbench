"""FROC 曲线计算（Phase 6.3）。

横轴：每张图允许的假阳性数（FP/image）
纵轴：病灶级别敏感度 / 召回率

这是医疗 CAD 系统的临床标准评估方式，对应"在可容忍的误报下能检出多少真病灶"。

TODO（Phase 6.3）：实现 + 产出 FROC 曲线图（五模型叠加）。
"""
from __future__ import annotations


def compute_froc(preds, gts, *, num_images):
    """返回 (fps_per_image, sensitivity) 两条曲线点。"""
    raise NotImplementedError("Phase 6.3 实现")


def plot_froc(curves, out_path):
    raise NotImplementedError("Phase 6.3 实现")


if __name__ == "__main__":
    pass