"""结构化错误分析（Phase 7）。

产出：
  - 按类别 AP 拆分，最难/最易类别解释
  - FP 溯源：模型真错 vs 1/3 医生标注被判 FP 的合理检测
  - FN 溯源：小尺寸漏检 vs 解剖结构重叠漏检，各 3-5 张对比图
  - 图像级误报：完全正常胸片上误报"有异常"的比例
  - 四模型在同一张难例上的对比图

TODO（Phase 7）：实现。输出落到 outputs/bad_cases/。
"""
from __future__ import annotations


def analyze_fp_origin(preds, raw_annotations):
    raise NotImplementedError("Phase 7.2 实现")


def analyze_fn_origin(preds, gts):
    raise NotImplementedError("Phase 7.3 实现")


def image_level_false_positive(preds, normal_image_ids):
    raise NotImplementedError("Phase 7.4 实现")


if __name__ == "__main__":
    pass