"""VinDr-CXR 类别映射（供全项目复用）。

来源：Kaggle `vinbigdata-chest-xray-abnormalities-detection` 竞赛。
本表为标准映射，实际 class_id 以首次读取 train.csv/classes.csv 后核对为准
（Phase 1 完成后冻结本文件）。
"""

# class_id -> 英文名
CLASS_NAMES = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",  # 图像级标签，不参与 box 回归
}

NUM_CLASSES = 14  # 不含 No finding

# 简历/README 备注用
CLASS_NOTES = {
    0: "样本较多，目标偏大，预期易检测",
    5: "ILD：边界模糊，弥漫性，预期较难",
    8: "Nodule/Mass：小目标，预期较难",
    12: "Pneumothorax：样本稀少，长尾类别",
}