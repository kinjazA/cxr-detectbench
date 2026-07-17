"""VinDr-CXR 类别映射（供全项目复用）。

来源：Kaggle `vinbigdata-chest-xray-abnormalities-detection` 竞赛。
本表为标准映射，实际 class_id 以首次读取 train.csv 后核对为准
（Phase 1 已核对：与下方完全一致，已冻结）。

**标注事实**：17 位放射科医生池（R1–R17），每张异常图由其中恰好 3 位独立标注
（非计划初稿所述"固定 3 名医生"）。多标注融合仍按 image_id 聚合多框。
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