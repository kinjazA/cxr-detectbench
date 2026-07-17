"""多标注融合（Phase 2.3）。

VinDr-CXR 每张图由 17 位放射科医生池中的 3 位独立标注，同一区域最多 3 个框。
本脚本产出三版 COCO 标注，用于 Phase 2.4 消融实验：

  - labels_coco/raw/   17 位医生的框全保留（同一 image_id 的多个 rad_id）
  - labels_coco/wbf/   ensemble-boxes 的 weighted_boxes_fusion（IoU 阈值默认 0.5）
  - labels_coco/nms/   NMS 去重

Phase 2.4 消融：YOLOv8n 少量 epoch 分别在三版上训练，对比 val mAP，
锁定最终融合策略；IoU 阈值可再做一次消融。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm


def boxes_to_normalized(boxes, img_width, img_height):
    """Convert [x_min, y_min, x_max, y_max] to normalized [0, 1]."""
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, [0, 2]] /= img_width
    boxes[:, [1, 3]] /= img_height
    return boxes.tolist()


def boxes_to_abs(boxes, img_width, img_height):
    """Convert normalized [0, 1] to absolute [x_min, y_min, x_max, y_max]."""
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, [0, 2]] *= img_width
    boxes[:, [1, 3]] *= img_height
    return boxes.tolist()


def nms_single_class(boxes, scores, iou_thr=0.5):
    """Simple NMS for single class (numpy implementation)."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def fuse_annotations_per_image(
    annotations_for_img,
    img_width,
    img_height,
    fusion_mode="wbf",
    iou_thr=0.5,
):
    """对单张图的多个标注做融合。

    Args:
        annotations_for_img: list of dict, 每个 dict 是一个标注框 (x_min, y_min, x_max, y_max, class_id)
        img_width, img_height: 图像尺寸（用于归一化）
        fusion_mode: "raw" | "wbf" | "nms"
        iou_thr: WBF 或 NMS 的 IoU 阈值

    Returns:
        融合后的标注列表 (list of dict)
    """
    if fusion_mode == "raw":
        # 不融合，全保留
        return annotations_for_img

    # 按 class_id 分组（因为 WBF/NMS 都是 per-class 的）
    class_groups = defaultdict(list)
    for ann in annotations_for_img:
        class_groups[ann["class_id"]].append(ann)

    fused = []

    for class_id, anns in class_groups.items():
        boxes = [
            [a["x_min"], a["y_min"], a["x_max"], a["y_max"]]
            for a in anns
        ]
        # 每个框的置信度（多标注融合时可认为每个医生的框置信度相同）
        scores = [1.0] * len(boxes)

        if fusion_mode == "wbf":
            # WBF 需要归一化坐标
            boxes_norm = boxes_to_normalized(boxes, img_width, img_height)
            # WBF expects list of lists (for multiple models), here we have single "model" (all radiologists)
            boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
                [boxes_norm],
                [scores],
                [[class_id] * len(boxes)],
                iou_thr=iou_thr,
                skip_box_thr=0.0,  # 不跳过任何框
            )
            # 转回绝对坐标
            boxes_abs = boxes_to_abs(boxes_wbf, img_width, img_height)

            for box, score in zip(boxes_abs, scores_wbf):
                fused.append({
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                    "class_id": class_id,
                    "score": score,
                })

        elif fusion_mode == "nms":
            keep_indices = nms_single_class(boxes, scores, iou_thr)
            for idx in keep_indices:
                ann = anns[idx]
                fused.append({
                    "x_min": ann["x_min"],
                    "y_min": ann["y_min"],
                    "x_max": ann["x_max"],
                    "y_max": ann["y_max"],
                    "class_id": ann["class_id"],
                    "score": 1.0,
                })

    return fused


def build_coco_json(
    train_csv_path: str,
    images_meta_path: str,
    output_path: str,
    fusion_mode: str = "wbf",
    iou_thr: float = 0.5,
):
    """构建 COCO 格式标注。

    Args:
        train_csv_path: train.csv 路径
        images_meta_path: images.csv 路径（DICOM metadata，含分辨率）
        output_path: 输出 COCO json 路径
        fusion_mode: "raw" | "wbf" | "nms"
        iou_thr: WBF/NMS IoU 阈值
    """
    df = pd.read_csv(train_csv_path)
    images_df = pd.read_csv(images_meta_path)

    # 从 images.csv 获取分辨率（列名因数据集版本而异，需要自动适配）
    print(f"[DEBUG] images.csv columns: {list(images_df.columns)}")

    # 适配 image_id 列名
    id_col = None
    for col in ['image_id', 'SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']:
        if col in images_df.columns:
            id_col = col
            break
    if id_col is None:
        id_col = images_df.columns[0]  # fallback to first column
    if id_col != 'image_id':
        images_df.rename(columns={id_col: 'image_id'}, inplace=True)

    # 适配宽高列名
    width_col = None
    height_col = None
    for col in images_df.columns:
        if col.lower() in ('columns', 'width', 'imagewidth', 'cols'):
            width_col = col
        if col.lower() in ('rows', 'height', 'imageheight'):
            height_col = col
    # 如果没有精确匹配，尝试模糊匹配
    if width_col is None:
        for col in images_df.columns:
            if 'column' in col.lower() or 'width' in col.lower():
                width_col = col
                break
    if height_col is None:
        for col in images_df.columns:
            if 'row' in col.lower() or 'height' in col.lower():
                height_col = col
                break
    # 最后兜底：用倒数第二列和最后一列
    if width_col is None and height_col is None:
        if len(images_df.columns) >= 2:
            width_col = images_df.columns[-2]
            height_col = images_df.columns[-1]

    if width_col is None or height_col is None:
        print(f"[ERROR] Cannot identify width/height columns in images.csv! Columns: {list(images_df.columns)}")
        raise KeyError(f"Need 'Columns'/'Rows' or similar in {list(images_df.columns)}")

    print(f"[DEBUG] Using image_id={id_col}, width={width_col}, height={height_col}")

    img_size_map = {}
    for _, row in images_df.iterrows():
        img_size_map[row['image_id']] = (int(row[width_col]), int(row[height_col]))  # (width, height)

    # 过滤掉 No finding（class_id=14）
    df_boxes = df[df['class_id'] != 14].copy()

    # 按 image_id 分组
    grouped = df_boxes.groupby('image_id')

    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # categories（0-13，不含 No finding=14）
    from class_names import CLASS_NAMES
    for cid in range(14):
        coco["categories"].append({
            "id": cid,
            "name": CLASS_NAMES[cid],
        })

    annotation_id = 1

    for image_id in tqdm(df['image_id'].unique(), desc=f"Building COCO ({fusion_mode})"):
        if image_id not in img_size_map:
            print(f"Warning: {image_id} not in images.csv, skipping")
            continue

        img_width, img_height = img_size_map[image_id]

        # COCO images entry
        coco["images"].append({
            "id": image_id,
            "file_name": f"{image_id}.png",
            "width": img_width,
            "height": img_height,
        })

        # 该图的所有标注（来自多个医生）
        if image_id not in grouped.groups:
            # No finding 图，没有框
            continue

        img_group = grouped.get_group(image_id)
        annotations_raw = []
        for _, row in img_group.iterrows():
            annotations_raw.append({
                "x_min": row['x_min'],
                "y_min": row['y_min'],
                "x_max": row['x_max'],
                "y_max": row['y_max'],
                "class_id": int(row['class_id']),
            })

        # 融合
        annotations_fused = fuse_annotations_per_image(
            annotations_raw,
            img_width,
            img_height,
            fusion_mode=fusion_mode,
            iou_thr=iou_thr,
        )

        # 转成 COCO annotations
        for ann in annotations_fused:
            x_min, y_min, x_max, y_max = ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]
            width = x_max - x_min
            height = y_max - y_min

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["class_id"],
                "bbox": [x_min, y_min, width, height],  # COCO format: [x, y, w, h]
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f"Saved {fusion_mode} COCO to {output_path}")
    print(f"  Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")


def main():
    parser = argparse.ArgumentParser(description="Multi-annotator fusion for VinDr-CXR")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="../data/raw/train.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default="../data/raw/images.csv",
        help="Path to images.csv (DICOM metadata with Rows/Columns)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/processed/labels_coco",
        help="Output directory for COCO jsons",
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.5,
        help="IoU threshold for WBF/NMS",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 产出三版标注
    for mode in ["raw", "wbf", "nms"]:
        out_path = output_dir / mode / "annotations.json"
        build_coco_json(
            args.train_csv,
            args.images_csv,
            str(out_path),
            fusion_mode=mode,
            iou_thr=args.iou_thr,
        )

    print("\nDone! Generated 3 fusion variants:")
    print(f"  - raw:  {output_dir / 'raw' / 'annotations.json'}")
    print(f"  - wbf:  {output_dir / 'wbf' / 'annotations.json'}")
    print(f"  - nms:  {output_dir / 'nms' / 'annotations.json'}")


if __name__ == "__main__":
    main()
