"""COCO → YOLO 格式转换（Phase 2.5）。

从最终选定的融合标注版本（Phase 2.4 消融后确定），生成：
  - 完整 COCO json（给 MMDetection，已在 label_fusion.py 生成）
  - YOLO txt（给 Ultralytics，归一化的 <cls> <cx> <cy> <w> <h>）

保证两种格式共享同一张图像集与同一划分（Phase 3 产出的 split）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def coco_to_yolo_format(coco_json_path: str, output_dir: str):
    """将 COCO json 转换成 YOLO txt 格式。

    Args:
        coco_json_path: COCO annotations.json 路径
        output_dir: YOLO txt 输出目录（每张图一个 <image_id>.txt）
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 image_id -> (width, height) 映射
    img_info = {}
    for img in coco['images']:
        img_info[img['id']] = (img['width'], img['height'])

    # 按 image_id 分组 annotations
    from collections import defaultdict
    img_annotations = defaultdict(list)
    for ann in coco['annotations']:
        img_annotations[ann['image_id']].append(ann)

    print(f"Converting COCO to YOLO format...")
    print(f"  Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")

    for image_id, anns in tqdm(img_annotations.items(), desc="Writing YOLO labels"):
        if image_id not in img_info:
            print(f"Warning: image_id={image_id} not in images list, skipping")
            continue

        img_width, img_height = img_info[image_id]

        yolo_lines = []
        for ann in anns:
            # COCO bbox: [x, y, w, h] (top-left corner + width/height)
            x, y, w, h = ann['bbox']
            class_id = ann['category_id']

            # YOLO format: <class> <cx> <cy> <w> <h> (normalized, center-based)
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 保存
        out_path = output_dir / f"{image_id}.txt"
        with open(out_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print(f"Done! YOLO labels saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Path to COCO annotations.json (e.g., labels_coco/wbf/annotations.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for YOLO txt files (e.g., labels_yolo/wbf/)",
    )

    args = parser.parse_args()

    coco_to_yolo_format(args.coco_json, args.output_dir)


if __name__ == "__main__":
    main()
