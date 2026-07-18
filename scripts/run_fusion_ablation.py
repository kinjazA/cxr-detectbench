"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.

Uses standard YOLO directory structure:
  ablation/{mode}/
    data.yaml        (path: ., train: images/train, val: images/val)
    images/train/    -> symlink to original PNG directory
    images/val/      -> symlink to original PNG directory
    labels/train/    -> train split YOLO label .txt files
    labels/val/      -> val split YOLO label .txt files

YOLO finds labels by replacing /images/ with /labels/ in each image path:
  ablation/wbf/images/train/0005e8e37.png  →  ablation/wbf/labels/train/0005e8e37.txt  ✓
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def create_temp_split(train_csv_path, output_dir, test_size=0.2, seed=42):
    """80/20 random split for ablation."""
    df = pd.read_csv(train_csv_path)
    image_ids = df['image_id'].unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=test_size, random_state=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'image_id': train_ids}).to_csv(output_dir / 'train.csv', index=False)
    pd.DataFrame({'image_id': val_ids}).to_csv(output_dir / 'val.csv', index=False)

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    return set(train_ids), set(val_ids)


def prepare_run(fusion_mode, coco_json, base_dir, train_ids, val_ids, images_src):
    """Set up standard YOLO dataset:

      ablation/{mode}/
        data.yaml
        images/train/  -> symlink
        images/val/    -> symlink
        labels/train/  -> txt files
        labels/val/    -> txt files
    """
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    base_dir = Path(base_dir)
    work_dir = base_dir / 'ablation' / fusion_mode
    if work_dir.exists():
        shutil.rmtree(work_dir)

    # Create directory structure
    img_train_dir = work_dir / 'images' / 'train'
    img_val_dir = work_dir / 'images' / 'val'
    lbl_train_dir = work_dir / 'labels' / 'train'
    lbl_val_dir = work_dir / 'labels' / 'val'

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True)

    # Symlink images directories (directory-level = instant)
    images_src = Path(images_src).resolve()
    # Remove the empty dirs and symlink
    img_train_dir.rmdir()
    os.symlink(str(images_src), str(img_train_dir), target_is_directory=True)
    img_val_dir.rmdir()
    os.symlink(str(images_src), str(img_val_dir), target_is_directory=True)

    # Generate ALL YOLO labels to temp
    temp_dir = work_dir / '_temp'
    temp_dir.mkdir()
    coco_to_yolo_format(coco_json, str(temp_dir))

    # Split labels into train/val (only for images with labels)
    with open(coco_json) as f:
        coco = json.load(f)

    train_cnt = 0
    val_cnt = 0
    for img in coco['images']:
        img_id = img['id']
        txt_file = f"{img_id}.txt"
        src = temp_dir / txt_file
        if not src.exists():
            continue
        if img_id in train_ids:
            shutil.copy2(str(src), str(lbl_train_dir / txt_file))
            train_cnt += 1
        elif img_id in val_ids:
            shutil.copy2(str(src), str(lbl_val_dir / txt_file))
            val_cnt += 1

    shutil.rmtree(temp_dir)

    # Write YOLO data.yaml
    from class_names import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_path = work_dir / 'data.yaml'
    yaml_path.write_text(f"""path: {work_dir}
train: images/train
val: images/val

nc: 14
names: {names}
""")

    print(f"  Labels: {train_cnt} train, {val_cnt} val")
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Fusion ablation experiment")
    parser.add_argument('--coco_dir', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--output', type=str, default='phase2_fusion_ablation.csv')
    args = parser.parse_args()

    base_dir = Path(args.coco_dir).parent
    images_dir = Path(args.images_dir)
    if images_dir.is_symlink():
        images_dir = images_dir.resolve()
    images_dir = images_dir.resolve()

    # Verify images exist
    png_count = len(list(images_dir.glob('*.png')))
    print(f"Images: {images_dir} ({png_count} PNGs)")

    # 1. Split (same for all 3 fusion variants)
    print("\n" + "=" * 60)
    print("Step 1: 80/20 split")
    print("=" * 60)
    train_ids, val_ids = create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # 2. Train
    results = {}
    for i, mode in enumerate(['raw', 'wbf', 'nms']):
        print(f"\n{'='*60}")
        print(f"Step 2.{i+1}: YOLOv8n + {mode}")
        print(f"{'='*60}")

        coco_json = Path(args.coco_dir) / mode / 'annotations.json'
        yaml_path = prepare_run(mode, str(coco_json), str(base_dir),
                                train_ids, val_ids, str(images_dir))

        try:
            model = YOLO('yolov8n.pt')
            model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, name=f'fusion_{mode}',
                        project='runs/ablation', exist_ok=True, patience=0, verbose=True)
            m = model.val()
            results[mode] = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map)}
        except Exception as e:
            print(f"\nERROR {mode}: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {'mAP50': 0.0, 'mAP50-95': 0.0}

        print(f"\n{mode}: mAP@0.5={results[mode]['mAP50']:.4f}"
              f"  mAP@0.5:0.95={results[mode]['mAP50-95']:.4f}")

    # 3. Summary
    print(f"\n{'='*60}\nResults\n{'='*60}")
    df = pd.DataFrame(results).T.reset_index(names='fusion_mode')
    print(df.to_string(index=False))
    df.to_csv(args.output, index=False)
    valid = df[df['mAP50'] > 0]
    if len(valid):
        best = valid.loc[valid['mAP50'].idxmax(), 'fusion_mode']
        print(f"\nBest: {best}")


if __name__ == '__main__':
    main()
