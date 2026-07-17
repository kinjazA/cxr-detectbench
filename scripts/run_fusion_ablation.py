"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.

Strategy (no per-file symlink/copy):
  - One directory-level symlink for ALL images (proven to work on Kaggle)
  - Train/val splits: same images dir, different labels dirs
  - YOLO auto-skips images without a label file (works as negative samples)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def create_temp_split(train_csv_path: str, output_dir: str, test_size=0.2, seed=42):
    """80/20 random split for ablation."""
    df = pd.read_csv(train_csv_path)
    image_ids = df['image_id'].unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=test_size, random_state=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'image_id': train_ids}).to_csv(output_dir / 'train.csv', index=False)
    pd.DataFrame({'image_id': val_ids}).to_csv(output_dir / 'val.csv', index=False)

    print(f"Created temp split: {len(train_ids)} train, {len(val_ids)} val")
    return set(train_ids), set(val_ids)


def convert_all_coco_to_yolo(coco_json_path: str, output_dir: str):
    """Convert COCO to YOLO labels for ALL images."""
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format
    coco_to_yolo_format(coco_json_path, output_dir)


def setup_dataset(fusion_mode: str, coco_json: str, base_dir: str,
                  train_ids: set, val_ids: set, images_src: str):
    """Set up YOLO dataset with directory-level symlink + split labels.

    Final structure:
      ablation/{mode}/
        dataset.yaml
        train/
          images/  -> symlink to original PNG directory
          labels/  -> label txt for train images only
        val/
          images/  -> symlink to original PNG directory
          labels/  -> label txt for val images only

    YOLO scans all images in each split but only uses those with a matching label file.
    """
    base_dir = Path(base_dir)
    work_dir = base_dir / 'ablation' / fusion_mode
    images_src_path = Path(images_src).resolve()

    # Clean and create directories
    if work_dir.exists():
        shutil.rmtree(work_dir)

    train_img_dir = work_dir / 'train' / 'images'
    train_lbl_dir = work_dir / 'train' / 'labels'
    val_img_dir = work_dir / 'val' / 'images'
    val_lbl_dir = work_dir / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Directory-level symlink for images (PROVEN TO WORK on Kaggle)
    #    symlink the entire PNG dir into each split's images/ subdir
    png_subdir = train_img_dir / 'pngs'
    if not png_subdir.exists():
        os.symlink(str(images_src_path), str(png_subdir), target_is_directory=True)

    png_subdir_val = val_img_dir / 'pngs'
    if not png_subdir_val.exists():
        os.symlink(str(images_src_path), str(png_subdir_val), target_is_directory=True)

    # 2. Generate ALL YOLO labels to temp dir
    temp_dir = work_dir / '_temp_labels'
    temp_dir.mkdir(parents=True, exist_ok=True)
    convert_all_coco_to_yolo(coco_json, str(temp_dir))

    # 3. Copy labels to train/val (tiny files, very fast)
    import json
    with open(coco_json) as f:
        coco = json.load(f)

    train_count = 0
    val_count = 0

    for img in coco['images']:
        img_id = img['id']
        txt_file = f"{img_id}.txt"
        src_txt = temp_dir / txt_file

        if not src_txt.exists():
            continue  # No-finding image, no label needed

        if img_id in train_ids:
            shutil.copy2(src_txt, train_lbl_dir / txt_file)
            train_count += 1
        elif img_id in val_ids:
            shutil.copy2(src_txt, val_lbl_dir / txt_file)
            val_count += 1

    # Clean temp
    shutil.rmtree(temp_dir)

    # 4. Write YOLO data.yaml
    from class_names import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_path = work_dir / 'dataset.yaml'
    yaml_content = f"""# Fusion ablation: {fusion_mode}
path: {work_dir}
train: train
val: val

nc: 14
names: {names}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"  Train labels: {train_count}, Val labels: {val_count}")
    print(f"  Images via symlink -> {images_src_path}")
    print(f"  Config: {yaml_path}")

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

    coco_dir = Path(args.coco_dir)
    images_dir = Path(args.images_dir)
    base_dir = coco_dir.parent

    # Resolve images path
    if images_dir.is_symlink():
        images_dir = images_dir.resolve()
    images_dir = images_dir.resolve()

    # Verify images exist
    png_files = list(images_dir.glob('*.png'))
    print(f"Images dir: {images_dir} ({len(png_files)} PNG files)")

    # 1. Split
    print("\n" + "=" * 60)
    print("Step 1: Temporary train/val split (80/20)")
    print("=" * 60)
    train_ids, val_ids = create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # 2. Train on each fusion variant
    results = {}

    for i, fusion_mode in enumerate(['raw', 'wbf', 'nms']):
        print("\n" + "=" * 60)
        print(f"Step 2.{i+1}: Training YOLOv8n with {fusion_mode} fusion")
        print("=" * 60)

        coco_json = coco_dir / fusion_mode / 'annotations.json'

        yaml_path = setup_dataset(
            fusion_mode, str(coco_json), str(base_dir),
            train_ids, val_ids, str(images_dir)
        )

        try:
            model = YOLO('yolov8n.pt')
            model.train(
                data=yaml_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                name=f'fusion_ablation_{fusion_mode}',
                project='runs/fusion_ablation',
                exist_ok=True,
                patience=0,
                verbose=True,
            )
            metrics = model.val()
            results[fusion_mode] = {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
            }
        except Exception as e:
            print(f"\nERROR training {fusion_mode}: {e}")
            import traceback
            traceback.print_exc()
            results[fusion_mode] = {'mAP50': 0.0, 'mAP50-95': 0.0}

        print(f"\n{fusion_mode}: mAP@0.5={results[fusion_mode]['mAP50']:.4f}"
              f"  mAP@0.5:0.95={results[fusion_mode]['mAP50-95']:.4f}")

    # 3. Summary
    print("\n" + "=" * 60)
    print("Step 3: Results Summary")
    print("=" * 60)

    df = pd.DataFrame(results).T
    df.index.name = 'fusion_mode'
    df = df.reset_index()
    print("\n" + df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    valid = df[df['mAP50'] > 0]
    if len(valid) > 0:
        best = valid.loc[valid['mAP50'].idxmax(), 'fusion_mode']
        best_map = valid.loc[valid['mAP50'].idxmax(), 'mAP50']
        print(f"\nBest fusion: {best} (mAP@0.5={best_map:.4f})")


if __name__ == '__main__':
    main()
