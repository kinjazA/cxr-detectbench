"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def create_temp_split(train_csv_path: str, output_dir: str, test_size=0.2, seed=42):
    """Create temporary 80/20 train/val split for ablation."""
    df = pd.read_csv(train_csv_path)
    image_ids = df['image_id'].unique()

    train_ids, val_ids = train_test_split(image_ids, test_size=test_size, random_state=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'image_id': train_ids}).to_csv(output_dir / 'train.csv', index=False)
    pd.DataFrame({'image_id': val_ids}).to_csv(output_dir / 'val.csv', index=False)

    print(f"Created temp split: {len(train_ids)} train, {len(val_ids)} val")
    return set(train_ids), set(val_ids)


def prepare_yolo_structure(fusion_mode: str, coco_json: str, base_dir: str,
                           train_ids: set, val_ids: set, images_src: str):
    """Prepare YOLO train/val directory structure with images and labels.

    Creates:
        {base_dir}/ablation/{fusion_mode}/train/images/  (symlinks or copies)
        {base_dir}/ablation/{fusion_mode}/train/labels/
        {base_dir}/ablation/{fusion_mode}/val/images/
        {base_dir}/ablation/{fusion_mode}/val/labels/
    """
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    base_dir = Path(base_dir)
    work_dir = base_dir / 'ablation' / fusion_mode

    # Clean any previous run
    if work_dir.exists():
        shutil.rmtree(work_dir)

    # Create directories
    (work_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (work_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
    (work_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (work_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)

    # Generate all YOLO labels to a temp dir first
    temp_dir = work_dir / '_temp_labels'
    temp_dir.mkdir(parents=True, exist_ok=True)
    coco_to_yolo_format(coco_json, str(temp_dir))

    # Split labels into train/val
    import json
    with open(coco_json) as f:
        coco = json.load(f)

    images_src = Path(images_src)
    train_count = 0
    val_count = 0

    for img in coco['images']:
        img_id = img['id']
        txt_file = f"{img_id}.txt"
        src_txt = temp_dir / txt_file
        src_img = images_src / f"{img_id}.png"

        if img_id in train_ids:
            dst_txt = work_dir / 'train' / 'labels' / txt_file
            dst_img = work_dir / 'train' / 'images' / f"{img_id}.png"
            train_count += 1
        elif img_id in val_ids:
            dst_txt = work_dir / 'val' / 'labels' / txt_file
            dst_img = work_dir / 'val' / 'images' / f"{img_id}.png"
            val_count += 1
        else:
            continue

        # Copy label
        if src_txt.exists():
            shutil.copy2(src_txt, dst_txt)

        # Symlink image (much faster than copy for 15000 images)
        if src_img.exists() and not dst_img.exists():
            try:
                os.symlink(src_img.resolve(), dst_img)
            except OSError:
                # Fallback to copy if symlink fails
                shutil.copy2(src_img, dst_img)

    # Clean up temp
    shutil.rmtree(temp_dir)

    print(f"Prepared YOLO structure ({fusion_mode}): {train_count} train, {val_count} val")
    return work_dir


def create_yolo_yaml(fusion_mode: str, work_dir: str) -> str:
    """Create YOLO dataset config yaml."""
    from class_names import CLASS_NAMES

    # Detection: 14 classes (0-13), exclude "No finding" (14)
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_content = f"""path: {work_dir}
train: train
val: val

nc: 14
names: {names}
"""

    yaml_path = Path(work_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Fusion ablation experiment")
    parser.add_argument('--coco_dir', type=str, required=True, help='COCO labels directory')
    parser.add_argument('--images_dir', type=str, required=True, help='Images directory')
    parser.add_argument('--train_csv', type=str, required=True, help='train.csv path')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default='phase2_fusion_ablation.csv', help='Output CSV')

    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    images_dir = Path(args.images_dir)
    base_dir = coco_dir.parent

    # Create temporary split
    print("\n" + "="*60)
    print("Step 1: Creating temporary train/val split")
    print("="*60)
    train_ids, val_ids = create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # Train on each fusion variant
    results = {}

    for i, fusion_mode in enumerate(['raw', 'wbf', 'nms']):
        print("\n" + "="*60)
        print(f"Step 2.{i+1}: Training with {fusion_mode} fusion")
        print("="*60)

        coco_json = coco_dir / fusion_mode / 'annotations.json'

        # Prepare YOLO directory structure
        print(f"\nPreparing YOLO structure for {fusion_mode}...")
        work_dir = prepare_yolo_structure(
            fusion_mode, str(coco_json), str(base_dir),
            train_ids, val_ids, str(images_dir)
        )

        # Create YOLO config
        yaml_path = create_yolo_yaml(fusion_mode, str(work_dir))

        # Train
        print(f"\nTraining YOLOv8n with {fusion_mode} fusion...")
        model = YOLO('yolov8n.pt')

        try:
            train_results = model.train(
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

            # Validate
            metrics = model.val()

            results[fusion_mode] = {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
            }
        except Exception as e:
            print(f"\nERROR training {fusion_mode}: {e}")
            results[fusion_mode] = {
                'mAP50': 0.0,
                'mAP50-95': 0.0,
            }

        print(f"\n{fusion_mode} results:")
        print(f"  mAP@0.5      = {results[fusion_mode]['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95 = {results[fusion_mode]['mAP50-95']:.4f}")

    # Save results
    print("\n" + "="*60)
    print("Step 3: Summary")
    print("="*60)

    df = pd.DataFrame(results).T
    df.index.name = 'fusion_mode'
    df = df.reset_index()

    print("\n" + df.to_string(index=False))

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    valid = df[df['mAP50'] > 0]
    if len(valid) > 0:
        best_mode = valid.loc[valid['mAP50'].idxmax(), 'fusion_mode']
        best_map = valid.loc[valid['mAP50'].idxmax(), 'mAP50']
        print(f"\nRecommended fusion strategy: {best_mode} (mAP@0.5={best_map:.4f})")
    else:
        print("\nWARNING: All fusion variants failed training.")


if __name__ == '__main__':
    main()
