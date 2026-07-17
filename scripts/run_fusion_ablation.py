"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.

Key design: uses YOLO's train/val text-file listing instead of copying images.
No symlinks, no image duplication. Images stay at original location.
"""
import argparse
import json
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


def convert_coco_to_yolo(coco_json_path: str, output_dir: str):
    """Convert COCO annotations to YOLO format."""
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format
    coco_to_yolo_format(coco_json_path, output_dir)


def prepare_fusion_run(fusion_mode: str, coco_json: str, base_dir: str,
                       train_ids: set, val_ids: set, images_abs_path: str):
    """Prepare YOLO dataset without copying images.

    Creates:
      labels_yolo/{mode}/          - all YOLO label txt files
      ablation/{mode}/{mode}.yaml  - YOLO config pointing to original images
      ablation/{mode}/train.txt    - list of train image filenames
      ablation/{mode}/val.txt      - list of val image filenames

    Images stay at their original location (images_abs_path).
    YOLO reads them directly.
    """
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    base_dir = Path(base_dir)
    work_dir = base_dir / 'ablation' / fusion_mode
    work_dir.mkdir(parents=True, exist_ok=True)

    # Generate all YOLO labels
    labels_dir = base_dir / 'labels_yolo' / fusion_mode
    labels_dir.mkdir(parents=True, exist_ok=True)
    coco_to_yolo_format(coco_json, str(labels_dir))

    # Build train/val text files listing image paths
    import json
    with open(coco_json) as f:
        coco = json.load(f)

    images_abs_path = Path(images_abs_path)

    train_lines = []
    val_lines = []

    for img in coco['images']:
        img_id = img['id']
        img_file = f"{img_id}.png"

        # Verify image exists
        full_img_path = images_abs_path / img_file
        if not full_img_path.exists():
            continue

        if img_id in train_ids:
            train_lines.append(str(full_img_path))
        elif img_id in val_ids:
            val_lines.append(str(full_img_path))

    # Write train/val image lists (absolute paths)
    with open(work_dir / 'train_images.txt', 'w') as f:
        f.write('\n'.join(train_lines))
    with open(work_dir / 'val_images.txt', 'w') as f:
        f.write('\n'.join(val_lines))

    # Create YOLO yaml
    from class_names import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_content = f"""# YOLO dataset config - images stay at original location
train: {work_dir / 'train_images.txt'}
val: {work_dir / 'val_images.txt'}

# Labels are at the same path, just with .txt extension
# YOLO will look for labels next to images; we override via label location

nc: 14
names: {names}
"""

    yaml_path = work_dir / f'{fusion_mode}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Prepared {fusion_mode}: {len(train_lines)} train, {len(val_lines)} val")
    print(f"  Labels: {labels_dir}")
    print(f"  Config: {yaml_path}")

    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Fusion ablation experiment")
    parser.add_argument('--coco_dir', type=str, required=True, help='COCO labels directory')
    parser.add_argument('--images_dir', type=str, required=True, help='Images directory (original PNGs)')
    parser.add_argument('--train_csv', type=str, required=True, help='train.csv path')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default='phase2_fusion_ablation.csv', help='Output CSV')

    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    images_dir = Path(args.images_dir)
    base_dir = coco_dir.parent

    # Resolve images_dir to absolute path (it might be a symlink)
    if images_dir.is_symlink():
        images_dir = images_dir.resolve()
        print(f"Images dir resolved to: {images_dir}")
    elif not images_dir.is_absolute():
        images_dir = images_dir.resolve()

    print(f"Images directory: {images_dir}")

    # Create temporary split
    print("\n" + "=" * 60)
    print("Step 1: Creating temporary train/val split")
    print("=" * 60)
    train_ids, val_ids = create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # Train on each fusion variant
    results = {}

    for i, fusion_mode in enumerate(['raw', 'wbf', 'nms']):
        print("\n" + "=" * 60)
        print(f"Step 2.{i+1}: Training with {fusion_mode} fusion")
        print("=" * 60)

        coco_json = coco_dir / fusion_mode / 'annotations.json'

        # Prepare YOLO run
        print(f"\nPreparing {fusion_mode}...")
        yaml_path = prepare_fusion_run(
            fusion_mode, str(coco_json), str(base_dir),
            train_ids, val_ids, str(images_dir)
        )

        # Train
        print(f"\nTraining YOLOv8n with {fusion_mode} fusion...")
        print(f"  Config: {yaml_path}")

        try:
            model = YOLO('yolov8n.pt')

            train_results = model.train(
                data=yaml_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                name=f'fusion_ablation_{fusion_mode}',
                project='runs/fusion_ablation',
                exist_ok=True,
                patience=5,  # early stop if no improvement
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
            import traceback
            traceback.print_exc()
            results[fusion_mode] = {'mAP50': 0.0, 'mAP50-95': 0.0}

        print(f"\n{fusion_mode}: mAP@0.5={results[fusion_mode]['mAP50']:.4f}, "
              f"mAP@0.5:0.95={results[fusion_mode]['mAP50-95']:.4f}")

    # Save results
    print("\n" + "=" * 60)
    print("Step 3: Summary")
    print("=" * 60)

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
        print(f"\nRecommended fusion: {best_mode} (mAP@0.5={best_map:.4f})")
    else:
        print("\nWARNING: All fusion variants failed. Check training logs above.")


if __name__ == '__main__':
    main()
