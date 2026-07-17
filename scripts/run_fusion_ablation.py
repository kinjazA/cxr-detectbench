"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.
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
    return train_ids, val_ids


def convert_coco_to_yolo(coco_json_path: str, output_dir: str):
    """Convert COCO annotations to YOLO format."""
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    coco_to_yolo_format(coco_json_path, output_dir)


def create_yolo_yaml(fusion_mode: str, base_dir: str) -> str:
    """Create YOLO dataset config yaml."""
    from class_names import CLASS_NAMES

    yaml_content = f"""path: {base_dir}
train: labels_yolo_temp/{fusion_mode}
val: labels_yolo_temp/{fusion_mode}

nc: 14
names: {list(CLASS_NAMES.values())}
"""

    yaml_path = Path(base_dir) / f'fusion_{fusion_mode}.yaml'
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
    create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # Train on each fusion variant
    results = {}

    for fusion_mode in ['raw', 'wbf', 'nms']:
        print("\n" + "="*60)
        print(f"Step 2.{list(['raw', 'wbf', 'nms']).index(fusion_mode) + 1}: Training with {fusion_mode} fusion")
        print("="*60)

        # Convert to YOLO format
        coco_json = coco_dir / fusion_mode / 'annotations.json'
        yolo_dir = base_dir / 'labels_yolo_temp' / fusion_mode

        print(f"\nConverting {fusion_mode} COCO to YOLO...")
        convert_coco_to_yolo(str(coco_json), str(yolo_dir))

        # Create YOLO config
        yaml_path = create_yolo_yaml(fusion_mode, str(base_dir))

        # Train
        print(f"\nTraining YOLOv8n with {fusion_mode} fusion...")
        model = YOLO('yolov8n.pt')

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

    best_mode = df.loc[df['mAP50'].idxmax(), 'fusion_mode']
    best_map = df.loc[df['mAP50'].idxmax(), 'mAP50']
    print(f"\nRecommended fusion strategy: {best_mode} (mAP@0.5={best_map:.4f})")


if __name__ == '__main__':
    main()
