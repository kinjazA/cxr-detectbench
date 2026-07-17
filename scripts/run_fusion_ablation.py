"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.

Uses YOLOv8's text-file listing mode: each line in train.txt/val.txt is an absolute
image path. No symlinks, no directory structure tricks. Labels sit alongside images
(as *.txt files) in the YOLO labels directory.
"""
import argparse
import json
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

    print(f"Created split: {len(train_ids)} train, {len(val_ids)} val")
    return set(train_ids), set(val_ids)


def prepare_run(fusion_mode, coco_json, base_dir, train_ids, val_ids, images_dir):
    """Prepare YOLO dataset in text-file mode.

    Layout:
      ablation/{mode}/
        train_images.txt   - absolute paths to train PNGs
        val_images.txt     - absolute paths to val PNGs
        labels/            - YOLO label txts (train+val split)
        dataset.yaml

    YOLO reads train_images.txt to know which images to train on.
    Labels sit in labels/ and YOLO discovers them by replacing 'images' → 'labels'
    in the path. Since we separate train/val via text files, labels just go in one dir.
    """
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    base_dir = Path(base_dir)
    work_dir = base_dir / 'ablation' / fusion_mode
    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    labels_dir = work_dir / 'labels'
    labels_dir.mkdir()

    # Convert ALL labels to the labels/ directory
    coco_to_yolo_format(coco_json, str(labels_dir))

    # Read COCO to know which image_ids have labels
    with open(coco_json) as f:
        coco = json.load(f)

    images_dir = Path(images_dir).resolve()
    has_label = {img['id'] for img in coco['images']}

    train_paths = []
    val_paths = []

    for img in coco['images']:
        img_id = img['id']
        png_file = f"{img_id}.png"
        full_path = images_dir / png_file

        # Only include images that actually exist
        if not full_path.exists():
            continue

        if img_id in train_ids:
            train_paths.append(str(full_path))
        elif img_id in val_ids:
            val_paths.append(str(full_path))

    # Write image path lists
    with open(work_dir / 'train_images.txt', 'w') as f:
        f.write('\n'.join(train_paths))

    with open(work_dir / 'val_images.txt', 'w') as f:
        f.write('\n'.join(val_paths))

    # YOLO data.yaml: when train is a .txt file, each line = one image path
    from class_names import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_path = work_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""# Fusion ablation: {fusion_mode}
path: .
train: {work_dir / 'train_images.txt'}
val: {work_dir / 'val_images.txt'}

nc: 14
names: {names}
""")

    print(f"  Train images: {len(train_paths)}, Val images: {len(val_paths)}")
    print(f"  Labels dir: {labels_dir}")
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

    png_count = len(list(images_dir.glob('*.png')))
    print(f"Images: {images_dir} ({png_count} PNGs)")

    # 1. Split
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

        print(f"\n{mode}: mAP@0.5={results[mode]['mAP50']:.4f}  mAP@0.5:0.95={results[mode]['mAP50-95']:.4f}")

    # 3. Summary
    print(f"\n{'='*60}\nResults\n{'='*60}")
    df = pd.DataFrame(results).T.reset_index(names='fusion_mode')
    print(df.to_string(index=False))
    df.to_csv(args.output, index=False)

    valid = df[df['mAP50'] > 0]
    if len(valid):
        best = valid.loc[valid['mAP50'].idxmax(), 'fusion_mode']
        print(f"\nBest: {best} (mAP@0.5={valid.loc[valid['mAP50'].idxmax(), 'mAP50']:.4f})")


if __name__ == '__main__':
    main()
