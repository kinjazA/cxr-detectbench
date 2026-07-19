"""Run fusion ablation experiment with YOLOv8n.

Quick 20-epoch training on 3 fusion variants (raw/wbf/nms) to select best strategy.

V15 fix — copies images to real directory (NOT symlink) so YOLO's img2label_paths
replacement (images/ → labels/) works correctly. Uses labels/ directory-level symlink
to switch between variants without rewriting data.yaml.

Data layout:
  ablation/
    data.yaml             (shared across all variants)
    images/train/         (REAL dir — copied PNGs for train split)
    images/val/           (REAL dir — copied PNGs for val split)
    labels/               (DIRECTORY SYMLINK → current variant's labels/)
    raw/labels/train/     (YOLO .txt files for raw fusion)
    raw/labels/val/       (YOLO .txt files for raw fusion)
    wbf/labels/train/     (YOLO .txt files for wbf fusion)
    wbf/labels/val/       (YOLO .txt files for wbf fusion)
    nms/labels/train/     (YOLO .txt files for nms fusion)
    nms/labels/val/       (YOLO .txt files for nms fusion)
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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


def copy_images_split(images_src, ablation_dir, train_ids, val_ids):
    """Copy PNG images to ablation/images/{train,val}/ (one-time, shared by variants).

    Uses REAL directories (not symlinks) so YOLO's Path.resolve() preserves
    the /images/ path segment, which img2label_paths needs for /labels/ replacement.
    """
    images_dir = ablation_dir / 'images'
    train_dir = images_dir / 'train'
    val_dir = images_dir / 'val'

    # Check if already copied
    if train_dir.exists() and val_dir.exists():
        train_count = len(list(train_dir.glob('*.png')))
        val_count = len(list(val_dir.glob('*.png')))
        total_existing = train_count + val_count
        total_needed = len(train_ids) + len(val_ids)
        if total_existing >= total_needed:
            print(f"Images already exist: {train_count} train, {val_count} val ({total_existing}/{total_needed})")
            return images_dir

    images_src = Path(images_src).resolve()
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(images_src.glob('*.png'))
    total = len(pngs)
    print(f"Copying {total} PNG images from {images_src}...")

    # Use bulk copy via cp (avoids ARG_MAX by copying dir contents)
    # Create temp directory for all images
    all_images_dir = ablation_dir / '_images_all'
    all_images_dir.mkdir(parents=True, exist_ok=True)

    # Use cp with source dir contents (not listing individual files)
    result = subprocess.run(
        ['cp', '-r', f'{images_src}/.', f'{all_images_dir}/'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # fallback: use python copy
        print(f"cp bulk copy failed ({result.stderr.strip()}), using shutil per-file...")
        for png in tqdm(pngs, desc="Copying images"):
            shutil.copy2(str(png), str(all_images_dir / png.name))
    else:
        print(f"Bulk copied to {all_images_dir}")
        print(f"  (checking: {len(list(all_images_dir.glob('*.png')))} PNGs)")

    # Sort by split
    train_count = 0
    val_count = 0
    for png in tqdm(all_images_dir.glob('*.png'), desc="Sorting by split"):
        image_id = png.stem
        if image_id in train_ids:
            shutil.move(str(png), str(train_dir / png.name))
            train_count += 1
        elif image_id in val_ids:
            shutil.move(str(png), str(val_dir / png.name))
            val_count += 1
        else:
            png.unlink()  # not in either split (shouldn't happen)

    shutil.rmtree(all_images_dir)

    print(f"Images: {train_count} train, {val_count} val")
    return images_dir


def prepare_variant(fusion_mode, coco_json, ablation_dir, train_ids, val_ids):
    """Generate YOLO labels for a variant and update the labels symlink.

    1. Generates YOLO .txt labels to ablation/{mode}/labels/{train,val}/
    2. Updates ablation/labels/ symlink → ablation/{mode}/labels/
       (YOLO's img2label_paths produces ablation/labels/... paths,
        and OS-level symlink following makes them resolve transparently)
    """
    import sys
    sys.path.insert(0, 'scripts')
    from convert_coco_yolo import coco_to_yolo_format

    coco_json = Path(coco_json)
    mode_dir = ablation_dir / fusion_mode
    labels_dir = mode_dir / 'labels'

    # Generate labels if not already done
    if not (labels_dir / 'train').exists() or not (labels_dir / 'val').exists():
        print(f"Generating YOLO labels for {fusion_mode}...")

        # Generate all YOLO labels to temp
        temp_dir = mode_dir / '_temp'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        coco_to_yolo_format(str(coco_json), str(temp_dir))

        # Split into train/val
        lbl_train = labels_dir / 'train'
        lbl_val = labels_dir / 'val'
        lbl_train.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)

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
                shutil.copy2(str(src), str(lbl_train / txt_file))
                train_cnt += 1
            elif img_id in val_ids:
                shutil.copy2(str(src), str(lbl_val / txt_file))
                val_cnt += 1

        shutil.rmtree(temp_dir)
        print(f"  Labels: {train_cnt} train, {val_cnt} val")
    else:
        train_cnt = len(list((labels_dir / 'train').glob('*.txt')))
        val_cnt = len(list((labels_dir / 'val').glob('*.txt')))
        print(f"  Labels already exist: {train_cnt} train, {val_cnt} val")

    # Update ablation/labels/ symlink to point to this variant's labels
    labels_link = ablation_dir / 'labels'
    if labels_link.is_symlink() or labels_link.exists():
        labels_link.unlink()
    os.symlink(str(mode_dir / 'labels'), str(labels_link), target_is_directory=True)
    print(f"  labels/ symlink → {mode_dir.name}/labels/")


def create_data_yaml(ablation_dir):
    """Create shared data.yaml at ablation/data.yaml."""
    import sys
    sys.path.insert(0, 'scripts')
    from class_names import CLASS_NAMES
    names = [CLASS_NAMES[i] for i in range(14)]

    yaml_path = ablation_dir / 'data.yaml'
    yaml_path.write_text(f"""path: {ablation_dir}
train: images/train
val: images/val

nc: 14
names: {names}
""")
    return str(yaml_path)


def verify_setup(ablation_dir):
    """Verify that image and label files are findable by YOLO."""
    images_train = list((ablation_dir / 'images' / 'train').glob('*.png'))
    images_val = list((ablation_dir / 'images' / 'val').glob('*.png'))
    labels_link = ablation_dir / 'labels'

    print(f"\nSetup verification:")
    print(f"  images/train: {len(images_train)} PNGs")
    print(f"  images/val:   {len(images_val)} PNGs")
    print(f"  labels/       → {os.readlink(str(labels_link)) if labels_link.is_symlink() else 'NOT SYMLINK'}")

    # Check a few labels through the symlink
    train_txts = sorted((labels_link / 'train').glob('*.txt'))
    val_txts = sorted((labels_link / 'val').glob('*.txt'))
    print(f"  labels/train:  {len(train_txts)} txt files")
    print(f"  labels/val:    {len(val_txts)} txt files")

    if len(train_txts) > 0:
        # Verify the first label file is non-empty
        first = train_txts[0]
        content = first.read_text().strip()
        print(f"  Sample label ({first.name}): {content[:80]}")


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
    ablation_dir = base_dir / 'ablation'
    images_src = Path(args.images_dir).resolve()

    # Verify source images
    src_pngs = list(images_src.glob('*.png'))
    print(f"Source images: {images_src} ({len(src_pngs)} PNGs)")

    # 1. Split (same for all 3 fusion variants)
    print("\n" + "=" * 60)
    print("Step 1: 80/20 split")
    print("=" * 60)
    train_ids, val_ids = create_temp_split(args.train_csv, base_dir / 'splits_temp')

    # 2. Copy images (one-time)
    print("\n" + "=" * 60)
    print("Step 2: Copy images to real directory")
    print("=" * 60)
    images_dir = copy_images_split(str(images_src), ablation_dir, train_ids, val_ids)

    # 3. Create shared data.yaml
    yaml_path = create_data_yaml(ablation_dir)
    print(f"\ndata.yaml: {yaml_path}")

    # 4. Train each variant
    results = {}
    for i, mode in enumerate(['raw', 'wbf', 'nms']):
        print(f"\n{'='*60}")
        print(f"Step 4.{i+1}: YOLOv8n + {mode}")
        print(f"{'='*60}")

        coco_json = Path(args.coco_dir) / mode / 'annotations.json'
        prepare_variant(mode, str(coco_json), ablation_dir, train_ids, val_ids)
        verify_setup(ablation_dir)

        try:
            model = YOLO('yolov8n.pt')
            model.train(data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, name=f'fusion_{mode}',
                        project='runs/ablation', exist_ok=True, patience=0, verbose=True)
            m = model.val()
            results[mode] = {'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map)}
            print(f"\n{mode}: mAP@0.5={results[mode]['mAP50']:.4f}"
                  f"  mAP@0.5:0.95={results[mode]['mAP50-95']:.4f}")
        except Exception as e:
            print(f"\nERROR {mode}: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {'mAP50': 0.0, 'mAP50-95': 0.0}

    # 5. Summary
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
