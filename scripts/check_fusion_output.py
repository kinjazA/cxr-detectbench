"""Check fusion output statistics."""
import json
from pathlib import Path

coco_dir = Path('data/processed/labels_coco')

print("=" * 60)
print("Fusion Output Statistics")
print("=" * 60)

for mode in ['raw', 'wbf', 'nms']:
    json_path = coco_dir / mode / 'annotations.json'
    if not json_path.exists():
        print(f"{mode:4s}: NOT FOUND")
        continue

    with open(json_path) as f:
        data = json.load(f)

    n_images = len(data['images'])
    n_boxes = len(data['annotations'])
    avg_boxes = n_boxes / n_images if n_images > 0 else 0

    print(f"{mode:4s}: {n_images:5d} images, {n_boxes:6d} boxes, avg {avg_boxes:.2f} boxes/image")

print("=" * 60)
