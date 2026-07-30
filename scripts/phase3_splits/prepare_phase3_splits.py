"""Prepare Phase 3 EDA tables and reproducible train/val/test splits.

This script intentionally uses only the Python standard library so it can run in
minimal local or Kaggle environments before the training stack is installed.
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


CLASS_IDS = list(range(14))
NORMAL_LABEL = "normal"


def read_train_csv(path: Path):
    rows = []
    image_labels = defaultdict(set)
    annotation_counts = Counter()
    image_ids = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "class_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            image_id = row["image_id"]
            class_id = int(row["class_id"])
            image_ids.add(image_id)
            rows.append(row)
            if class_id != 14:
                image_labels[image_id].add(class_id)
                annotation_counts[class_id] += 1

    for image_id in image_ids:
        image_labels.setdefault(image_id, set())

    return rows, dict(image_labels), annotation_counts


def image_id_from_metadata_row(row: dict[str, str]) -> str | None:
    for col in ("image_id", "SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID"):
        value = row.get(col)
        if value:
            return Path(value).stem
    fname = row.get("fname")
    if fname:
        return Path(fname).stem
    return None


def read_image_metadata(path: Path | None):
    if path is None or not path.exists():
        return {}, []

    sizes = {}
    fieldnames = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name for name in (reader.fieldnames or []) if name]
        for row in reader:
            image_id = image_id_from_metadata_row(row)
            if not image_id:
                continue

            if "dim0" in row and "dim1" in row and row["dim0"] and row["dim1"]:
                height = int(float(row["dim0"]))
                width = int(float(row["dim1"]))
            elif "Rows" in row and "Columns" in row and row["Rows"] and row["Columns"]:
                height = int(float(row["Rows"]))
                width = int(float(row["Columns"]))
            else:
                continue
            sizes[image_id] = (width, height)

    return sizes, fieldnames


def labels_for_image(label_set: set[int]) -> tuple:
    if not label_set:
        return (NORMAL_LABEL,)
    return tuple(sorted(label_set))


def build_units(image_labels, groups_by_image=None):
    if not groups_by_image:
        return {image_id: [image_id] for image_id in sorted(image_labels)}

    units = defaultdict(list)
    for image_id in sorted(image_labels):
        units[groups_by_image.get(image_id, image_id)].append(image_id)
    return dict(units)


def unit_labels(unit_images, image_labels):
    labels = set()
    for image_id in unit_images:
        labels.update(image_labels[image_id])
    return labels_for_image(labels)


def target_split_sizes(n_items: int, train_ratio: float, val_ratio: float):
    train_n = int(round(n_items * train_ratio))
    val_n = int(round(n_items * val_ratio))
    test_n = n_items - train_n - val_n
    if min(train_n, val_n, test_n) <= 0:
        raise ValueError(f"Invalid split sizes: train={train_n}, val={val_n}, test={test_n}")
    return {"train": train_n, "val": val_n, "test": test_n}


def greedy_multilabel_split(image_labels, seed, train_ratio, val_ratio, groups_by_image=None):
    units = build_units(image_labels, groups_by_image)
    split_names = ("train", "val", "test")
    target_ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    target_sizes = target_split_sizes(len(image_labels), train_ratio, val_ratio)

    unit_label_map = {unit: unit_labels(images, image_labels) for unit, images in units.items()}
    label_totals = Counter()
    for labels in unit_label_map.values():
        for label in labels:
            label_totals[label] += 1

    split_images = {name: [] for name in split_names}
    split_label_counts = {name: Counter() for name in split_names}
    split_image_counts = Counter()

    rng = random.Random(seed)
    ordered_units = list(units)
    rng.shuffle(ordered_units)
    ordered_units.sort(
        key=lambda unit: (
            min(label_totals[label] for label in unit_label_map[unit]),
            -len(unit_label_map[unit]),
            unit,
        )
    )

    for unit in ordered_units:
        images = units[unit]
        labels = unit_label_map[unit]
        best_split = None
        best_score = None
        for split in split_names:
            would_size = split_image_counts[split] + len(images)
            overfill = max(0, would_size - target_sizes[split])
            size_score = (would_size - target_sizes[split]) ** 2 - (
                split_image_counts[split] - target_sizes[split]
            ) ** 2
            score = size_score * 0.05 + overfill * 1000
            for label in labels:
                target_label = label_totals[label] * target_ratios[split]
                before = split_label_counts[split][label]
                after = before + 1
                score += (after - target_label) ** 2 - (before - target_label) ** 2
            if best_score is None or score < best_score:
                best_split = split
                best_score = score

        split_images[best_split].extend(images)
        split_image_counts[best_split] += len(images)
        for label in labels:
            split_label_counts[best_split][label] += 1

    for split in split_names:
        split_images[split].sort()

    return split_images


def exact_slice(items, split_sizes):
    out = {}
    start = 0
    for split in ("train", "val", "test"):
        end = start + split_sizes[split]
        out[split] = items[start:end]
        start = end
    return out


def stratify_abnormal_images(image_labels, image_ids, seed, train_ratio, val_ratio):
    split_names = ("train", "val", "test")
    target_ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    target_sizes = target_split_sizes(len(image_ids), train_ratio, val_ratio)

    label_to_images = defaultdict(set)
    for image_id in image_ids:
        for label in image_labels[image_id]:
            label_to_images[label].add(image_id)

    label_totals = {label: len(ids) for label, ids in label_to_images.items()}
    desired_label_counts = {
        label: {split: label_totals[label] * target_ratios[split] for split in split_names}
        for label in label_totals
    }
    split_images = {name: [] for name in split_names}
    remaining_capacity = dict(target_sizes)

    rng = random.Random(seed)
    unassigned = set(image_ids)

    while unassigned:
        active_labels = [
            label for label, ids in label_to_images.items()
            if ids & unassigned
        ]
        if not active_labels:
            raise RuntimeError("Unassigned abnormal images have no labels.")

        label = min(
            active_labels,
            key=lambda item: (
                len(label_to_images[item] & unassigned),
                label_totals[item],
                item,
            ),
        )
        candidates = list(label_to_images[label] & unassigned)
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda image_id: (
                -len(image_labels[image_id]),
                image_id,
            )
        )
        image_id = candidates[0]
        labels = image_labels[image_id]

        possible_splits = [split for split in split_names if remaining_capacity[split] > 0]
        if not possible_splits:
            raise RuntimeError(f"No split has capacity for image_id={image_id}")

        best_split = max(
            possible_splits,
            key=lambda split: (
                desired_label_counts[label][split],
                sum(desired_label_counts[other][split] for other in labels),
                remaining_capacity[split],
            ),
        )

        split_images[best_split].append(image_id)
        remaining_capacity[best_split] -= 1
        unassigned.remove(image_id)
        for other in labels:
            desired_label_counts[other][best_split] -= 1

    if any(value != 0 for value in remaining_capacity.values()):
        raise RuntimeError(f"Split capacities not exhausted: {remaining_capacity}")

    return split_images


def stratified_image_split(image_labels, seed, train_ratio, val_ratio):
    """Split normal images exactly and abnormal images with multilabel balance."""
    rng = random.Random(seed)
    normal_ids = sorted(image_id for image_id, labels in image_labels.items() if not labels)
    abnormal_ids = sorted(image_id for image_id, labels in image_labels.items() if labels)

    rng.shuffle(normal_ids)
    normal_splits = exact_slice(normal_ids, target_split_sizes(len(normal_ids), train_ratio, val_ratio))
    abnormal_splits = stratify_abnormal_images(
        image_labels=image_labels,
        image_ids=abnormal_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    split_images = {}
    for split in ("train", "val", "test"):
        split_images[split] = sorted(normal_splits[split] + abnormal_splits[split])
    return split_images


def class_count_score(counts, targets, labels, splits):
    score = 0.0
    for label in labels:
        for split in splits:
            target = targets[label][split]
            denom = max(target, 1.0)
            score += ((counts[split][label] - target) ** 2) / denom
    return score


def improve_abnormal_split_by_swaps(
    split_images,
    image_labels,
    seed,
    train_ratio,
    val_ratio,
    iterations,
):
    if iterations <= 0:
        return split_images

    split_names = ("train", "val", "test")
    target_ratios = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    abnormal_by_split = {
        split: [image_id for image_id in image_ids if image_labels[image_id]]
        for split, image_ids in split_images.items()
    }

    label_totals = Counter()
    for labels in image_labels.values():
        for label in labels:
            label_totals[label] += 1
    targets = {
        label: {split: label_totals[label] * target_ratios[split] for split in split_names}
        for label in label_totals
    }

    counts = {split: Counter() for split in split_names}
    for split, image_ids in abnormal_by_split.items():
        for image_id in image_ids:
            for label in image_labels[image_id]:
                counts[split][label] += 1

    rng = random.Random(seed + 1009)
    for _ in range(iterations):
        split_a, split_b = rng.sample(split_names, 2)
        if not abnormal_by_split[split_a] or not abnormal_by_split[split_b]:
            continue

        idx_a = rng.randrange(len(abnormal_by_split[split_a]))
        idx_b = rng.randrange(len(abnormal_by_split[split_b]))
        image_a = abnormal_by_split[split_a][idx_a]
        image_b = abnormal_by_split[split_b][idx_b]
        labels_a = image_labels[image_a]
        labels_b = image_labels[image_b]
        if labels_a == labels_b:
            continue

        affected = labels_a | labels_b
        before = class_count_score(counts, targets, affected, (split_a, split_b))

        for label in labels_a:
            counts[split_a][label] -= 1
            counts[split_b][label] += 1
        for label in labels_b:
            counts[split_b][label] -= 1
            counts[split_a][label] += 1

        after = class_count_score(counts, targets, affected, (split_a, split_b))
        if after < before:
            abnormal_by_split[split_a][idx_a] = image_b
            abnormal_by_split[split_b][idx_b] = image_a
        else:
            for label in labels_a:
                counts[split_a][label] += 1
                counts[split_b][label] -= 1
            for label in labels_b:
                counts[split_b][label] += 1
                counts[split_a][label] -= 1

    normal_by_split = {
        split: [image_id for image_id in image_ids if not image_labels[image_id]]
        for split, image_ids in split_images.items()
    }
    return {
        split: sorted(normal_by_split[split] + abnormal_by_split[split])
        for split in split_names
    }


def write_split_csvs(output_dir: Path, split_images, image_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, image_ids in split_images.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_id", "has_abnormal", "num_classes", "class_ids"]
            )
            writer.writeheader()
            for image_id in image_ids:
                labels = sorted(image_labels[image_id])
                writer.writerow(
                    {
                        "image_id": image_id,
                        "has_abnormal": int(bool(labels)),
                        "num_classes": len(labels),
                        "class_ids": "|".join(str(x) for x in labels),
                    }
                )


def split_summary(split_images, image_labels):
    rows = []
    for split, image_ids in split_images.items():
        total = len(image_ids)
        abnormal = sum(1 for image_id in image_ids if image_labels[image_id])
        normal = total - abnormal
        rows.append(
            {
                "split": split,
                "images": total,
                "abnormal_images": abnormal,
                "normal_images": normal,
                "abnormal_rate": abnormal / total if total else 0.0,
            }
        )
    return rows


def class_image_distribution(split_images, image_labels):
    rows = []
    total_by_class = {
        class_id: sum(1 for labels in image_labels.values() if class_id in labels)
        for class_id in CLASS_IDS
    }
    for class_id in CLASS_IDS:
        row = {"class_id": class_id, "total_images": total_by_class[class_id]}
        for split, image_ids in split_images.items():
            count = sum(1 for image_id in image_ids if class_id in image_labels[image_id])
            row[f"{split}_images"] = count
            row[f"{split}_rate_of_class"] = (
                count / total_by_class[class_id] if total_by_class[class_id] else 0.0
            )
        rows.append(row)
    return rows


def class_annotation_distribution(split_images, train_rows):
    image_to_split = {}
    for split, image_ids in split_images.items():
        for image_id in image_ids:
            image_to_split[image_id] = split

    counts = {class_id: Counter() for class_id in CLASS_IDS}
    for row in train_rows:
        class_id = int(row["class_id"])
        if class_id == 14:
            continue
        split = image_to_split[row["image_id"]]
        counts[class_id][split] += 1

    rows = []
    for class_id in CLASS_IDS:
        total = sum(counts[class_id].values())
        out = {"class_id": class_id, "total_annotations": total}
        for split in ("train", "val", "test"):
            value = counts[class_id][split]
            out[f"{split}_annotations"] = value
            out[f"{split}_rate_of_class"] = value / total if total else 0.0
        rows.append(out)
    return rows


def max_rate_deviation(rows, target_ratios, count_key):
    max_error = 0.0
    max_item = None
    for row in rows:
        if int(row[count_key]) == 0:
            continue
        for split, target_rate in target_ratios.items():
            rate = float(row[f"{split}_rate_of_class"])
            error = abs(rate - target_rate)
            if error > max_error:
                max_error = error
                max_item = (row["class_id"], split, rate, target_rate)
    return max_error, max_item


def validate_split_integrity(split_images, image_labels, train_ratio, val_ratio):
    expected_sizes = target_split_sizes(len(image_labels), train_ratio, val_ratio)
    seen = {}
    for split, image_ids in split_images.items():
        if len(image_ids) != expected_sizes[split]:
            raise ValueError(
                f"{split} split size mismatch: expected={expected_sizes[split]}, got={len(image_ids)}"
            )
        for image_id in image_ids:
            if image_id in seen:
                raise ValueError(f"image_id={image_id} appears in both {seen[image_id]} and {split}")
            seen[image_id] = split

    missing = sorted(set(image_labels) - set(seen))
    extra = sorted(set(seen) - set(image_labels))
    if missing or extra:
        raise ValueError(
            f"Split coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def validate_class_balance(class_rows, target_ratios, max_error):
    error, item = max_rate_deviation(class_rows, target_ratios, "total_images")
    if item and error > max_error:
        class_id, split, rate, target_rate = item
        raise ValueError(
            "Class image split balance exceeds threshold: "
            f"class_id={class_id}, split={split}, rate={rate:.4f}, "
            f"target={target_rate:.4f}, error={error:.4f}, max={max_error:.4f}"
        )


def percentile(values, q):
    if not values:
        return ""
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bbox_size_summary(train_rows, image_sizes):
    by_class = defaultdict(lambda: {"w": [], "h": [], "area": []})
    for row in train_rows:
        class_id = int(row["class_id"])
        if class_id == 14:
            continue
        size = image_sizes.get(row["image_id"])
        if not size:
            continue
        img_w, img_h = size
        box_w = (float(row["x_max"]) - float(row["x_min"])) / img_w
        box_h = (float(row["y_max"]) - float(row["y_min"])) / img_h
        by_class[class_id]["w"].append(box_w)
        by_class[class_id]["h"].append(box_h)
        by_class[class_id]["area"].append(box_w * box_h)

    rows = []
    for class_id in CLASS_IDS:
        values = by_class[class_id]
        row = {"class_id": class_id, "boxes_with_size": len(values["area"])}
        for key in ("w", "h", "area"):
            for name, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
                value = percentile(values[key], q)
                row[f"{key}_{name}"] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def svg_document(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}'
        '.title{font-size:18px;font-weight:700}.label{font-size:12px}'
        '.tick{font-size:11px;fill:#52606d}.note{font-size:11px;fill:#62748a}</style>\n'
        f"{body}\n</svg>\n"
    )


def write_class_split_svg(path: Path, class_rows):
    width = 920
    row_h = 30
    top = 74
    left = 160
    bar_w = 620
    bar_h = 16
    height = top + len(class_rows) * row_h + 56
    colors = {"train": "#2f80ed", "val": "#27ae60", "test": "#f2994a"}

    body = [
        '<text x="24" y="32" class="title">Phase 3 class image distribution by split</text>',
        '<text x="24" y="52" class="note">Stacked bars show each class split into train/val/test image counts.</text>',
    ]
    legend_x = left
    for split in ("train", "val", "test"):
        body.append(f'<rect x="{legend_x}" y="58" width="12" height="12" fill="{colors[split]}"/>')
        body.append(f'<text x="{legend_x + 18}" y="69" class="tick">{split}</text>')
        legend_x += 72

    for idx, row in enumerate(class_rows):
        y = top + idx * row_h
        total = int(row["total_images"])
        label = f"class {row['class_id']} ({total})"
        body.append(f'<text x="24" y="{y + 13}" class="label">{html.escape(label)}</text>')
        x = left
        for split in ("train", "val", "test"):
            count = int(row[f"{split}_images"])
            segment_w = 0 if total == 0 else bar_w * count / total
            body.append(
                f'<rect x="{x:.2f}" y="{y}" width="{segment_w:.2f}" height="{bar_h}" '
                f'fill="{colors[split]}"/>'
            )
            x += segment_w
        body.append(
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" '
            'fill="none" stroke="#cbd2d9" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left + bar_w + 14}" y="{y + 13}" class="tick">'
            f"{row['train_images']} / {row['val_images']} / {row['test_images']}</text>"
        )

    write_svg(path, svg_document(width, height, "\n".join(body)))


def write_bbox_area_svg(path: Path, bbox_rows):
    width = 920
    row_h = 30
    top = 72
    left = 160
    bar_w = 620
    bar_h = 16
    height = top + len(bbox_rows) * row_h + 48
    values = [float(row["area_p50"]) for row in bbox_rows if row["area_p50"] != ""]
    max_value = max(values) if values else 1.0

    body = [
        '<text x="24" y="32" class="title">Phase 3 bbox median area by class</text>',
        '<text x="24" y="52" class="note">Area is normalized by image width and height; smaller bars indicate small-object pressure.</text>',
    ]
    for idx, row in enumerate(bbox_rows):
        y = top + idx * row_h
        value = float(row["area_p50"]) if row["area_p50"] != "" else 0.0
        segment_w = bar_w * value / max_value if max_value else 0
        body.append(f'<text x="24" y="{y + 13}" class="label">class {row["class_id"]}</text>')
        body.append(
            f'<rect x="{left}" y="{y}" width="{segment_w:.2f}" height="{bar_h}" '
            'fill="#9b51e0"/>'
        )
        body.append(
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" '
            'fill="none" stroke="#cbd2d9" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left + bar_w + 14}" y="{y + 13}" class="tick">{value:.4f}</text>'
        )

    write_svg(path, svg_document(width, height, "\n".join(body)))


def format_float(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(
    path: Path,
    summary_rows,
    class_rows,
    class_ann_rows,
    bbox_rows,
    metadata_columns,
    group_col,
    target_ratios,
):
    image_error, image_error_item = max_rate_deviation(
        class_rows, target_ratios, "total_images"
    )
    ann_error, ann_error_item = max_rate_deviation(
        class_ann_rows, target_ratios, "total_annotations"
    )

    lines = [
        "# Phase 3 Split Report",
        "",
        "## Split Summary",
        "",
        "| split | images | abnormal_images | normal_images | abnormal_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['images']} | {row['abnormal_images']} | "
            f"{row['normal_images']} | {row['abnormal_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Class Image Distribution",
            "",
            "| class_id | total | train | val | test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['class_id']} | {row['total_images']} | {row['train_images']} | "
            f"{row['val_images']} | {row['test_images']} |"
        )

    lines.extend(
        [
            "",
            "## Balance Diagnostics",
            "",
            f"- Max per-class image split-rate deviation: {image_error:.4f}"
            + (f" (class {image_error_item[0]}, {image_error_item[1]})" if image_error_item else ""),
            f"- Max per-class annotation split-rate deviation: {ann_error:.4f}"
            + (f" (class {ann_error_item[0]}, {ann_error_item[1]})" if ann_error_item else ""),
            "- Image-level balance is the acceptance criterion for this split. Annotation-level balance is reported as a diagnostic because one image can contain multiple boxes.",
        ]
    )

    lines.extend(
        [
            "",
            "## BBox Size Summary",
            "",
            "Values are normalized by image width/height. Empty values mean image metadata was unavailable.",
            "",
            "| class_id | boxes | w_p50 | h_p50 | area_p50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bbox_rows:
        lines.append(
            f"| {row['class_id']} | {row['boxes_with_size']} | "
            f"{format_float(row['w_p50'])} | {format_float(row['h_p50'])} | "
            f"{format_float(row['area_p50'])} |"
        )

    lines.extend(
        [
            "",
            "## Grouping",
            "",
        ]
    )
    if group_col:
        lines.append(f"Requested group column: `{group_col}`.")
    else:
        lines.append(
            "No patient/study group column was used. The current local metadata does not expose a "
            "patient_id/study_id key, so this split is image-level stratified."
        )
    lines.append("")
    lines.append(f"Metadata columns seen: `{', '.join(metadata_columns)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", default="data/raw/train.csv")
    parser.add_argument("--images_csv", default="data/raw/images.csv")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_col", default=None)
    parser.add_argument(
        "--swap_iterations",
        type=int,
        default=200000,
        help="Number of deterministic abnormal-image swaps used to improve per-class balance.",
    )
    parser.add_argument(
        "--max_class_rate_error",
        type=float,
        default=0.03,
        help="Maximum allowed absolute deviation from target split rates for per-class image counts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_csv = Path(args.train_csv)
    images_csv = Path(args.images_csv) if args.images_csv else None
    output_dir = Path(args.output_dir)

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Ratios must satisfy train>0, val>0, train+val<1")

    train_rows, image_labels, _ = read_train_csv(train_csv)
    image_sizes, metadata_columns = read_image_metadata(images_csv)

    if args.group_col:
        raise NotImplementedError(
            "Grouped splitting is only allowed after a concrete patient/study column is confirmed. "
            f"Requested group_col={args.group_col!r}; available metadata columns={metadata_columns}"
        )

    split_images = stratified_image_split(
        image_labels=image_labels,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_images = improve_abnormal_split_by_swaps(
        split_images=split_images,
        image_labels=image_labels,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        iterations=args.swap_iterations,
    )

    validate_split_integrity(split_images, image_labels, args.train_ratio, args.val_ratio)
    summary_rows = split_summary(split_images, image_labels)
    class_image_rows = class_image_distribution(split_images, image_labels)
    class_ann_rows = class_annotation_distribution(split_images, train_rows)
    target_ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": 1.0 - args.train_ratio - args.val_ratio,
    }
    validate_class_balance(
        class_image_rows,
        target_ratios,
        args.max_class_rate_error,
    )
    bbox_rows = bbox_size_summary(train_rows, image_sizes)

    write_split_csvs(output_dir, split_images, image_labels)
    write_csv(output_dir / "split_summary.csv", summary_rows)
    write_csv(output_dir / "class_image_distribution_by_split.csv", class_image_rows)
    write_csv(output_dir / "class_annotation_distribution_by_split.csv", class_ann_rows)
    write_csv(output_dir / "bbox_size_summary_by_class.csv", bbox_rows)
    write_class_split_svg(output_dir / "class_image_distribution_by_split.svg", class_image_rows)
    write_bbox_area_svg(output_dir / "bbox_median_area_by_class.svg", bbox_rows)
    write_report(
        output_dir / "split_report.md",
        summary_rows,
        class_image_rows,
        class_ann_rows,
        bbox_rows,
        metadata_columns,
        args.group_col,
        target_ratios,
    )

    print(f"Wrote split files and reports to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['split']}: images={row['images']}, abnormal={row['abnormal_images']}, "
            f"normal={row['normal_images']}, abnormal_rate={row['abnormal_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
