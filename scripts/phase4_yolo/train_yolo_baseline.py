"""Train a reproducible Ultralytics YOLO baseline and write a compact summary.

This is the Phase 4 baseline entry point. It intentionally keeps P100-friendly
defaults and copies only small artifacts plus best/last checkpoints to the
summary directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_dataset_summary(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        return str(dst_dir / src.name)
    return None


def class_metric_rows(metrics):
    box = getattr(metrics, "box", None)
    if box is None:
        return []

    names = getattr(metrics, "names", {}) or {}
    maps = getattr(box, "maps", None)
    ap_class_index = getattr(box, "ap_class_index", None)
    if maps is None or ap_class_index is None:
        return []

    all_ap = getattr(box, "all_ap", None)
    rows = []
    for idx, class_id in enumerate(ap_class_index):
        class_id = int(class_id)
        row = {
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "map50_95": safe_float(maps[class_id]) if class_id < len(maps) else None,
        }
        if all_ap is not None and idx < len(all_ap) and len(all_ap[idx]) > 0:
            row["map50"] = safe_float(all_ap[idx][0])
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


def write_metrics(summary_dir: Path, metrics, args, save_dir: Path):
    box = metrics.box
    overall = {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "workers": args.workers,
        "cache": args.cache,
        "run_name": args.name,
        "save_dir": str(save_dir),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": safe_float(getattr(box, "mp", None)),
        "recall": safe_float(getattr(box, "mr", None)),
    }

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "metrics.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (summary_dir / "metrics.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in overall.items()) + "\n",
        encoding="utf-8",
    )
    write_csv(summary_dir / "per_class_metrics.csv", class_metric_rows(metrics))


def copy_run_artifacts(save_dir: Path, summary_dir: Path, include_last: bool):
    copied = []
    for name in ("args.yaml", "results.csv"):
        copied_path = copy_if_exists(save_dir / name, summary_dir)
        if copied_path:
            copied.append(copied_path)

    weights_dir = summary_dir / "weights"
    copied_path = copy_if_exists(save_dir / "weights" / "best.pt", weights_dir)
    if copied_path:
        copied.append(copied_path)
    if include_last:
        copied_path = copy_if_exists(save_dir / "weights" / "last.pt", weights_dir)
        if copied_path:
            copied.append(copied_path)
    return copied


def train(args):
    from ultralytics import YOLO

    project = Path(args.project).resolve()
    summary_dir = Path(args.summary_dir).resolve()

    model = YOLO(args.model)
    train_result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(project),
        exist_ok=args.exist_ok,
        patience=0,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        cache=args.cache,
        plots=False,
        save_period=-1,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(
        getattr(train_result, "save_dir", None)
        or getattr(trainer, "save_dir", None)
        or project / args.name
    )
    best_weight = save_dir / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Expected best checkpoint was not created: {best_weight}")

    val_model = YOLO(str(best_weight))
    metrics = val_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="val",
        workers=args.workers,
        plots=False,
    )

    write_metrics(summary_dir, metrics, args, save_dir)
    dataset_summary = Path(args.dataset_summary)
    if dataset_summary.exists():
        shutil.copy2(dataset_summary, summary_dir / "dataset_summary.csv")
    copied = copy_run_artifacts(save_dir, summary_dir, args.include_last)

    print(f"BASELINE_SAVE_DIR={save_dir}")
    print(f"BASELINE_SUMMARY_DIR={summary_dir}")
    print(f"BASELINE_MAP50={float(metrics.box.map50):.6f}")
    print(f"BASELINE_MAP50_95={float(metrics.box.map):.6f}")
    for path in copied:
        print(f"BASELINE_ARTIFACT={path}")
    for row in read_dataset_summary(summary_dir / "dataset_summary.csv"):
        print(
            "BASELINE_DATASET="
            f"{row['split']},{row['images']},{row['linked_images']},"
            f"{row['label_files']},{row['boxes']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml path")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="/kaggle/working/yolo_baseline_runs")
    parser.add_argument("--name", default="yolov8n_wbf_phase3_img640_ep50")
    parser.add_argument("--summary_dir", default="/kaggle/working/yolo_baseline_summary")
    parser.add_argument(
        "--dataset_summary",
        default="data/processed/yolo_wbf_phase3/dataset_summary.csv",
    )
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exist_ok", action="store_true")
    parser.add_argument("--include_last", action="store_true")
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
