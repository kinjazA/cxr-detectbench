"""PNG → CLAHE 增强 → 灰度转 3 通道（Phase 2.1）。

数据源：corochann/vinbigdata-chest-xray-original-png（第三方已从 DICOM 转 PNG）
输入：原始 PNG（灰度，保留原始分辨率）
输出：data/processed/images_png/<image_id>.png（CLAHE 后 + 3 通道）

灰度转 3 通道方案：复制通道（cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)）
理由：
  - 医疗影像保留原始灰度信息语义，不引入伪彩色映射的额外假设
  - 预训练权重（ImageNet）期望 3 通道输入，复制通道是最简单的适配
  - 不增加计算开销（仅内存 3 倍）

Phase 2.2：保留一组处理前后对比图（供 README 展示）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def apply_clahe_to_image(img: np.ndarray, clip_limit=2.0, tile_size=8) -> np.ndarray:
    """对单张灰度图做 CLAHE 增强。

    Args:
        img: 灰度图 (H, W)，uint8 或 uint16
        clip_limit: CLAHE 对比度限制
        tile_size: 网格大小

    Returns:
        增强后的灰度图 (H, W)，uint8
    """
    # 如果是 uint16，先归一化到 uint8
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img)
    return enhanced


def gray_to_3ch(img: np.ndarray) -> np.ndarray:
    """灰度转 3 通道（复制通道方案）。"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    else:
        return img  # 已经是 3 通道


def save_comparison_figure(
    original_path: Path,
    processed_img: np.ndarray,
    out_path: Path,
):
    """保存处理前后对比图（Phase 2.2）。"""
    import matplotlib.pyplot as plt

    # 读原图
    original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CLAHE Enhancement Comparison', fontsize=16)

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original PNG (grayscale)')
    axes[0, 0].axis('off')

    # CLAHE 后的灰度图（取处理图的一个通道）
    enhanced_gray = processed_img[:, :, 0] if len(processed_img.shape) == 3 else processed_img
    axes[0, 1].imshow(enhanced_gray, cmap='gray')
    axes[0, 1].set_title('After CLAHE')
    axes[0, 1].axis('off')

    # 直方图对比
    axes[1, 0].hist(original.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlim(0, 255)

    axes[1, 1].hist(enhanced_gray.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[1, 1].set_title('Enhanced Histogram')
    axes[1, 1].set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply CLAHE to chest X-ray PNGs")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to raw PNG images (e.g., /kaggle/input/vinbigdata-chest-xray-original-png/train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/processed/images_png",
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/raw/train.csv",
        help="Path to train.csv (for getting image_id list)",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="CLAHE tile grid size",
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save before/after comparison figure (Phase 2.2)",
    )
    parser.add_argument(
        "--comparison_sample",
        type=str,
        default=None,
        help="Specific image_id for comparison figure (random if None)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读 train.csv 获取所有 image_id
    df = pd.read_csv(args.csv_path)
    image_ids = df['image_id'].unique()

    print(f"Processing {len(image_ids)} images from {input_dir}")
    print(f"Output to {output_dir}")
    print(f"CLAHE params: clip_limit={args.clip_limit}, tile_size={args.tile_size}")

    comparison_saved = False

    for image_id in tqdm(image_ids, desc="Applying CLAHE"):
        # 输入文件（原始 PNG 可能在 train/ 子目录，也可能直接在根目录）
        png_path = input_dir / f"{image_id}.png"
        if not png_path.exists():
            png_path = input_dir / "train" / f"{image_id}.png"
        if not png_path.exists():
            print(f"Warning: {image_id}.png not found, skipping")
            continue

        # 读取灰度图
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: failed to read {png_path}, skipping")
            continue

        # CLAHE
        enhanced = apply_clahe_to_image(img, args.clip_limit, args.tile_size)

        # 转 3 通道
        img_3ch = gray_to_3ch(enhanced)

        # 保存
        out_path = output_dir / f"{image_id}.png"
        cv2.imwrite(str(out_path), img_3ch)

        # Phase 2.2: 保存对比图（只做一次）
        if args.save_comparison and not comparison_saved:
            if args.comparison_sample is None or image_id == args.comparison_sample:
                comparison_path = output_dir.parent / "clahe_comparison.png"
                save_comparison_figure(png_path, img_3ch, comparison_path)
                comparison_saved = True

    print(f"\nDone! Processed images saved to {output_dir}")
    if args.save_comparison and comparison_saved:
        print(f"Comparison figure saved to {output_dir.parent / 'clahe_comparison.png'}")


if __name__ == "__main__":
    main()
