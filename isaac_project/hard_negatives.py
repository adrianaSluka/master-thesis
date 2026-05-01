"""
Composite carrier-drone crops onto background images to generate
hard-negative training data for YOLO.

Takes transparent PNG crops from the previous step and pastes them
onto random backgrounds with randomization:
  - Position: bottom 40% of the image (where carrier parts naturally appear)
  - Scale: random zoom in/out
  - Horizontal flip: 50% chance
  - Brightness/contrast jitter on the crop to match varied lighting
  - Optional: slight rotation for realism

Outputs:
  - Final composited images (JPG)
  - YOLO-format label files (empty .txt = no objects = hard negative)
  - Or optionally with a "carrier" class label if you want YOLO to learn to ignore it

Usage:
    python compose_hard_negatives.py \
        --crops_dir ./hard_negatives \
        --bg_dir ./backgrounds \
        --output_dir ./yolo_hard_negatives \
        --num_images 3000 \
        --out_w 960 --out_h 540
"""

import argparse
import cv2
import numpy as np
import random
from pathlib import Path
import json

def dump_flat_json_per_row(data: dict, dir: Path, name: str):
    path = dir / name
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(data.items())
        for idx, (k, v) in enumerate(items):
            line = json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
            if idx < len(items) - 1:
                line += ","
            f.write("  " + line + "\n")
        f.write("}\n")

def load_crops(crops_dir):
    """Load all RGBA PNG crops."""
    crops_dir = Path(crops_dir)
    paths = sorted([
        p for p in crops_dir.iterdir()
        if p.suffix.lower() == ".png"
    ])
    print(f"Found {len(paths)} crops in {crops_dir}")
    return paths


def load_backgrounds(bg_dir):
    """Load all background image paths."""
    bg_dir = Path(bg_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in bg_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(paths)} backgrounds in {bg_dir}")
    return paths


def load_rgba(path):
    """Load image as RGBA numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        # Grayscale, shouldn't happen but handle
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # No alpha channel
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge([b, g, r, a])
    return img  # BGRA


def random_crop_background(bg_path, out_w, out_h):
    """Load background and random-crop to output size."""
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return None

    bh, bw = bg.shape[:2]

    # If background is smaller than output, resize up
    if bw < out_w or bh < out_h:
        scale = max(out_w / bw, out_h / bh) * 1.1
        bg = cv2.resize(bg, (int(bw * scale), int(bh * scale)))
        bh, bw = bg.shape[:2]

    # Random crop
    x = random.randint(0, bw - out_w)
    y = random.randint(0, bh - out_h)
    crop = bg[y:y + out_h, x:x + out_w]
    return crop


def augment_crop(crop_bgra, scale_range=(0.5, 1.5), max_rotation=5):
    """
    Augment a carrier drone crop:
      - Random horizontal flip
      - Random scale
      - Random slight rotation
      - Brightness/contrast jitter
    Returns augmented BGRA image.
    """
    h, w = crop_bgra.shape[:2]

    # --- Horizontal flip (50% chance) ---
    if random.random() < 0.5:
        crop_bgra = cv2.flip(crop_bgra, 1)

    # --- Random scale ---
    scale = random.uniform(*scale_range)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    crop_bgra = cv2.resize(crop_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # --- Random slight rotation ---
    if max_rotation > 0:
        angle = random.uniform(-max_rotation, max_rotation)
        rh, rw = crop_bgra.shape[:2]
        M = cv2.getRotationMatrix2D((rw / 2, rh / 2), angle, 1.0)
        # Compute new bounding size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_rw = int(rh * sin + rw * cos)
        new_rh = int(rh * cos + rw * sin)
        M[0, 2] += (new_rw - rw) / 2
        M[1, 2] += (new_rh - rh) / 2
        crop_bgra = cv2.warpAffine(crop_bgra, M, (new_rw, new_rh),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

    # --- Brightness/contrast jitter (color channels only, not alpha) ---
    bgr = crop_bgra[:, :, :3].astype(np.float32)
    alpha = crop_bgra[:, :, 3]

    # Brightness shift
    brightness = random.uniform(-30, 30)
    bgr += brightness

    # Contrast
    contrast = random.uniform(0.7, 1.3)
    mean = bgr.mean()
    bgr = (bgr - mean) * contrast + mean

    bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    result = np.zeros_like(crop_bgra)
    result[:, :, :3] = bgr
    result[:, :, 3] = alpha
    return result


def composite(bg_bgr, crop_bgra, position):
    """
    Alpha-composite crop onto background at given position.
    Position is (x, y) of top-left corner of crop on background.
    """
    px, py = position
    bh, bw = bg_bgr.shape[:2]
    ch, cw = crop_bgra.shape[:2]
    print('bh, bw', bh, bw)
    print('ch, cw', ch, cw)

    # Compute overlap region
    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(bw, px + cw)
    y2 = min(bh, py + ch)

    if x2 <= x1 or y2 <= y1:
        return bg_bgr

    # Crop region that falls within background
    cx1 = x1 - px
    cy1 = y1 - py
    cx2 = cx1 + (x2 - x1)
    cy2 = cy1 + (y2 - y1)

    # Alpha blending
    crop_region = crop_bgra[cy1:cy2, cx1:cx2]
    bg_region = bg_bgr[y1:y2, x1:x2]

    alpha = crop_region[:, :, 3:4].astype(np.float32) / 255.0
    fg = crop_region[:, :, :3].astype(np.float32)
    bg_f = bg_region.astype(np.float32)

    blended = (fg * alpha + bg_f * (1.0 - alpha)).astype(np.uint8)
    bg_bgr[y1:y2, x1:x2] = blended

    return bg_bgr


def get_crop_bbox_yolo(position, crop_h, crop_w, crop_alpha, img_w, img_h):
    """
    Compute YOLO-format bbox [cx, cy, w, h] normalized,
    based on actual visible (non-transparent) pixels.
    Returns None if too small.
    """
    px, py = position

    # Find non-transparent pixels in crop
    ys, xs = np.where(crop_alpha > 128)
    if len(xs) == 0:
        return None

    # In image coordinates
    abs_x1 = px + int(xs.min())
    abs_y1 = py + int(ys.min())
    abs_x2 = px + int(xs.max())
    abs_y2 = py + int(ys.max())

    # Clamp to image
    abs_x1 = max(0, abs_x1)
    abs_y1 = max(0, abs_y1)
    abs_x2 = min(img_w, abs_x2)
    abs_y2 = min(img_h, abs_y2)

    bw = abs_x2 - abs_x1
    bh = abs_y2 - abs_y1

    if bw < 5 or bh < 5:
        return None

    cx = (abs_x1 + bw / 2) / img_w
    cy = (abs_y1 + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h

    return [cx, cy, nw, nh]

def trim_transparent(img_bgra, threshold=10):
    alpha = img_bgra[:, :, 3]
    row_mask = alpha.max(axis=1) >= threshold
    col_mask = alpha.max(axis=0) >= threshold
    if not row_mask.any() or not col_mask.any():
        return None
    y1 = int(np.argmax(row_mask))
    y2 = int(len(row_mask) - np.argmax(row_mask[::-1]))
    x1 = int(np.argmax(col_mask))
    x2 = int(len(col_mask) - np.argmax(col_mask[::-1]))
    return img_bgra[y1:y2, x1:x2].copy()

def generate_hard_negatives(
    crops_dir,
    bg_dir,
    output_dir,
    num_images=3000,
    out_w=960,
    out_h=540,
    scale_range=(0.5, 1.5),
    max_rotation=5,
    label_carrier=False,
    carrier_class_id=1,
    max_crops_per_image=3,
    bottom_fraction=0.4,
):
    """
    Main generation loop.

    Args:
        crops_dir: Directory with transparent carrier PNG crops
        bg_dir: Directory with background images
        output_dir: Where to save composited images + labels
        num_images: How many hard negatives to generate
        out_w, out_h: Output image dimensions
        scale_range: (min_scale, max_scale) for crop augmentation
        max_rotation: Max rotation in degrees
        label_carrier: If True, write YOLO bbox labels for carrier parts
                       If False, write empty .txt (pure hard negatives)
        carrier_class_id: Class ID for carrier if label_carrier=True
        max_crops_per_image: Max number of carrier crops to paste per image
        bottom_fraction: Carrier parts placed in bottom X fraction of image
    """
    output_dir = Path(output_dir)
    img_dir = output_dir 
    #lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    #lbl_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = load_crops(crops_dir)
    bg_paths = load_backgrounds(bg_dir)

    if not crop_paths or not bg_paths:
        print("Need both crops and backgrounds!")
        return
    
    labels = {}

    for i in range(num_images):
        # Random background
        bg = random_crop_background(random.choice(bg_paths), out_w, out_h)
        if bg is None:
            continue

        # Paste 1-N carrier crops
        n_crops = random.randint(1, max_crops_per_image)

        for _ in range(n_crops):
            crop_bgra = load_rgba(random.choice(crop_paths))
            crop_bgra = trim_transparent(crop_bgra)
            if crop_bgra is None:
                continue

            # Augment
            crop_aug = augment_crop(crop_bgra, scale_range, max_rotation)
            ch, cw = crop_aug.shape[:2]

            # Position: bottom portion of image, random horizontal
            # The carrier part should be partially visible (can go off-edge)
            min_y = out_h-ch#int(out_h * (1.0 - bottom_fraction))

            px = random.randint(-cw // 4, out_w - cw * 3 // 4)
            #print(min_y, out_h, ch, out_h - ch // 4)
            py = random.randint(min_y, out_h-ch//2)#out_h - ch // 3)
            # Composite
            bg = composite(bg, crop_aug, (px, py))

            # Label
            # if label_carrier:
            #     bbox = get_crop_bbox_yolo(
            #         (px, py), ch, cw,
            #         crop_aug[:, :, 3], out_w, out_h
            #     )
            #     if bbox:
            #         labels.append(f"{carrier_class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")


        # Save image
        labels[str(i)] = []
        stem = f"rgb_{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save label (empty = no target drone = hard negative)
        # with open(lbl_dir / f"{stem}.txt", "w") as f:
        #     if labels:
        #         f.write("\n".join(labels) + "\n")
        #     # else: empty file = pure hard negative

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{num_images} generated")

    print(f"\nDone: {num_images} images in {img_dir}")
    dump_flat_json_per_row(labels, output_dir, 'scene_gt_info.json')
    dump_flat_json_per_row(labels, output_dir, 'scene_gt.json')
    if not label_carrier:
        print("      Labels are EMPTY (pure hard negatives, no bbox)")
    else:
        print(f"      Carrier labeled as class {carrier_class_id}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Compose hard negatives for YOLO")
    # parser.add_argument("--crops_dir", type=str, required=True,
    #                     help="Dir with transparent carrier PNG crops")
    # parser.add_argument("--bg_dir", type=str, required=True,
    #                     help="Dir with background images")
    # parser.add_argument("--output_dir", type=str, default="./yolo_hard_negatives")
    # parser.add_argument("--num_images", type=int, default=3000)
    # parser.add_argument("--out_w", type=int, default=960)
    # parser.add_argument("--out_h", type=int, default=540)
    # parser.add_argument("--scale_min", type=float, default=0.5)
    # parser.add_argument("--scale_max", type=float, default=1.5)
    # parser.add_argument("--max_rotation", type=float, default=5)
    # parser.add_argument("--label_carrier", action="store_true",
    #                     help="If set, label carrier parts with bboxes (class 1). "
    #                          "Otherwise empty labels (pure hard negatives).")
    # parser.add_argument("--carrier_class_id", type=int, default=1)
    # parser.add_argument("--max_crops_per_image", type=int, default=3)
    # parser.add_argument("--bottom_fraction", type=float, default=0.4,
    #                     help="Place crops in bottom X fraction of image (default 0.4)")

    # args = parser.parse_args()

    generate_hard_negatives(
        crops_dir='/home/user/Desktop/hard_negatives_final',
        bg_dir='/home/user/Desktop/backgrounds/backgrounds',
        output_dir='/home/user/Desktop/hard_negatives_output/000023',
        num_images=2000,
        out_w=960,
        out_h=540,
        scale_range=(0.2, 0.6),
        max_rotation=5,
        #label_carrier=args.label_carrier,
        #carrier_class_id=args.carrier_class_id,
        max_crops_per_image=1,
        bottom_fraction=0.4,
    ) 