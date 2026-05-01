"""
Create a 3x3 grid figure for thesis: 3 rows of samples, columns = RGB | Mask | Depth.
Usage: python make_grid.py --data_dir /path/to/train_pbr/000001 --output grid.png
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_rgb(rgb_dir, idx):
    path = rgb_dir / idx #f"rgb_{idx:06d}.jpg"
    if not path.exists():
        path = rgb_dir / idx#f"{idx:06d}.jpg"
    return Image.open(path).convert("RGB")


def load_mask(mask_dir, idx):
    path = mask_dir / f"{idx:06d}_000000.png"
    return Image.open(path).convert("L")

def load_debug(rgb_dir, idx):
    path = rgb_dir / f"{idx:06d}_debug.jpg"
    if not path.exists():
        path = rgb_dir / f"{idx:06d}.jpg"
    return Image.open(path).convert("RGB")


# def load_depth(depth_dir, idx):
#     img = Image.open(depth_dir / f"{idx:06d}.png")
#     arr = np.array(img, dtype=np.float32)
#     # Normalize for visualization
#     valid = arr[arr > 0]
#     if len(valid) > 0:
#         vmin, vmax = valid.min(), valid.max()
#         arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
#     arr[np.array(img) == 0] = 0  # Keep background black
#     return arr

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, required=True,
    #                     help="Path to BOP scene dir (e.g. train_pbr/000001)")
    # parser.add_argument("--indices", type=int, nargs=3, default=[0, 50, 100],
    #                     help="Three frame indices to display")
    # parser.add_argument("--output", type=str, default="grid.png")
    # args = parser.parse_args()

    data_dir = Path('/home/user/Desktop/figures')
    rgb_dir = data_dir / "screenshots"
    #mask_dir = data_dir / "mask"

    fig, axes = plt.subplots(2, 3, figsize=(13, 4))

    col_titles = ["Keypoints", "EPnP", "Final Pose"]
    images = ['screenshot_1_1.png', 'screenshot_1_2.png', 'screenshot_1_3.png',
              'screenshot_2_1.png', 'screenshot_2_2.png', 'screenshot_2_3.png']

    for row, idx in enumerate(images):
        rgb = load_rgb(rgb_dir, idx)
        rgb_crop = crop_center(rgb, 900, 299)
        print(rgb.size)
        #mask = load_mask(mask_dir, idx)
        #depth = load_debug(rgb_dir, idx)
        a = 1 if row > 2 else 0
        print(row%3, a)
        axes[a, row%3].imshow(rgb_crop)
        #axes[row, 1].imshow(mask, cmap="gray")
        #axes[row, 2].imshow(depth, cmap="inferno")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")


    plt.subplots_adjust(left=0.01, right=0.95, top=0.85, bottom=0.1, wspace=0.01, hspace=0.01)
    plt.savefig('/home/user/Desktop/figures/figure_2', dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {'/home/user/Desktop/figures'}")


if __name__ == "__main__":
    main()