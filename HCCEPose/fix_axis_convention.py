"""
fix_scene_gt.py

Patches already-generated scene_gt.json files to correct two bugs:

  Bug 1 — coordinate convention:
    Isaac Sim camera uses OpenGL convention (Z out of screen, Y up).
    BOP / VisPy expects Z into screen, Y down.
    Fix: apply flip = diag(1, -1, -1) to both R and t.

  Bug 2 — unit scale:
    Isaac Sim stage units are meters; BOP PLY files are typically in mm.
    Fix: multiply t by UNIT_SCALE (default 1000 for m→mm).

Usage:
    python fix_scene_gt.py --dataset_path /path/to/debug_output

    If your PLY vertices are in centimeters instead of millimetres, pass:
    python fix_scene_gt.py --dataset_path /path/to/debug_output --unit_scale 100
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


FLIP_YZ = np.diag([1.0, -1.0, -1.0])   # OpenGL → BOP axis convention


def fix_entry(entry: dict, unit_scale: float) -> dict:
    """Apply coordinate-convention flip and unit rescale to one GT entry."""
    entry = dict(entry)  # shallow copy

    R = np.array(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(entry["cam_t_m2c"], dtype=np.float64).reshape(3)

    R_fixed = FLIP_YZ @ R
    t_fixed = FLIP_YZ @ t * unit_scale

    entry["cam_R_m2c"] = R_fixed.flatten().tolist()
    entry["cam_t_m2c"] = t_fixed.flatten().tolist()

    return entry


def fix_scene_gt_json(json_path: Path, unit_scale: float, dry_run: bool = False):
    with open(json_path, "r") as f:
        data = json.load(f)

    # scene_gt.json  →  { "0": [ {entry}, ... ], "1": [...], ... }
    fixed = {}
    for frame_key, entries in data.items():
        fixed[frame_key] = [fix_entry(e, unit_scale) for e in entries]

    if dry_run:
        # Print first frame as a sanity check
        first_key = next(iter(fixed))
        print(f"  [dry_run] first frame ({first_key}):")
        print(f"    t_original : {data[first_key][0]['cam_t_m2c'][:3]}")
        print(f"    t_fixed    : {fixed[first_key][0]['cam_t_m2c'][:3]}")
        return

    # Backup original
    backup_path = json_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)
        print(f"  backed up → {backup_path.name}")

    with open(json_path, "w") as f:
        json.dump(fixed, f)
    print(f"  fixed    → {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                        help="Root of the BOP dataset (contains train_pbr/)")
    parser.add_argument("--folder_name", default="train_pbr",
                        help="Sub-folder to process (default: train_pbr)")
    parser.add_argument("--unit_scale", type=float, default=1,
                        help="Multiply t by this to convert to mm. "
                             "1000 if Isaac stage = meters (default), "
                             "100 if Isaac stage = centimeters.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would change without writing files.")
    args = parser.parse_args()

    folder = Path(args.dataset_path) / args.folder_name
    print('folder', folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    scene_dirs = sorted(folder.glob("*/"))
    if not scene_dirs:
        raise FileNotFoundError(f"No scene sub-folders found under {folder}")

    print(f"unit_scale = {args.unit_scale}  (dry_run={args.dry_run})")
    print(f"Processing {len(scene_dirs)} scene(s) in {folder}\n")

    for scene_dir in scene_dirs:
        gt_path = scene_dir / "scene_gt.json"
        if not gt_path.exists():
            print(f"  skip (no scene_gt.json): {scene_dir.name}")
            continue
        print(f"scene: {scene_dir.name}")
        fix_scene_gt_json(gt_path, args.unit_scale, dry_run=args.dry_run)

    print("\nDone.")
    if not args.dry_run:
        print("Original files backed up as *.json.bak — delete them once you've verified the output labels look correct.")


if __name__ == "__main__":
    main()