"""
Renumber seg_2 data so it continues from seg_1's last frame (302).

- Merges all scene_gt*.json files from archive2/scene_gt/ into one dict
- Merges all scene_camera*.json files from archive2/scene_cameras/ into one dict
- Shifts all keys by OFFSET (302) so seg_2 frame 1 -> 303, frame 2 -> 304, etc.
- Renames seg_2 images accordingly (frame_000001.jpg -> frame_000303.jpg)
- Writes scene_gt_seg2.json and scene_camera_seg2.json with new keys
"""

import json
import os
import shutil
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────
OFFSET = 0#302  # seg_1 max frame number

# Adjust these paths if needed
BASE = Path("/home/user/Desktop/isaac_project/real_images/frames_960_new_video")
ARCHIVE2 = BASE / "archive2"
SEG2_SRC = BASE #/ "seg_2"

# Where to put renamed images
SEG2_DST = BASE / "seg_1_renumbered"

# Where to save the new JSONs
JSON_OUT = BASE

GT_DIR = ARCHIVE2 / "scene_gt"
CAM_DIR = ARCHIVE2 / "scene_cameras"

# ── HELPERS ────────────────────────────────────────────────────────────

def merge_jsons_from_dir(directory: Path, prefix_list: str) -> dict:
    """
    Merge all JSON files in a directory that start with the given prefix.
    All keys are collected into a single dict.
    Duplicate keys: later files overwrite earlier ones (shouldn't happen
    if the range files are non-overlapping).
    """
    merged = {}
    json_files = [directory / pref for pref in prefix_list]#sorted(directory.glob(f"{prefix}*.json"))
    print(f"  Found {len(json_files)} JSON files matching '{prefix_list}*' in {directory}:")
    for jf in json_files:
        print(f"    - {jf.name}")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
    return merged


def shift_keys(data: dict, offset: int) -> dict:
    """Shift all integer keys by offset, preserving sort order."""
    shifted = {}
    for k, v in data.items():
        new_key = str(int(k) + offset)
        shifted[new_key] = v
    # Sort by integer key
    return dict(sorted(shifted.items(), key=lambda x: int(x[0])))


# ── MERGE JSONs ────────────────────────────────────────────────────────
print("=" * 60)
print("Merging scene_gt JSONs...")
gt_merged = merge_jsons_from_dir(GT_DIR, ['scene_gt1-26.json', 'scene_gt26-126.json', 'scene_gt131-206.json', 'scene_gt211-301.json', 'scene_gt3.json'])#['scene_gt.json', 'scene_gt1.json'])
print(f"  Total keys in merged scene_gt: {len(gt_merged)}")
print(f"  Key range: {min(gt_merged, key=lambda k: int(k))} - {max(gt_merged, key=lambda k: int(k))}")

print("\nMerging scene_camera JSONs...")
cam_merged = merge_jsons_from_dir(CAM_DIR, ['scene_camera1-26.json', 'scene_camera26-126.json', 'scene_camera131-206.json', 'scene_camera211-301.json', 'scene_camera3.json'])
print(f"  Total keys in merged scene_camera: {len(cam_merged)}")
print(f"  Key range: {min(cam_merged, key=lambda k: int(k))} - {max(cam_merged, key=lambda k: int(k))}")

# ── INNER JOIN ─────────────────────────────────────────────────────────
common_keys = set(gt_merged.keys()) & set(cam_merged.keys())
print(f"\nInner join: {len(common_keys)} common keys")

gt_joined = {k: gt_merged[k] for k in common_keys}
cam_joined = {k: cam_merged[k] for k in common_keys}

# ── SHIFT KEYS ─────────────────────────────────────────────────────────
gt_shifted = shift_keys(gt_joined, 0) #OFFSET)
cam_shifted = shift_keys(cam_joined, 0) #OFFSET)

print(f"\nAfter shifting by {OFFSET}:")
print(f"  scene_gt  key range: {min(gt_shifted, key=lambda k: int(k))} - {max(gt_shifted, key=lambda k: int(k))}")
print(f"  scene_cam key range: {min(cam_shifted, key=lambda k: int(k))} - {max(cam_shifted, key=lambda k: int(k))}")

# ── WRITE NEW JSONs ────────────────────────────────────────────────────
gt_out = JSON_OUT / "scene_gt_seg1.json"
cam_out = JSON_OUT / "scene_camera_seg1.json"

with open(gt_out, "w", encoding="utf-8") as f:
    json.dump(gt_shifted, f, indent=2, ensure_ascii=False)
print(f"\nWrote {gt_out} ({len(gt_shifted)} entries)")

with open(cam_out, "w", encoding="utf-8") as f:
    json.dump(cam_shifted, f, indent=2, ensure_ascii=False)
print(f"Wrote {cam_out} ({len(cam_shifted)} entries)")

# ── RENAME SEG_2 IMAGES ───────────────────────────────────────────────
SEG2_DST.mkdir(parents=True, exist_ok=True)

# Build set of valid old frame numbers (before shifting)
valid_old_keys = set(int(k) for k in common_keys)

renamed_count = 0
skipped_count = 0

# Find all jpg/png images in seg_2
image_files = sorted(SEG2_SRC.glob("frame_*.jpg")) + sorted(SEG2_SRC.glob("frame_*.png"))

for img_path in image_files:
    # Extract number from filename like frame_000001.jpg
    stem = img_path.stem  # e.g. "frame_000001"
    ext = img_path.suffix  # e.g. ".jpg"

    try:
        old_num = int(stem.split("_")[-1])
    except ValueError:
        print(f"  Skipping (can't parse number): {img_path.name}")
        skipped_count += 1
        continue

    if old_num not in valid_old_keys:
        skipped_count += 1
        continue

    new_num = old_num + OFFSET
    new_name = f"frame_{new_num:06d}{ext}"
    dst_path = SEG2_DST / new_name
    shutil.copy2(img_path, dst_path)
    renamed_count += 1

print(f"\nRenamed {renamed_count} images -> {SEG2_DST}")
print(f"Skipped {skipped_count} images (not in JSON keys)")
print("=" * 60)
print("Done!")