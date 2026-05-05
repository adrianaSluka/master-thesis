"""
Merge seg_1 and seg_2 renumbered data into a single folder.
- Copies all images from seg_1_renumbered and seg_2_renumbered into frames_960_5fps_new
- Merges scene_camera_seg1.json + scene_camera_seg2.json -> scene_camera.json
- Merges scene_gt_seg1.json + scene_gt_seg2.json -> scene_gt.json
"""

import json
import shutil
from pathlib import Path

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

# ── PATHS ──────────────────────────────────────────────────────────────
SRC = Path("/home/user/Desktop/isaac_project/real_images/frames_960_new_video")
DST = Path("/home/user/Desktop/isaac_project/real_images/frames_960_5fps_new")

SEG1_IMAGES = SRC / "seg_1_renumbered"
SEG2_IMAGES = SRC / "seg_2_renumbered"

CAM_SEG1 = SRC / "scene_camera_seg1.json"
CAM_SEG2 = SRC / "scene_camera_seg2.json"
GT_SEG1  = SRC / "scene_gt_seg1.json"
GT_SEG2  = SRC / "scene_gt_seg2.json"

DST.mkdir(parents=True, exist_ok=True)

# ── COPY IMAGES ────────────────────────────────────────────────────────
copied = 0
for folder in [SEG1_IMAGES, SEG2_IMAGES]:
    for img in sorted(folder.glob("*.*")):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            shutil.copy2(img, DST / img.name)
            copied += 1
    print(f"Copied from {folder.name}: done")

print(f"Total images copied: {copied}")

# ── MERGE JSONs ────────────────────────────────────────────────────────
for seg1_path, seg2_path, out_name in [
    (CAM_SEG1, CAM_SEG2, "scene_camera.json"),
    (GT_SEG1,  GT_SEG2,  "scene_gt.json"),
]:
    with open(seg1_path, "r") as f:
        d1 = json.load(f)
    with open(seg2_path, "r") as f:
        d2 = json.load(f)

    # Check for key collisions
    overlap = set(d1.keys()) & set(d2.keys())
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping keys in {out_name}: {sorted(overlap, key=int)[:10]}...")

    merged = {**d1, **d2}
    merged = dict(sorted(merged.items(), key=lambda x: int(x[0])))

    out_path = DST / out_name
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(merged, f, indent=2, ensure_ascii=False)

    dump_flat_json_per_row(merged, DST, out_name)

    print(f"Wrote {out_path} ({len(merged)} entries, range {min(merged, key=lambda k: int(k))}-{max(merged, key=lambda k: int(k))})")

print("Done!")