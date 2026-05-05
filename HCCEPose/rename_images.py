from pathlib import Path

#DATASET_ROOT = Path("/home/user/Desktop/isaac_project/debug_output_E_hard_negatives")
DATASET_ROOT = Path("/home/user/Desktop/isaac_project/real_images/")
DRY_RUN = False  # change to False to actually rename

count = 0

for rgb_dir in DATASET_ROOT.glob("frames_960_5fps_new"):
    for img_path in rgb_dir.iterdir():
        if not img_path.is_file():
            continue

        name = img_path.name
        if not (name.startswith("rgb_") | name.startswith("frame_")):
            continue

        if name.startswith("rgb_"):
            new_name = name[len("rgb_"):]   # remove "rgb_"
        if name.startswith("frame_"):
            new_name = name[len("frame_"):]   # remove "rgb_"
        
        new_path = img_path.with_name(new_name)

        if new_path.exists():
            raise FileExistsError(f"Target already exists: {new_path}")

        print(f"{img_path} -> {new_path}")
        count += 1

        if not DRY_RUN:
            img_path.rename(new_path)

print(f"Found {count} files")
if DRY_RUN:
    print("Dry run only. Set DRY_RUN = False to apply changes.")