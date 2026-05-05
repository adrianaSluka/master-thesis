"""
Batch evaluation script for HccePose drone pose estimation.

Evaluates on two datasets:
  1. PBR (synthetic) — held-out scene 000022 with ground-truth poses
     Metrics: detection rate, ADD, MSPD
  2. Real images — no ground truth
     Metrics: detection rate only

Saves 2D and 6D visualisation images in separate output folders.
"""

import cv2
import json
import os
import sys
import glob
import trimesh
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths  (edit these to match your machine)
# ---------------------------------------------------------------------------
#DATASET_PATH = "/home/user/Desktop/isaac_project/debug_output_D"
#DATASET_PATH = "/home/user/Desktop/isaac_project/debug_output_E_hard_negatives"
''' dataset path includes path to all folders with only 000001-000022 (without hard negatives) but with trained YOLO 
    same path as in s4_p2 after train'''
dataset_path = '/xxx/xxx/debug_output_C'



# PBR held-out scene
PBR_SCENE_DIR = os.path.join(DATASET_PATH, "train_pbr", "000022")
PBR_RGB_DIR = os.path.join(PBR_SCENE_DIR, "rgb")
PBR_MASK_DIR = os.path.join(PBR_SCENE_DIR, "mask")
PBR_MASK_VISIB_DIR = os.path.join(PBR_SCENE_DIR, "mask_visib")

# Real images
''' path to folder with real data and coresponding annotations'''
REAL_IMAGE_DIR = "/xxx/xxx/frames_960_5fps/"
REAL_ANNOTATIONS_DIR = "/xxx/xxx/frames_960_5fps/"
# REAL_IMAGE_DIR = "/home/user/Desktop/isaac_project/real_images/frames_960_5fps/"
# REAL_ANNOTATIONS_DIR = "/home/user/Desktop/isaac_project/real_images/frames_960_5fps/"


# 3D model used to compute ADD / MSPD (PLY expected by BOP convention)
MODEL_PATH = os.path.join(DATASET_PATH, "models", "obj_000001.ply")

# Output folders for visualisations
'''path to the folder, where output will be saved'''
OUTPUT_PBR = "/xxx/xxx/pbr"
OUTPUT_REAL = "/xxx/xxx/real"
#OUTPUT_PBR = "/home/user/Desktop/results2_F_efficientnet/pbr"
#OUTPUT_REAL = "/home/user/Desktop/results2_E_hard_negatives/real"
#OUTPUT_REAL = "/home/user/Desktop/results2_E_hard_negatives/real_new_video"


os.makedirs(OUTPUT_PBR, exist_ok=True)
os.makedirs(OUTPUT_REAL, exist_ok=True)

OBJ_ID = 1
CUDA_DEVICE = "0"

# YOLO detection confidence threshold
CONF_THRESHOLD = 0.8

# ADD / MSPD thresholds
ADD_THRESHOLD_FRACTION = 0.10  # 10 % of model diameter
MSPD_TAU = 20.0  # pixels — BOP default for MSPD correctness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dump_flat_json_per_row(data: dict, dir: Path, name: str):
    path = dir / name
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(data.items())
        for idx, (k, v) in enumerate(items):
            line = json.dumps(k, ensure_ascii=False, cls=NumpyEncoder) + ": " + json.dumps(v, ensure_ascii=False, cls=NumpyEncoder)
            if idx < len(items) - 1:
                line += ","
            f.write("  " + line + "\n")
        f.write("}\n")

def load_scene_json(path):
    """Load a BOP-style per-frame JSON (scene_gt / scene_camera / scene_gt_info)."""
    with open(path, "r") as f:
        return json.load(f)


def load_model_points(ply_path, n_sample=2000):
    """Load PLY model vertices; subsample if needed."""
    mesh = trimesh.load(ply_path, process=False)
    pts = np.array(mesh.vertices, dtype=np.float64)
    if len(pts) > n_sample:
        idx = np.random.default_rng(42).choice(len(pts), n_sample, replace=False)
        pts = pts[idx]
    return pts


def compute_model_diameter(pts):
    """Brute-force diameter (max pairwise distance) — fine for ≤ a few k pts."""
    from scipy.spatial.distance import pdist
    return float(pdist(pts).max())


def compute_add(R_pred, t_pred, R_gt, t_gt, pts):
    """Average Distance of Model Points (ADD)."""
    pred = (R_pred @ pts.T).T + t_pred.reshape(1, 3)
    gt = (R_gt @ pts.T).T + t_gt.reshape(1, 3)
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def project_points(K, R, t, pts):
    """Project 3D points to 2D using pinhole camera (BOP convention)."""
    pts_cam = (R @ pts.T).T + t.reshape(1, 3)
    # BOP / OpenCV: x right, y down, z forward
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def compute_mspd(K, R_pred, t_pred, R_gt, t_gt, pts):
    """
    Maximum Symmetry-aware Projection Distance (MSPD).
    For an asymmetric object this is simply the max over model points of
    the 2-D reprojection error.  (Symmetry handling omitted — the drone
    is assumed asymmetric.)
    """
    proj_pred = project_points(K, R_pred, t_pred, pts)
    proj_gt = project_points(K, R_gt, t_gt, pts)
    dists = np.linalg.norm(proj_pred - proj_gt, axis=1)
    return float(dists.max())

def compute_mssd(R_pred, t_pred, R_gt, t_gt, pts):
    pred = (R_pred @ pts.T).T + t_pred.reshape(1, 3)
    gt = (R_gt @ pts.T).T + t_gt.reshape(1, 3)
    dists = np.linalg.norm(pred - gt, axis=1)
    return float(dists.max())


def cam_k_list_to_matrix(cam_k_list):
    """Convert the flat 9-element cam_K list from BOP JSON to a 3×3 numpy array."""
    return np.array(cam_k_list, dtype=np.float64).reshape(3, 3)


def collect_images(directory, extensions=("*.jpg", "*.jpeg", "*.png")):
    """Recursively collect image paths, sorted."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(paths)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Setup HccePose tester (done once) ----------------------------------
    sys.path.insert(0, os.getcwd())
    from HccePose.bop_loader import bop_dataset
    from HccePose.tester import Tester

    bop_ds = bop_dataset(DATASET_PATH)
    tester = Tester(bop_ds, show_op=True, CUDA_DEVICE=CUDA_DEVICE, efficientnet_key='b4')

    # -- Load 3D model for ADD / MSPD --------------------------------------
    model_pts = load_model_points(MODEL_PATH)
    model_diameter = compute_model_diameter(model_pts)
    add_threshold = ADD_THRESHOLD_FRACTION * model_diameter
    print(f"Model diameter: {model_diameter:.2f}  |  ADD threshold (10%%): {add_threshold:.2f}")

    # ======================================================================
    #  1.  PBR evaluation
    # ======================================================================
    print("\n" + "=" * 70)
    print("  PBR EVALUATION  —  scene 000022")
    print("=" * 70)

    scene_gt = load_scene_json(os.path.join(PBR_SCENE_DIR, "scene_gt.json"))
    scene_cam = load_scene_json(os.path.join(PBR_SCENE_DIR, "scene_camera.json"))

    pbr_images = collect_images(PBR_RGB_DIR)[:50]
    if not pbr_images:
        print(f"WARNING: no images found in {PBR_RGB_DIR}")

    n_total_pbr = 0
    n_detected_pbr = 0
    add_values = []
    mspd_values = []
    mssd_values = []
    add_correct = 0
    mspd_correct = 0
    mssd_correct = 0
    predictions_pbr_dict = {}
    predictions_pbr_stats = {}



    for img_path in pbr_images:
        predicted_pbr_values = {}
        stats_pbr_values = {}
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        # Frame key in JSON — strip leading zeros / "rgb_" prefix
        # BOP BasicWriter names files rgb_000000.jpg; JSON keys are "0", "1", …
        frame_key = stem.replace("rgb_", "").lstrip("0") or "0"

        if frame_key not in scene_gt:
            print(f"  [skip] no GT for frame key '{frame_key}' ({fname})")
            continue

        n_total_pbr += 1
        gt_entry = scene_gt[frame_key][0]  # single object
        cam_entry = scene_cam[frame_key]

        K = cam_k_list_to_matrix(cam_entry["cam_K"])
        R_gt = np.array(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_gt = np.array(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3)

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [skip] cannot read {img_path}")
            continue

        results = tester.predict(K, image, [OBJ_ID],
                                 conf=CONF_THRESHOLD,
                                 confidence_threshold=CONF_THRESHOLD)

        detected = results is not None and 1 in results
        for name, val in results.items():
            # print('name', name, '\n')
            # print('value', val, '\n')
            if name == 1:
                n_detected_pbr += 1

        # Check if a pose was actually returned
        has_pose = False
        R_pred, t_pred = None, None
        # for key, val in results.items():
        #     print("result key", key)
        #     if key == 1:
        #         for i, j in val.items():
        #             print('key 2', i)

        if detected and OBJ_ID in results:
            pose_list = results[1]
            #print(pose_list[0])
            # if len(pose_list) > 0:
            #pose = pose_list[0]  # best detection
            print('Rts', pose_list['Rts'])
            T_pred = pose_list['Rts'][0]
            R_pred = T_pred[:3, :3]
            t_pred = T_pred[:3, 3]
            predicted_pbr_values['T_m2c'] = T_pred
            predicted_pbr_values['R_m2c'] = R_pred
            predicted_pbr_values['t_m2c'] = t_pred

            # R_pred = np.array(pose["R"], dtype=np.float64).reshape(3, 3)
            # t_pred = np.array(pose["t"], dtype=np.float64).reshape(3)
            has_pose = True
            predictions_pbr_dict[str(frame_key)] = predicted_pbr_values


        # if detected:
        #     n_detected_pbr += 1

        if has_pose:
            print('t_pred', t_pred)
            print('t_gt', t_gt)
            add_val = compute_add(R_pred, t_pred, R_gt, t_gt, model_pts)
            mspd_val = compute_mspd(K, R_pred, t_pred, R_gt, t_gt, model_pts)
            mssd_val = compute_mssd(R_pred, t_pred, R_gt, t_gt, model_pts)

            stats_pbr_values['ADD'] = add_val
            stats_pbr_values['MSPD'] = mspd_val
            stats_pbr_values['MSSD'] = mssd_val
            stats_pbr_values['detections'] = len(pose_list['Rts'])

            add_values.append(add_val)
            mspd_values.append(mspd_val)
            mssd_values.append(mssd_val)
            if add_val < add_threshold:
                add_correct += 1
            if mspd_val < MSPD_TAU:
                mspd_correct += 1
            if mssd_val < add_threshold:
                mssd_correct += 1
            predictions_pbr_stats[str(frame_key)] = stats_pbr_values


        # Save visualisations
        out_stem = f"{frame_key.zfill(6)}"
        if detected and "show_2D_results" in results:
            cv2.imwrite(os.path.join(OUTPUT_PBR, f"{out_stem}_2d.jpg"),
                        results["show_2D_results"])
        if detected and "show_6D_vis1" in results:
            cv2.imwrite(os.path.join(OUTPUT_PBR, f"{out_stem}_6d.jpg"),
                        results["show_6D_vis1"])

        if n_total_pbr % 50 == 0:
            print(f"  processed {n_total_pbr} PBR images …")

        #dump_flat_json_per_row(predictions_pbr_dict, Path('/home/user/Desktop/results2_D'), Path('scene_pred_synthetic.json'))
        #dump_flat_json_per_row(predictions_pbr_dict, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred_synthetic.json'))
        dump_flat_json_per_row(predictions_pbr_dict, Path(OUTPUT_PBR), Path('scene_pred_synthetic.json'))

        #dump_flat_json_per_row(predictions_pbr_stats, Path('/home/user/Desktop/results2_D'), Path('scene_pred_stats_synthetic.json'))
        #dump_flat_json_per_row(predictions_pbr_stats, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred_stats_synthetic.json'))
        dump_flat_json_per_row(predictions_pbr_stats, Path('OUTPUT_PBR'), Path('scene_pred_stats_synthetic.json'))






    # -- PBR summary --------------------------------------------------------
    print(f"\nPBR results ({n_total_pbr} images):")
    print(f"  Detection rate : {n_detected_pbr}/{n_total_pbr}"
          f"  ({100 * n_detected_pbr / max(n_total_pbr, 1):.1f}%)")
    if add_values:
        print(f"  ADD  mean      : {np.mean(add_values):.3f}")
        print(f"  ADD  median    : {np.median(add_values):.3f}")
        print(f"  ADD  <10%diam  : {add_correct}/{len(add_values)}"
              f"  ({100 * add_correct / len(add_values):.1f}%)")
        print(f"  MSPD mean      : {np.mean(mspd_values):.2f} px")
        print(f"  MSPD median    : {np.median(mspd_values):.2f} px")
        print(f"  MSPD <{MSPD_TAU}px   : {mspd_correct}/{len(mspd_values)}"
              f"  ({100 * mspd_correct / len(mspd_values):.1f}%)")
        print(f"  MSSD mean      : {np.mean(mssd_values):.2f} px")
        print(f"  MSSD median    : {np.median(mssd_values):.2f} px")
        print(f"  MSSD <{add_correct}px   : {mssd_correct}/{len(mssd_values)}"
              f"  ({100 * mssd_correct / len(mssd_values):.1f}%)")
    else:
        print("  (no poses recovered — cannot compute ADD/MSPD)")

    # ======================================================================
    #  2.  Real-image evaluation
    # ======================================================================
    print("\n" + "=" * 70)
    print("  REAL IMAGE EVALUATION")
    print("=" * 70)

    # Camera intrinsics for the real camera (same as in original inference.py)
    fx, fy = 759.0000013653175, 758.9999936891988
    cx, cy = 480.0, 270.0
    K_real = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)

    real_images = collect_images(REAL_IMAGE_DIR)#[:3]
    scene_gt = load_scene_json(os.path.join(REAL_ANNOTATIONS_DIR, "scene_gt.json"))
    scene_cam = load_scene_json(os.path.join(REAL_ANNOTATIONS_DIR, "scene_camera.json"))
    if not real_images:
        print(f"WARNING: no images found in {REAL_IMAGE_DIR}")

    n_total_real = 0
    n_detected_real = 0
    add_real_values = []
    mspd_real_values = []
    mssd_real_values = []
    add_real_correct = 0
    mspd_real_correct = 0
    mssd_real_correct = 0
    predictions_dict = {}
    predictions_stats = {}

    for img_path in real_images:
        predicted_values = {}
        stats_values = {}
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        img_num = str(int(stem))


        n_total_real += 1
        gt_entry = scene_gt[img_num][0]  # single object
        cam_entry = scene_cam[img_num]

        K = cam_k_list_to_matrix(cam_entry["cam_K"])
        R_gt = np.array(gt_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_gt = np.array(gt_entry["cam_t_m2c"], dtype=np.float64).reshape(3)

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [skip] cannot read {img_path}")
            continue

        results = tester.predict(K_real, image, [OBJ_ID],
                                 conf=CONF_THRESHOLD,
                                 confidence_threshold=CONF_THRESHOLD)

        #detected = results is not None #and np.max(results['show_2D_results']) > 0#"show_2D_results" in results
        detected = results is not None and 1 in results
        print("IMAGE PATH", '\n')
        print(img_path, '\n')


        for name, val in results.items():
            # print('name', name, '\n')
            # print('value', val, '\n')
            if name == 1:
                n_detected_real += 1

        has_pose = False
        R_pred, t_pred = None, None
        if detected and OBJ_ID in results:
            print('IMG NUM', img_num)
            pose_list = results[1]

            #print(pose_list[0])
            # if len(pose_list) > 0:
            #     pose = pose_list[0]  # best detection
            print('Rts', pose_list['Rts'])
            T_pred = pose_list['Rts'][0]
            R_pred = T_pred[:3, :3]
            t_pred = T_pred[:3, 3]
            predicted_values['T_m2c'] = T_pred
            predicted_values['R_m2c'] = R_pred
            predicted_values['t_m2c'] = t_pred

            # R_pred = np.array(pose["R"], dtype=np.float64).reshape(3, 3)
            # t_pred = np.array(pose["t"], dtype=np.float64).reshape(3)
            has_pose = True
            predictions_dict[str(img_num)] = predicted_values
        

        if has_pose:
            print('t_pred', t_pred)
            print('t_gt', t_gt)
            add_val = compute_add(R_pred, t_pred, R_gt, t_gt, model_pts)
            print('ADD VAL', add_val)
            mspd_val = compute_mspd(K, R_pred, t_pred, R_gt, t_gt, model_pts)
            mssd_val = compute_mssd(R_pred, t_pred, R_gt, t_gt, model_pts)


            stats_values['ADD'] = add_val
            stats_values['MSPD'] = mspd_val
            stats_values['MSSD'] = mssd_val
            stats_values['detections'] = len(pose_list['Rts'])

            add_real_values.append(add_val)
            mspd_real_values.append(mspd_val)
            mssd_real_values.append(mssd_val)

            if add_val < add_threshold:
                add_real_correct += 1
            if mspd_val < MSPD_TAU:
                mspd_real_correct += 1
            if mssd_val < add_threshold:
                mssd_real_correct += 1
            predictions_stats[str(img_num)] = stats_values

        # Save visualisations
        #if detected and "show_2D_results" in results:
        cv2.imwrite(os.path.join(OUTPUT_REAL, f"{stem}_2d.jpg"),
                    results["show_2D_results"])
        #if detected and "show_6D_vis1" in results:
        cv2.imwrite(os.path.join(OUTPUT_REAL, f"{stem}_6d.jpg"),
                    results["show_6D_vis1"])

        if n_total_real % 50 == 0:
            print(f"  processed {n_total_real} real images …")

    # -- Real summary -------------------------------------------------------
    print(f"\nReal results ({n_total_real} images):")
    print(f"  Detection rate : {n_detected_real}/{n_total_real}"
          f"  ({100 * n_detected_real / max(n_total_real, 1):.1f}%)")

    if add_real_values:
        print(f"  ADD  mean      : {np.mean(add_real_values):.3f}")
        print(f"  ADD  median    : {np.median(add_real_values):.3f}")
        print(f"  ADD  <10%diam  : {add_real_correct}/{len(add_real_values)}"
              f"  ({100 * add_real_correct / len(add_real_values):.1f}%)")
        print(f"  MSPD mean      : {np.mean(mspd_real_values):.2f} px")
        print(f"  MSPD median    : {np.median(mspd_real_values):.2f} px")
        print(f"  MSPD <{MSPD_TAU}px   : {mspd_real_correct}/{len(mspd_real_values)}"
              f"  ({100 * mspd_real_correct / len(mspd_real_values):.1f}%)")
        print(f"  MSSD mean      : {np.mean(mssd_real_values):.2f} px")
        print(f"  MSSD median    : {np.median(mssd_real_values):.2f} px")
        print(f"  MSSD <{MSPD_TAU}px   : {mssd_real_correct}/{len(mssd_real_values)}"
              f"  ({100 * mssd_real_correct / len(mssd_real_values):.1f}%)")
    
    else:
        print("  (no poses recovered — cannot compute ADD/MSPD)")

    # ======================================================================
    #  Final combined report
    # ======================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    report = {
        "pbr": {
            "n_images": n_total_pbr,
            "n_detected": n_detected_pbr,
            "detection_rate": round(n_detected_pbr / max(n_total_pbr, 1), 4),
            "add_mean": round(float(np.mean(add_values)), 4) if add_values else None,
            "add_median": round(float(np.median(add_values)), 4) if add_values else None,
            "add_accuracy_10pct": round(add_correct / max(len(add_values), 1), 4) if add_values else None,
            "mspd_mean_px": round(float(np.mean(mspd_values)), 4) if mspd_values else None,
            "mspd_median_px": round(float(np.median(mspd_values)), 4) if mspd_values else None,
            "mspd_accuracy": round(mspd_correct / max(len(mspd_values), 1), 4) if mspd_values else None,
        },
        "real": {
            "n_images": n_total_real,
            "n_detected": n_detected_real,
            "detection_rate": round(n_detected_real / max(n_total_real, 1), 4),
        },
    }

    report_path = "/home/user/Desktop/results/eval_report.json"
    #predictions_dict_path = '/home/user/Desktop/isaac_project/real_images/scene_pred.json'
    #predictions_stats_path = '/home/user/Desktop/isaac_project/real_images/scene_pred_stats.json'
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    #dump_flat_json_per_row(predictions_dict, Path('/home/user/Desktop/isaac_project/real_images'), Path('scene_pred.json'))
    #dump_flat_json_per_row(predictions_dict, Path('/home/user/Desktop/results2_D'), Path('scene_pred.json'))
    #dump_flat_json_per_row(predictions_dict, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred.json'))
    #dump_flat_json_per_row(predictions_dict, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred_new_video.json'))
    dump_flat_json_per_row(predictions_dict, Path(OUTPUT_REAL), Path('scene_pred.json'))



    
    #dump_flat_json_per_row(predictions_stats, Path('/home/user/Desktop/isaac_project/real_images'), Path('scene_pred_stats.json'))
    #dump_flat_json_per_row(predictions_stats, Path('/home/user/Desktop/results2_D'), Path('scene_pred_stats.json'))
    #dump_flat_json_per_row(predictions_stats, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred_stats.json'))
    #dump_flat_json_per_row(predictions_stats, Path('/home/user/Desktop/results2_E_hard_negatives'), Path('scene_pred_stats_new_video.json'))
    dump_flat_json_per_row(predictions_stats, Path(OUTPUT_REAL), Path('scene_pred_stats.json'))



    print(f"Report saved to {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()