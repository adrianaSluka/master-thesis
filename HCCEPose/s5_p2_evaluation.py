"""
Pose Estimation Evaluation Script
==================================
Reads two JSON files (synthetic & real) with per-image metrics:
  {"0": {"ADD": 0.011, "MSPD": 0.689, "MSSD": 0.014, "detections": 1}, ...}

Computes:
  - ADD / MSSD / MSPD: mean, median
  - Recall at BOP thresholds (5%–50% diameter for ADD/MSSD; 5–50 px × r for MSPD)
  - Average Recall (AR) for each metric
  - BOP AR = (AR_MSSD + AR_MSPD) / 2
  - AUC for ADD (area under the recall-vs-threshold curve)
  - Average detections per image

Plots:
  1. AUC curve for ADD (synthetic vs real overlaid)
  2. 2×3 histogram grid (rows: synthetic / real; columns: ADD / MSSD / MSPD)

Usage:
  python evaluate_pose.py
  # or with custom paths:
  python evaluate_pose.py --syn path/to/synthetic.json --real path/to/real.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import trimesh



# ─── Config ───────────────────────────────────────────────────────────────
#DEFAULT_SYN  = "/home/user/Desktop/results2_D/scene_pred_stats_synthetic.json"
#DEFAULT_REAL = "/home/user/Desktop/results2_D/scene_pred_stats.json"
# DEFAULT_SYN  = "/home/user/Desktop/results2_E_hard_negatives/scene_pred_stats_synthetic.json"
# DEFAULT_REAL = "/home/user/Desktop/results2_E_hard_negatives/scene_pred_stats.json"
# DEFAULT_SYN  = "/home/user/Desktop/results2_E_hard_negatives/scene_pred_stats_synthetic_new_video.json"
# DEFAULT_REAL = "/home/user/Desktop/results2_E_hard_negatives/scene_pred_stats_new_video.json"
# DATASET_PATH = "/home/user/Desktop/isaac_project/debug_output_E_hard_negatives"
'''dataset path includes path to all folders with only 000001-000022 (without hard negatives) (with trained YOLO and HccePose)'''
'''DEFAULT_SYN and DEFAULT_REAL are paths to outputs of s5_p1_inference'''
DATASET_PATH = '/xxx/xxx/debug_output_C'
DEFAULT_SYN  = "/xxx/xxx/scene_pred_stats_synthetic.json"
DEFAULT_REAL = "/xxx/xxx/scene_pred_stats.json"



# BOP thresholds
DIAMETER_FRACTIONS = np.arange(0.05, 0.55, 0.05)   # 0.05 … 0.50  (10 steps)
MSPD_R = 960 / 640                                  # resolution scaling factor
MSPD_BASE_PX = np.arange(5, 55, 5)                  # 5 … 50 base pixels
MSPD_THRESHOLDS = MSPD_BASE_PX * MSPD_R             # 7.5, 15, …, 75
MODEL_PATH = os.path.join(DATASET_PATH, "models", "obj_000001.ply")


# AUC sweep
AUC_MAX = 0.5       # sweep ADD threshold from 0 to this (diameter fraction)
AUC_STEPS = 1000


# ─── Helpers ──────────────────────────────────────────────────────────────

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

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_arrays(data):
    """Pull per-image metric arrays from the JSON dict."""
    add_vals  = []
    mssd_vals = []
    mspd_vals = []
    det_vals  = []

    for key in sorted(data.keys(), key=lambda k: int(k)):
        entry = data[key]
        if "ADD" in entry:
            add_vals.append(entry["ADD"])
        if "MSSD" in entry:
            mssd_vals.append(entry["MSSD"])
        if "MSPD" in entry:
            mspd_vals.append(entry["MSPD"])
        det_vals.append(entry.get("detections", 0))

    return (np.array(add_vals),  np.array(mssd_vals),
            np.array(mspd_vals), np.array(det_vals))


def recall_at_threshold(values, threshold):
    """Fraction of values strictly below threshold."""
    if len(values) == 0:
        return 0.0
    return float(np.sum(values < threshold)) / len(values)


def compute_auc_curve(add_vals, max_thresh=AUC_MAX, n_steps=AUC_STEPS, model_diameter=2.11):
    """
    AUC for ADD
    -----------
    Sweep threshold τ from 0 to max_thresh.  At each τ compute:
        recall(τ) = fraction of images where ADD < τ
    This gives a monotonically non-decreasing curve.
    AUC = area under this curve, normalised by max_thresh so AUC ∈ [0, 1].

    A method whose errors are clustered near zero gets high AUC even if a
    few outliers exist; a method with all errors just below the cutoff gets
    lower AUC despite identical single-threshold recall.
    """
    thresholds = np.linspace(0, max_thresh, n_steps + 1)
    recalls = np.array([recall_at_threshold(add_vals, t*model_diameter) for t in thresholds])
    auc = float(np.trapz(recalls, thresholds) / max_thresh)
    return thresholds, recalls, auc


def compute_all_metrics(data, add_threshold, model_diameter):
    add, mssd, mspd, det = extract_arrays(data)
    n = len(det)

    # Recall at BOP thresholds
    add_recalls  = [recall_at_threshold(add,  t) for t in add_threshold]
    mssd_recalls = [recall_at_threshold(mssd, t) for t in add_threshold]
    mspd_recalls = [recall_at_threshold(mspd, t) for t in MSPD_THRESHOLDS]

    # Average Recall
    ar_add  = float(np.mean(add_recalls))
    ar_mssd = float(np.mean(mssd_recalls))
    ar_mspd = float(np.mean(mspd_recalls))
    ar_bop  = (ar_mssd + ar_mspd) / 2.0

    # AUC for ADD
    auc_thresh, auc_recall, auc_val = compute_auc_curve(add, max_thresh=AUC_MAX, n_steps=AUC_STEPS, model_diameter=model_diameter)

    return {
        "n": n,
        "add": add,   "mssd": mssd,   "mspd": mspd,   "det": det,
        "add_mean":  float(np.mean(add))  if len(add)  else None,
        "add_med":   float(np.median(add))if len(add)  else None,
        "mssd_mean": float(np.mean(mssd)) if len(mssd) else None,
        "mssd_med":  float(np.median(mssd))if len(mssd)else None,
        "mspd_mean": float(np.mean(mspd)) if len(mspd) else None,
        "mspd_med":  float(np.median(mspd))if len(mspd)else None,
        "avg_det":   float(np.mean(det)),
        "add_recalls":  add_recalls,
        "mssd_recalls": mssd_recalls,
        "mspd_recalls": mspd_recalls,
        "ar_add": ar_add, "ar_mssd": ar_mssd, "ar_mspd": ar_mspd, "ar_bop": ar_bop,
        "auc_thresh": auc_thresh, "auc_recall": auc_recall, "auc": auc_val,
    }


# ─── Printing ─────────────────────────────────────────────────────────────

def print_metrics(label, m, add_threshold):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Images evaluated:       {m['n']}")
    print(f"  Avg detections / image: {m['avg_det']:.2f}")
    print()
    print(f"  ADD   mean: {m['add_mean']:.6f}   median: {m['add_med']:.6f}")
    print(f"  MSSD  mean: {m['mssd_mean']:.6f}   median: {m['mssd_med']:.6f}")
    print(f"  MSPD  mean: {m['mspd_mean']:.6f}   median: {m['mspd_med']:.6f}")
    print()
    print(f"  ADD  AUC (0–{AUC_MAX}d): {m['auc'] * 100:.2f}%")
    print()

    # Recall table
    header = "  Thresh |   ADD    |   MSSD   |   MSPD"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, frac in enumerate(add_threshold):
        pct = int(frac * 100)
        mspd_px = MSPD_THRESHOLDS[i]
        print(f"  {pct:3d}%   | {m['add_recalls'][i]*100:6.1f}%  | "
              f"{m['mssd_recalls'][i]*100:6.1f}%  | "
              f"{m['mspd_recalls'][i]*100:6.1f}%  (< {mspd_px:.1f} px)")

    print()
    print(f"  AR ADD:           {m['ar_add']  * 100:.2f}%")
    print(f"  AR MSSD:          {m['ar_mssd'] * 100:.2f}%")
    print(f"  AR MSPD:          {m['ar_mspd'] * 100:.2f}%")
    print(f"  BOP AR (MSSD+MSPD)/2: {m['ar_bop'] * 100:.2f}%")


# ─── Plotting ─────────────────────────────────────────────────────────────

def plot_auc(syn_m, real_m, out_path):
    """Plot 1: ADD AUC curves overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if syn_m is not None:
        ax.plot(syn_m["auc_thresh"], syn_m["auc_recall"],
                color="#6366f1", linewidth=2,
                label=f"Synthetic  (AUC = {syn_m['auc']*100:.1f}%)")
        ax.fill_between(syn_m["auc_thresh"], syn_m["auc_recall"],
                        alpha=0.12, color="#6366f1")

    if real_m is not None:
        print(real_m["auc_thresh"])
        print(real_m["auc_recall"])
        ax.plot(real_m["auc_thresh"], real_m["auc_recall"],
                color="#f97316", linewidth=2,
                label=f"Real  (AUC = {real_m['auc']*100:.1f}%)")
        ax.fill_between(real_m["auc_thresh"], real_m["auc_recall"],
                        alpha=0.12, color="#f97316")

    ax.set_xlabel("ADD threshold (fraction of diameter)")
    ax.set_ylabel("Recall")
    ax.set_title("ADD AUC Curve")
    ax.set_xlim(0, AUC_MAX)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_histograms(syn_m, real_m, out_path, n_bins=30):
    """
    Plot 2: 2×3 histogram grid.
    Row 0 = Synthetic,  Row 1 = Real
    Columns = ADD, MSSD, MSPD
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    configs = [
        ("ADD",  "add",  "diameter fraction"),
        ("MSSD", "mssd", "diameter fraction"),
        ("MSPD", "mspd", "pixels"),
    ]

    datasets = [
        ("Synthetic", syn_m,  "#6366f1"),
        ("Real",      real_m, "#f97316"),
    ]
    

    for row, (ds_label, m, color) in enumerate(datasets):
        for col, (metric_name, key, unit) in enumerate(configs):
            ax = axes[row, col]
            if m is not None and len(m[key]) > 0:
                vals = m[key]
                ax.hist(vals, bins=n_bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
                ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1, label=f"mean={np.mean(vals):.4f}")
                ax.axvline(np.median(vals), color="green", linestyle="--", linewidth=1, label=f"median={np.median(vals):.4f}")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")

            ax.set_title(f"{ds_label} + Hard Negatives — {metric_name}", fontsize=11)
            if metric_name=='MSPD':
                ax.set_xlabel(f"{metric_name} ({'pixels'})", fontsize=9)
            else:
                ax.set_xlabel(f"{metric_name} ({'meters'})", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)

    fig.suptitle("Error Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    model_pts = load_model_points(MODEL_PATH)
    model_diameter = compute_model_diameter(model_pts)
    add_threshold = [model_diameter*i for i in DIAMETER_FRACTIONS]
    parser = argparse.ArgumentParser(description="Evaluate 6D pose estimation results")

    #out_dir = Path('/home/user/Desktop/results2_D')
    #out_dir = Path('/home/user/Desktop/results2_E_hard_negatives')
    out_dir = Path('/xxx/xxx/results')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & compute
    syn_m, real_m = None, None

    syn_path = Path(DEFAULT_SYN)#Path(args.syn)
    if syn_path.exists():
        print(f"Loading synthetic: {syn_path}")
        syn_m = compute_all_metrics(load_json(syn_path), add_threshold, model_diameter)
        print_metrics("SYNTHETIC DATA", syn_m, add_threshold)
    else:
        print(f"Synthetic file not found: {syn_path}")

    real_path = Path(DEFAULT_REAL)#Path(args.real)
    if real_path.exists():
        print(f"Loading real: {real_path}")
        real_m = compute_all_metrics(load_json(real_path), add_threshold, model_diameter)
        print_metrics("REAL DATA", real_m, add_threshold)
    else:
        print(f"Real file not found: {real_path}")

    if syn_m is None and real_m is None:
        print("No data loaded — nothing to plot.")
        return

    # Plots
    print("\nGenerating plots...")
    plot_auc(syn_m, real_m, out_dir / "add_auc_curve.png")
    plot_histograms(syn_m, real_m, out_dir / "error_histograms.png")
    print("\nDone.")


if __name__ == "__main__":
    main()