#!/usr/bin/env python3
"""
Parse HCCEPose training logs: losses (every Nth iteration), evaluation
errors (per-level HCCE error rates for front/back x/y/z), and max_acc
(ADD(-S) pose accuracy — the best per-object score across the test set
at each evaluation checkpoint).

Generates line-plot PNGs and optionally saves CSVs.

Usage
-----
  python parse_hccepose_log.py log4.file                       # default: every 500 iters
  python parse_hccepose_log.py log4.file --step 1000           # every 1000 iters
  python parse_hccepose_log.py log4.file --csv                 # also write CSVs
  python parse_hccepose_log.py log4.file --no-plot             # skip plot generation

What is max_acc?
----------------
HCCEPose evaluates on a test set every 500 training iterations.  For each
evaluation it computes the ADD(-S) accuracy (fraction of poses whose average
vertex error is < 10 % of the object diameter) on multiple sub-splits and
reports the per-split scores as an array.  `max acc` is the *best* score
across those splits — i.e. the peak ADD(-S) accuracy at that checkpoint.
It grows from ~0 at the start to ~0.97 for a well-trained model.
"""

import re
import csv
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── regex patterns ──────────────────────────────────────────────────────────
_LOSS_RE = re.compile(
    r"iteration_step:\s*(\d+)\s+"
    r"loss_front:\s*([\d.eE+-]+)\s+"
    r"loss_back:\s*([\d.eE+-]+)\s+"
    r"loss_mask:\s*([\d.eE+-]+)\s+"
    r"total_loss:\s*([\d.eE+-]+)"
)

_MAX_ACC_RE = re.compile(r"max acc:\s+([\d.eE+-]+)")

_ACC_ARRAY_RE = re.compile(r"^\[([\d.\s\n]+)\]\s*$")

_STEP_RE = re.compile(r"step(\d+)")

_ERROR_RE = re.compile(
    r"(front|back)\(([xyz])\)\s*error:\s*\[([\d.\s]+)\]"
)


# ── parsing ─────────────────────────────────────────────────────────────────
def parse_log(log_path, loss_step=500):
    losses = []          # list of dicts
    evals = []           # list of dicts  {step, max_acc, errors: {front_x: [...], ...}}

    current_eval = {}
    acc_array_buf = None  # accumulates multi-line accuracy arrays

    with open(log_path, "r") as f:
        for line in f:
            # ── accumulate accuracy array (may span 2 lines) ──
            if acc_array_buf is not None:
                acc_array_buf += " " + line.strip()
                if "]" in line:
                    # parse the complete array
                    inner = acc_array_buf.split("[")[1].split("]")[0]
                    current_eval["acc_all"] = [float(v) for v in inner.split()]
                    acc_array_buf = None
                continue

            # ── detect start of accuracy array (line starting with '[' and digits) ──
            stripped = line.strip()
            if stripped.startswith("[") and "error" not in line and "rank" not in line:
                # Quick sanity check: first char after '[' should be a digit or space
                inner_start = stripped[1:].lstrip()
                if inner_start and (inner_start[0].isdigit() or inner_start[0] == '-'):
                    acc_array_buf = stripped
                    if "]" in stripped:
                        inner = acc_array_buf.split("[")[1].split("]")[0]
                        current_eval["acc_all"] = [float(v) for v in inner.split()]
                        acc_array_buf = None
                    continue

            # ── losses ──
            m = _LOSS_RE.search(line)
            if m:
                it = int(m.group(1))
                if it % loss_step == 0:
                    losses.append({
                        "iteration": it,
                        "loss_front": float(m.group(2)),
                        "loss_back": float(m.group(3)),
                        "loss_mask": float(m.group(4)),
                        "total_loss": float(m.group(5)),
                    })
                continue

            # ── max_acc ──
            m = _MAX_ACC_RE.search(line)
            if m:
                current_eval["max_acc"] = float(m.group(1))
                continue

            # ── checkpoint path → extract step number ──
            m = _STEP_RE.search(line)
            if m and "best check point" in line:
                current_eval["step"] = int(m.group(1))
                continue

            # ── per-component errors ──
            m = _ERROR_RE.search(line)
            if m:
                surface = m.group(1)   # front / back
                comp = m.group(2)      # x / y / z
                vals = [float(v) for v in m.group(3).split()]
                key = f"{surface}_{comp}"
                current_eval.setdefault("errors", {})[key] = vals

                # after back(z) we have all 6 → flush
                if key == "back_z" and "step" in current_eval and "max_acc" in current_eval:
                    evals.append(current_eval)
                    current_eval = {}

    return losses, evals


# ── pretty-print ────────────────────────────────────────────────────────────
def print_losses(losses):
    hdr = f"{'iter':>8} | {'loss_front':>12} | {'loss_back':>12} | {'loss_mask':>12} | {'total_loss':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in losses:
        print(
            f"{r['iteration']:>8d} | "
            f"{r['loss_front']:>12.6f} | "
            f"{r['loss_back']:>12.6f} | "
            f"{r['loss_mask']:>12.6f} | "
            f"{r['total_loss']:>12.6f}"
        )


def print_evals(evals):
    print(f"\n{'step':>8} | {'max_acc':>8} | error levels (1-8) ...")
    print("-" * 90)
    for e in evals:
        print(f"{e['step']:>8d} | {e['max_acc']:>8.4f} |", end="")
        # just show front_x as a representative
        vals = e["errors"].get("front_x", [])
        print("  front_x:", " ".join(f"{v:.2f}" for v in vals))


# ── CSV export ──────────────────────────────────────────────────────────────
def save_losses_csv(losses, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "loss_front", "loss_back", "loss_mask", "total_loss"])
        w.writeheader()
        w.writerows(losses)


def save_evals_csv(evals, path):
    fieldnames = ["step", "max_acc"]
    for surface in ("front", "back"):
        for comp in "xyz":
            for lvl in range(1, 9):
                fieldnames.append(f"{surface}_{comp}_L{lvl}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in evals:
            row = {"step": e["step"], "max_acc": e["max_acc"]}
            for surface in ("front", "back"):
                for comp in "xyz":
                    key = f"{surface}_{comp}"
                    vals = e["errors"].get(key, [0] * 8)
                    for i, v in enumerate(vals):
                        row[f"{key}_L{i+1}"] = v
            w.writerow(row)


# ── plotting ────────────────────────────────────────────────────────────────
def plot_all(losses, evals, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Training losses ──────────────────────────────────────────────
    if losses:
        iters = [r["iteration"] for r in losses[1:]]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("HCCEPose Training Losses", fontsize=15, fontweight="bold")

        for ax, key, color, title in zip(
            axes.flat,
            ["total_loss", "loss_front", "loss_back", "loss_mask"],
            ["#2d2d2d", "#e74c3c", "#3498db", "#2ecc71"],
            ["Total Loss", "Front Surface Loss", "Back Surface Loss", "Mask Loss"],
        ):
            vals = [r[key] for r in losses[1:]]
            ax.plot(iters, vals, color=color, linewidth=1.2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(out_dir / "losses.png", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir / 'losses.png'}")

    # ── 2. max_acc (ADD(-S) accuracy) over training ─────────────────────
    if evals:
        steps = [e["step"] for e in evals]
        accs = [e["max_acc"] for e in evals]

        # Gather all sub-split arrays
        has_splits = all("acc_all" in e for e in evals)

        fig, ax = plt.subplots(figsize=(12, 5))

        if has_splits:
            all_arrs = np.array([e["acc_all"] for e in evals])  # (n_evals, 15)
            n_splits = all_arrs.shape[1]

            # Individual sub-split lines (transparent)
            for j in range(n_splits):
                ax.plot(steps, all_arrs[:, j],
                        color="#aaaaaa", linewidth=0.6, alpha=0.35)

            # Min–max shaded band
            arr_min = all_arrs.min(axis=1)
            arr_max = all_arrs.max(axis=1)
            ax.fill_between(steps, arr_min, arr_max,
                            color="#e74c3c", alpha=0.12, label="min–max range")

            # Mean line
            arr_mean = all_arrs.mean(axis=1)
            ax.plot(steps, arr_mean, color="#3498db", linewidth=1.5,
                    label="mean across splits")

            # Max line (what "max acc" reports)
            ax.plot(steps, accs, color="#e74c3c", linewidth=2.0,
                    marker="o", markersize=3, label="max acc (best split)")

            ax.legend(fontsize=9, loc="lower right")
        else:
            ax.plot(steps, accs, color="#e74c3c", linewidth=1.5,
                    marker="o", markersize=3)

        ax.set_title("ADD(-S) Pose Accuracy over Training\n"
                     "(15 test sub-splits shown individually in gray)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ADD(-S) Accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "max_acc.png", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir / 'max_acc.png'}")

    # ── 3. Per-level HCCE errors (front & back, x/y/z) ─────────────────
    if evals:
        n_levels = len(evals[0]["errors"].get("front_x", []))
        steps = [e["step"] for e in evals]
        colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_levels))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "HCCE Per-Level Error Rates  (lower = better)\n"
            "Each line is one encoding level (L1 = coarsest … L8 = finest)",
            fontsize=13, fontweight="bold",
        )

        for row_idx, surface in enumerate(("front", "back")):
            for col_idx, comp in enumerate("xyz"):
                ax = axes[row_idx, col_idx]
                key = f"{surface}_{comp}"
                for lvl in range(n_levels):
                    vals = [e["errors"][key][lvl] for e in evals]
                    ax.plot(steps, vals, color=colors[lvl], linewidth=1.1,
                            label=f"L{lvl+1}", alpha=0.85)
                ax.set_title(f"{surface}({comp})", fontsize=11)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Error Rate")
                ax.set_ylim(-0.02, 0.55)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 2:
                    ax.legend(fontsize=8, loc="upper right", ncol=2)

        plt.tight_layout()
        fig.savefig(out_dir / "hcce_errors.png", dpi=150)
        plt.close(fig)
        print(f"  saved {out_dir / 'hcce_errors.png'}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parse & visualize HCCEPose training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("log_file", help="Path to log file")
    parser.add_argument("--step", type=int, default=500,
                        help="Sample losses every N iterations (default: 500)")
    parser.add_argument("--csv", action="store_true",
                        help="Also write losses.csv and evals.csv")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory for output plots & CSVs (default: cwd)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    losses, evals = parse_log(args.log_file, loss_step=args.step)

    print(f"Parsed {len(losses)} loss entries, {len(evals)} eval checkpoints\n")
    print_losses(losses)
    print_evals(evals)

    if args.csv:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_losses_csv(losses, out / "losses.csv")
        save_evals_csv(evals, out / "evals.csv")
        print(f"\nCSVs saved to {out}")

    if not args.no_plot:
        print("\nGenerating plots...")
        plot_all(losses, evals, args.out_dir)
        print("Done.")


if __name__ == "__main__":
    main()