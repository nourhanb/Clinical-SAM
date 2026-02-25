#!/usr/bin/env python3
"""
Plot Phase I PPO convergence: mean episodic return vs PPO update.
Reads CSV from Phase I training (outputs/logs/phase1_ppo_train_metrics.csv)
and saves figure to outputs/plots/phase1_ppo_convergence.png (and optionally .pdf).

Usage:
    python tools/plot_phase1_convergence.py --csv outputs/logs/phase1_ppo_train_metrics.csv --out outputs/plots/phase1_ppo_convergence.png
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(y, window=5):
    """Moving average; window must be odd. Returns array same length; edges use partial window."""
    if window < 2 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser(description="Plot Phase I PPO convergence")
    parser.add_argument("--csv", type=str, default="outputs/logs/phase1_ppo_train_metrics.csv",
                        help="Path to phase1_ppo_train_metrics.csv")
    parser.add_argument("--out", type=str, default="outputs/plots/phase1_ppo_convergence.png",
                        help="Output figure path (.png or .pdf)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Moving average window for mean curve (0 to disable). Default 5.")
    parser.add_argument("--no-raw", action="store_true",
                        help="Do not plot raw curve (only smoothed).")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}. Run Phase I training first to generate it.")

    df = pd.read_csv(args.csv)
    if "update" not in df.columns or "ep_return_mean" not in df.columns:
        raise ValueError(f"CSV must have columns 'update' and 'ep_return_mean'. Found: {list(df.columns)}")

    # Drop rows where ep_return_mean is missing (e.g. empty string from no episodes that iteration)
    df = df.replace("", np.nan)
    df = df.dropna(subset=["ep_return_mean"])
    if df.empty:
        raise ValueError("No valid rows with ep_return_mean in CSV.")

    updates = df["update"].values.astype(int)
    mean_ret = df["ep_return_mean"].values.astype(float)
    has_std = "ep_return_std" in df.columns
    std_ret = df["ep_return_std"].values.astype(float) if has_std else None
    if has_std and (std_ret is not None):
        # Replace nan std with 0 for shading
        std_ret = np.nan_to_num(std_ret, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Raw curve (faint) if requested and smoothing is on
    if not args.no_raw and args.smooth and args.smooth >= 2:
        ax.plot(updates, mean_ret, color="tab:blue", alpha=0.25, linewidth=0.8, label="Raw")

    # Smoothed or raw main curve
    if args.smooth and args.smooth >= 2:
        mean_smooth = smooth(mean_ret, window=args.smooth)
        ax.plot(updates, mean_smooth, color="tab:blue", linewidth=1.5, label="Mean (smoothed)" if not args.no_raw else "Mean episodic return")
    else:
        ax.plot(updates, mean_ret, color="tab:blue", linewidth=1.5, label="Mean episodic return")

    # Shaded region: std or 95% CI (1.96*std/sqrt(n) would need n; we use std as variability across rollouts)
    if has_std and std_ret is not None and np.any(std_ret > 0):
        if args.smooth and args.smooth >= 2:
            std_smooth = smooth(std_ret, window=args.smooth)
            upper = mean_smooth + std_smooth
            lower = mean_smooth - std_smooth
        else:
            upper = mean_ret + std_ret
            lower = mean_ret - std_ret
        ax.fill_between(updates, lower, upper, color="tab:blue", alpha=0.2, label="±1 std")

    ax.set_xlabel("PPO Updates", fontsize=11)
    ax.set_ylabel("Mean Episodic Return", fontsize=11)
    ax.set_title("Phase I PPO Convergence", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Optionally save PDF alongside if output was PNG
    if out_path.lower().endswith(".png"):
        pdf_path = out_path[:-4] + ".pdf"
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
