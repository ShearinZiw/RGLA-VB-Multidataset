from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phm_proposed_pipeline import metric_dict


REQUIRED_COLUMNS = {"cut", "y_true", "y_pred"}


def resolve_wear_target(frame: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    scale = float(np.nanmax(np.abs(frame[["y_true", "y_pred"]].to_numpy(dtype=float))))
    return "vb_norm" if scale <= 2.0 else "vb"


def smooth_prediction(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def plot_curves(
    frame: pd.DataFrame,
    smooth: pd.Series,
    output_path: Path,
    wear_target: str,
    title: str | None,
    dpi: int,
) -> None:
    fig, (curve_ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]},
        constrained_layout=True,
    )
    cut = frame["cut"].to_numpy(dtype=float)
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    y_smooth = smooth.to_numpy(dtype=float)
    cut_spacing = float(np.median(np.diff(cut))) if len(cut) > 1 else 1.0
    raw_bar_width = 0.82 * cut_spacing
    smooth_bar_width = 0.40 * cut_spacing

    curve_ax.plot(cut, y_true, color="#202124", linewidth=2.2, label="Ground truth")
    curve_ax.plot(cut, y_pred, color="#56B4E9", linewidth=1.0, alpha=0.42, label="Prediction")
    curve_ax.plot(cut, y_smooth, color="#0072B2", linewidth=2.0, label="Smoothed prediction")
    error_ax.bar(
        cut,
        np.abs(y_pred - y_true),
        width=raw_bar_width,
        color="#E69F00",
        alpha=0.35,
        linewidth=0,
        label="Raw",
        zorder=2,
    )
    error_ax.bar(
        cut,
        np.abs(y_smooth - y_true),
        width=smooth_bar_width,
        color="#D55E00",
        alpha=0.88,
        linewidth=0,
        label="Smoothed",
        zorder=3,
    )

    high_threshold = 0.75 * y_true[-1] if wear_target == "vb" else 0.75
    high_mask = y_true >= high_threshold
    if np.any(high_mask):
        high_start = float(cut[np.flatnonzero(high_mask)[0]])
        for axis in (curve_ax, error_ax):
            axis.axvspan(high_start, float(cut[-1]), color="#F0E442", alpha=0.10, linewidth=0)

    curve_ax.set_title(title or output_path.stem.replace("_curves", ""), fontsize=13)
    curve_ax.set_ylabel("VB (um)" if wear_target == "vb" else "Normalized VB")
    error_ax.set_ylabel("Absolute error")
    error_ax.set_xlabel("Cut")
    curve_ax.legend(loc="best", frameon=False, ncol=3)
    error_ax.legend(loc="best", frameon=False, ncol=2)
    for axis in (curve_ax, error_ax):
        axis.grid(True, color="#DADCE0", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_metrics(output_dir: Path, stem: str, metrics: dict[str, float | int]) -> None:
    row = pd.DataFrame([metrics])
    row.to_csv(output_dir / f"{stem}_metrics.csv", index=False)
    (output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    lines = [
        "# Prediction Metrics",
        "",
        "| MAE | RMSE | R2 | Endpoint | High-wear MAE | Monotonic violations | Pearson |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['r2']:.4f} | "
        f"{metrics['endpoint_error']:.4f} | {metrics['high_wear_mae']:.4f} | "
        f"{int(metrics['monotonic_violations'])} | {metrics['pearson']:.4f} |",
    ]
    (output_dir / f"{stem}_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    frame = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input prediction CSV is empty")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Input contains missing cut, y_true, or y_pred values")
    frame = frame[["cut", "y_true", "y_pred"]].sort_values("cut").reset_index(drop=True)
    if frame["cut"].duplicated().any():
        raise ValueError("Input contains duplicate cut values")

    wear_target = resolve_wear_target(frame, args.wear_target)
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    metrics = metric_dict(y_true, y_pred, no_norm=wear_target == "vb")
    smooth = smooth_prediction(frame["y_pred"], args.smooth_window)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    figure_path = output_dir / f"{stem}_curves.png"
    plot_curves(frame, smooth, figure_path, wear_target, args.title, args.dpi)
    write_metrics(output_dir, stem, metrics)

    summary = {
        "input_csv": str(input_path),
        "figure": str(figure_path),
        "metrics_csv": str(output_dir / f"{stem}_metrics.csv"),
        "metrics_json": str(output_dir / f"{stem}_metrics.json"),
        "metrics_markdown": str(output_dir / f"{stem}_metrics.md"),
        "wear_target": wear_target,
        "smooth_window": args.smooth_window,
        "metrics_use_raw_y_pred": True,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and evaluate a cut,y_true,y_pred prediction CSV."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", help="Defaults to <input-dir>/evaluation.")
    parser.add_argument("--wear-target", choices=("auto", "vb", "vb_norm"), default="auto")
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("smooth-window must be a positive odd integer")
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    evaluate(args)


if __name__ == "__main__":
    main()
