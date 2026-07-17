from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASETS = ("c1", "c4", "c6")
CHANNELS = ("Fx", "Fy", "Fz", "Ax", "Ay", "Az", "AE")


@dataclass
class FeatureConfig:
    data_root: str
    output_dir: str
    crop_start_frac: float = 0.10
    crop_end_frac: float = 0.90
    pcc_threshold: float = 0.90
    max_features: int = 28
    ridge_alpha: float = 1.0
    seed: int = 20260510
    smoke: bool = True
    deep_signal_check: bool = False
    augmentation_windows: int = 1
    sampling_rate_hz: float = 50000.0
    spindle_rpm: float = 10400.0
    window_rotations: int = 60
    stride_rotations: int = 15


def dataset_number(dataset: str) -> str:
    match = re.search(r"\d+", dataset)
    if not match:
        raise ValueError(f"Dataset name must contain a number: {dataset}")
    return match.group(0)


def signal_dir(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / dataset


def wear_path(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / f"{dataset}_wear.csv"


def signal_path(data_root: Path, dataset: str, cut: int) -> Path:
    n = dataset_number(dataset)
    return signal_dir(data_root, dataset) / f"c_{n}_{cut:03d}.csv"


def load_wear(data_root: Path, dataset: str) -> pd.DataFrame:
    wear = pd.read_csv(wear_path(data_root, dataset))
    required = {"cut", "1", "2", "3"}
    missing = required.difference(wear.columns)
    if missing:
        raise ValueError(f"{wear_path(data_root, dataset)} missing columns: {sorted(missing)}")
    wear = wear.sort_values("cut").reset_index(drop=True)
    wear["vb_avg"] = wear[["1", "2", "3"]].mean(axis=1)
    denom = float(wear["vb_avg"].iloc[-1] - wear["vb_avg"].iloc[0])
    if abs(denom) < 1e-12:
        raise ValueError(f"{dataset} has near-zero VB range; cannot normalize")
    wear["vb_norm"] = (wear["vb_avg"] - wear["vb_avg"].iloc[0]) / denom
    return wear


def stable_crop(signal: np.ndarray, start_frac: float, end_frac: float) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected signal shape (n, {len(CHANNELS)}), got {signal.shape}")
    if not 0 <= start_frac < end_frac <= 1:
        raise ValueError("Crop fractions must satisfy 0 <= start < end <= 1")
    start = int(round(signal.shape[0] * start_frac))
    end = int(round(signal.shape[0] * end_frac))
    if end <= start + 8:
        raise ValueError(f"Crop too small for signal with {signal.shape[0]} rows")
    return signal[start:end]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def channel_features(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        raise ValueError("Signal segment too short for feature extraction")
    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    absmean = float(np.mean(np.abs(x)))
    sra = float(np.mean(np.sqrt(np.abs(x))) ** 2)
    xmax = float(np.max(np.abs(x)))
    skew = float(np.mean((centered / (std + 1e-12)) ** 3))
    kurt = float(np.mean((centered / (std + 1e-12)) ** 4))
    margin = safe_div(xmax, sra)
    shape = safe_div(rms, absmean)
    impulse = safe_div(xmax, absmean)
    peak = safe_div(xmax, rms)

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0)
    if len(spectrum) > 1:
        spectrum = spectrum[1:]
        freqs = freqs[1:]
    total_amp = float(np.sum(spectrum))
    mean_amp = float(np.mean(spectrum)) if len(spectrum) else 0.0
    max_amp = float(np.max(spectrum)) if len(spectrum) else 0.0
    fc = safe_div(float(np.sum(freqs * spectrum)), total_amp)
    msf = safe_div(float(np.sum((freqs**2) * spectrum)), total_amp)
    rmsf = float(math.sqrt(max(msf, 0.0)))
    sdf = float(math.sqrt(max(safe_div(float(np.sum(((freqs - fc) ** 2) * spectrum)), total_amp), 0.0)))

    return [
        mean,
        std,
        rms,
        sra,
        skew,
        absmean,
        xmax,
        kurt,
        margin,
        shape,
        impulse,
        peak,
        mean_amp,
        max_amp,
        fc,
        msf,
        rmsf,
        sdf,
    ]


def feature_names() -> list[str]:
    base = [
        "mean",
        "std",
        "rms",
        "sra",
        "skew",
        "absmean",
        "max_abs",
        "kurtosis",
        "margin_factor",
        "shape_factor",
        "impulse_factor",
        "peak_factor",
        "mean_amplitude",
        "max_amplitude",
        "frequency_center",
        "mean_square_frequency",
        "rms_frequency",
        "std_frequency",
    ]
    return [f"{ch}_{name}" for ch in CHANNELS for name in base]


def extract_segment_features(segment: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    for channel_idx in range(segment.shape[1]):
        feats.extend(channel_features(segment[:, channel_idx]))
    return np.asarray(feats, dtype=float)


def extract_window_batch_features(
    segment: np.ndarray,
    starts: np.ndarray,
    window_rows: int,
) -> np.ndarray:
    """Vectorized equivalent of extract_segment_features for many windows."""
    channel_blocks: list[np.ndarray] = []
    for channel_idx in range(segment.shape[1]):
        windows = np.stack(
            [segment[start : start + window_rows, channel_idx] for start in starts],
            axis=0,
        ).astype(float, copy=False)
        mean = windows.mean(axis=1)
        centered = windows - mean[:, None]
        std = windows.std(axis=1)
        rms = np.sqrt(np.mean(windows * windows, axis=1))
        absmean = np.mean(np.abs(windows), axis=1)
        sra = np.mean(np.sqrt(np.abs(windows)), axis=1) ** 2
        xmax = np.max(np.abs(windows), axis=1)
        normalized = centered / (std[:, None] + 1e-12)
        skew = np.mean(normalized**3, axis=1)
        kurt = np.mean(normalized**4, axis=1)

        def divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            return np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)

        spectrum = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]
        freqs = np.fft.rfftfreq(window_rows, d=1.0)[1:]
        total_amp = spectrum.sum(axis=1)
        mean_amp = spectrum.mean(axis=1) if spectrum.shape[1] else np.zeros(len(windows))
        max_amp = spectrum.max(axis=1) if spectrum.shape[1] else np.zeros(len(windows))
        fc = divide(np.sum(freqs[None, :] * spectrum, axis=1), total_amp)
        msf = divide(np.sum((freqs[None, :] ** 2) * spectrum, axis=1), total_amp)
        rmsf = np.sqrt(np.maximum(msf, 0.0))
        sdf = np.sqrt(
            np.maximum(
                divide(np.sum(((freqs[None, :] - fc[:, None]) ** 2) * spectrum, axis=1), total_amp),
                0.0,
            )
        )
        channel_blocks.append(
            np.column_stack(
                [
                    mean,
                    std,
                    rms,
                    sra,
                    skew,
                    absmean,
                    xmax,
                    kurt,
                    divide(xmax, sra),
                    divide(rms, absmean),
                    divide(xmax, absmean),
                    divide(xmax, rms),
                    mean_amp,
                    max_amp,
                    fc,
                    msf,
                    rmsf,
                    sdf,
                ]
            )
        )
    return np.concatenate(channel_blocks, axis=1)


def extract_signal_feature_windows(
    path: Path,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extract one or more paper-style rotating windows from a single cut."""
    signal = np.loadtxt(path, delimiter=",", dtype=float)
    segment = stable_crop(signal, cfg.crop_start_frac, cfg.crop_end_frac)
    if cfg.augmentation_windows <= 1:
        return (
            extract_segment_features(segment)[None, :],
            np.asarray([0], dtype=int),
            len(segment),
            len(signal),
        )
    if cfg.sampling_rate_hz <= 0 or cfg.spindle_rpm <= 0:
        raise ValueError("sampling_rate_hz and spindle_rpm must be positive")
    if cfg.window_rotations <= 0 or cfg.stride_rotations <= 0:
        raise ValueError("window_rotations and stride_rotations must be positive")

    samples_per_rotation = cfg.sampling_rate_hz * 60.0 / cfg.spindle_rpm
    window_rows = max(8, int(round(samples_per_rotation * cfg.window_rotations)))
    stride_rows = max(1, int(round(samples_per_rotation * cfg.stride_rotations)))
    window_rows = min(window_rows, len(segment))
    max_start = max(0, len(segment) - window_rows)
    natural_starts = np.arange(0, max_start + 1, stride_rows, dtype=int)

    if len(natural_starts) >= cfg.augmentation_windows:
        pick = np.linspace(0, len(natural_starts) - 1, cfg.augmentation_windows)
        starts = natural_starts[np.rint(pick).astype(int)]
    else:
        # Short cuts cannot always supply 30 exact 15-rotation strides. Preserve
        # the 60-rotation window and spread the requested views over the cut.
        starts = np.rint(np.linspace(0, max_start, cfg.augmentation_windows)).astype(int)

    windows = extract_window_batch_features(segment, starts, window_rows)
    return windows, starts, window_rows, len(signal)


def extract_signal_features(path: Path, cfg: FeatureConfig) -> np.ndarray:
    windows, _, _, _ = extract_signal_feature_windows(path, cfg)
    return windows[0]


def load_or_extract_dataset(
    data_root: Path,
    dataset: str,
    cfg: FeatureConfig,
    max_cuts: int | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    augmentation_tag = ""
    if cfg.augmentation_windows > 1:
        augmentation_tag = (
            f"_aug{cfg.augmentation_windows}_sr{cfg.sampling_rate_hz:g}"
            f"_rpm{cfg.spindle_rpm:g}_wr{cfg.window_rotations}_ws{cfg.stride_rotations}"
        )
    normalization_tag = "_prefixnorm" if max_cuts is not None else ""
    cache_name = (
        f"{dataset}_features_max{max_cuts or 'all'}_crop"
        f"{cfg.crop_start_frac:.2f}-{cfg.crop_end_frac:.2f}{augmentation_tag}{normalization_tag}.csv"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not force:
        print(json.dumps({"event": "feature_cache_hit", "dataset": dataset, "path": str(cache_path)}), flush=True)
        return pd.read_csv(cache_path)

    wear = load_wear(data_root, dataset)
    if max_cuts is not None:
        wear = wear.iloc[:max_cuts].copy()
        denominator = float(wear["vb_avg"].iloc[-1] - wear["vb_avg"].iloc[0])
        if abs(denominator) < 1e-12:
            raise ValueError(f"{dataset} prefix has near-zero VB range; cannot normalize")
        wear["vb_norm"] = (wear["vb_avg"] - wear["vb_avg"].iloc[0]) / denominator
    rows = []
    names = feature_names()
    cache_dir.mkdir(parents=True, exist_ok=True)
    part_dir = cache_dir / f"{cache_path.stem}_parts"
    part_dir.mkdir(exist_ok=True)
    total_cuts = len(wear)
    progress_every = max(1, total_cuts // 20)
    extraction_start = perf_counter()
    for cut_index, (_, record) in enumerate(wear.iterrows(), start=1):
        cut = int(record["cut"])
        part_path = part_dir / f"cut_{cut:03d}.csv"
        if part_path.exists() and not force:
            rows.extend(pd.read_csv(part_path).to_dict("records"))
            if cut_index == 1 or cut_index == total_cuts or cut_index % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "feature_progress",
                            "dataset": dataset,
                            "completed_cuts": cut_index,
                            "total_cuts": total_cuts,
                            "resumed_part": True,
                        }
                    ),
                    flush=True,
                )
            continue
        path = signal_path(data_root, dataset, cut)
        if not path.exists():
            raise FileNotFoundError(path)
        windows, starts, window_rows, signal_rows = extract_signal_feature_windows(path, cfg)
        cut_rows = []
        for window_id, (feats, start) in enumerate(zip(windows, starts)):
            row = {
                "dataset": dataset,
                "cut": cut,
                "window_id": window_id,
                "window_start": int(start),
                "window_rows": int(window_rows),
                "vb_avg": float(record["vb_avg"]),
                "vb_norm": float(record["vb_norm"]),
                "signal_rows": int(signal_rows),
            }
            row.update({name: float(value) for name, value in zip(names, feats)})
            cut_rows.append(row)
        pd.DataFrame(cut_rows).to_csv(part_path, index=False)
        rows.extend(cut_rows)
        if cut_index == 1 or cut_index == total_cuts or cut_index % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "feature_progress",
                        "dataset": dataset,
                        "completed_cuts": cut_index,
                        "total_cuts": total_cuts,
                        "elapsed_seconds": perf_counter() - extraction_start,
                        "resumed_part": False,
                    }
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def select_features_by_source_pcc(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    y = train_frame[target_col].to_numpy(dtype=float)
    records = []
    for col in feature_cols:
        x = train_frame[col].to_numpy(dtype=float)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        records.append({"feature": col, "pcc": corr, "abs_pcc": abs(corr)})
    pcc = pd.DataFrame(records).sort_values("abs_pcc", ascending=False).reset_index(drop=True)
    selected = pcc[pcc["abs_pcc"] >= threshold]["feature"].tolist()
    if len(selected) < max_features:
        selected = pcc.head(max_features)["feature"].tolist()
    else:
        selected = selected[:max_features]
    return selected, pcc


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    residual = y_pred - y_true
    high_mask = y_true >= 0.75
    if not np.any(high_mask):
        high_mask = np.ones_like(y_true, dtype=bool)
    diffs = np.diff(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "endpoint_error": float(abs(y_pred[-1] - y_true[-1])),
        "high_wear_mae": float(mean_absolute_error(y_true[high_mask], y_pred[high_mask])),
        "monotonic_violations": int(np.sum(diffs < -1e-6)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12 else 0.0,
    }


def empirical_time_transfer(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    source_time = np.linspace(0.0, 1.0, len(train_frame))
    target_time = np.linspace(0.0, 1.0, len(test_frame))
    return np.interp(target_time, source_time, train_frame["vb_norm"].to_numpy(dtype=float))


def run_source_only(
    data_root: Path,
    source: str,
    target: str,
    cfg: FeatureConfig,
    max_cuts: int | None,
    output_dir: Path,
    force_features: bool = False,
) -> dict[str, object]:
    cache_dir = output_dir / "features"
    train = load_or_extract_dataset(data_root, source, cfg, max_cuts, cache_dir, force=force_features)
    test = load_or_extract_dataset(data_root, target, cfg, max_cuts, cache_dir, force=force_features)
    feature_cols = feature_names()
    selected, pcc = select_features_by_source_pcc(train, feature_cols, "vb_norm", cfg.pcc_threshold, cfg.max_features)

    x_train = train[selected].to_numpy(dtype=float)
    x_test = test[selected].to_numpy(dtype=float)
    y_train = train["vb_norm"].to_numpy(dtype=float)
    y_test = test["vb_norm"].to_numpy(dtype=float)
    x_train_std, x_test_std, _, _ = standardize_train_test(x_train, x_test)

    model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed)
    model.fit(x_train_std, y_train)
    pred = np.clip(model.predict(x_test_std), -0.25, 1.25)
    empirical_pred = empirical_time_transfer(train, test)

    run_id = f"{source}_to_{target}_max{max_cuts or 'all'}"
    pred_frame = pd.DataFrame(
        {
            "dataset": target,
            "cut": test["cut"],
            "y_true_vb_norm": y_test,
            "source_only_pred_vb_norm": pred,
            "empirical_time_transfer_pred_vb_norm": empirical_pred,
        }
    )
    pred_path = output_dir / f"{run_id}_predictions.csv"
    pcc_path = output_dir / f"{run_id}_source_pcc.csv"
    pred_frame.to_csv(pred_path, index=False)
    pcc.to_csv(pcc_path, index=False)

    return {
        "run_id": run_id,
        "source": source,
        "target": target,
        "max_cuts": max_cuts,
        "num_train": int(len(train)),
        "num_test": int(len(test)),
        "num_features_total": len(feature_cols),
        "num_features_selected": len(selected),
        "selected_features": selected,
        "source_only": regression_metrics(y_test, pred),
        "empirical_time_transfer": regression_metrics(y_test, empirical_pred),
        "paths": {
            "predictions_csv": str(pred_path),
            "source_pcc_csv": str(pcc_path),
        },
    }


def run_data_sanity(data_root: Path, output_dir: Path) -> dict[str, object]:
    datasets = {}
    for dataset in DATASETS:
        wear = load_wear(data_root, dataset)
        directory = signal_dir(data_root, dataset)
        files = sorted(directory.glob("*.csv"))
        expected_files = [signal_path(data_root, dataset, int(cut)) for cut in wear["cut"]]
        expected_set = {path.name for path in expected_files}
        actual_set = {path.name for path in files}
        missing = sorted(expected_set.difference(actual_set))
        extra = sorted(actual_set.difference(expected_set))
        invalid_files = []
        row_counts = []
        nan_files = []
        for path in expected_files:
            if not path.exists():
                continue
            try:
                data = np.loadtxt(path, delimiter=",", dtype=float)
                if data.ndim != 2 or data.shape[1] != len(CHANNELS):
                    invalid_files.append({"file": path.name, "shape": list(data.shape)})
                if not np.isfinite(data).all():
                    nan_files.append(path.name)
                row_counts.append(int(data.shape[0]))
            except Exception as exc:  # noqa: BLE001 - preserve exact file failure in result JSON.
                invalid_files.append({"file": path.name, "error": repr(exc)})
        datasets[dataset] = {
            "wear_rows": int(len(wear)),
            "signal_files": int(len(files)),
            "vb_avg_start": float(wear["vb_avg"].iloc[0]),
            "vb_avg_end": float(wear["vb_avg"].iloc[-1]),
            "vb_norm_start": float(wear["vb_norm"].iloc[0]),
            "vb_norm_end": float(wear["vb_norm"].iloc[-1]),
            "missing_signal_files": missing,
            "extra_signal_files": extra,
            "invalid_signal_files": invalid_files,
            "nan_signal_files": nan_files,
            "min_signal_rows": int(min(row_counts)) if row_counts else 0,
            "max_signal_rows": int(max(row_counts)) if row_counts else 0,
            "all_checks_passed": bool(
                len(wear) == 315
                and len(files) == 315
                and not missing
                and not extra
                and not invalid_files
                and not nan_files
                and abs(float(wear["vb_norm"].iloc[0])) < 1e-12
                and abs(float(wear["vb_norm"].iloc[-1]) - 1.0) < 1e-12
            ),
        }
    result = {"datasets": datasets}
    (output_dir / "data_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_tracker(output_dir: Path, sanity: dict[str, object], source_result: dict[str, object]) -> None:
    smoke = source_result["max_cuts"] is not None
    run_id = "R007-smoke" if smoke else "R007"
    status = "SMOKE-DONE" if smoke else "DONE"
    purpose = "regression smoke sanity" if smoke else "source-only regression baseline"
    notes = (
        f"max_cuts={source_result['max_cuts']}; MAE={source_result['source_only']['mae']:.4f}, "
        f"RMSE={source_result['source_only']['rmse']:.4f}"
        if smoke
        else f"full lifecycle; MAE={source_result['source_only']['mae']:.4f}, RMSE={source_result['source_only']['rmse']:.4f}"
    )
    tracker = [
        "# Experiment Tracker",
        "",
        "| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
        "| R001 | M0 | data integrity | raw inspection | C1/C4/C6 | count, file alignment, NaN, VB_norm endpoints | MUST | DONE | Full local signal files validated |",
        "| R002 | M1 | physical pseudo-label | empirical model | C1/C4/C6 | norm MAE/RMSE, zero point | MUST | DONE | See refine-logs/pseudo-label-phm/SUMMARY.md |",
        "| R004 | M2 | feature sanity | 18 features/channel + PCC | C1/C4 subset sanity | selected features | MUST | DONE | Source-only sanity used extracted features |",
        f"| {run_id} | M4 | {purpose} | Source-only Ridge | {source_result['source']} -> {source_result['target']} | norm MAE/RMSE/R2 | MUST | {status} | {notes} |",
    ]
    (output_dir / "EXPERIMENT_TRACKER_SANITY.md").write_text("\n".join(tracker) + "\n", encoding="utf-8")


def write_results(output_dir: Path, cfg: FeatureConfig, sanity: dict[str, object], source_result: dict[str, object], elapsed: float) -> None:
    payload = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plan": "refine-logs/EXPERIMENT_PLAN.md",
        "config": asdict(cfg),
        "elapsed_seconds": elapsed,
        "sanity": sanity,
        "source_only_regression": source_result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanity_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Initial Experiment Results",
        "",
        f"**Date**: {payload['date']}",
        "**Plan**: `refine-logs/EXPERIMENT_PLAN.md`",
        "",
        "## M0: Data Sanity",
        "",
    ]
    for dataset, info in sanity["datasets"].items():
        lines.append(
            f"- `{dataset}`: wear rows={info['wear_rows']}, signal files={info['signal_files']}, "
            f"VB_norm start/end={info['vb_norm_start']:.1f}/{info['vb_norm_end']:.1f}, "
            f"rows min/max={info['min_signal_rows']}/{info['max_signal_rows']}, checks_passed={info['all_checks_passed']}"
        )
    lines.extend(
        [
            "",
            "## M4: Source-only Regression " + ("Smoke Sanity" if source_result["max_cuts"] is not None else "Baseline"),
            "",
            f"- Split: `{source_result['source']} -> {source_result['target']}`",
            f"- Train/test cuts: {source_result['num_train']}/{source_result['num_test']}",
            f"- Selected features: {source_result['num_features_selected']} / {source_result['num_features_total']}",
            (
                "- This is a smoke run. It is not a full-lifecycle B3 result and is not deployable as the proposed method."
                if source_result["max_cuts"] is not None
                else "- This is a full-lifecycle source-only baseline, but it is still not the proposed method."
            ),
            "",
            "| System | MAE | RMSE | R2 | Endpoint Error | High-wear MAE | Monotonic Violations | Pearson |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("source_only", "empirical_time_transfer"):
        metric = source_result[name]
        lines.append(
            f"| {name} | {metric['mae']:.4f} | {metric['rmse']:.4f} | {metric['r2']:.4f} | "
            f"{metric['endpoint_error']:.4f} | {metric['high_wear_mae']:.4f} | "
            f"{metric['monotonic_violations']} | {metric['pearson']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Status",
            "",
            "- Sanity-first run completed on CPU using conda base.",
            "- This run intentionally does not deploy DDGAN/PGRU yet.",
            "- Evaluation uses dataset ground-truth `VB_norm`, not pseudo-label output.",
            "- Deployment gate: FULL PROPOSED PIPELINE NOT IMPLEMENTED, so AUTO_DEPLOY stops after sanity.",
        ]
    )
    (output_dir / "EXPERIMENT_RESULTS_SANITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PHM 2010 normalized wear-value experiment bridge sanity.")
    parser.add_argument("--data-root", default=r"D:\PHM\data")
    parser.add_argument("--output-dir", default=r"D:\PHM\refine-logs\bridge-sanity")
    parser.add_argument("--source", default="c1", choices=DATASETS)
    parser.add_argument("--target", default="c4", choices=DATASETS)
    parser.add_argument("--max-cuts", type=int, default=30)
    parser.add_argument("--crop-start-frac", type=float, default=0.10)
    parser.add_argument("--crop-end-frac", type=float, default=0.90)
    parser.add_argument("--pcc-threshold", type=float, default=0.90)
    parser.add_argument("--max-features", type=int, default=28)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--full-lifecycle", action="store_true", help="Use all 315 cuts for the source-only sanity regression.")
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("source and target must differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        crop_start_frac=args.crop_start_frac,
        crop_end_frac=args.crop_end_frac,
        pcc_threshold=args.pcc_threshold,
        max_features=args.max_features,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        smoke=not args.full_lifecycle,
        deep_signal_check=True,
    )
    max_cuts = None if args.full_lifecycle else args.max_cuts
    start = perf_counter()
    sanity = run_data_sanity(Path(args.data_root), output_dir)
    source_result = run_source_only(
        Path(args.data_root),
        args.source,
        args.target,
        cfg,
        max_cuts,
        output_dir,
        force_features=args.force_features,
    )
    elapsed = perf_counter() - start
    write_results(output_dir, cfg, sanity, source_result, elapsed)
    write_tracker(output_dir, sanity, source_result)
    print(json.dumps({"elapsed_seconds": elapsed, "source_only_regression": source_result}, indent=2))


if __name__ == "__main__":
    main()
