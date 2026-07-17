"""Adapter for the NASA Ames milling wear data.

The adapter preserves raw continuous VB values and missing-label masks. It does
not interpolate labels and does not normalize VB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


SIGNAL_COLUMNS = (
    "smcAC",
    "smcDC",
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
)
REQUIRED_RAW_COLUMNS = (
    "case",
    "run",
    "VB",
    "time",
    "DOC",
    "feed",
    "material",
) + SIGNAL_COLUMNS
MATERIAL_NAMES = {1: "cast_iron", 2: "steel"}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"NASA milling file is missing columns: {missing}")
    if len(frame) != 167:
        raise ValueError(f"Expected 167 NASA runs, found {len(frame)}")
    if frame.duplicated(["case", "run"]).any():
        raise ValueError("NASA case/run identity is not unique")
    return frame.sort_values(["case", "run"], kind="stable").reset_index(drop=True)


def build_run_table(path: str | Path, include_signal_features: bool = True) -> pd.DataFrame:
    raw = load_raw(path)
    material_id = raw["material"].map(MATERIAL_NAMES)
    if material_id.isna().any():
        unknown = sorted(raw.loc[material_id.isna(), "material"].unique().tolist())
        raise ValueError(f"Unknown NASA material codes: {unknown}")

    result = pd.DataFrame(
        {
            "dataset_id": "nasa_milling",
            "domain_id": material_id,
            "machine_id": "nasa_mill",
            "tool_id": raw["case"].map(lambda value: f"case_{int(value):02d}"),
            "material_id": material_id,
            "condition_id": [
                f"doc_{doc:g}_feed_{feed:g}" for doc, feed in zip(raw["DOC"], raw["feed"], strict=True)
            ],
            "sequence_id": raw["case"].map(lambda value: f"case_{int(value):02d}"),
            "progress_value": raw["time"].astype(float),
            "progress_type": "elapsed_time",
            "cut_index": raw["run"].astype(int),
            "vb_value": raw["VB"].astype(float),
            "vb_unit": "native_unverified",
            "vb_um": np.nan,
            "label_mask": raw["VB"].notna(),
            "label_origin": np.where(raw["VB"].notna(), "measured", "missing"),
            "doc": raw["DOC"].astype(float),
            "feed": raw["feed"].astype(float),
        }
    )

    if include_signal_features:
        for signal_name in SIGNAL_COLUMNS:
            summaries = [_summarize_signal(values) for values in raw[signal_name]]
            for feature_name in summaries[0]:
                result[f"{signal_name}__{feature_name}"] = [item[feature_name] for item in summaries]
    return result


def _summarize_signal(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Each NASA signal must be a finite, non-empty one-dimensional array")
    rms = float(np.sqrt(np.mean(np.square(array))))
    peak = float(np.max(np.abs(array)))
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "rms": rms,
        "peak_to_peak": float(np.ptp(array)),
        "skew": float(skew(array, bias=False)),
        "kurtosis": float(kurtosis(array, fisher=True, bias=False)),
        "crest_factor": peak / max(rms, np.finfo(np.float64).eps),
        "length": float(array.size),
    }
