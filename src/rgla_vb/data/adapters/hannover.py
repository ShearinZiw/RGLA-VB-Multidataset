"""Non-destructive discovery helpers for the Hannover multi-machine dataset.

The full HDF5 field mapping is intentionally deferred until the user's exact
dataset version is connected. These helpers inventory files and expose HDF5
structure without loading large sensor arrays into memory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


TOOL_TO_MACHINE = {
    **{f"T{i}": "M1" for i in range(1, 4)},
    **{f"T{i}": "M2" for i in range(4, 7)},
    **{f"T{i}": "M3" for i in range(7, 10)},
}
TOOL_PATTERN = re.compile(r"(?<![A-Z0-9])T(?:OOL)?[_-]?([1-9])(?![0-9])", re.IGNORECASE)
MACHINE_PATTERN = re.compile(r"(?<![A-Z0-9])M(?:ACHINE)?[_-]?([1-3])(?![0-9])", re.IGNORECASE)
RUN_PATTERN = re.compile(r"(?:RUN|CUT|PROCESS)[_-]?(\d+)", re.IGNORECASE)


def parse_identity(path: str | Path) -> dict[str, str | int | None]:
    text = Path(path).stem.upper()
    tool_match = TOOL_PATTERN.search(text)
    machine_match = MACHINE_PATTERN.search(text)
    run_match = RUN_PATTERN.search(text)
    tool_id = f"T{tool_match.group(1)}" if tool_match else None
    machine_id = f"M{machine_match.group(1)}" if machine_match else TOOL_TO_MACHINE.get(tool_id)
    if tool_id and machine_id and TOOL_TO_MACHINE[tool_id] != machine_id:
        raise ValueError(f"Contradictory Hannover identity in filename: {path}")
    return {
        "machine_id": machine_id,
        "tool_id": tool_id,
        "run_index": int(run_match.group(1)) if run_match else None,
    }


def build_inventory(root: str | Path) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted((*root.rglob("*.h5"), *root.rglob("*.hdf5")))
    if not files:
        raise ValueError(f"No HDF5 files found under {root}")
    rows = []
    for path in files:
        identity = parse_identity(path)
        rows.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                **identity,
            }
        )
    return pd.DataFrame(rows)


def probe_hdf5(path: str | Path) -> list[dict[str, Any]]:
    """List groups/datasets, shapes and dtypes without reading array payloads."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                records.append(
                    {
                        "name": name,
                        "kind": "dataset",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )
            else:
                records.append({"name": name, "kind": "group", "shape": None, "dtype": None})

        handle.visititems(visitor)
    return records
