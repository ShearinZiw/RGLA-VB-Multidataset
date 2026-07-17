"""Canonical lifecycle table schema shared by dataset adapters."""

from __future__ import annotations

IDENTITY_COLUMNS = (
    "dataset_id",
    "domain_id",
    "machine_id",
    "tool_id",
    "material_id",
    "condition_id",
    "sequence_id",
)

PROGRESS_COLUMNS = ("progress_value", "progress_type")
LABEL_COLUMNS = ("vb_value", "vb_unit", "vb_um", "label_mask", "label_origin")
REQUIRED_COLUMNS = IDENTITY_COLUMNS + PROGRESS_COLUMNS + LABEL_COLUMNS

LABEL_ORIGINS = frozenset({"measured", "provider_interpolated", "missing"})
PROGRESS_TYPES = frozenset({"cut", "run", "contact_time", "elapsed_time", "removal_volume"})


def vb_to_um(value: float, unit: str) -> float:
    """Convert a verified VB value to micrometres without normalization."""
    if unit == "um":
        return float(value)
    if unit == "mm":
        return float(value) * 1000.0
    raise ValueError(f"Unsupported or unverified VB unit: {unit!r}")
