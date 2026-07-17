"""Resolve dataset roots without copying external raw data into the project."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.json"


class DatasetRegistry:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path).resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.datasets: dict[str, dict[str, Any]] = payload["datasets"]

    def names(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    def spec(self, name: str) -> dict[str, Any]:
        try:
            return self.datasets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset {name!r}; choose from {self.names()}") from exc

    def resolve_root(self, name: str, require_present: bool = False) -> Path | None:
        spec = self.spec(name)
        env_value = os.environ.get(spec["root_env"])
        raw_root = env_value or spec.get("default_root")
        if raw_root is None:
            if require_present:
                raise FileNotFoundError(
                    f"{name} has no configured root. Set {spec['root_env']} to its external directory."
                )
            return None
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        if require_present and not root.exists():
            raise FileNotFoundError(f"Configured {name} root does not exist: {root}")
        return root

    def validate(self, name: str, require_present: bool = False) -> dict[str, Any]:
        root = self.resolve_root(name, require_present=require_present)
        result: dict[str, Any] = {"dataset": name, "root": str(root) if root else None}
        if root is None or not root.exists():
            result.update({"status": "not_configured", "present": False})
            return result

        if name == "phm2010":
            required = self.spec(name)["required_subdirectories"]
            missing = [item for item in required if not (root / item).is_dir()]
            result.update({"status": "ok" if not missing else "invalid", "present": True, "missing": missing})
        elif name == "hannover":
            files = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))
            result.update(
                {
                    "status": "ok" if files else "invalid",
                    "present": True,
                    "hdf5_files": len(files),
                    "expected_hdf5_runs": self.spec(name)["expected_hdf5_runs"],
                }
            )
        elif name == "nasa_milling":
            path = root / self.spec(name)["filename"]
            digest = _sha256(path) if path.is_file() else None
            expected = self.spec(name)["sha256"]
            result.update(
                {
                    "status": "ok" if digest == expected else "invalid",
                    "present": path.is_file(),
                    "file": str(path),
                    "sha256": digest,
                    "checksum_matches": digest == expected,
                }
            )
        else:
            raise AssertionError(f"No validator implemented for {name}")

        if require_present and result["status"] != "ok":
            raise ValueError(f"Dataset validation failed: {result}")
        return result


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
