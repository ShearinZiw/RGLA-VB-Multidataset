from pathlib import Path

import h5py

from rgla_vb.data.adapters.hannover import build_inventory, parse_identity, probe_hdf5


def test_hannover_identity_parser_infers_machine() -> None:
    assert parse_identity("T8_run_0042.h5") == {"machine_id": "M3", "tool_id": "T8", "run_index": 42}
    assert parse_identity("M2-T5-process17.hdf5") == {"machine_id": "M2", "tool_id": "T5", "run_index": 17}


def test_hannover_inventory_and_probe(tmp_path: Path) -> None:
    path = tmp_path / "M1_T2_run_0003.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("signals/force", data=[1.0, 2.0])
    inventory = build_inventory(tmp_path)
    assert inventory.loc[0, "tool_id"] == "T2"
    records = probe_hdf5(path)
    force = next(item for item in records if item["name"] == "signals/force")
    assert force["shape"] == [2]
