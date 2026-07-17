from pathlib import Path

from rgla_vb.data.registry import DatasetRegistry


def test_registry_resolves_project_local_nasa() -> None:
    registry = DatasetRegistry()
    root = registry.resolve_root("nasa_milling", require_present=True)
    assert root is not None
    assert root.name == "nasa_milling"
    assert registry.validate("nasa_milling", require_present=True)["checksum_matches"] is True


def test_registry_resolves_external_phm() -> None:
    result = DatasetRegistry().validate("phm2010", require_present=True)
    assert result["status"] == "ok"
    assert Path(result["root"]).name == "data"
