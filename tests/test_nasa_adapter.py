from pathlib import Path

from rgla_vb.data.adapters.nasa_milling import SIGNAL_COLUMNS, build_run_table


DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa_milling" / "data.parquet"


def test_nasa_schema_and_missing_labels() -> None:
    table = build_run_table(DATA, include_signal_features=False)
    assert len(table) == 167
    assert int(table["label_mask"].sum()) == 146
    assert table["sequence_id"].nunique() == 16
    assert set(table["material_id"]) == {"cast_iron", "steel"}
    assert table.loc[~table["label_mask"], "vb_value"].isna().all()
    assert table["vb_unit"].eq("native_unverified").all()


def test_nasa_signal_features_are_finite() -> None:
    table = build_run_table(DATA, include_signal_features=True)
    feature_columns = [column for column in table if "__" in column]
    assert len(feature_columns) == len(SIGNAL_COLUMNS) * 8
    assert table[feature_columns].notna().all().all()
