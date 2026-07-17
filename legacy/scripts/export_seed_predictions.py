from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SYSTEM = "proposed_ddgan_pgru_finetune"


def locate_seed_dir(run_root: Path, seed: int) -> Path:
    candidate = run_root / f"seed_{seed}"
    if candidate.is_dir():
        return candidate
    if run_root.is_dir() and (run_root / "paper_regression_results.json").exists():
        return run_root
    raise FileNotFoundError(f"Cannot find seed_{seed} under {run_root}")


def locate_prediction_csv(seed_dir: Path, result: dict[str, object]) -> Path:
    configured = Path(result["regression"]["predictions_csv"])
    if configured.exists():
        return configured
    cfg = result["config"]
    fallback = seed_dir / f"{cfg['source']}_to_{cfg['target']}_paper_regression_predictions.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Prediction CSV is unavailable: {configured} or {fallback}")


def available_systems(frame: pd.DataFrame) -> list[str]:
    return sorted(column[: -len("_pred")] for column in frame.columns if column.endswith("_pred"))


def export_predictions(args: argparse.Namespace) -> Path:
    seed_dir = locate_seed_dir(Path(args.run_root).resolve(), args.seed)
    result_path = seed_dir / "paper_regression_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = result["config"]
    if int(cfg["seed"]) != args.seed:
        raise ValueError(f"Result seed {cfg['seed']} does not match requested seed {args.seed}")
    for name in ("source", "target"):
        expected = getattr(args, name)
        if expected and cfg[name].lower() != expected.lower():
            raise ValueError(f"Result {name}={cfg[name]} does not match requested {expected}")

    prediction_path = locate_prediction_csv(seed_dir, result)
    frame = pd.read_csv(prediction_path)
    if "split" in frame.columns and (frame["split"] == "target_suffix_test").any():
        frame = frame.loc[frame["split"] == "target_suffix_test"].copy()
    systems = available_systems(frame)
    if args.system not in systems:
        raise ValueError(f"Unknown system {args.system}; available systems: {systems}")

    true_columns = [column for column in frame.columns if column.startswith("y_true_")]
    if len(true_columns) != 1:
        raise ValueError(f"Expected one y_true column, found {true_columns}")
    true_column = true_columns[0]
    exported = frame[["cut", true_column, f"{args.system}_pred"]].rename(
        columns={true_column: "y_true", f"{args.system}_pred": "y_pred"}
    )
    exported = exported.sort_values("cut")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else seed_dir
        / "exports"
        / f"{cfg['source']}_to_{cfg['target']}_seed_{args.seed}_predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False)
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export standardized target-wear predictions for one completed experiment seed."
    )
    parser.add_argument("--run-root", required=True, help="Multi-seed root or a seed directory.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source", help="Optional source-domain validation, for example c4.")
    parser.add_argument("--target", help="Optional target-domain validation, for example c6.")
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=f"System name. Default: {DEFAULT_SYSTEM}.",
    )
    parser.add_argument("--output", help="Output CSV path. Defaults to seed_dir/exports/.")
    export_predictions(parser.parse_args())


if __name__ == "__main__":
    main()
