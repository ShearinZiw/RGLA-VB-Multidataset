from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "endpoint_error",
    "high_wear_mae",
    "monotonic_violations",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def valid_result(path: Path, seed: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_state = json.loads((path.parent / "RUN_STATE.json").read_text(encoding="utf-8"))
        return (
            payload["config"]["seed"] == seed
            and bool(payload["regression"]["metrics"])
            and run_state.get("status") == "complete"
            and "evaluation" in run_state.get("completed_stages", [])
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_command(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
    pipeline = Path(__file__).with_name("phm_paper_regression_pipeline.py")
    command = [
        sys.executable,
        "-u",
        str(pipeline),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--output-dir",
        str(seed_dir.resolve()),
        "--feature-cache-dir",
        str(Path(args.feature_cache_dir).resolve()),
        "--data-sanity-path",
        str(Path(args.data_sanity_path).resolve()),
        "--source",
        args.source,
        "--target",
        args.target,
        "--wear-target",
        args.wear_target,
        "--target-prefix-fraction",
        str(args.target_prefix_fraction),
        "--lambda-target-prefix",
        str(args.lambda_target_prefix),
        "--target-prefix-strategy",
        args.target_prefix_strategy,
        "--target-prefix-finetune-epochs",
        str(args.target_prefix_finetune_epochs),
        "--lr-target-prefix-finetune",
        str(args.lr_target_prefix_finetune),
        "--target-prefix-finetune-scope",
        args.target_prefix_finetune_scope,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--augmentation-windows",
        str(args.augmentation_windows),
        "--pseudo-epochs",
        str(args.pseudo_epochs),
        "--gan-epochs",
        str(args.gan_epochs),
        "--adapt-epochs",
        str(args.adapt_epochs),
        "--finetune-epochs",
        str(args.finetune_epochs),
        "--lifecycle-batch-size",
        str(args.lifecycle_batch_size),
        "--pseudo-batch-size",
        str(args.pseudo_batch_size),
        "--noise-dim",
        str(args.noise_dim),
        "--gan-hidden-dim",
        str(args.gan_hidden_dim),
        "--pgru-hidden1",
        str(args.pgru_hidden1),
        "--pgru-hidden2",
        str(args.pgru_hidden2),
        "--pgru-implementation",
        args.pgru_implementation,
        "--lr-pseudo",
        str(args.lr_pseudo),
        "--lr-g",
        str(args.lr_g),
        "--lr-d",
        str(args.lr_d),
        "--lr-reg",
        str(args.lr_reg),
        "--lr-finetune",
        str(args.lr_finetune),
        "--lambda-js",
        str(args.lambda_js),
        "--lambda-domain",
        str(args.lambda_domain),
        "--lambda-s8",
        str(args.lambda_s8),
        "--lambda-coeff-anchor",
        str(args.lambda_coeff_anchor),
        "--lambda-pseudo-consistency",
        str(args.lambda_pseudo_consistency),
        "--lambda-smooth",
        str(args.lambda_smooth),
        "--lambda-source-replay",
        str(args.lambda_source_replay),
        "--lambda-target-pseudo",
        str(args.lambda_target_pseudo),
        "--finetune-base",
        args.finetune_base,
        "--generated-multiplier",
        str(args.generated_multiplier),
        "--pseudo-filter-quantile",
        str(args.pseudo_filter_quantile),
        "--pseudo-filter-min-sets",
        str(args.pseudo_filter_min_sets),
        "--log-every",
        str(args.log_every),
    ]
    command.extend(["--full-lifecycle"] if args.max_cuts is None else ["--max-cuts", str(args.max_cuts)])
    if args.feature_attention:
        command.append("--feature-attention")
    if args.pretrained_root:
        command.extend(
            ["--pretrained-dir", str((Path(args.pretrained_root) / f"seed_{seed}").resolve())]
        )
    if args.pseudo_filter_max_mae is not None:
        command.extend(["--pseudo-filter-max-mae", str(args.pseudo_filter_max_mae)])
    command.append("--deterministic" if args.deterministic else "--no-deterministic")
    if args.resume:
        command.append("--resume")
    return command


def aggregate_results(output_root: Path, seeds: list[int]) -> dict[str, object]:
    payloads = []
    rows = []
    quality_rows = []
    filter_rows = []
    for seed in seeds:
        result_path = output_root / f"seed_{seed}" / "paper_regression_results.json"
        if not valid_result(result_path, seed):
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payloads.append(payload)
        for system, metrics in payload["regression"]["metrics"].items():
            rows.append({"seed": seed, "system": system, **metrics})
        quality_rows.append({"seed": seed, **payload["generation"]["quality"]})
        filter_rows.append({"seed": seed, **payload["generation"]["filter"]})
    if not rows:
        raise RuntimeError("No completed seed results are available for aggregation")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "multi_seed_metrics.csv", index=False)
    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_csv(output_root / "multi_seed_generation_quality.csv", index=False)
    filter_frame = pd.DataFrame(filter_rows)
    filter_frame.to_csv(output_root / "multi_seed_generation_filter.csv", index=False)
    metric_columns = [column for column in frame.columns if column not in {"seed", "system"}]
    summary: dict[str, object] = {
        "completed_seeds": sorted(frame["seed"].unique().astype(int).tolist()),
        "requested_seeds": seeds,
        "systems": {},
        "generation_quality": {},
        "generation_filter": {},
        "source": payloads[0]["config"]["source"],
        "target": payloads[0]["config"]["target"],
        "wear_target": payloads[0]["config"]["wear_target"],
        "target_prefix_fraction": payloads[0]["config"].get("target_prefix_fraction", 0.0),
        "target_prefix_strategy": payloads[0]["config"].get("target_prefix_strategy", "joint"),
        "metrics_scope": payloads[0].get("evaluation", {}).get("metrics_scope", "full_target"),
    }
    for system, group in frame.groupby("system", sort=True):
        system_summary = {}
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            best_index = int(np.argmin(values) if metric in LOWER_IS_BETTER else np.argmax(values))
            system_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "best": float(values[best_index]),
                "best_seed": int(group.iloc[best_index]["seed"]),
            }
        summary["systems"][system] = system_summary
    for metric in quality_frame.columns:
        if metric == "seed":
            continue
        values = quality_frame[metric].to_numpy(dtype=float)
        summary["generation_quality"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    for metric in filter_frame.select_dtypes(include=[np.number, "bool"]).columns:
        if metric == "seed":
            continue
        values = filter_frame[metric].to_numpy(dtype=float)
        summary["generation_filter"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    summary["elapsed_seconds_sum"] = float(sum(payload["elapsed_seconds"] for payload in payloads))
    write_json(output_root / "multi_seed_summary.json", summary)

    selected_metrics = ["mae", "rmse", "r2", "endpoint_error", "high_wear_mae", "pearson"]
    lines = [
        f"# {summary['source'].upper()} to {summary['target'].upper()} Multi-seed Continuous {summary['wear_target']} Results",
        "",
        f"Completed seeds: `{', '.join(map(str, summary['completed_seeds']))}`",
        f"Target labeled prefix: `{100.0 * summary['target_prefix_fraction']:.1f}%`; "
        f"strategy: `{summary['target_prefix_strategy']}`; metric scope: `{summary['metrics_scope']}`.",
        "",
        "| System | MAE | RMSE | R2 | Endpoint error | High-wear MAE | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system, values in summary["systems"].items():
        cells = []
        for metric in selected_metrics:
            item = values[metric]
            cells.append(f"{item['mean']:.4f} +/- {item['std']:.4f}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    proposed = summary["systems"].get("proposed_ddgan_pgru_finetune")
    if proposed:
        lines.extend(
            [
                "",
                "## Seed Selection",
                "",
                f"Lowest proposed MAE: seed `{proposed['mae']['best_seed']}` "
                f"with `{proposed['mae']['best']:.4f}`.",
                "",
                "Report mean +/- sample standard deviation as the primary result; the best seed is diagnostic only.",
            ]
        )
    (output_root / "MULTI_SEED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential local multi-seed runner for paper regression.")
    parser.add_argument("--data-root", default=r"D:\PHM\data")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--data-sanity-path", required=True)
    parser.add_argument("--pretrained-root")
    parser.add_argument("--source", default="c1")
    parser.add_argument("--target", default="c4")
    parser.add_argument("--wear-target", choices=("vb", "vb_norm"), default="vb_norm")
    parser.add_argument("--target-prefix-fraction", type=float, default=0.0)
    parser.add_argument("--lambda-target-prefix", type=float, default=1.0)
    parser.add_argument("--target-prefix-strategy", choices=("joint", "finetune"), default="joint")
    parser.add_argument("--target-prefix-finetune-epochs", type=int, default=100)
    parser.add_argument("--lr-target-prefix-finetune", type=float, default=1e-4)
    parser.add_argument(
        "--target-prefix-finetune-scope",
        choices=("head", "recurrent_head"),
        default="recurrent_head",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260510, 20260511, 20260512])
    parser.add_argument("--max-cuts", type=int)
    parser.add_argument("--full-lifecycle", action="store_true")
    parser.add_argument("--augmentation-windows", type=int, default=30)
    parser.add_argument("--pseudo-epochs", type=int, default=200)
    parser.add_argument("--gan-epochs", type=int, default=400)
    parser.add_argument("--adapt-epochs", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--lifecycle-batch-size", type=int, default=5)
    parser.add_argument("--pseudo-batch-size", type=int, default=5)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--gan-hidden-dim", type=int, default=64)
    parser.add_argument("--pgru-hidden1", type=int, default=120)
    parser.add_argument("--pgru-hidden2", type=int, default=240)
    parser.add_argument("--pgru-implementation", choices=("fused", "cell"), default="fused")
    parser.add_argument("--lr-pseudo", type=float, default=1e-3)
    parser.add_argument("--lr-g", type=float, default=2e-3)
    parser.add_argument("--lr-d", type=float, default=2e-5)
    parser.add_argument("--lr-reg", type=float, default=5e-3)
    parser.add_argument("--lr-finetune", type=float, default=5e-3)
    parser.add_argument("--lambda-js", type=float, default=1.0)
    parser.add_argument("--lambda-domain", type=float, default=0.1)
    parser.add_argument("--lambda-s8", type=float, default=0.08)
    parser.add_argument("--lambda-coeff-anchor", type=float, default=0.01)
    parser.add_argument("--lambda-pseudo-consistency", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-source-replay", type=float, default=0.0)
    parser.add_argument("--lambda-target-pseudo", type=float, default=0.0)
    parser.add_argument("--finetune-base", choices=("source", "domain"), default="domain")
    parser.add_argument("--generated-multiplier", type=int, default=1)
    parser.add_argument("--pseudo-filter-quantile", type=float, default=0.70)
    parser.add_argument("--pseudo-filter-max-mae", type=float)
    parser.add_argument("--pseudo-filter-min-sets", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("source and target must differ")
    if args.full_lifecycle and args.max_cuts is not None:
        raise ValueError("--full-lifecycle and --max-cuts cannot be used together")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.pretrained_root and not Path(args.pretrained_root).is_dir():
        raise FileNotFoundError(args.pretrained_root)
    if not 0.0 <= args.target_prefix_fraction < 1.0:
        raise ValueError("target-prefix-fraction must be in [0, 1)")
    if args.target_prefix_fraction > 0 and args.wear_target != "vb":
        raise ValueError("target prefix supervision requires --wear-target vb")
    if args.lambda_target_prefix < 0:
        raise ValueError("lambda-target-prefix must be non-negative")
    if args.target_prefix_strategy == "finetune" and args.target_prefix_fraction <= 0:
        raise ValueError("target-prefix-strategy finetune requires a positive target prefix")
    if args.target_prefix_finetune_epochs < 1:
        raise ValueError("target-prefix-finetune-epochs must be positive")
    if args.lr_target_prefix_finetune <= 0:
        raise ValueError("lr-target-prefix-finetune must be positive")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.feature_cache_dir).mkdir(parents=True, exist_ok=True)
    state_path = output_root / "MULTI_SEED_STATE.json"
    state = {
        "status": "running",
        "source": args.source,
        "target": args.target,
        "seeds": args.seeds,
        "jobs": {},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["jobs"] = previous.get("jobs", {})
        now = datetime.now()
        for job in state["jobs"].values():
            if job.get("status") == "running" and job.get("started_at"):
                interrupted = max(
                    0.0,
                    (now - datetime.fromisoformat(job["started_at"])).total_seconds(),
                )
                job["elapsed_seconds_accumulated"] = float(
                    job.get("elapsed_seconds_accumulated", 0.0)
                ) + interrupted
                job["status"] = "interrupted"
    write_json(state_path, state)

    failed = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        result_path = seed_dir / "paper_regression_results.json"
        if args.resume and valid_result(result_path, seed):
            state["jobs"][str(seed)] = {"status": "complete", "skipped_existing": True}
            write_json(state_path, state)
            print(json.dumps({"event": "seed_skipped", "seed": seed}), flush=True)
            continue
        command = build_command(args, seed, seed_dir)
        log_path = seed_dir / "training.log"
        previous_elapsed = float(
            state["jobs"].get(str(seed), {}).get("elapsed_seconds_accumulated", 0.0)
        )
        state["jobs"][str(seed)] = {
            "status": "running",
            "log": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds_accumulated": previous_elapsed,
        }
        write_json(state_path, state)
        print(json.dumps({"event": "seed_start", "seed": seed, "log": str(log_path)}), flush=True)
        started = perf_counter()
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(f"[seed {seed}] {line}", end="", flush=True)
            return_code = process.wait()
        elapsed = perf_counter() - started
        complete = return_code == 0 and valid_result(result_path, seed)
        state["jobs"][str(seed)] = {
            "status": "complete" if complete else "failed",
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "elapsed_seconds_accumulated": previous_elapsed + elapsed,
            "log": str(log_path),
        }
        write_json(state_path, state)
        if not complete:
            failed.append(seed)
            if not args.continue_on_error:
                break

    completed = [seed for seed in args.seeds if valid_result(output_root / f"seed_{seed}" / "paper_regression_results.json", seed)]
    summary = aggregate_results(output_root, completed) if completed else None
    state["status"] = (
        "complete" if not failed and len(completed) == len(args.seeds) else "partial_failure"
    )
    state["completed_seeds"] = completed
    state["failed_seeds"] = failed
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state["summary"] = str(output_root / "multi_seed_summary.json") if summary else None
    write_json(state_path, state)
    print(json.dumps({"event": "multiseed_complete", "completed": completed, "failed": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
