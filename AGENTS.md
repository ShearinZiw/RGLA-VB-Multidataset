# Agent Execution Contract

This project will be implemented in small tasks by cost-efficient coding models. Follow these rules exactly.

## Non-negotiable invariants

1. The prediction task is continuous raw `VB` regression. Never convert it to classification.
2. Never use `VB_norm` as the training target. Unit conversion to `vb_um` is allowed only when the source unit is recorded.
3. Target-domain labels are unavailable during UDA training. Evaluation labels must not enter scaling, feature selection, pseudo-labeling, checkpoint selection, or early stopping.
4. Split by complete tool/case/machine identities before creating windows. Never split windows from one lifecycle across train and test.
5. Fit scalers and feature selectors on source training data only.
6. Do not modify `legacy/scripts/` unless the current task explicitly lists an allowed legacy file.
7. Do not silently drop Hannover T8, machine-2 aliased channels, missing NASA labels, or failed runs. Record exclusions with a reason.
8. Every experiment must write its resolved config, seed, git status, metrics, predictions, and label-visibility audit.

## Task protocol

- Read `README.md`, `docs/00_PROJECT_OVERVIEW.md`, and exactly one file from `tasks/`.
- Change only files listed under `Allowed files` in that task.
- Run every command listed under `Required checks`.
- Do not redesign formulas, filenames, schemas, or CLI arguments.
- If a required input is absent, stop with a clear error; do not invent data.
- Finish by writing the task's requested completion report under `outputs/task_reports/`.

## Scope control

Do not combine tasks. A task marked dependent on another task cannot be started until the dependency's tests pass. Avoid broad refactors, new frameworks, automatic hyperparameter search, and optional dependencies.
