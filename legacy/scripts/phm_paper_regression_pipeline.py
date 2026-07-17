from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from phm_bridge import (
    DATASETS,
    FeatureConfig,
    feature_names,
    load_or_extract_dataset,
    run_data_sanity,
    select_features_by_source_pcc,
)
from phm_proposed_pipeline import (
    PAPER_COEFFS_RAW,
    grad_reverse,
    make_tensor,
    metric_dict,
    pick_device,
    set_seed,
    torch_json,
)


@dataclass
class PaperRegressionConfig:
    data_root: str
    output_dir: str
    feature_cache_dir: str | None
    data_sanity_path: str | None
    pretrained_dir: str | None
    source: str
    target: str
    max_cuts: int | None
    crop_start_frac: float
    crop_end_frac: float
    pcc_threshold: float
    max_features: int
    wear_target: str
    target_prefix_fraction: float
    lambda_target_prefix: float
    target_prefix_strategy: str
    target_prefix_finetune_epochs: int
    lr_target_prefix_finetune: float
    target_prefix_finetune_scope: str
    augmentation_windows: int
    sampling_rate_hz: float
    spindle_rpm: float
    window_rotations: int
    stride_rotations: int
    seed: int
    device: str
    deterministic: bool
    pseudo_epochs: int
    gan_epochs: int
    adapt_epochs: int
    finetune_epochs: int
    lifecycle_batch_size: int
    pseudo_batch_size: int
    noise_dim: int
    gan_hidden_dim: int
    pgru_hidden1: int
    pgru_hidden2: int
    pgru_implementation: str
    lr_pseudo: float
    lr_g: float
    lr_d: float
    lr_reg: float
    lr_finetune: float
    lambda_js: float
    lambda_domain: float
    lambda_s8: float
    lambda_coeff_anchor: float
    lambda_pseudo_consistency: float
    lambda_smooth: float
    lambda_source_replay: float
    lambda_target_pseudo: float
    finetune_base: str
    generated_multiplier: int
    pseudo_filter_quantile: float
    pseudo_filter_max_mae: float | None
    pseudo_filter_min_sets: int
    feature_attention: bool
    force_features: bool
    resume: bool
    log_every: int


@dataclass
class LifecycleData:
    source_sequences: np.ndarray
    target_sequences: np.ndarray
    source_y_norm: np.ndarray
    target_y_norm: np.ndarray
    source_y_raw: np.ndarray
    target_y_raw: np.ndarray
    source_cuts: np.ndarray
    target_cuts: np.ndarray
    selected_features: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_info: dict[str, object]


def batch_indices(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), device=device)


def source_wear_target(data: LifecycleData, cfg: PaperRegressionConfig) -> np.ndarray:
    return data.source_y_raw if cfg.wear_target == "vb" else data.source_y_norm


def target_wear_target(data: LifecycleData, cfg: PaperRegressionConfig) -> np.ndarray:
    return data.target_y_raw if cfg.wear_target == "vb" else data.target_y_norm


def target_prefix_count(data: LifecycleData, cfg: PaperRegressionConfig) -> int:
    if cfg.target_prefix_fraction <= 0:
        return 0
    return min(
        len(data.target_cuts) - 1,
        max(1, int(math.ceil(len(data.target_cuts) * cfg.target_prefix_fraction))),
    )


def target_prefix_loss_scale(data: LifecycleData, cfg: PaperRegressionConfig) -> float:
    return cfg.lambda_target_prefix * target_prefix_count(data, cfg) / len(data.target_cuts)


def pretraining_target_prefix_count(data: LifecycleData, cfg: PaperRegressionConfig) -> int:
    if cfg.target_prefix_strategy != "joint":
        return 0
    return target_prefix_count(data, cfg)


def target_curve_w0(data: LifecycleData, cfg: PaperRegressionConfig) -> float:
    if pretraining_target_prefix_count(data, cfg) > 0:
        return float(data.target_y_raw[0])
    return float(data.source_y_raw[0])


def wear_column_name(cfg: PaperRegressionConfig) -> str:
    return "vb_avg" if cfg.wear_target == "vb" else "vb_norm"


def prediction_bounds(cfg: PaperRegressionConfig) -> tuple[float, float]:
    return (-50.0, 500.0) if cfg.wear_target == "vb" else (-0.25, 1.25)


def log_epoch(
    stage: str,
    epoch: int,
    total_epochs: int,
    totals: dict[str, float],
    steps: int,
    log_every: int,
) -> None:
    if log_every <= 0:
        return
    if epoch != 0 and epoch + 1 != total_epochs and (epoch + 1) % log_every != 0:
        return
    payload = {
        "event": "epoch",
        "stage": stage,
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        **{key: value / steps for key, value in totals.items()},
    }
    print(json.dumps(payload), flush=True)


def frame_to_lifecycle_sequences(
    frame: pd.DataFrame,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "window_id" not in frame.columns:
        frame = frame.copy()
        frame["window_id"] = 0
    cuts = np.sort(frame["cut"].unique()).astype(np.float32)
    sequences = []
    for _, group in frame.groupby("window_id", sort=True):
        ordered = group.sort_values("cut")
        if len(ordered) != len(cuts) or not np.array_equal(ordered["cut"].to_numpy(), cuts):
            raise ValueError("Every augmented window must contain one row for every cut")
        sequences.append(ordered[selected].to_numpy(dtype=np.float32))
    labels = frame.sort_values(["window_id", "cut"]).drop_duplicates("cut").sort_values("cut")
    return (
        np.stack(sequences, axis=0),
        labels["vb_norm"].to_numpy(dtype=np.float32),
        labels["vb_avg"].to_numpy(dtype=np.float32),
        cuts,
    )


def load_lifecycle_data(cfg: PaperRegressionConfig) -> LifecycleData:
    feature_cfg = FeatureConfig(
        data_root=cfg.data_root,
        output_dir=cfg.output_dir,
        crop_start_frac=cfg.crop_start_frac,
        crop_end_frac=cfg.crop_end_frac,
        pcc_threshold=cfg.pcc_threshold,
        max_features=cfg.max_features,
        seed=cfg.seed,
        smoke=cfg.max_cuts is not None,
        deep_signal_check=False,
        augmentation_windows=cfg.augmentation_windows,
        sampling_rate_hz=cfg.sampling_rate_hz,
        spindle_rpm=cfg.spindle_rpm,
        window_rotations=cfg.window_rotations,
        stride_rotations=cfg.stride_rotations,
    )
    output_dir = Path(cfg.output_dir)
    cache_dir = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir else output_dir / "features"
    source_frame = load_or_extract_dataset(
        Path(cfg.data_root), cfg.source, feature_cfg, cfg.max_cuts, cache_dir, force=cfg.force_features
    )
    target_frame = load_or_extract_dataset(
        Path(cfg.data_root), cfg.target, feature_cfg, cfg.max_cuts, cache_dir, force=cfg.force_features
    )
    available = [name for name in feature_names() if name in source_frame.columns]
    selected, pcc = select_features_by_source_pcc(
        source_frame, available, wear_column_name(cfg), cfg.pcc_threshold, cfg.max_features
    )
    pcc_path = output_dir / f"{cfg.source}_to_{cfg.target}_paper_source_pcc.csv"
    pcc.to_csv(pcc_path, index=False)

    source_seq, source_y_norm, source_y_raw, source_cuts = frame_to_lifecycle_sequences(
        source_frame, selected
    )
    target_seq, target_y_norm, target_y_raw, target_cuts = frame_to_lifecycle_sequences(
        target_frame, selected
    )
    mean = source_seq.reshape(-1, source_seq.shape[-1]).mean(axis=0)
    std = source_seq.reshape(-1, source_seq.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    source_seq = (source_seq - mean) / std
    target_seq = (target_seq - mean) / std
    return LifecycleData(
        source_sequences=source_seq.astype(np.float32),
        target_sequences=target_seq.astype(np.float32),
        source_y_norm=source_y_norm,
        target_y_norm=target_y_norm,
        source_y_raw=source_y_raw,
        target_y_raw=target_y_raw,
        source_cuts=source_cuts,
        target_cuts=target_cuts,
        selected_features=selected,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        feature_info={
            "pcc_path": str(pcc_path),
            "num_total_features": len(available),
            "num_selected_features": len(selected),
            "source_lifecycle_sets": len(source_seq),
            "target_lifecycle_sets": len(target_seq),
            "cuts_per_lifecycle": len(source_cuts),
            "tensor_layout": "augmented_lifecycle_set x cut x selected_feature",
        },
    )


class PhysicsCoefficientNet(nn.Module):
    """Paper-style 315*28 -> 128 -> 64 -> 4 physical coefficient mapper."""

    def __init__(self, cuts: int, feature_dim: int, base_coefficients: tuple[float, float, float, float]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cuts * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("base", torch.tensor(base_coefficients, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = 0.35 * torch.tanh(self.net(x))
        return self.base.unsqueeze(0) * torch.exp(delta)


def build_pseudo_label_network(
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> PhysicsCoefficientNet:
    paper_coefficients = PAPER_COEFFS_RAW[cfg.source]
    return PhysicsCoefficientNet(
        len(data.source_cuts),
        len(data.selected_features),
        (
            float(paper_coefficients["A"]),
            float(paper_coefficients["B"]),
            float(paper_coefficients["C"]),
            float(paper_coefficients["D"]),
        ),
    ).to(device)


def physical_curves_torch(
    coefficients: torch.Tensor,
    cuts: torch.Tensor,
    w0: float,
    normalize: bool,
) -> torch.Tensor:
    a = coefficients[:, 0:1]
    b = coefficients[:, 1:2]
    c = coefficients[:, 2:3]
    d = coefficients[:, 3:4]
    t = cuts.unsqueeze(0)
    raw = w0 + a * t * torch.exp(torch.clamp(b * t, -80.0, 80.0)) + c * torch.pow(t, d)
    if not normalize:
        return raw
    denominator = torch.clamp(raw[:, -1:] - raw[:, :1], min=1e-6)
    return (raw - raw[:, :1]) / denominator


def s8_constraint_loss(coefficients: torch.Tensor, t_end: float) -> torch.Tensor:
    a, b, c, d = [coefficients[:, i] for i in range(4)]
    k = -b / torch.clamp(d, min=1e-6)
    endpoint_power = c * torch.pow(torch.full_like(d, t_end), d)
    exp_peak = -a * b * math.exp(-1.0)
    return (
        torch.relu((k - 0.03) / 0.03).square().mean()
        + torch.relu((endpoint_power - 160.0) / 160.0).square().mean()
        + torch.relu((exp_peak - 80.0) / 80.0).square().mean()
    )


def train_pseudo_label_network(
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[PhysicsCoefficientNet, np.ndarray, pd.DataFrame, dict[str, list[float]]]:
    source_w0 = float(data.source_y_raw[0])
    target_w0 = target_curve_w0(data, cfg)
    prefix_count = pretraining_target_prefix_count(data, cfg)
    prefix_scale = target_prefix_loss_scale(data, cfg) if prefix_count > 0 else 0.0
    model = build_pseudo_label_network(data, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_pseudo, weight_decay=1e-4)
    xs = make_tensor(data.source_sequences, device)
    xt = make_tensor(data.target_sequences, device)
    cuts = make_tensor(data.source_cuts, device)
    target_cuts = make_tensor(data.target_cuts, device)
    y_source = make_tensor(source_wear_target(data, cfg), device)
    y_target = make_tensor(target_wear_target(data, cfg), device)
    batch_size = min(cfg.pseudo_batch_size, len(xs), len(xt))
    history = {"loss": [], "curve": [], "target_prefix": [], "s8": [], "anchor": []}

    for epoch in range(cfg.pseudo_epochs):
        steps = max(1, math.ceil(len(xs) / batch_size))
        totals = {key: 0.0 for key in history}
        for _ in range(steps):
            indices = batch_indices(len(xs), batch_size, device)
            coefficients = model(xs[indices])
            curves = physical_curves_torch(
                coefficients, cuts, source_w0, normalize=cfg.wear_target == "vb_norm"
            )
            curve_loss = F.smooth_l1_loss(curves, y_source.unsqueeze(0).expand_as(curves))
            target_prefix_loss = torch.zeros((), device=device)
            if prefix_count > 0:
                target_indices = batch_indices(len(xt), batch_size, device)
                target_coefficients = model(xt[target_indices])
                target_curves = physical_curves_torch(
                    target_coefficients,
                    target_cuts,
                    target_w0,
                    normalize=cfg.wear_target == "vb_norm",
                )
                target_prefix_loss = F.smooth_l1_loss(
                    target_curves[:, :prefix_count],
                    y_target[:prefix_count].unsqueeze(0).expand(batch_size, -1),
                )
            s8_loss = s8_constraint_loss(coefficients, float(data.source_cuts[-1]))
            anchor_loss = torch.mean(torch.log(torch.clamp(coefficients / model.base, min=1e-8)).square())
            loss = (
                curve_loss
                + prefix_scale * target_prefix_loss
                + cfg.lambda_s8 * s8_loss
                + cfg.lambda_coeff_anchor * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["curve"] += float(curve_loss.detach().cpu())
            totals["target_prefix"] += float(target_prefix_loss.detach().cpu())
            totals["s8"] += float(s8_loss.detach().cpu())
            totals["anchor"] += float(anchor_loss.detach().cpu())
        for key in history:
            history[key].append(totals[key] / steps)
        log_epoch("pseudo_label", epoch, cfg.pseudo_epochs, totals, steps, cfg.log_every)

    model.eval()
    with torch.no_grad():
        source_coefficients = model(xs)
        target_coefficients = model(xt)
        target_curves = physical_curves_torch(
            target_coefficients,
            make_tensor(data.target_cuts, device),
            target_w0,
            normalize=cfg.wear_target == "vb_norm",
        )
    low, high = (0.0, 500.0) if cfg.wear_target == "vb" else (0.0, 1.0)
    target_curves_np = np.clip(target_curves.cpu().numpy(), low, high)

    records = []
    for domain, values in (
        (cfg.source, source_coefficients.cpu().numpy()),
        (cfg.target, target_coefficients.cpu().numpy()),
    ):
        for set_id, row in enumerate(values):
            records.append(
                {
                    "domain": domain,
                    "lifecycle_set": set_id,
                    "W0_source_anchor": source_w0,
                    "W0_target_curve_anchor": target_w0,
                    "A": row[0],
                    "B": row[1],
                    "C": row[2],
                    "D": row[3],
                }
            )
    return model, target_curves_np.astype(np.float32), pd.DataFrame(records), history


class LifecycleGenerator(nn.Module):
    def __init__(self, noise_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(noise_dim + 1, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor, pseudo_wear: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat([z, pseudo_wear.unsqueeze(-1)], dim=-1).transpose(1, 2)
        return self.net(conditioned).transpose(1, 2)


class LifecycleDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2)).squeeze(-1)
        return self.output(h).squeeze(-1)


def build_lifecycle_ddgan(
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[LifecycleGenerator, LifecycleDiscriminator]:
    feature_dim = data.source_sequences.shape[-1]
    return (
        LifecycleGenerator(cfg.noise_dim, feature_dim, cfg.gan_hidden_dim).to(device),
        LifecycleDiscriminator(feature_dim, cfg.gan_hidden_dim).to(device),
    )


def soft_histogram_js(x: torch.Tensor, y: torch.Tensor, bins: int = 16) -> torch.Tensor:
    centers = torch.linspace(-3.5, 3.5, bins, device=x.device, dtype=x.dtype)
    bandwidth = 0.45

    def histogram(values: torch.Tensor) -> torch.Tensor:
        assignment = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / bandwidth).square())
        hist = assignment.mean(dim=(0, 1))
        return hist / torch.clamp(hist.sum(dim=-1, keepdim=True), min=1e-8)

    p = histogram(x)
    q = histogram(y)
    midpoint = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p + 1e-8) - torch.log(midpoint + 1e-8)), dim=-1)
    kl_q = torch.sum(q * (torch.log(q + 1e-8) - torch.log(midpoint + 1e-8)), dim=-1)
    return 0.5 * (kl_p + kl_q).mean()


def train_lifecycle_ddgan(
    data: LifecycleData,
    target_pseudo_sets: np.ndarray,
    pseudo_model: PhysicsCoefficientNet,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[LifecycleGenerator, LifecycleDiscriminator, dict[str, list[float]]]:
    generator, discriminator = build_lifecycle_ddgan(data, cfg, device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))
    xs = make_tensor(data.source_sequences, device)
    xt = make_tensor(data.target_sequences, device)
    pseudo = make_tensor(target_pseudo_sets, device)
    pseudo_model.eval()
    for parameter in pseudo_model.parameters():
        parameter.requires_grad = False
    cuts = make_tensor(data.target_cuts, device)
    batch_size = min(cfg.lifecycle_batch_size, len(xs), len(xt))
    history = {
        "d_loss": [],
        "g_loss": [],
        "adversarial": [],
        "js": [],
        "pseudo_consistency": [],
    }
    best_score = float("inf")
    best_epoch = 0
    best_generator_state: dict[str, torch.Tensor] | None = None
    best_discriminator_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.gan_epochs):
        steps = max(1, math.ceil(max(len(xs), len(xt)) / batch_size))
        totals = {key: 0.0 for key in history}
        for _ in range(steps):
            source_idx = batch_indices(len(xs), batch_size, device)
            target_idx = batch_indices(len(xt), batch_size, device)
            source_batch = xs[source_idx]
            target_batch = xt[target_idx]
            pseudo_batch = pseudo[target_idx]
            z = torch.randn(batch_size, pseudo.shape[1], cfg.noise_dim, device=device)
            fake = generator(z, pseudo_batch)

            opt_d.zero_grad(set_to_none=True)
            target_logits = discriminator(target_batch)
            source_logits = discriminator(source_batch)
            fake_logits = discriminator(fake.detach())
            target_loss = F.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits))
            source_loss = F.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_loss = (target_loss + source_loss + fake_loss) / 3.0
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(fake)
            adversarial = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            js_loss = soft_histogram_js(fake, target_batch)
            fake_coefficients = pseudo_model(fake)
            fake_pseudo_curve = physical_curves_torch(
                fake_coefficients,
                cuts,
                w0=target_curve_w0(data, cfg),
                normalize=cfg.wear_target == "vb_norm",
            )
            pseudo_consistency = F.smooth_l1_loss(fake_pseudo_curve, pseudo_batch)
            g_loss = (
                adversarial
                + cfg.lambda_js * js_loss
                + cfg.lambda_pseudo_consistency * pseudo_consistency
            )
            g_loss.backward()
            opt_g.step()
            totals["d_loss"] += float(d_loss.detach().cpu())
            totals["g_loss"] += float(g_loss.detach().cpu())
            totals["adversarial"] += float(adversarial.detach().cpu())
            totals["js"] += float(js_loss.detach().cpu())
            totals["pseudo_consistency"] += float(pseudo_consistency.detach().cpu())
        for key in history:
            history[key].append(totals[key] / steps)
        log_epoch("ddgan", epoch, cfg.gan_epochs, totals, steps, cfg.log_every)
        selection_score = (
            totals["adversarial"]
            + cfg.lambda_js * totals["js"]
            + cfg.lambda_pseudo_consistency * totals["pseudo_consistency"]
        ) / steps
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            best_generator_state = {
                key: value.detach().cpu().clone() for key, value in generator.state_dict().items()
            }
            best_discriminator_state = {
                key: value.detach().cpu().clone() for key, value in discriminator.state_dict().items()
            }
    if best_generator_state is None or best_discriminator_state is None:
        raise RuntimeError("DDGAN training produced no selectable checkpoint")
    generator.load_state_dict(best_generator_state)
    discriminator.load_state_dict(best_discriminator_state)
    history["selected_epoch"] = [float(best_epoch)]
    history["selected_score"] = [best_score]
    announce("ddgan_checkpoint_selected", epoch=best_epoch, score=best_score)
    return generator, discriminator, history


@torch.no_grad()
def generate_lifecycle_sets(
    generator: LifecycleGenerator,
    target_pseudo_sets: np.ndarray,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    generator.eval()
    count = max(1, len(target_pseudo_sets) * cfg.generated_multiplier)
    pseudo_sets = np.tile(target_pseudo_sets, (cfg.generated_multiplier, 1))[:count].astype(np.float32)
    pseudo = make_tensor(pseudo_sets, device)
    generated = []
    for start in range(0, count, cfg.lifecycle_batch_size):
        end = min(start + cfg.lifecycle_batch_size, count)
        labels = pseudo[start:end]
        z = torch.randn(end - start, pseudo.shape[1], cfg.noise_dim, device=device)
        generated.append(generator(z, labels).cpu().numpy())
    x = np.concatenate(generated, axis=0).astype(np.float32)
    return x, pseudo_sets


@torch.no_grad()
def filter_generated_lifecycle_sets(
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    pseudo_model: PhysicsCoefficientNet,
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    pseudo_model.eval()
    scores = []
    cuts = make_tensor(data.target_cuts, device)
    for start in range(0, len(generated_x), cfg.lifecycle_batch_size):
        end = min(start + cfg.lifecycle_batch_size, len(generated_x))
        features = make_tensor(generated_x[start:end], device)
        conditioned = make_tensor(generated_y[start:end], device)
        coefficients = pseudo_model(features)
        reconstructed = physical_curves_torch(
            coefficients,
            cuts,
            w0=target_curve_w0(data, cfg),
            normalize=cfg.wear_target == "vb_norm",
        )
        scores.append(torch.mean(torch.abs(reconstructed - conditioned), dim=1).cpu().numpy())
    consistency_mae = np.concatenate(scores).astype(np.float64)
    quantile_threshold = float(np.quantile(consistency_mae, cfg.pseudo_filter_quantile))
    requested_threshold = quantile_threshold
    if cfg.pseudo_filter_max_mae is not None:
        requested_threshold = min(requested_threshold, cfg.pseudo_filter_max_mae)
    kept = consistency_mae <= requested_threshold
    minimum = min(max(1, cfg.pseudo_filter_min_sets), len(consistency_mae))
    relaxed_for_minimum = False
    if int(kept.sum()) < minimum:
        kept = np.zeros(len(consistency_mae), dtype=bool)
        kept[np.argsort(consistency_mae, kind="stable")[:minimum]] = True
        relaxed_for_minimum = True
    effective_threshold = float(consistency_mae[kept].max())
    score_frame = pd.DataFrame(
        {
            "candidate_set": np.arange(len(consistency_mae), dtype=int),
            "pseudo_consistency_mae": consistency_mae,
            "kept": kept,
        }
    )
    report = {
        "candidate_sets": int(len(consistency_mae)),
        "kept_sets": int(kept.sum()),
        "rejected_sets": int((~kept).sum()),
        "keep_ratio": float(kept.mean()),
        "quantile": cfg.pseudo_filter_quantile,
        "quantile_threshold": quantile_threshold,
        "configured_max_mae": cfg.pseudo_filter_max_mae,
        "effective_threshold": effective_threshold,
        "minimum_sets": minimum,
        "threshold_relaxed_for_minimum": relaxed_for_minimum,
        "candidate_mae_mean": float(consistency_mae.mean()),
        "candidate_mae_std": float(consistency_mae.std()),
        "kept_mae_mean": float(consistency_mae[kept].mean()),
        "kept_mae_max": effective_threshold,
    }
    announce("generated_sets_filtered", **report)
    return generated_x[kept], generated_y[kept], score_frame, report


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.net(x), dim=-1)
        return x * weights * x.shape[-1], weights


class PhysicsGuidedRecurrentCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, hidden_dim * 2)
        self.power_path = nn.Linear(input_dim, hidden_dim)
        self.exponential_path = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_delta: torch.Tensor,
        h_previous: torch.Tensor,
    ) -> torch.Tensor:
        update, forget = torch.sigmoid(self.gates(torch.cat([x, h_previous], dim=-1))).chunk(2, dim=-1)
        power_term = torch.tanh(self.power_path(x_delta))
        sigma = torch.sigmoid(self.exponential_path(x_delta)).clamp_min(1e-4)
        exponential_term = torch.tanh(torch.exp(-1.0 / sigma))
        physics_state = power_term + h_previous * power_term + h_previous * exponential_term
        candidate = torch.tanh(physics_state)
        return (1.0 - update) * (forget * h_previous) + update * candidate


class BidirectionalPGRULayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim % 2:
            raise ValueError("PGRU output dimension must be even for bidirectional concatenation")
        direction_dim = output_dim // 2
        self.forward_cell = PhysicsGuidedRecurrentCell(input_dim, direction_dim)
        self.backward_cell = PhysicsGuidedRecurrentCell(input_dim, direction_dim)
        self.direction_dim = direction_dim

    def _run(self, x: torch.Tensor, cell: PhysicsGuidedRecurrentCell, reverse: bool) -> torch.Tensor:
        batch, steps, _ = x.shape
        hidden = torch.zeros(batch, self.direction_dim, device=x.device, dtype=x.dtype)
        deltas = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        outputs: list[torch.Tensor] = []
        positions = range(steps - 1, -1, -1) if reverse else range(steps)
        for position in positions:
            current = x[:, position]
            hidden = cell(current, deltas[:, position], hidden)
            outputs.append(hidden)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._run(x, self.forward_cell, False), self._run(x, self.backward_cell, True)],
            dim=-1,
        )


class FusedBidirectionalPGRULayer(nn.Module):
    """cuDNN recurrence with vectorized paper-style power/exponential guidance."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim % 2:
            raise ValueError("PGRU output dimension must be even for bidirectional concatenation")
        self.recurrent = nn.GRU(
            input_dim,
            output_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.power_path = nn.Linear(input_dim, output_dim)
        self.exponential_path = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.recurrent.flatten_parameters()
        recurrent, _ = self.recurrent(x)
        deltas = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        power_term = torch.tanh(self.power_path(deltas))
        sigma = torch.sigmoid(self.exponential_path(deltas)).clamp_min(1e-4)
        exponential_term = torch.tanh(torch.exp(-1.0 / sigma))
        return torch.tanh(recurrent + power_term + recurrent * power_term + recurrent * exponential_term)


class PaperPGRUFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        use_attention: bool,
        implementation: str,
    ) -> None:
        super().__init__()
        self.attention = FeatureAttention(feature_dim) if use_attention else None
        layer = FusedBidirectionalPGRULayer if implementation == "fused" else BidirectionalPGRULayer
        self.pgru1 = layer(feature_dim, hidden1)
        self.pgru2 = layer(hidden1, hidden2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention = None
        if self.attention is not None:
            x, attention = self.attention(x)
        return self.pgru2(self.pgru1(x)), attention


class PaperPGRURegressor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden1: int,
        hidden2: int,
        use_attention: bool,
        implementation: str,
    ) -> None:
        super().__init__()
        self.feature_extractor = PaperPGRUFeatureExtractor(
            feature_dim, hidden1, hidden2, use_attention, implementation
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(hidden2, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        grl_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features, attention = self.feature_extractor(x)
        wear = self.regression_head(features).squeeze(-1)
        domain = self.domain_head(grad_reverse(features.mean(dim=1), grl_lambda))
        return wear, domain, attention


def build_pgru_regressor(
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> PaperPGRURegressor:
    return PaperPGRURegressor(
        len(data.selected_features),
        cfg.pgru_hidden1,
        cfg.pgru_hidden2,
        cfg.feature_attention,
        cfg.pgru_implementation,
    ).to(device)


def sequence_smoothness_loss(prediction: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 3:
        return torch.zeros((), device=prediction.device)
    second = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    return second.square().mean()


def train_adaptation_stage(
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
    use_domain: bool,
) -> tuple[PaperPGRURegressor, dict[str, list[float]]]:
    model = build_pgru_regressor(data, cfg, device)
    if cfg.wear_target == "vb":
        output_layer = model.regression_head[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Expected the regression head to end with nn.Linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, float(np.mean(source_wear_target(data, cfg))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_reg, weight_decay=1e-4)
    xs = make_tensor(data.source_sequences, device)
    xt = make_tensor(data.target_sequences, device)
    y_source = make_tensor(source_wear_target(data, cfg), device)
    y_target = make_tensor(target_wear_target(data, cfg), device)
    prefix_count = pretraining_target_prefix_count(data, cfg)
    prefix_scale = target_prefix_loss_scale(data, cfg) if prefix_count > 0 else 0.0
    batch_size = min(cfg.lifecycle_batch_size, len(xs), len(xt))
    history = {
        "loss": [],
        "regression": [],
        "target_prefix": [],
        "domain": [],
        "smooth": [],
    }
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.adapt_epochs):
        steps = max(1, math.ceil(max(len(xs), len(xt)) / batch_size))
        totals = {key: 0.0 for key in history}
        for _ in range(steps):
            source_idx = batch_indices(len(xs), batch_size, device)
            source_pred, source_domain, _ = model(xs[source_idx], grl_lambda=1.0 if use_domain else 0.0)
            source_target = y_source.unsqueeze(0).expand_as(source_pred)
            regression = F.smooth_l1_loss(source_pred, source_target)
            target_prefix_loss = torch.zeros((), device=device)
            domain_loss = torch.zeros((), device=device)
            loss = regression
            target_domain = None
            if use_domain or prefix_count > 0:
                target_idx = batch_indices(len(xt), batch_size, device)
                target_pred, target_domain, _ = model(
                    xt[target_idx], grl_lambda=1.0 if use_domain else 0.0
                )
                if prefix_count > 0:
                    target_prefix_loss = F.smooth_l1_loss(
                        target_pred[:, :prefix_count],
                        y_target[:prefix_count].unsqueeze(0).expand(batch_size, -1),
                    )
                    loss = loss + prefix_scale * target_prefix_loss
            if use_domain:
                if target_domain is None:
                    raise RuntimeError("Target domain logits were not computed")
                domain_logits = torch.cat([source_domain, target_domain], dim=0)
                domain_labels = torch.cat(
                    [
                        torch.zeros(batch_size, dtype=torch.long, device=device),
                        torch.ones(batch_size, dtype=torch.long, device=device),
                    ]
                )
                domain_loss = F.cross_entropy(domain_logits, domain_labels)
                loss = loss + cfg.lambda_domain * domain_loss
            smooth = sequence_smoothness_loss(source_pred)
            if cfg.lambda_smooth > 0:
                loss = loss + cfg.lambda_smooth * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["regression"] += float(regression.detach().cpu())
            totals["target_prefix"] += float(target_prefix_loss.detach().cpu())
            totals["domain"] += float(domain_loss.detach().cpu())
            totals["smooth"] += float(smooth.detach().cpu())
        for key in history:
            history[key].append(totals[key] / steps)
        stage = "domain_adaptation" if use_domain else "source_only"
        log_epoch(stage, epoch, cfg.adapt_epochs, totals, steps, cfg.log_every)
        selection_score = totals["regression"] / steps
        if prefix_count > 0:
            selection_score += prefix_scale * totals["target_prefix"] / steps
        if use_domain:
            selection_score += cfg.lambda_domain * abs(totals["domain"] / steps - math.log(2.0))
        if cfg.lambda_smooth > 0:
            selection_score += cfg.lambda_smooth * totals["smooth"] / steps
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"{stage} training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    history["selected_epoch"] = [float(best_epoch)]
    history["selected_score"] = [best_score]
    announce(f"{stage}_checkpoint_selected", epoch=best_epoch, score=best_score)
    return model, history


def fine_tune_regression_head(
    base_model: PaperPGRURegressor,
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    data: LifecycleData,
    target_pseudo_sets: np.ndarray,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[PaperPGRURegressor, dict[str, list[float]]]:
    model = copy.deepcopy(base_model)
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False
    for parameter in model.domain_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.regression_head.parameters(), lr=cfg.lr_finetune, weight_decay=1e-4
    )
    xg = make_tensor(generated_x, device)
    yg = make_tensor(generated_y, device)
    xs = make_tensor(data.source_sequences, device)
    ys = make_tensor(source_wear_target(data, cfg), device)
    xt = make_tensor(data.target_sequences, device)
    yt = make_tensor(target_pseudo_sets, device)
    y_target_true = make_tensor(target_wear_target(data, cfg), device)
    prefix_count = pretraining_target_prefix_count(data, cfg)
    prefix_scale = target_prefix_loss_scale(data, cfg) if prefix_count > 0 else 0.0
    batch_size = min(cfg.lifecycle_batch_size, len(xg), len(xs), len(xt))
    history = {
        "loss": [],
        "generated": [],
        "source_replay": [],
        "target_pseudo": [],
        "target_prefix": [],
    }
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    model.train()
    for epoch in range(cfg.finetune_epochs):
        active_lengths = [len(xg)]
        if cfg.lambda_source_replay > 0:
            active_lengths.append(len(xs))
        if cfg.lambda_target_pseudo > 0 or prefix_count > 0:
            active_lengths.append(len(xt))
        steps = max(1, math.ceil(max(active_lengths) / batch_size))
        totals = {key: 0.0 for key in history}
        for _ in range(steps):
            generated_idx = batch_indices(len(xg), batch_size, device)
            generated_prediction, _, _ = model(xg[generated_idx])
            generated_loss = F.smooth_l1_loss(generated_prediction, yg[generated_idx])
            source_loss = torch.zeros((), device=device)
            if cfg.lambda_source_replay > 0:
                source_idx = batch_indices(len(xs), batch_size, device)
                source_prediction, _, _ = model(xs[source_idx])
                source_target = ys.unsqueeze(0).expand_as(source_prediction)
                source_loss = F.smooth_l1_loss(source_prediction, source_target)
            target_pseudo_loss = torch.zeros((), device=device)
            target_prefix_loss = torch.zeros((), device=device)
            if cfg.lambda_target_pseudo > 0 or prefix_count > 0:
                target_idx = batch_indices(len(xt), batch_size, device)
                target_prediction, _, _ = model(xt[target_idx])
            if cfg.lambda_target_pseudo > 0:
                target_pseudo_loss = F.smooth_l1_loss(target_prediction, yt[target_idx])
            if prefix_count > 0:
                target_prefix_loss = F.smooth_l1_loss(
                    target_prediction[:, :prefix_count],
                    y_target_true[:prefix_count].unsqueeze(0).expand(batch_size, -1),
                )
            loss = (
                generated_loss
                + cfg.lambda_source_replay * source_loss
                + cfg.lambda_target_pseudo * target_pseudo_loss
                + prefix_scale * target_prefix_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.regression_head.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["generated"] += float(generated_loss.detach().cpu())
            totals["source_replay"] += float(source_loss.detach().cpu())
            totals["target_pseudo"] += float(target_pseudo_loss.detach().cpu())
            totals["target_prefix"] += float(target_prefix_loss.detach().cpu())
        for key in history:
            history[key].append(totals[key] / steps)
        log_epoch(
            "generated_target_finetune",
            epoch,
            cfg.finetune_epochs,
            totals,
            steps,
            cfg.log_every,
        )
        selection_score = totals["loss"] / steps
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.regression_head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Fine-tuning produced no selectable checkpoint")
    model.regression_head.load_state_dict(best_state)
    history["selected_epoch"] = [float(best_epoch)]
    history["selected_score"] = [best_score]
    announce("fine_tuning_checkpoint_selected", epoch=best_epoch, score=best_score)
    return model, history


def fine_tune_on_target_prefix(
    base_model: PaperPGRURegressor,
    data: LifecycleData,
    cfg: PaperRegressionConfig,
    device: torch.device,
) -> tuple[PaperPGRURegressor, dict[str, list[float]]]:
    prefix_count = target_prefix_count(data, cfg)
    if prefix_count <= 0:
        raise ValueError("Target-prefix fine-tuning requires a positive target prefix")

    model = copy.deepcopy(base_model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.regression_head.parameters():
        parameter.requires_grad = True
    if cfg.target_prefix_finetune_scope == "recurrent_head":
        for parameter in model.feature_extractor.pgru2.parameters():
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr_target_prefix_finetune,
        weight_decay=1e-4,
    )
    xt = make_tensor(data.target_sequences, device)
    yt = make_tensor(target_wear_target(data, cfg)[:prefix_count], device)
    xs = make_tensor(data.source_sequences, device)
    ys = make_tensor(source_wear_target(data, cfg), device)
    batch_size = min(cfg.lifecycle_batch_size, len(xt), len(xs))
    history = {"loss": [], "target_prefix": [], "source_replay": [], "smooth": []}
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    model.train()

    for epoch in range(cfg.target_prefix_finetune_epochs):
        steps = max(1, math.ceil(max(len(xt), len(xs)) / batch_size))
        totals = {key: 0.0 for key in history}
        for _ in range(steps):
            target_idx = batch_indices(len(xt), batch_size, device)
            target_prediction, _, _ = model(xt[target_idx])
            target_loss = F.smooth_l1_loss(
                target_prediction[:, :prefix_count],
                yt.unsqueeze(0).expand(batch_size, -1),
            )

            source_loss = torch.zeros((), device=device)
            if cfg.lambda_source_replay > 0:
                source_idx = batch_indices(len(xs), batch_size, device)
                source_prediction, _, _ = model(xs[source_idx])
                source_loss = F.smooth_l1_loss(
                    source_prediction,
                    ys.unsqueeze(0).expand_as(source_prediction),
                )

            smooth = sequence_smoothness_loss(target_prediction)
            loss = (
                cfg.lambda_target_prefix * target_loss
                + cfg.lambda_source_replay * source_loss
                + cfg.lambda_smooth * smooth
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["target_prefix"] += float(target_loss.detach().cpu())
            totals["source_replay"] += float(source_loss.detach().cpu())
            totals["smooth"] += float(smooth.detach().cpu())

        for key in history:
            history[key].append(totals[key] / steps)
        log_epoch(
            "target_prefix_finetune",
            epoch,
            cfg.target_prefix_finetune_epochs,
            totals,
            steps,
            cfg.log_every,
        )
        selection_score = totals["loss"] / steps
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Target-prefix fine-tuning produced no selectable checkpoint")
    model.load_state_dict(best_state)
    history["selected_epoch"] = [float(best_epoch)]
    history["selected_score"] = [best_score]
    announce("target_prefix_finetune_checkpoint_selected", epoch=best_epoch, score=best_score)
    return model, history


@torch.no_grad()
def predict_lifecycle(
    model: PaperPGRURegressor,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    x = make_tensor(sequences, device)
    predictions = []
    attentions = []
    for start in range(0, len(x), batch_size):
        prediction, _, attention = model(x[start : start + batch_size])
        predictions.append(prediction.cpu().numpy())
        if attention is not None:
            attentions.append(attention.cpu().numpy())
    stacked = np.concatenate(predictions, axis=0)
    attention_mean = np.concatenate(attentions, axis=0).mean(axis=(0, 1)) if attentions else None
    return stacked.mean(axis=0), stacked.std(axis=0), attention_mean


def write_long_generated_csv(
    path: Path,
    generated_x: np.ndarray,
    generated_y: np.ndarray,
    cuts: np.ndarray,
    features: list[str],
    cfg: PaperRegressionConfig,
) -> None:
    rows = []
    for set_id in range(len(generated_x)):
        frame = pd.DataFrame(generated_x[set_id], columns=features)
        frame.insert(0, f"pseudo_{wear_column_name(cfg)}", generated_y[set_id])
        frame.insert(0, "cut", cuts)
        frame.insert(0, "generated_set", set_id)
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def write_summary(output_dir: Path, result: dict[str, object]) -> None:
    cfg = result["config"]
    metrics = result["regression"]["metrics"]
    evaluation = result["evaluation"]
    lines = [
        "# Paper-aligned Continuous VB Regression Results",
        "",
        f"**Split**: `{cfg['source']} -> {cfg['target']}`  ",
        f"**Cuts**: `{cfg['max_cuts'] or 'all'}`  ",
        f"**Augmented lifecycle sets**: `{cfg['augmentation_windows']}`  ",
        f"**Wear target**: `{cfg['wear_target']}`  ",
        f"**Target labeled prefix**: `{evaluation['target_prefix_cuts']}` cuts "
        f"(`{100.0 * cfg['target_prefix_fraction']:.1f}%`)  ",
        f"**Target prefix strategy**: `{cfg['target_prefix_strategy']}`  ",
        f"**Metric scope**: `{evaluation['metrics_scope']}` "
        f"(`{evaluation['target_suffix_cuts']}` cuts)  ",
        f"**Task deviation from paper**: three-stage classification is replaced by continuous `{cfg['wear_target']}` regression.",
        "",
        "## Paper Alignment",
        "",
        "- Sliding-window expansion is organized as complete lifecycle feature sets.",
        "- The pseudo-label FC network maps a flattened lifecycle feature set to `A/B/C/D`.",
        "- DDGAN discriminates target-real from source-real and generated feature sets with a J-S regularizer.",
        "- Two bidirectional physics-guided recurrent layers operate along the cut axis.",
        f"- Fine-tuning starts from the `{cfg['finetune_base']}` model, freezes the feature extractor, "
        "and combines generated target data with configured source replay and target pseudo distillation.",
        "",
        "## Metrics",
        "",
        "| System | MAE | RMSE | R2 | Endpoint Error | High-wear MAE | Monotonic Violations | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        lines.append(
            f"| {name} | {metric['mae']:.4f} | {metric['rmse']:.4f} | {metric['r2']:.4f} | "
            f"{metric['endpoint_error']:.4f} | {metric['high_wear_mae']:.4f} | "
            f"{metric['monotonic_violations']} | {metric['pearson']:.4f} |"
        )
    if evaluation["target_prefix_cuts"] > 0 and cfg["target_prefix_strategy"] == "joint":
        lines.extend(
            [
                "",
                "Compatibility note: `source_only_pgru` remains the stable artifact key, but in "
                "prefix-supervised runs it receives source labels plus the labeled target prefix.",
            ]
        )
    elif evaluation["target_prefix_cuts"] > 0:
        lines.extend(
            [
                "",
                "- `source_only_pgru` and `proposed_before_target_prefix_finetune` do not use target "
                "VB labels; only the final proposed model receives the labeled target prefix.",
            ]
        )
    lines.extend(
        [
            "",
            "## Leakage Guard",
            "",
            (
                f"- Only the first {evaluation['target_prefix_cuts']} target VB labels enter training; "
                f"all {evaluation['target_suffix_cuts']} suffix labels are evaluation-only."
                if evaluation["target_prefix_cuts"] > 0
                else "- Target VB is used only after training for metric computation."
            ),
            (
                "- Target pseudo-labels are predicted without target VB supervision; target labels enter only the final prefix fine-tune."
                if cfg["target_prefix_strategy"] == "finetune"
                else "- Target pseudo-labels are predicted from sensor features using source supervision and any configured labeled target prefix."
            ),
            "- No target suffix label, target endpoint, or target normalization coefficient enters training.",
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{output_dir / 'paper_regression_results.json'}`",
            f"- Predictions: `{result['regression']['predictions_csv']}`",
            f"- Pseudo labels: `{result['pseudo_labeling']['pseudo_labels_csv']}`",
            f"- Physical coefficients: `{result['pseudo_labeling']['coefficients_csv']}`",
            f"- Generated lifecycle features: `{result['generation']['generated_features_csv']}`",
            f"- Generated-set consistency scores: `{result['generation']['filter_scores_csv']}`",
        ]
    )
    if result["regression"].get("attention_summary_csv"):
        lines.append(f"- Feature attention: `{result['regression']['attention_summary_csv']}`")
    (output_dir / "PAPER_REGRESSION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(torch_json(value), indent=2), encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


PRETRAINED_COMPATIBILITY_FIELDS = (
    "source",
    "target",
    "max_cuts",
    "wear_target",
    "target_prefix_fraction",
    "lambda_target_prefix",
    "target_prefix_strategy",
    "augmentation_windows",
    "crop_start_frac",
    "crop_end_frac",
    "pcc_threshold",
    "max_features",
    "sampling_rate_hz",
    "spindle_rpm",
    "window_rotations",
    "stride_rotations",
    "seed",
    "noise_dim",
    "gan_hidden_dim",
    "pgru_hidden1",
    "pgru_hidden2",
    "pgru_implementation",
    "feature_attention",
)

BACKWARD_CONFIG_DEFAULTS = {
    "target_prefix_fraction": 0.0,
    "lambda_target_prefix": 1.0,
    "target_prefix_strategy": "joint",
    "target_prefix_finetune_epochs": 100,
    "lr_target_prefix_finetune": 1e-4,
    "target_prefix_finetune_scope": "recurrent_head",
}


def saved_config_value(saved: dict[str, object], field: str) -> object:
    return saved.get(field, BACKWARD_CONFIG_DEFAULTS.get(field))


def validate_pretrained_dir(cfg: PaperRegressionConfig) -> Path | None:
    if not cfg.pretrained_dir:
        return None
    pretrained_dir = Path(cfg.pretrained_dir)
    config_path = pretrained_dir / "RUN_CONFIG.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    saved = load_json(config_path)
    current = asdict(cfg)
    changed = [
        field
        for field in PRETRAINED_COMPATIBILITY_FIELDS
        if saved_config_value(saved, field) != current.get(field)
    ]
    if changed:
        raise ValueError(f"Pretrained run is incompatible for fields: {changed}")
    return pretrained_dir


def mark_stage(output_dir: Path, stage: str, status: str = "running") -> None:
    state_path = output_dir / "RUN_STATE.json"
    state = load_json(state_path) if state_path.exists() else {"completed_stages": []}
    completed = list(state.get("completed_stages", []))
    if stage and stage not in completed:
        completed.append(stage)
    state.update(
        {
            "status": status,
            "completed_stages": completed,
            "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_json(state_path, state)


def load_weights(path: Path, model: nn.Module, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def save_torch_rng(path: Path) -> None:
    state: dict[str, object] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def restore_torch_rng(path: Path) -> None:
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def announce(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def run_pipeline(cfg: PaperRegressionConfig) -> dict[str, object]:
    start = perf_counter()
    set_seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "RUN_CONFIG.json"
    current_config = asdict(cfg)
    if cfg.resume and config_path.exists():
        saved_config = load_json(config_path)
        ignored = {"resume", "log_every", "force_features"}
        changed = [
            key
            for key, value in current_config.items()
            if key not in ignored and saved_config_value(saved_config, key) != value
        ]
        if changed:
            raise ValueError(f"Cannot resume with changed configuration fields: {changed}")
    write_json(config_path, current_config)
    mark_stage(output_dir, "", status="running")
    announce("run_start", seed=cfg.seed, output_dir=str(output_dir), device=str(device))
    if cfg.data_sanity_path:
        sanity_source = Path(cfg.data_sanity_path)
        if not sanity_source.exists():
            raise FileNotFoundError(sanity_source)
        sanity = load_json(sanity_source)
        write_json(output_dir / "data_sanity.json", sanity)
        announce("data_sanity_reused", source=str(sanity_source))
    else:
        sanity = run_data_sanity(Path(cfg.data_root), output_dir)
    data = load_lifecycle_data(cfg)
    announce(
        "data_ready",
        source_sets=len(data.source_sequences),
        target_sets=len(data.target_sequences),
        cuts=len(data.source_cuts),
        features=len(data.selected_features),
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    pretrained_dir = validate_pretrained_dir(cfg)
    pretrained_checkpoint_dir = pretrained_dir / "checkpoints" if pretrained_dir else None
    if pretrained_dir:
        announce("pretrained_stages_available", source=str(pretrained_dir))

    coefficient_path = output_dir / f"{cfg.source}_to_{cfg.target}_physical_coefficients.csv"
    pseudo_path = output_dir / f"{cfg.source}_to_{cfg.target}_target_pseudo_labels.csv"
    pseudo_state_path = checkpoint_dir / "pseudo_label_net.pt"
    pseudo_rng_path = checkpoint_dir / "rng_after_pseudo.pt"
    pseudo_sets_path = checkpoint_dir / "target_pseudo_sets.npy"
    pseudo_history_path = checkpoint_dir / "pseudo_history.json"
    can_resume_pseudo = all(
        path.exists()
        for path in (pseudo_state_path, pseudo_sets_path, pseudo_history_path, coefficient_path)
    )
    if pretrained_dir:
        assert pretrained_checkpoint_dir is not None
        pretrained_coefficient_path = (
            pretrained_dir / f"{cfg.source}_to_{cfg.target}_physical_coefficients.csv"
        )
        required = (
            pretrained_checkpoint_dir / "pseudo_label_net.pt",
            pretrained_checkpoint_dir / "target_pseudo_sets.npy",
            pretrained_checkpoint_dir / "pseudo_history.json",
            pretrained_coefficient_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing pretrained pseudo-label artifacts: {missing}")
        pseudo_model = build_pseudo_label_network(data, cfg, device)
        load_weights(pretrained_checkpoint_dir / "pseudo_label_net.pt", pseudo_model, device)
        target_pseudo_sets = np.load(pretrained_checkpoint_dir / "target_pseudo_sets.npy")
        coefficients = pd.read_csv(pretrained_coefficient_path)
        pseudo_history = load_json(pretrained_checkpoint_dir / "pseudo_history.json")
        restore_torch_rng(pretrained_checkpoint_dir / "rng_after_pseudo.pt")
        coefficients.to_csv(coefficient_path, index=False)
        announce("stage_reused", stage="pseudo_label", source=str(pretrained_dir))
    elif cfg.resume and can_resume_pseudo:
        pseudo_model = build_pseudo_label_network(data, cfg, device)
        load_weights(pseudo_state_path, pseudo_model, device)
        target_pseudo_sets = np.load(pseudo_sets_path)
        coefficients = pd.read_csv(coefficient_path)
        pseudo_history = load_json(pseudo_history_path)
        restore_torch_rng(pseudo_rng_path)
        announce("stage_resumed", stage="pseudo_label")
    else:
        pseudo_model, target_pseudo_sets, coefficients, pseudo_history = train_pseudo_label_network(
            data, cfg, device
        )
        coefficients.to_csv(coefficient_path, index=False)
        torch.save(pseudo_model.state_dict(), pseudo_state_path)
        np.save(pseudo_sets_path, target_pseudo_sets)
        write_json(pseudo_history_path, pseudo_history)
        save_torch_rng(pseudo_rng_path)
        mark_stage(output_dir, "pseudo_label")
        announce("stage_complete", stage="pseudo_label")
    target_pseudo = target_pseudo_sets.mean(axis=0)
    target_pseudo_std = target_pseudo_sets.std(axis=0)
    pseudo_frames = []
    for set_id, curve in enumerate(target_pseudo_sets):
        pseudo_column = f"pseudo_{wear_column_name(cfg)}"
        pseudo_frames.append(
            pd.DataFrame(
                {
                    "lifecycle_set": set_id,
                    "cut": data.target_cuts,
                    pseudo_column: curve,
                    f"{pseudo_column}_mean": target_pseudo,
                    f"{pseudo_column}_std": target_pseudo_std,
                }
            )
        )
    pd.concat(pseudo_frames, ignore_index=True).to_csv(pseudo_path, index=False)

    generator_path = checkpoint_dir / "lifecycle_dd_generator.pt"
    discriminator_path = checkpoint_dir / "lifecycle_dd_discriminator.pt"
    gan_history_path = checkpoint_dir / "gan_history.json"
    gan_rng_path = checkpoint_dir / "rng_after_gan.pt"
    generated_x_path = checkpoint_dir / "generated_x.npy"
    generated_y_path = checkpoint_dir / "generated_y.npy"
    filter_report_path = checkpoint_dir / "generated_filter_report.json"
    filter_scores_path = output_dir / f"{cfg.source}_to_{cfg.target}_generated_filter_scores.csv"
    can_resume_gan = all(
        path.exists()
        for path in (
            generator_path,
            discriminator_path,
            gan_history_path,
            generated_x_path,
            generated_y_path,
            filter_report_path,
            filter_scores_path,
        )
    )
    if pretrained_dir:
        assert pretrained_checkpoint_dir is not None
        pretrained_filter_scores_path = (
            pretrained_dir / f"{cfg.source}_to_{cfg.target}_generated_filter_scores.csv"
        )
        required = (
            pretrained_checkpoint_dir / "lifecycle_dd_generator.pt",
            pretrained_checkpoint_dir / "lifecycle_dd_discriminator.pt",
            pretrained_checkpoint_dir / "gan_history.json",
            pretrained_checkpoint_dir / "generated_x.npy",
            pretrained_checkpoint_dir / "generated_y.npy",
            pretrained_checkpoint_dir / "generated_filter_report.json",
            pretrained_filter_scores_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing pretrained DDGAN artifacts: {missing}")
        generator, discriminator = build_lifecycle_ddgan(data, cfg, device)
        load_weights(pretrained_checkpoint_dir / "lifecycle_dd_generator.pt", generator, device)
        load_weights(
            pretrained_checkpoint_dir / "lifecycle_dd_discriminator.pt", discriminator, device
        )
        gan_history = load_json(pretrained_checkpoint_dir / "gan_history.json")
        generated_x = np.load(pretrained_checkpoint_dir / "generated_x.npy")
        generated_y = np.load(pretrained_checkpoint_dir / "generated_y.npy")
        filter_report = load_json(pretrained_checkpoint_dir / "generated_filter_report.json")
        filter_scores = pd.read_csv(pretrained_filter_scores_path)
        restore_torch_rng(pretrained_checkpoint_dir / "rng_after_gan.pt")
        filter_scores.to_csv(filter_scores_path, index=False)
        announce("stage_reused", stage="ddgan", source=str(pretrained_dir))
    elif cfg.resume and can_resume_gan:
        generator, discriminator = build_lifecycle_ddgan(data, cfg, device)
        load_weights(generator_path, generator, device)
        load_weights(discriminator_path, discriminator, device)
        gan_history = load_json(gan_history_path)
        generated_x = np.load(generated_x_path)
        generated_y = np.load(generated_y_path)
        filter_report = load_json(filter_report_path)
        filter_scores = pd.read_csv(filter_scores_path)
        restore_torch_rng(gan_rng_path)
        announce("stage_resumed", stage="ddgan")
    else:
        generator, discriminator, gan_history = train_lifecycle_ddgan(
            data, target_pseudo_sets, pseudo_model, cfg, device
        )
        candidate_x, candidate_y = generate_lifecycle_sets(
            generator, target_pseudo_sets, cfg, device
        )
        generated_x, generated_y, filter_scores, filter_report = filter_generated_lifecycle_sets(
            candidate_x, candidate_y, pseudo_model, data, cfg, device
        )
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        np.save(generated_x_path, generated_x)
        np.save(generated_y_path, generated_y)
        write_json(gan_history_path, gan_history)
        write_json(filter_report_path, filter_report)
        filter_scores.to_csv(filter_scores_path, index=False)
        save_torch_rng(gan_rng_path)
        mark_stage(output_dir, "ddgan")
        announce("stage_complete", stage="ddgan")
    generated_path = output_dir / f"{cfg.source}_to_{cfg.target}_generated_lifecycle_features.csv"
    write_long_generated_csv(
        generated_path,
        generated_x,
        generated_y,
        data.target_cuts,
        data.selected_features,
        cfg,
    )

    source_state_path = checkpoint_dir / "source_only_pgru.pt"
    adaptation_state_path = checkpoint_dir / "domain_adaptation_pgru.pt"
    adaptation_history_path = checkpoint_dir / "adaptation_histories.json"
    adaptation_rng_path = checkpoint_dir / "rng_after_adaptation.pt"
    can_resume_adaptation = all(
        path.exists() for path in (source_state_path, adaptation_state_path, adaptation_history_path)
    )
    if pretrained_dir:
        assert pretrained_checkpoint_dir is not None
        required = (
            pretrained_checkpoint_dir / "source_only_pgru.pt",
            pretrained_checkpoint_dir / "domain_adaptation_pgru.pt",
            pretrained_checkpoint_dir / "adaptation_histories.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing pretrained adaptation artifacts: {missing}")
        source_model = build_pgru_regressor(data, cfg, device)
        adaptation_model = build_pgru_regressor(data, cfg, device)
        load_weights(pretrained_checkpoint_dir / "source_only_pgru.pt", source_model, device)
        load_weights(
            pretrained_checkpoint_dir / "domain_adaptation_pgru.pt", adaptation_model, device
        )
        adaptation_histories = load_json(pretrained_checkpoint_dir / "adaptation_histories.json")
        source_history = adaptation_histories["source_only"]
        adaptation_history = adaptation_histories["domain_adaptation"]
        restore_torch_rng(pretrained_checkpoint_dir / "rng_after_adaptation.pt")
        announce("stage_reused", stage="adaptation", source=str(pretrained_dir))
    elif cfg.resume and can_resume_adaptation:
        source_model = build_pgru_regressor(data, cfg, device)
        adaptation_model = build_pgru_regressor(data, cfg, device)
        load_weights(source_state_path, source_model, device)
        load_weights(adaptation_state_path, adaptation_model, device)
        adaptation_histories = load_json(adaptation_history_path)
        source_history = adaptation_histories["source_only"]
        adaptation_history = adaptation_histories["domain_adaptation"]
        restore_torch_rng(adaptation_rng_path)
        announce("stage_resumed", stage="adaptation")
    else:
        source_model, source_history = train_adaptation_stage(data, cfg, device, use_domain=False)
        adaptation_model, adaptation_history = train_adaptation_stage(data, cfg, device, use_domain=True)
        torch.save(source_model.state_dict(), source_state_path)
        torch.save(adaptation_model.state_dict(), adaptation_state_path)
        write_json(
            adaptation_history_path,
            {"source_only": source_history, "domain_adaptation": adaptation_history},
        )
        save_torch_rng(adaptation_rng_path)
        mark_stage(output_dir, "adaptation")
        announce("stage_complete", stage="adaptation")

    generated_state_name = (
        "pre_prefix_finetuned_pgru.pt"
        if cfg.target_prefix_strategy == "finetune"
        else "proposed_finetuned_pgru.pt"
    )
    generated_history_name = (
        "pre_prefix_finetune_history.json"
        if cfg.target_prefix_strategy == "finetune"
        else "finetune_history.json"
    )
    generated_state_path = checkpoint_dir / generated_state_name
    finetune_history_path = checkpoint_dir / generated_history_name
    finetune_rng_path = checkpoint_dir / "rng_after_finetune.pt"
    if cfg.resume and generated_state_path.exists() and finetune_history_path.exists():
        proposed_model = build_pgru_regressor(data, cfg, device)
        load_weights(generated_state_path, proposed_model, device)
        finetune_history = load_json(finetune_history_path)
        restore_torch_rng(finetune_rng_path)
        announce("stage_resumed", stage="fine_tuning")
    else:
        finetune_base_model = source_model if cfg.finetune_base == "source" else adaptation_model
        proposed_model, finetune_history = fine_tune_regression_head(
            finetune_base_model,
            generated_x,
            generated_y,
            data,
            target_pseudo_sets,
            cfg,
            device,
        )
        torch.save(proposed_model.state_dict(), generated_state_path)
        write_json(finetune_history_path, finetune_history)
        save_torch_rng(finetune_rng_path)
        mark_stage(output_dir, "fine_tuning")
        announce("stage_complete", stage="fine_tuning")

    before_prefix_finetune_model: PaperPGRURegressor | None = None
    target_prefix_finetune_history: dict[str, list[float]] | None = None
    if cfg.target_prefix_strategy == "finetune":
        before_prefix_finetune_model = proposed_model
        prefix_state_path = checkpoint_dir / "proposed_finetuned_pgru.pt"
        prefix_history_path = checkpoint_dir / "target_prefix_finetune_history.json"
        prefix_rng_path = checkpoint_dir / "rng_after_target_prefix_finetune.pt"
        if cfg.resume and prefix_state_path.exists() and prefix_history_path.exists():
            proposed_model = build_pgru_regressor(data, cfg, device)
            load_weights(prefix_state_path, proposed_model, device)
            target_prefix_finetune_history = load_json(prefix_history_path)
            restore_torch_rng(prefix_rng_path)
            announce("stage_resumed", stage="target_prefix_finetune")
        else:
            proposed_model, target_prefix_finetune_history = fine_tune_on_target_prefix(
                before_prefix_finetune_model,
                data,
                cfg,
                device,
            )
            torch.save(proposed_model.state_dict(), prefix_state_path)
            write_json(prefix_history_path, target_prefix_finetune_history)
            save_torch_rng(prefix_rng_path)
            mark_stage(output_dir, "target_prefix_finetune")
            announce("stage_complete", stage="target_prefix_finetune")

    model_map = {
        "source_only_pgru": source_model,
        "domain_adaptation_pgru": adaptation_model,
        "proposed_ddgan_pgru_finetune": proposed_model,
    }
    if before_prefix_finetune_model is not None:
        model_map["proposed_before_target_prefix_finetune"] = before_prefix_finetune_model
    predictions = {}
    prediction_std = {}
    attention_values = {}
    for name, model in model_map.items():
        mean, std, attention = predict_lifecycle(
            model, data.target_sequences, device, cfg.lifecycle_batch_size
        )
        low, high = prediction_bounds(cfg)
        predictions[name] = np.clip(mean, low, high)
        prediction_std[name] = std
        if attention is not None:
            attention_values[name] = attention
    predictions["physics_pseudo_label"] = target_pseudo
    prediction_std["physics_pseudo_label"] = target_pseudo_std
    y_true = target_wear_target(data, cfg)
    prefix_count = target_prefix_count(data, cfg)
    evaluation_slice = slice(prefix_count, None)
    metrics = {
        name: metric_dict(
            y_true[evaluation_slice],
            prediction[evaluation_slice],
            no_norm=cfg.wear_target == "vb",
        )
        for name, prediction in predictions.items()
    }

    if prefix_count > 0:
        split = np.where(
            np.arange(len(y_true)) < prefix_count,
            "target_prefix_train",
            "target_suffix_test",
        )
    else:
        split = np.full(len(y_true), "target_evaluation", dtype=object)

    prediction_frame = pd.DataFrame(
        {
            "cut": data.target_cuts,
            "split": split,
            f"y_true_{wear_column_name(cfg)}": y_true,
            **{f"{name}_pred": value for name, value in predictions.items()},
            **{f"{name}_std": value for name, value in prediction_std.items()},
        }
    )
    prediction_path = output_dir / f"{cfg.source}_to_{cfg.target}_paper_regression_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    attention_path: Path | None = None
    if attention_values:
        attention_frame = pd.DataFrame({"feature": data.selected_features})
        for name, values in attention_values.items():
            attention_frame[f"{name}_mean_weight"] = values
        attention_path = output_dir / f"{cfg.source}_to_{cfg.target}_feature_attention.csv"
        attention_frame.to_csv(attention_path, index=False)

    with torch.no_grad():
        generated_tensor = make_tensor(generated_x, device)
        target_tensor = make_tensor(data.target_sequences, device)
        source_tensor = make_tensor(data.source_sequences, device)
        generation_quality = {
            "soft_js_generated_target": float(soft_histogram_js(generated_tensor, target_tensor).cpu()),
            "soft_js_generated_source": float(soft_histogram_js(generated_tensor, source_tensor).cpu()),
            "target_probability_generated": float(torch.sigmoid(discriminator(generated_tensor)).mean().cpu()),
            "target_probability_target": float(torch.sigmoid(discriminator(target_tensor)).mean().cpu()),
            "target_probability_source": float(torch.sigmoid(discriminator(source_tensor)).mean().cpu()),
        }
        generated_coefficients = pseudo_model(generated_tensor)
        generated_pseudo_curves = physical_curves_torch(
            generated_coefficients,
            make_tensor(data.target_cuts, device),
            w0=target_curve_w0(data, cfg),
            normalize=cfg.wear_target == "vb_norm",
        )
        generation_quality["generated_pseudo_consistency_mae"] = float(
            torch.mean(
                torch.abs(generated_pseudo_curves - make_tensor(generated_y, device))
            ).cpu()
        )

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(pseudo_model.state_dict(), model_dir / "physics_pseudo_label_net.pt")
    torch.save(generator.state_dict(), model_dir / "lifecycle_dd_generator.pt")
    torch.save(discriminator.state_dict(), model_dir / "lifecycle_dd_discriminator.pt")
    torch.save(source_model.state_dict(), model_dir / "source_only_pgru.pt")
    torch.save(adaptation_model.state_dict(), model_dir / "domain_adaptation_pgru.pt")
    torch.save(proposed_model.state_dict(), model_dir / "proposed_finetuned_pgru.pt")
    if before_prefix_finetune_model is not None:
        torch.save(
            before_prefix_finetune_model.state_dict(),
            model_dir / "proposed_before_target_prefix_finetune.pt",
        )

    result = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": perf_counter() - start,
        "config": asdict(cfg),
        "device": str(device),
        "sanity": sanity,
        "task": {
            "paper_task": "three-stage tool wear classification",
            "implemented_task": f"continuous {cfg.wear_target} regression per cut",
            "wear_target": cfg.wear_target,
            "intentional_deviation": "FC output 3 is replaced by a scalar regression output at every cut",
            "system_key_note": (
                "source_only_pgru is retained as a backward-compatible artifact key; "
                "with target_prefix_fraction > 0 it also receives labeled target-prefix supervision"
                if prefix_count > 0 and cfg.target_prefix_strategy == "joint"
                else "source_only_pgru receives source labels only"
            ),
        },
        "evaluation": {
            "target_prefix_fraction": cfg.target_prefix_fraction,
            "target_prefix_strategy": cfg.target_prefix_strategy,
            "target_prefix_cuts": prefix_count,
            "target_suffix_cuts": len(y_true) - prefix_count,
            "target_prefix_last_cut": float(data.target_cuts[prefix_count - 1])
            if prefix_count > 0
            else None,
            "target_suffix_first_cut": float(data.target_cuts[prefix_count]),
            "metrics_scope": "target_suffix_only" if prefix_count > 0 else "full_target",
        },
        "feature_info": data.feature_info,
        "leakage_guard": {
            "target_ground_truth_used_for_training": prefix_count > 0,
            "target_ground_truth_used_for_evaluation_only": prefix_count == 0,
            "target_prefix_ground_truth_used_for_training": prefix_count > 0,
            "target_prefix_ground_truth_training_stage": (
                "target_prefix_finetune"
                if cfg.target_prefix_strategy == "finetune" and prefix_count > 0
                else "joint_training"
                if prefix_count > 0
                else None
            ),
            "target_suffix_ground_truth_used_for_training": False,
            "target_prefix_cuts": prefix_count,
            "target_suffix_cuts": len(y_true) - prefix_count,
            "target_normalization_used_for_training": False,
            "raw_vb_mode": cfg.wear_target == "vb",
        },
        "pseudo_labeling": {
            "network": "flattened lifecycle FC: cuts*28 -> 128 -> 64 -> A/B/C/D",
            "history": pseudo_history,
            "num_target_pseudo_sets": len(target_pseudo_sets),
            "pseudo_labels_csv": str(pseudo_path),
            "coefficients_csv": str(coefficient_path),
        },
        "generation": {
            "network": "lifecycle convolutional DDGAN with target/source/generated discriminator losses",
            "history": gan_history,
            "quality": generation_quality,
            "filter": filter_report,
            "filter_scores_csv": str(filter_scores_path),
            "num_generated_lifecycle_sets": len(generated_x),
            "generated_features_csv": str(generated_path),
        },
        "regression": {
            "network": f"two-layer bidirectional PGRU sequence regressor ({cfg.pgru_implementation})",
            "training": (
                f"{cfg.finetune_base} pretraining base -> freeze extractor -> generated-target "
                f"fine-tuning (source replay={cfg.lambda_source_replay}, "
                f"real-target pseudo distillation={cfg.lambda_target_pseudo}, "
                f"target prefix fraction={cfg.target_prefix_fraction}, "
                f"target prefix strategy={cfg.target_prefix_strategy})"
            ),
            "feature_attention": cfg.feature_attention,
            "histories": {
                "source_only": source_history,
                "domain_adaptation": adaptation_history,
                "fine_tuning": finetune_history,
                "target_prefix_fine_tuning": target_prefix_finetune_history,
            },
            "metrics": metrics,
            "predictions_csv": str(prediction_path),
            "attention_summary_csv": str(attention_path) if attention_path else None,
        },
    }
    (output_dir / "paper_regression_results.json").write_text(
        json.dumps(torch_json(result), indent=2), encoding="utf-8"
    )
    write_summary(output_dir, result)
    mark_stage(output_dir, "evaluation", status="complete")
    announce("run_complete", seed=cfg.seed, elapsed_seconds=result["elapsed_seconds"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-aligned PHM DDGAN + PGRU pipeline with continuous VB regression."
    )
    parser.add_argument("--data-root", default=r"D:\PHM\data")
    parser.add_argument("--output-dir", default=r"D:\PHM\refine-logs\paper-regression-smoke")
    parser.add_argument("--feature-cache-dir")
    parser.add_argument("--data-sanity-path")
    parser.add_argument("--pretrained-dir")
    parser.add_argument("--source", default="c1", choices=DATASETS)
    parser.add_argument("--target", default="c4", choices=DATASETS)
    parser.add_argument("--max-cuts", type=int, default=30)
    parser.add_argument("--full-lifecycle", action="store_true")
    parser.add_argument("--crop-start-frac", type=float, default=0.10)
    parser.add_argument("--crop-end-frac", type=float, default=0.90)
    parser.add_argument("--pcc-threshold", type=float, default=0.90)
    parser.add_argument("--max-features", type=int, default=28)
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
    parser.add_argument("--augmentation-windows", type=int, default=30)
    parser.add_argument("--sampling-rate-hz", type=float, default=50000.0)
    parser.add_argument("--spindle-rpm", type=float, default=10400.0)
    parser.add_argument("--window-rotations", type=int, default=60)
    parser.add_argument("--stride-rotations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--feature-attention", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("source and target must differ")
    if args.augmentation_windows < 1:
        raise ValueError("augmentation_windows must be positive")
    if args.generated_multiplier < 1:
        raise ValueError("generated_multiplier must be positive")
    if not 0.0 < args.pseudo_filter_quantile <= 1.0:
        raise ValueError("pseudo_filter_quantile must be in (0, 1]")
    if args.pseudo_filter_max_mae is not None and args.pseudo_filter_max_mae <= 0:
        raise ValueError("pseudo_filter_max_mae must be positive")
    if args.pseudo_filter_min_sets < 1:
        raise ValueError("pseudo_filter_min_sets must be positive")
    if args.lr_finetune <= 0:
        raise ValueError("lr_finetune must be positive")
    if not 0.0 <= args.target_prefix_fraction < 1.0:
        raise ValueError("target-prefix-fraction must be in [0, 1)")
    if args.target_prefix_fraction > 0 and args.wear_target != "vb":
        raise ValueError("target prefix supervision requires --wear-target vb to prevent endpoint leakage")
    if args.lambda_target_prefix < 0:
        raise ValueError("lambda-target-prefix must be non-negative")
    if args.target_prefix_strategy == "finetune" and args.target_prefix_fraction <= 0:
        raise ValueError("target-prefix-strategy finetune requires a positive target prefix")
    if args.target_prefix_finetune_epochs < 1:
        raise ValueError("target-prefix-finetune-epochs must be positive")
    if args.lr_target_prefix_finetune <= 0:
        raise ValueError("lr-target-prefix-finetune must be positive")
    if args.lambda_source_replay < 0 or args.lambda_target_pseudo < 0:
        raise ValueError("fine-tuning loss weights must be non-negative")
    cfg = PaperRegressionConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        feature_cache_dir=args.feature_cache_dir,
        data_sanity_path=args.data_sanity_path,
        pretrained_dir=args.pretrained_dir,
        source=args.source,
        target=args.target,
        max_cuts=None if args.full_lifecycle else args.max_cuts,
        crop_start_frac=args.crop_start_frac,
        crop_end_frac=args.crop_end_frac,
        pcc_threshold=args.pcc_threshold,
        max_features=args.max_features,
        wear_target=args.wear_target,
        target_prefix_fraction=args.target_prefix_fraction,
        lambda_target_prefix=args.lambda_target_prefix,
        target_prefix_strategy=args.target_prefix_strategy,
        target_prefix_finetune_epochs=args.target_prefix_finetune_epochs,
        lr_target_prefix_finetune=args.lr_target_prefix_finetune,
        target_prefix_finetune_scope=args.target_prefix_finetune_scope,
        augmentation_windows=args.augmentation_windows,
        sampling_rate_hz=args.sampling_rate_hz,
        spindle_rpm=args.spindle_rpm,
        window_rotations=args.window_rotations,
        stride_rotations=args.stride_rotations,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        pseudo_epochs=args.pseudo_epochs,
        gan_epochs=args.gan_epochs,
        adapt_epochs=args.adapt_epochs,
        finetune_epochs=args.finetune_epochs,
        lifecycle_batch_size=args.lifecycle_batch_size,
        pseudo_batch_size=args.pseudo_batch_size,
        noise_dim=args.noise_dim,
        gan_hidden_dim=args.gan_hidden_dim,
        pgru_hidden1=args.pgru_hidden1,
        pgru_hidden2=args.pgru_hidden2,
        pgru_implementation=args.pgru_implementation,
        lr_pseudo=args.lr_pseudo,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lr_reg=args.lr_reg,
        lr_finetune=args.lr_finetune,
        lambda_js=args.lambda_js,
        lambda_domain=args.lambda_domain,
        lambda_s8=args.lambda_s8,
        lambda_coeff_anchor=args.lambda_coeff_anchor,
        lambda_pseudo_consistency=args.lambda_pseudo_consistency,
        lambda_smooth=args.lambda_smooth,
        lambda_source_replay=args.lambda_source_replay,
        lambda_target_pseudo=args.lambda_target_pseudo,
        finetune_base=args.finetune_base,
        generated_multiplier=args.generated_multiplier,
        pseudo_filter_quantile=args.pseudo_filter_quantile,
        pseudo_filter_max_mae=args.pseudo_filter_max_mae,
        pseudo_filter_min_sets=args.pseudo_filter_min_sets,
        feature_attention=args.feature_attention,
        force_features=args.force_features,
        resume=args.resume,
        log_every=args.log_every,
    )
    result = run_pipeline(cfg)
    print(
        json.dumps(
            torch_json(
                {
                    "elapsed_seconds": result["elapsed_seconds"],
                    "metrics": result["regression"]["metrics"],
                    "generation_quality": result["generation"]["quality"],
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
