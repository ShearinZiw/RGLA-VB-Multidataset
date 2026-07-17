from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from phm_bridge import (
    DATASETS,
    FeatureConfig,
    load_or_extract_dataset,
    load_wear,
    regression_metrics,
    run_data_sanity,
    select_features_by_source_pcc,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def torch_json(value):
    if isinstance(value, dict):
        return {k: torch_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [torch_json(v) for v in value]
    if isinstance(value, tuple):
        return [torch_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


@dataclass
class PipelineConfig:
    data_root: str
    output_dir: str
    source: str
    target: str
    max_cuts: int | None
    crop_start_frac: float
    crop_end_frac: float
    pcc_threshold: float
    max_features: int
    seed: int
    device: str
    batch_size: int
    gan_epochs: int
    reg_epochs: int
    hidden_dim: int
    noise_dim: int
    lr_g: float
    lr_d: float
    lr_r: float
    lambda_domain: float
    lambda_generated: float
    lambda_smooth: float
    generated_multiplier: int
    pseudo_label_mode: str
    no_norm: bool
    pgru_pooling: str


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class DDGenerator(nn.Module):
    def __init__(self, noise_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 1 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, y, domain], dim=1))


class DDDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.real_head = nn.Linear(hidden_dim, 1)
        self.domain_head = nn.Linear(hidden_dim, 2)
        self.wear_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.real_head(h), self.domain_head(h), self.wear_head(h)


class PGRURegressor(nn.Module):
    """A compact PGRU-style regressor over ordered feature coordinates."""

    def __init__(self, feature_dim: int, hidden_dim: int, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported PGRU pooling mode: {pooling}")
        self.input_proj = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention_head = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            if pooling == "attention"
            else None
        )
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.domain_head = nn.Linear(hidden_dim, 2)
        self.feature_dim = feature_dim
        self.pooling = pooling

    def encode_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq = x.unsqueeze(-1)
        seq = self.input_proj(seq)
        out, _ = self.gru(seq)
        if self.attention_head is None:
            pooled = out.mean(dim=1)
            attention = None
        else:
            scores = self.attention_head(out).squeeze(-1)
            attention = torch.softmax(scores, dim=1)
            pooled = torch.bmm(attention.unsqueeze(1), out).squeeze(1)
        return self.feature_head(pooled), attention

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encode_with_attention(x)
        return h

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        wear = self.reg_head(h).squeeze(1)
        domain = self.domain_head(grad_reverse(h, grl_lambda))
        return wear, domain


def make_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def domain_one_hot(domain_id: int, n: int, device: torch.device) -> torch.Tensor:
    y = torch.zeros((n, 2), dtype=torch.float32, device=device)
    y[:, domain_id] = 1.0
    return y


def normalize_source_target(
    x_source: np.ndarray,
    x_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_source.mean(axis=0)
    std = x_source.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x_source - mean) / std, (x_target - mean) / std, mean, std


def time_prior(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


# ── Physical wear model W(t) = W0 + A·t·exp(B·t) + C·t^D ──────────────────

PAPER_COEFFS_RAW = {
    "c1": {"W0": 39.6, "A": 2.1586, "B": -0.02212, "C": 0.1151, "D": 1.216},
    "c4": {"W0": 24.2, "A": 1.0018, "B": -0.007401, "C": 5.540e-10, "D": 4.573},
    "c6": {"W0": 29.1, "A": 1.8935, "B": -0.01395, "C": 0.001093, "D": 2.079},
}


def _physical_wear(t: np.ndarray, w0: float, a: float, b: float, c: float, d: float) -> np.ndarray:
    return w0 + a * t * np.exp(np.clip(b * t, -80, 80)) + c * np.power(t, d)


def _fit_physical_wear(
    t: np.ndarray,
    y: np.ndarray,
    dataset: str,
    normalized: bool,
) -> tuple[np.ndarray, tuple[float, float, float, float, float]]:
    """Fit W(t) on source wear labels in either normalized or raw VB space.

    Returns (fitted_curve, (W0, A, B, C, D)) where fitted_curve matches the
    input t positions, and the tuple can be used to evaluate W(t) at new t.
    """
    n = len(t)
    t_float = t.astype(float)
    y_std = float(np.std(y)) if float(np.std(y)) > 1e-9 else 1.0
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, 1.0)

    # Multiple starts for robustness
    starts = []
    paper = PAPER_COEFFS_RAW[dataset]
    base_w0 = 0.0 if normalized else float(y[0])
    starts.append((base_w0, paper["A"], paper["B"], paper["C"], paper["D"]))
    starts.append((base_w0, 0.5 if normalized else 1.0, -0.01, 0.1, 1.5))
    starts.append((base_w0, 1.0 if normalized else 2.0, -0.02, 0.01, 2.0))
    rng = np.random.default_rng(20260510)
    for _ in range(8):
        starts.append((
            rng.uniform(-0.1, 0.1) if normalized else rng.uniform(y_min, y_max),
            float(np.exp(rng.uniform(-3, 2))),
            float(-np.exp(rng.uniform(-5, -2))),
            float(np.exp(rng.uniform(-6, 0))),
            float(np.exp(rng.uniform(-1, 2))),
        ))

    best_loss = float("inf")
    best_curve: np.ndarray | None = None
    best_coeffs: tuple[float, float, float, float, float] = (base_w0, 1.0, -0.01, 0.1, 1.5)

    for (w0, a_raw, b_raw, c_raw, d_raw) in starts:
        params = np.array([w0, math.log(max(a_raw, 1e-12)),
                           math.log(max(-b_raw, 1e-12)),
                           math.log(max(c_raw, 1e-18)),
                           math.log(max(d_raw, 1e-12))], dtype=float)
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        lr = np.array([0.01, 0.005, 0.005, 0.005, 0.003])

        for step in range(1, 3001):
            w0_val = params[0]
            a_val = float(np.exp(params[1]))
            b_val = float(-np.exp(params[2]))
            c_val = float(np.exp(params[3]))
            d_val = float(np.exp(params[4]))

            y_hat = _physical_wear(t_float, w0_val, a_val, b_val, c_val, d_val)
            residual = (y_hat - y) / y_std
            loss_val = float(np.mean(residual ** 2))

            if loss_val < best_loss:
                best_loss = loss_val
                best_curve = y_hat.copy()
                best_coeffs = (w0_val, a_val, b_val, c_val, d_val)

            eps = 1e-5
            grad = np.zeros(5)
            for i in range(5):
                pp = params.copy()
                pp[i] += eps
                wv = pp[0]; av = float(np.exp(pp[1])); bv = float(-np.exp(pp[2]))
                cv = float(np.exp(pp[3])); dv = float(np.exp(pp[4]))
                yp = _physical_wear(t_float, wv, av, bv, cv, dv)
                grad[i] = (float(np.mean(((yp - y) / y_std) ** 2)) - loss_val) / eps

            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad * grad)
            m_hat = m / (1 - 0.9 ** step)
            v_hat = v / (1 - 0.999 ** step)
            params -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            w0_lower, w0_upper = (-0.5, 0.5) if normalized else (y_min - 0.5 * y_span, y_max + 0.5 * y_span)
            amplitude_upper = 20 if normalized else max(20.0, 5.0 * y_span)
            params[0] = float(np.clip(params[0], w0_lower, w0_upper))
            params[1] = float(np.clip(params[1], math.log(1e-8), math.log(amplitude_upper)))
            params[2] = float(np.clip(params[2], math.log(1e-6), math.log(5)))
            params[3] = float(np.clip(params[3], math.log(1e-16), math.log(amplitude_upper)))
            params[4] = float(np.clip(params[4], math.log(0.1), math.log(6)))

    assert best_curve is not None
    return (np.clip(best_curve, 0.0, 1.0) if normalized else best_curve), best_coeffs


def smoothness_loss(y_pred: torch.Tensor) -> torch.Tensor:
    if y_pred.numel() < 3:
        return torch.zeros((), device=y_pred.device)
    second = y_pred[2:] - 2 * y_pred[1:-1] + y_pred[:-2]
    return torch.mean(second * second)


def batch_indices(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, n, (batch_size,), device=device)


def train_ddgan(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target_pseudo: np.ndarray,
    cfg: PipelineConfig,
    device: torch.device,
) -> tuple[DDGenerator, DDDiscriminator, dict[str, list[float]]]:
    feature_dim = x_source.shape[1]
    generator = DDGenerator(cfg.noise_dim, feature_dim, cfg.hidden_dim).to(device)
    discriminator = DDDiscriminator(feature_dim, cfg.hidden_dim).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    xs = make_tensor(x_source, device)
    xt = make_tensor(x_target, device)
    ys = make_tensor(y_source.reshape(-1, 1), device)
    yt = make_tensor(y_target_pseudo.reshape(-1, 1), device)
    history = {"d_loss": [], "g_loss": []}
    b = min(cfg.batch_size, len(x_source), len(x_target))

    for _epoch in range(cfg.gan_epochs):
        steps = max(1, math.ceil(max(len(x_source), len(x_target)) / b))
        d_epoch = 0.0
        g_epoch = 0.0
        for _ in range(steps):
            si = batch_indices(len(x_source), b, device)
            ti = batch_indices(len(x_target), b, device)
            real_x = torch.cat([xs[si], xt[ti]], dim=0)
            real_y = torch.cat([ys[si], yt[ti]], dim=0)
            real_domain_ids = torch.cat(
                [
                    torch.zeros(b, dtype=torch.long, device=device),
                    torch.ones(b, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            real_domains = torch.cat([domain_one_hot(0, b, device), domain_one_hot(1, b, device)], dim=0)
            z = torch.randn((2 * b, cfg.noise_dim), device=device)
            fake_x = generator(z, real_y, real_domains).detach()

            opt_d.zero_grad(set_to_none=True)
            real_logits, real_domain_logits, real_wear = discriminator(real_x)
            fake_logits, _, _ = discriminator(fake_x)
            d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_domain = F.cross_entropy(real_domain_logits, real_domain_ids)
            d_wear = F.mse_loss(real_wear, real_y)
            d_loss = d_real + d_fake + 0.25 * d_domain + 0.25 * d_wear
            d_loss.backward()
            opt_d.step()

            z = torch.randn((2 * b, cfg.noise_dim), device=device)
            opt_g.zero_grad(set_to_none=True)
            fake_x = generator(z, real_y, real_domains)
            fake_logits, fake_domain_logits, fake_wear = discriminator(fake_x)
            g_adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            g_domain = F.cross_entropy(fake_domain_logits, real_domain_ids)
            g_wear = F.mse_loss(fake_wear, real_y)
            g_loss = g_adv + 0.25 * g_domain + 0.5 * g_wear
            g_loss.backward()
            opt_g.step()
            d_epoch += float(d_loss.detach().cpu())
            g_epoch += float(g_loss.detach().cpu())
        history["d_loss"].append(d_epoch / steps)
        history["g_loss"].append(g_epoch / steps)
    return generator, discriminator, history


def train_regressor(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target_real: np.ndarray,
    x_generated: np.ndarray | None,
    y_generated: np.ndarray | None,
    cfg: PipelineConfig,
    device: torch.device,
    mode: str,
) -> tuple[PGRURegressor, dict[str, list[float]]]:
    feature_dim = x_source.shape[1]
    model = PGRURegressor(feature_dim, cfg.hidden_dim, pooling=cfg.pgru_pooling).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_r)
    xs = make_tensor(x_source, device)
    ys = make_tensor(y_source, device)
    xt = make_tensor(x_target_real, device)
    if x_generated is not None and y_generated is not None:
        xg = make_tensor(x_generated, device)
        yg = make_tensor(y_generated, device)
    else:
        xg = None
        yg = None

    b = min(cfg.batch_size, len(x_source), max(1, len(x_target_real)))
    history = {"loss": [], "reg_source": [], "domain": [], "reg_generated": []}
    for _epoch in range(cfg.reg_epochs):
        steps = max(1, math.ceil(max(len(x_source), len(x_target_real)) / b))
        totals = {"loss": 0.0, "reg_source": 0.0, "domain": 0.0, "reg_generated": 0.0}
        for _ in range(steps):
            si = batch_indices(len(x_source), b, device)
            ti = batch_indices(len(x_target_real), b, device)
            opt.zero_grad(set_to_none=True)
            pred_s, dom_s = model(xs[si], grl_lambda=cfg.lambda_domain if mode != "source_only" else 0.0)
            loss_source = F.smooth_l1_loss(pred_s, ys[si])
            loss = loss_source
            domain_loss = torch.zeros((), device=device)
            gen_loss = torch.zeros((), device=device)
            if mode in {"dann", "proposed"}:
                _, dom_t = model(xt[ti], grl_lambda=cfg.lambda_domain)
                dom_logits = torch.cat([dom_s, dom_t], dim=0)
                dom_labels = torch.cat(
                    [
                        torch.zeros(b, dtype=torch.long, device=device),
                        torch.ones(b, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                domain_loss = F.cross_entropy(dom_logits, dom_labels)
                loss = loss + cfg.lambda_domain * domain_loss
            if mode == "proposed" and xg is not None and yg is not None:
                gi = batch_indices(len(xg), b, device)
                pred_g, _ = model(xg[gi], grl_lambda=0.0)
                gen_loss = F.smooth_l1_loss(pred_g, yg[gi])
                loss = loss + cfg.lambda_generated * gen_loss
            if cfg.lambda_smooth > 0.0:
                full_pred, _ = model(xs, grl_lambda=0.0)
                loss = loss + cfg.lambda_smooth * smoothness_loss(full_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["reg_source"] += float(loss_source.detach().cpu())
            totals["domain"] += float(domain_loss.detach().cpu())
            totals["reg_generated"] += float(gen_loss.detach().cpu())
        for key in totals:
            history[key].append(totals[key] / steps)
    return model, history


@torch.no_grad()
def predict(model: PGRURegressor, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    xt = make_tensor(x, device)
    for start in range(0, len(x), batch_size):
        pred, _ = model(xt[start : start + batch_size], grl_lambda=0.0)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def summarize_attention(
    model: PGRURegressor,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | float] | None:
    if model.pooling != "attention":
        return None
    model.eval()
    xt = make_tensor(x, device)
    chunks = []
    for start in range(0, len(x), batch_size):
        _, attention = model.encode_with_attention(xt[start : start + batch_size])
        assert attention is not None
        chunks.append(attention.detach().cpu().numpy())
    weights = np.concatenate(chunks, axis=0)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / math.log(max(2, weights.shape[1]))
    return {
        "mean": weights.mean(axis=0),
        "std": weights.std(axis=0),
        "mean_normalized_entropy": float(normalized_entropy.mean()),
    }


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, no_norm: bool = False) -> dict[str, float | int]:
    if no_norm:
        y_pred = np.clip(y_pred, -50.0, 500.0)
    else:
        y_pred = np.clip(y_pred, -0.25, 1.25)
    high_mask = y_true >= (0.75 * y_true[-1] if no_norm else 0.75)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "endpoint_error": float(abs(y_pred[-1] - y_true[-1])),
        "high_wear_mae": float(mean_absolute_error(y_true[high_mask], y_pred[high_mask]))
        if np.any(high_mask)
        else float(mean_absolute_error(y_true, y_pred)),
        "monotonic_violations": int(np.sum(np.diff(y_pred) < -1e-6)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12
        else 0.0,
    }


def generate_target_features(
    generator: DDGenerator,
    n_target: int,
    cfg: PipelineConfig,
    device: torch.device,
    y_target_pseudo: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    generator.eval()
    n_gen = max(n_target, n_target * cfg.generated_multiplier)
    if y_target_pseudo is not None:
        base_labels = y_target_pseudo.astype(np.float32)
    else:
        base_labels = time_prior(n_target).astype(np.float32)
    labels = np.tile(base_labels, cfg.generated_multiplier)
    if len(labels) < n_gen:
        if y_target_pseudo is not None:
            labels = np.concatenate([labels, base_labels[: n_gen - len(labels)]])
        else:
            labels = np.concatenate([labels, time_prior(n_gen - len(labels)).astype(np.float32)])
    labels = labels[:n_gen].astype(np.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, n_gen, cfg.batch_size):
            end = min(start + cfg.batch_size, n_gen)
            y = make_tensor(labels[start:end].reshape(-1, 1), device)
            z = torch.randn((end - start, cfg.noise_dim), device=device)
            domain = domain_one_hot(1, end - start, device)
            chunks.append(generator(z, y, domain).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0), labels


def load_frames(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
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
    )
    out = Path(cfg.output_dir)
    cache_dir = out / "features"
    source_frame = load_or_extract_dataset(
        Path(cfg.data_root),
        cfg.source,
        feature_cfg,
        cfg.max_cuts,
        cache_dir,
        force=False,
    )
    target_frame = load_or_extract_dataset(
        Path(cfg.data_root),
        cfg.target,
        feature_cfg,
        cfg.max_cuts,
        cache_dir,
        force=False,
    )
    feature_cols = [
        col
        for col in source_frame.columns
        if col
        not in {
            "dataset",
            "cut",
            "window_id",
            "window_start",
            "window_rows",
            "vb_avg",
            "vb_norm",
            "signal_rows",
        }
    ]
    target_label_col = "vb_avg" if cfg.no_norm else "vb_norm"
    selected, pcc = select_features_by_source_pcc(
        source_frame,
        feature_cols,
        target_label_col,
        cfg.pcc_threshold,
        cfg.max_features,
    )
    pcc_path = out / f"{cfg.source}_to_{cfg.target}_source_pcc.csv"
    pcc.to_csv(pcc_path, index=False)
    info = {"pcc_path": str(pcc_path), "selected_features": selected, "num_total_features": len(feature_cols)}
    return source_frame, target_frame, selected, info


def run_pipeline(cfg: PipelineConfig) -> dict[str, object]:
    start = perf_counter()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = run_data_sanity(Path(cfg.data_root), output_dir)
    source_frame, target_frame, selected, feature_info = load_frames(cfg)

    x_source_raw = source_frame[selected].to_numpy(dtype=np.float32)
    x_target_raw = target_frame[selected].to_numpy(dtype=np.float32)
    if cfg.no_norm:
        y_source = source_frame["vb_avg"].to_numpy(dtype=np.float32)
        y_target_true = target_frame["vb_avg"].to_numpy(dtype=np.float32)
        if cfg.pseudo_label_mode == "physical":
            src_cuts = source_frame["cut"].to_numpy(dtype=float)
            tgt_cuts = target_frame["cut"].to_numpy(dtype=float)
            src_physical_raw, (w0, a, b, c, d) = _fit_physical_wear(
                src_cuts, y_source, cfg.source, normalized=False
            )
            y_source = src_physical_raw.astype(np.float32)
            y_target_pseudo = _physical_wear(tgt_cuts, w0, a, b, c, d).astype(np.float32)
        else:
            tgt_t = time_prior(len(target_frame))
            src_vb_range = float(y_source[-1] - y_source[0])
            src_vb_start = float(y_source[0])
            y_target_pseudo = (tgt_t * src_vb_range + src_vb_start).astype(np.float32)
        label_col = "vb_avg"
    else:
        y_source = source_frame["vb_norm"].to_numpy(dtype=np.float32)
        y_target_true = target_frame["vb_norm"].to_numpy(dtype=np.float32)
        if cfg.pseudo_label_mode == "physical":
            src_cuts = source_frame["cut"].to_numpy(dtype=float)
            tgt_cuts = target_frame["cut"].to_numpy(dtype=float)
            # Fit physical model on source VB_norm → get smooth curve + coefficients
            src_physical_norm, (w0, a, b, c, d) = _fit_physical_wear(
                src_cuts, y_source, cfg.source, normalized=True
            )
            y_source = src_physical_norm.astype(np.float32)
            # Evaluate source-fitted physical model at target cut positions
            tgt_physical = _physical_wear(tgt_cuts.astype(float), w0, a, b, c, d)
            y_target_pseudo = np.clip(tgt_physical, 0.0, 1.0).astype(np.float32)
        else:
            y_target_pseudo = time_prior(len(target_frame))
        label_col = "vb_norm"
    x_source, x_target, mean, std = normalize_source_target(x_source_raw, x_target_raw)

    generator, discriminator, gan_history = train_ddgan(
        x_source,
        y_source,
        x_target,
        y_target_pseudo,
        cfg,
        device,
    )
    generated_x, generated_y = generate_target_features(
        generator, len(x_target), cfg, device,
        y_target_pseudo=y_target_pseudo,
    )

    source_only_model, source_hist = train_regressor(
        x_source,
        y_source,
        x_target,
        None,
        None,
        cfg,
        device,
        mode="source_only",
    )
    dann_model, dann_hist = train_regressor(
        x_source,
        y_source,
        x_target,
        None,
        None,
        cfg,
        device,
        mode="dann",
    )
    proposed_model, proposed_hist = train_regressor(
        x_source,
        y_source,
        x_target,
        generated_x,
        generated_y,
        cfg,
        device,
        mode="proposed",
    )

    preds = {
        "source_only_pgru": predict(source_only_model, x_target, device, cfg.batch_size),
        "dann_pgru": predict(dann_model, x_target, device, cfg.batch_size),
        "proposed_ddgan_pgru": predict(proposed_model, x_target, device, cfg.batch_size),
        "empirical_time_transfer": np.interp(time_prior(len(target_frame)), time_prior(len(source_frame)), y_source),
    }
    metrics = {name: metric_dict(y_target_true, pred, no_norm=cfg.no_norm) for name, pred in preds.items()}

    attention_path: Path | None = None
    attention_diagnostics: dict[str, dict[str, float | str]] = {}
    if cfg.pgru_pooling == "attention":
        attention_frame = pd.DataFrame({"feature": selected})
        attention_models = {
            "source_only_pgru": source_only_model,
            "dann_pgru": dann_model,
            "proposed_ddgan_pgru": proposed_model,
        }
        for name, model in attention_models.items():
            summary = summarize_attention(model, x_target, device, cfg.batch_size)
            assert summary is not None
            mean_attention = np.asarray(summary["mean"])
            attention_frame[f"{name}_mean"] = mean_attention
            attention_frame[f"{name}_std"] = np.asarray(summary["std"])
            top_index = int(np.argmax(mean_attention))
            attention_diagnostics[name] = {
                "mean_normalized_entropy": float(summary["mean_normalized_entropy"]),
                "top_feature": selected[top_index],
                "top_feature_mean_weight": float(mean_attention[top_index]),
            }
        attention_path = output_dir / f"{cfg.source}_to_{cfg.target}_pgru_attention_summary.csv"
        attention_frame.to_csv(attention_path, index=False)

    label_col = "vb_avg" if cfg.no_norm else "vb_norm"
    pred_frame = pd.DataFrame(
        {
            "dataset": cfg.target,
            "cut": target_frame["cut"].to_numpy(),
            f"y_true_{label_col}": y_target_true,
            "target_pseudo_time_prior": y_target_pseudo,
            **{f"{name}_pred_{label_col}": pred for name, pred in preds.items()},
        }
    )
    pred_path = output_dir / f"{cfg.source}_to_{cfg.target}_proposed_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    generated_path = output_dir / f"{cfg.source}_to_{cfg.target}_generated_target_features.csv"
    generated_frame = pd.DataFrame(generated_x, columns=selected)
    pseudo_label_col = "pseudo_vb_avg" if cfg.no_norm else "pseudo_vb_norm"
    generated_frame.insert(0, pseudo_label_col, generated_y)
    generated_frame.to_csv(generated_path, index=False)

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), model_dir / "dd_generator.pt")
    torch.save(discriminator.state_dict(), model_dir / "dd_discriminator.pt")
    torch.save(source_only_model.state_dict(), model_dir / "source_only_pgru.pt")
    torch.save(dann_model.state_dict(), model_dir / "dann_pgru.pt")
    torch.save(proposed_model.state_dict(), model_dir / "proposed_ddgan_pgru.pt")

    elapsed = perf_counter() - start
    result = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "sanity": sanity,
        "feature_info": {
            **feature_info,
            "num_selected_features": len(selected),
            "normalization": "source mean/std only",
            "label_type": "vb_avg (raw μm)" if cfg.no_norm else "vb_norm [0,1]",
        },
        "leakage_guard": {
            "target_ground_truth_used_for_training": False,
            "target_ground_truth_used_for_evaluation_only": True,
            "target_pseudo_label_protocol": cfg.pseudo_label_mode,
            "note": (
                "Target pseudo labels use a C1-fitted physical wear curve, not target VB."
                if cfg.pseudo_label_mode == "physical"
                else (
                    "Target pseudo labels use normalized cut index time prior, not target VB."
                    if not cfg.no_norm
                    else "Target pseudo labels use time prior scaled by SOURCE VB range (non-leaking)."
                )
            ),
            "evaluation_target": "target VB_avg (raw micrometers)" if cfg.no_norm else "target VB_norm [0,1]",
            "evaluation_cut_count": int(len(target_frame)),
        },
        "generation": {
            "network": "DDGenerator + DDDiscriminator with real/fake, domain, and wear auxiliary heads",
            "num_generated_target_samples": int(len(generated_x)),
            "gan_history": gan_history,
            "generated_features_csv": str(generated_path),
        },
        "regression": {
            "network": (
                "PGRU-style GRU regressor over selected feature coordinates "
                f"with {cfg.pgru_pooling} pooling"
            ),
            "pooling": cfg.pgru_pooling,
            "attention_summary_csv": str(attention_path) if attention_path is not None else None,
            "attention_diagnostics": attention_diagnostics,
            "histories": {
                "source_only_pgru": source_hist,
                "dann_pgru": dann_hist,
                "proposed_ddgan_pgru": proposed_hist,
            },
            "metrics": metrics,
            "predictions_csv": str(pred_path),
        },
    }
    result_path = output_dir / "proposed_pipeline_results.json"
    result_path.write_text(json.dumps(torch_json(result), indent=2), encoding="utf-8")
    write_markdown_summary(output_dir, result)
    return result


def write_markdown_summary(output_dir: Path, result: dict[str, object]) -> None:
    cfg = result["config"]
    metrics = result["regression"]["metrics"]
    lines = [
        "# Proposed Pipeline Results",
        "",
        f"**Date**: {result['date']}",
        "**Plan**: `refine-logs/EXPERIMENT_PLAN.md`",
        "",
        "## Scope",
        "",
        f"- Split: `{cfg['source']} -> {cfg['target']}`",
        f"- Cuts: `{cfg['max_cuts'] or 'all'}`",
        f"- Device: `{result['device']}`",
        f"- Label type: `{result.get('feature_info', {}).get('label_type', 'vb_norm [0,1]')}`",
        f"- Selected features: {result['feature_info']['num_selected_features']} / {result['feature_info']['num_total_features']}",
        "- Target labels are used for evaluation only, never for generated target pseudo-label training.",
        f"- Target pseudo-label protocol: `{result['leakage_guard']['target_pseudo_label_protocol']}`.",
        f"- Evaluation target: `{result['leakage_guard']['evaluation_target']}` over {result['leakage_guard']['evaluation_cut_count']} cuts.",
        "",
        "## Networks Implemented",
        "",
        "- Generation: DDGAN-style generator/discriminator with real/fake, domain, and wear auxiliary heads.",
        "- Regression: PGRU-style GRU regressor with source-only, DANN, and DDGAN-generated-target proposed variants.",
        f"- PGRU pooling: `{cfg.get('pgru_pooling', 'mean')}`.",
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
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{output_dir / 'proposed_pipeline_results.json'}`",
            f"- Predictions CSV: `{result['regression']['predictions_csv']}`",
            f"- Generated target features CSV: `{result['generation']['generated_features_csv']}`",
            *(
                [f"- PGRU attention summary CSV: `{result['regression']['attention_summary_csv']}`"]
                if result["regression"].get("attention_summary_csv")
                else []
            ),
            f"- Models: `{output_dir / 'models'}`",
            "",
            "## Current Interpretation",
            "",
            "- This is now an end-to-end runnable proposed-pipeline implementation.",
            "- It is still a first engineering version; the target pseudo-label protocol is deliberately non-leaking but simpler than a learned target physical pseudo-label network.",
            "- Full-lifecycle and all six PHM transfer directions should be run before making paper-level claims.",
        ]
    )
    (output_dir / "PROPOSED_PIPELINE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_splits_summary(output_dir: Path, results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        cfg = result["config"]
        for system, metric in result["regression"]["metrics"].items():
            rows.append(
                {
                    "source": cfg["source"],
                    "target": cfg["target"],
                    "system": system,
                    **metric,
                    "result_dir": str(Path(cfg["output_dir"])),
                }
            )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "ALL_SPLITS_SUMMARY.csv", index=False)
    lines = [
        "# All Splits Proposed Pipeline Summary",
        "",
        "| Source | Target | System | MAE | RMSE | R2 | Endpoint Error | High-wear MAE | Monotonic Violations | Pearson |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['target']} | {row['system']} | {row['mae']:.4f} | "
            f"{row['rmse']:.4f} | {row['r2']:.4f} | {row['endpoint_error']:.4f} | "
            f"{row['high_wear_mae']:.4f} | {row['monotonic_violations']} | {row['pearson']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This aggregate is generated by `--all-splits`.",
            "- Target `VB_norm` is used for evaluation only.",
            "- Target pseudo labels use the non-leaking normalized cut-index time prior.",
        ]
    )
    (output_dir / "ALL_SPLITS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "ALL_SPLITS_RESULTS.json").write_text(json.dumps(torch_json(results), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PHM 2010 DDGAN + PGRU proposed normalized wear pipeline.")
    parser.add_argument("--data-root", default=r"D:\PHM\data")
    parser.add_argument("--output-dir", default=r"D:\PHM\refine-logs\proposed-pipeline-smoke")
    parser.add_argument("--source", default="c1", choices=DATASETS)
    parser.add_argument("--target", default="c4", choices=DATASETS)
    parser.add_argument("--max-cuts", type=int, default=30)
    parser.add_argument("--full-lifecycle", action="store_true")
    parser.add_argument("--crop-start-frac", type=float, default=0.10)
    parser.add_argument("--crop-end-frac", type=float, default=0.90)
    parser.add_argument("--pcc-threshold", type=float, default=0.90)
    parser.add_argument("--max-features", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gan-epochs", type=int, default=30)
    parser.add_argument("--reg-epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--lr-g", type=float, default=2e-3)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--lr-r", type=float, default=1e-3)
    parser.add_argument("--lambda-domain", type=float, default=0.1)
    parser.add_argument("--lambda-generated", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--generated-multiplier", type=int, default=2)
    parser.add_argument("--pseudo-label-mode", default="target_time_prior",
                        choices=["target_time_prior", "physical"])
    parser.add_argument("--no-norm", action="store_true", help="Use raw VB_avg (micrometers) instead of normalized VB_norm.")
    parser.add_argument(
        "--pgru-pooling",
        default="mean",
        choices=["mean", "attention"],
        help="Pool BiGRU outputs across selected feature coordinates.",
    )
    parser.add_argument("--all-splits", action="store_true", help="Run all six PHM source-target transfer directions.")
    args = parser.parse_args()

    if args.source == args.target and not args.all_splits:
        raise ValueError("source and target must differ")
    max_cuts = None if args.full_lifecycle else args.max_cuts
    base_kwargs = dict(
        data_root=args.data_root,
        max_cuts=max_cuts,
        crop_start_frac=args.crop_start_frac,
        crop_end_frac=args.crop_end_frac,
        pcc_threshold=args.pcc_threshold,
        max_features=args.max_features,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        gan_epochs=args.gan_epochs,
        reg_epochs=args.reg_epochs,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lr_r=args.lr_r,
        lambda_domain=args.lambda_domain,
        lambda_generated=args.lambda_generated,
        lambda_smooth=args.lambda_smooth,
        generated_multiplier=args.generated_multiplier,
        pseudo_label_mode=args.pseudo_label_mode,
        no_norm=args.no_norm,
        pgru_pooling=args.pgru_pooling,
    )
    if args.all_splits:
        base_output = Path(args.output_dir)
        results = []
        for source in DATASETS:
            for target in DATASETS:
                if source == target:
                    continue
                split_dir = base_output / f"{source}_to_{target}"
                cfg = PipelineConfig(output_dir=str(split_dir), source=source, target=target, **base_kwargs)
                results.append(run_pipeline(cfg))
        write_all_splits_summary(base_output, results)
        compact = [
            {
                "source": result["config"]["source"],
                "target": result["config"]["target"],
                "metrics": result["regression"]["metrics"],
            }
            for result in results
        ]
        print(json.dumps(torch_json({"num_splits": len(results), "splits": compact}), indent=2))
    else:
        cfg = PipelineConfig(output_dir=args.output_dir, source=args.source, target=args.target, **base_kwargs)
        result = run_pipeline(cfg)
        print(
            json.dumps(
                torch_json({"elapsed_seconds": result["elapsed_seconds"], "metrics": result["regression"]["metrics"]}),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
