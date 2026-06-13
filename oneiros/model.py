"""JEPA-style latent world model (PyTorch, CPU).

The model has two learned parts:

* an **encoder** ``f`` mapping an observation to a latent ``z`` (dim ~16-32);
* a **predictor** ``g`` mapping ``(z_t, action)`` to a predicted next latent
  ``z_hat_{t+1}``.

The training objective is the Joint Embedding Predictive Architecture loss:
predict the next observation's latent *in latent space*, against a stop-gradient
target produced by an EMA copy of the encoder. There is no pixel decoder and no
reconstruction loss. Naive latent prediction collapses to a constant, so a
VICReg-style variance + covariance penalty is applied to the online latents to
keep the representation spread out and decorrelated.

Two encoder backbones are provided:

* ``ImageEncoder`` — a tiny CNN over the rendered image observation.
* ``VectorEncoder`` — an MLP over the 6D state+goal observation, used as the
  reliable fallback documented in the README.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    obs_mode: str = "vector"  # "vector" or "image"
    latent_dim: int = 16
    action_dim: int = 2
    img_channels: int = 2  # agent + goal channels when obs_mode == "image"
    img_size: int = 32
    vector_dim: int = 6  # state(4) + goal(2)
    hidden: int = 128
    ema_decay: float = 0.99


class VectorEncoder(nn.Module):
    """MLP encoder over the 6D state+goal observation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.vector_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """Tiny CNN encoder over the rendered image observation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.img_channels, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
        )
        flat = 32 * (cfg.img_size // 8) * (cfg.img_size // 8)
        self.head = nn.Sequential(
            nn.Linear(flat, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.head(h)


class Predictor(nn.Module):
    """MLP predictor g: (z_t, action) -> z_hat_{t+1}."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # residual prediction: the predictor learns the latent *delta*, which
        # makes short-horizon dynamics easier to model and improves planning.
        return z + self.net(torch.cat([z, action], dim=-1))


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture world model.

    Holds the online encoder ``f``, the predictor ``g``, and an EMA ``f_target``
    used as the (stop-gradient) prediction target.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.obs_mode == "image":
            self.encoder: nn.Module = ImageEncoder(cfg)
        else:
            self.encoder = VectorEncoder(cfg)
        self.predictor = Predictor(cfg)
        # EMA target encoder: a frozen-by-momentum copy of the online encoder.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode an observation with the online encoder f."""
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode an observation with the EMA target encoder (stop-grad)."""
        return self.target_encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next latent from a latent and an action."""
        return self.predictor(z, action)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update: target <- decay * target + (1 - decay) * online."""
        d = self.cfg.ema_decay
        for tp, p in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            tp.mul_(d).add_(p, alpha=1.0 - d)
        for tb, b in zip(
            self.target_encoder.buffers(), self.encoder.buffers(), strict=True
        ):
            tb.copy_(b)


# -- VICReg-style regularizers --------------------------------------------


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge each latent dimension's std toward ``target_std`` (anti-collapse)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Push the off-diagonal latent covariance toward zero (decorrelation)."""
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / d


def jepa_loss(
    model: JEPA,
    obs_t: torch.Tensor,
    action: torch.Tensor,
    obs_next: torch.Tensor,
    var_weight: float,
    cov_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Compute the full JEPA training loss and a dict of component scalars."""
    z_t = model.encode(obs_t)
    z_next_target = model.encode_target(obs_next)  # stop-grad EMA target
    z_hat = model.predict(z_t, action)

    pred = F.mse_loss(z_hat, z_next_target)
    var = variance_loss(z_t) + variance_loss(model.encode(obs_next))
    cov = covariance_loss(z_t)

    total = pred + var_weight * var + cov_weight * cov
    return total, {
        "pred": float(pred.detach()),
        "var": float(var.detach()),
        "cov": float(cov.detach()),
        "total": float(total.detach()),
    }
