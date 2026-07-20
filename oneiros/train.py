"""Train the JEPA-style latent world model on random-policy rollouts.

Run as a module::

    python -m oneiros.train --obs-mode vector --steps 4000

Everything is seeded. A small checkpoint (under 2 MB for the default vector
model) is written to ``oneiros/checkpoint.pt`` so the demo and the inference /
planning tests can run without retraining.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import collect_rollouts
from .env import PointMassConfig
from .model import JEPA, ModelConfig, jepa_loss

# Resolved relative to the package so the console scripts (oneiros-mcp,
# oneiros-demo) and the MCP server work from any working directory, not
# just the repo root.
_PKG_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = str(_PKG_DIR / "checkpoint.pt")
IMAGE_CHECKPOINT_PATH = str(_PKG_DIR / "checkpoint_image.pt")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _obs_keys(obs_mode: str) -> tuple[str, str]:
    if obs_mode == "image":
        return "obs_img_t", "obs_img_next"
    return "obs_vec_t", "obs_vec_next"


def train(
    obs_mode: str = "vector",
    latent_dim: int = 16,
    steps: int = 4000,
    batch_size: int = 256,
    lr: float = 1e-3,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    delta_var_weight: float = 0.0,
    n_episodes: int = 200,
    horizon: int = 50,
    seed: int = 0,
    checkpoint_path: str = CHECKPOINT_PATH,
    env_config: PointMassConfig | None = None,
    frame_stack: int = 1,
    verbose: bool = True,
) -> dict:
    """Train the model and save a checkpoint. Returns a metrics dict."""
    set_seed(seed)

    data = collect_rollouts(
        n_episodes=n_episodes,
        horizon=horizon,
        seed=seed,
        env_config=env_config,
        frame_stack=frame_stack,
    )
    k_t, k_next = _obs_keys(obs_mode)
    obs_t = torch.from_numpy(data[k_t])
    obs_next = torch.from_numpy(data[k_next])
    action = torch.from_numpy(data["action"])
    n = obs_t.shape[0]

    cfg = ModelConfig(
        obs_mode=obs_mode,
        latent_dim=latent_dim,
        img_channels=2 * frame_stack if obs_mode == "image" else 2,
    )
    model = JEPA(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    t0 = time.time()
    last = {}
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        loss, comps = jepa_loss(
            model,
            obs_t[idx],
            action[idx],
            obs_next[idx],
            var_weight=var_weight,
            cov_weight=cov_weight,
            delta_var_weight=delta_var_weight,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        last = comps
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(
                f"step {step:5d}  total {comps['total']:.4f}  "
                f"pred {comps['pred']:.4f}  var {comps['var']:.4f}  "
                f"cov {comps['cov']:.4f}"
            )

    elapsed = time.time() - t0
    metrics = evaluate(model, data, obs_mode)
    metrics.update({"train_time_s": elapsed, **{f"final_{k}": v for k, v in last.items()}})

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "env_config": asdict(env_config) if env_config is not None else None,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    if verbose:
        print(f"saved checkpoint -> {checkpoint_path} ({elapsed:.1f}s)")
        print("metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


@torch.no_grad()
def evaluate(model: JEPA, data: dict, obs_mode: str) -> dict:
    """Evaluate predictor MSE vs an identity baseline and latent collapse.

    The identity baseline predicts ``z_hat = z_t`` (no-op dynamics). Beating it
    by a clear margin demonstrates the predictor learned real latent dynamics.
    """
    model.eval()
    k_t, k_next = _obs_keys(obs_mode)
    obs_t = torch.from_numpy(data[k_t])
    obs_next = torch.from_numpy(data[k_next])
    action = torch.from_numpy(data["action"])

    z_t = model.encode(obs_t)
    z_next = model.encode_target(obs_next)
    z_hat = model.predict(z_t, action)

    pred_mse = float(((z_hat - z_next) ** 2).mean())
    identity_mse = float(((z_t - z_next) ** 2).mean())
    per_dim_std = z_t.std(dim=0)
    mean_std = float(per_dim_std.mean())
    min_std = float(per_dim_std.min())

    return {
        "pred_mse": pred_mse,
        "identity_mse": identity_mse,
        "mse_ratio": pred_mse / max(identity_mse, 1e-9),
        "latent_mean_std": mean_std,
        "latent_min_std": min_std,
    }


def load_model(checkpoint_path: str = CHECKPOINT_PATH) -> JEPA:
    """Load a trained JEPA model from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = JEPA(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Oneiros JEPA world model.")
    p.add_argument("--obs-mode", choices=["vector", "image"], default="vector")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--var-weight", type=float, default=25.0)
    p.add_argument("--cov-weight", type=float, default=1.0)
    p.add_argument(
        "--delta-var-weight",
        type=float,
        default=0.0,
        help="temporal anti-collapse penalty; the committed image model uses 25.0",
    )
    p.add_argument(
        "--frame-stack",
        type=int,
        choices=[1, 2],
        default=1,
        help="frames per image observation; the committed image model uses 2",
    )
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    p.add_argument(
        "--env",
        choices=["default", "swift"],
        default="default",
        help="environment preset; 'swift' has large per-step motion + non-linear drag",
    )
    args = p.parse_args()

    train(
        env_config=PointMassConfig.swift() if args.env == "swift" else None,
        obs_mode=args.obs_mode,
        latent_dim=args.latent_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        delta_var_weight=args.delta_var_weight,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        frame_stack=args.frame_stack,
    )


if __name__ == "__main__":
    main()
