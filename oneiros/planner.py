"""Model-predictive control (MPC) in the learned latent space.

The planner never touches the environment dynamics directly. It encodes the
current observation and a goal observation into latents with the trained
encoder, then searches action sequences by rolling the *predictor* ``g`` forward
``H`` steps and scoring each candidate by the predicted-latent distance to the
goal latent. Only the first action of the best sequence is executed; the planner
replans every step (receding-horizon MPC).

Two search strategies are provided: ``random_shooting`` and ``cem`` (the
cross-entropy method, default), both hand-rolled with numpy/torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import JEPA


@dataclass
class PlannerConfig:
    horizon: int = 12
    n_samples: int = 256
    max_accel: float = 1.0
    method: str = "cem"  # "cem" or "random"
    cem_iters: int = 3
    cem_elite_frac: float = 0.1
    seed: int = 0


class LatentMPCPlanner:
    """Plans actions by rolling the latent predictor forward toward a goal."""

    def __init__(self, model: JEPA, config: PlannerConfig | None = None) -> None:
        self.model = model
        self.cfg = config or PlannerConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    @torch.no_grad()
    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.model.encode(x).squeeze(0)

    @torch.no_grad()
    def _score_sequences(
        self, z0: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Roll the predictor forward and return per-sequence cost.

        ``actions`` has shape ``(n, H, action_dim)``. The cost is the squared
        latent distance to the goal at the final step plus a small running-cost
        term that rewards progress along the trajectory.
        """
        n, h, _ = actions.shape
        z = z0.unsqueeze(0).expand(n, -1).contiguous()
        goal = z_goal.unsqueeze(0).expand(n, -1)
        running = torch.zeros(n)
        for t in range(h):
            z = self.model.predict(z, actions[:, t, :])
            running = running + ((z - goal) ** 2).mean(dim=1)
        terminal = ((z - goal) ** 2).mean(dim=1)
        return terminal + 0.1 * running / h

    @torch.no_grad()
    def plan(self, current_obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        """Return the first action of the best action sequence found."""
        return self.plan_sequence(current_obs, goal_obs)[0]

    @torch.no_grad()
    def plan_sequence(
        self, current_obs: np.ndarray, goal_obs: np.ndarray
    ) -> np.ndarray:
        """Return the full best action sequence, shape ``(horizon, action_dim)``.

        Receding-horizon callers use only the first action (``plan``); the
        full sequence is exposed so an agent can inspect or execute the
        model's entire imagined plan (the MCP ``plan_trajectory`` tool).
        """
        z0 = self._encode(current_obs)
        z_goal = self._encode(goal_obs)
        cfg = self.cfg
        a_dim = self.model.cfg.action_dim

        if cfg.method == "random":
            actions = self._sample_uniform(cfg.n_samples, cfg.horizon, a_dim)
            costs = self._score_sequences(z0, z_goal, actions)
            best = int(torch.argmin(costs))
            return actions[best].cpu().numpy()

        # cross-entropy method: iteratively refit a diagonal Gaussian to elites.
        mean = np.zeros((cfg.horizon, a_dim), dtype=np.float32)
        std = np.full((cfg.horizon, a_dim), cfg.max_accel * 0.7, dtype=np.float32)
        n_elite = max(4, int(cfg.n_samples * cfg.cem_elite_frac))
        best_seq = np.zeros((cfg.horizon, a_dim), dtype=np.float32)
        for _ in range(cfg.cem_iters):
            samples = self._rng.normal(
                mean[None], std[None], size=(cfg.n_samples, cfg.horizon, a_dim)
            ).astype(np.float32)
            samples = np.clip(samples, -cfg.max_accel, cfg.max_accel)
            actions = torch.from_numpy(samples)
            costs = self._score_sequences(z0, z_goal, actions).cpu().numpy()
            elite_idx = np.argsort(costs)[:n_elite]
            elites = samples[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
            best_seq = elites[0]
        return best_seq.astype(np.float32)

    def _sample_uniform(self, n: int, h: int, a_dim: int) -> torch.Tensor:
        s = self._rng.uniform(
            -self.cfg.max_accel, self.cfg.max_accel, size=(n, h, a_dim)
        ).astype(np.float32)
        return torch.from_numpy(s)
