"""Replay buffer of random-policy rollouts.

Collects ``(obs_t, action_t, obs_next)`` transition triples by running a random
policy in the point-mass environment. Both image and vector observations are
stored so the same buffer can train either encoder backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PointMass, PointMassConfig


@dataclass
class Transition:
    obs_img_t: np.ndarray
    obs_img_next: np.ndarray
    obs_vec_t: np.ndarray
    obs_vec_next: np.ndarray
    action: np.ndarray


def collect_rollouts(
    n_episodes: int,
    horizon: int,
    seed: int = 0,
    env_config: PointMassConfig | None = None,
) -> dict[str, np.ndarray]:
    """Collect random-policy rollouts into flat numpy arrays.

    Returns a dict with keys ``obs_img_t``, ``obs_img_next``, ``obs_vec_t``,
    ``obs_vec_next`` and ``action``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    env = PointMass(config=env_config or PointMassConfig(), seed=seed)

    img_t: list[np.ndarray] = []
    img_next: list[np.ndarray] = []
    vec_t: list[np.ndarray] = []
    vec_next: list[np.ndarray] = []
    acts: list[np.ndarray] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep + 1)
        for _ in range(horizon):
            o_img_t = env.render()
            o_vec_t = env.state_obs()
            a = rng.uniform(-env.config.max_accel, env.config.max_accel, size=2)
            env.step(a)
            o_img_next = env.render()
            o_vec_next = env.state_obs()

            img_t.append(o_img_t)
            img_next.append(o_img_next)
            vec_t.append(o_vec_t)
            vec_next.append(o_vec_next)
            acts.append(a.astype(np.float32))

    return {
        "obs_img_t": np.stack(img_t).astype(np.float32),
        "obs_img_next": np.stack(img_next).astype(np.float32),
        "obs_vec_t": np.stack(vec_t).astype(np.float32),
        "obs_vec_next": np.stack(vec_next).astype(np.float32),
        "action": np.stack(acts).astype(np.float32),
    }
