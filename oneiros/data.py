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
    frame_stack: int = 1,
) -> dict[str, np.ndarray]:
    """Collect random-policy rollouts into flat numpy arrays.

    Returns a dict with keys ``obs_img_t``, ``obs_img_next``, ``obs_vec_t``,
    ``obs_vec_next`` and ``action``. Deterministic given ``seed``.

    ``frame_stack=2`` concatenates the previous frame onto each image
    observation (channels double). A single frame shows position only --
    velocity is invisible, and the point-mass dynamics are second-order,
    so no single-frame predictor can recover the next state. Stacking two
    frames makes velocity observable from pixels, the same reason pixel
    world models from DQN to V-JEPA consume clips rather than stills. At
    an episode start the previous frame is the current frame (zero
    apparent velocity, which is exact: episodes start at rest).
    """
    if frame_stack not in (1, 2):
        raise ValueError(f"frame_stack must be 1 or 2, got {frame_stack}")
    rng = np.random.default_rng(seed)
    env = PointMass(config=env_config or PointMassConfig(), seed=seed)

    img_t: list[np.ndarray] = []
    img_next: list[np.ndarray] = []
    vec_t: list[np.ndarray] = []
    vec_next: list[np.ndarray] = []
    acts: list[np.ndarray] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep + 1)
        prev_frame = env.render()
        for _ in range(horizon):
            frame_t = env.render()
            o_img_t = (
                np.concatenate([frame_t, prev_frame], axis=0)
                if frame_stack == 2
                else frame_t
            )
            o_vec_t = env.state_obs()
            a = rng.uniform(-env.config.max_accel, env.config.max_accel, size=2)
            env.step(a)
            frame_next = env.render()
            o_img_next = (
                np.concatenate([frame_next, frame_t], axis=0)
                if frame_stack == 2
                else frame_next
            )
            o_vec_next = env.state_obs()
            prev_frame = frame_t

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
