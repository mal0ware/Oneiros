"""Reference baselines and episode runners for the honesty gates.

Two things live here, both deliberately outside the model package's core:

* **Image-observation episode runners** for the swift environment, where
  the agent sees only stacked frames (position + apparent velocity from
  pixels) and plans with latent MPC. The runner mirrors
  ``demo_agent.run_episode`` but builds image observations, including the
  goal observation (the goal rendered at rest, stacked with itself --
  exact, because the goal does not move).

* **A privileged fitted-linear-dynamics MPC baseline.** The linear model
  is fit by least squares on ``(state, action) -> next_state`` from the
  same random-policy data budget the world model trains on -- but it
  reads the TRUE 4D state, which the image model never sees. Beating (or
  matching) a privileged linear planner from pixels is therefore a
  strong claim, and on the swift environment the dynamics are non-linear
  (speed-proportional drag + wall bounces) so the linear hypothesis
  class is genuinely wrong, not a strawman.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import PointMass, PointMassConfig
from .model import JEPA
from .planner import LatentMPCPlanner, PlannerConfig

REACH_TOL = 0.12  # same "reached" tolerance the vector demo uses


def stacked_obs(frame_t: np.ndarray, frame_prev: np.ndarray) -> np.ndarray:
    """Concatenate two rendered frames along the channel axis."""
    return np.concatenate([frame_t, frame_prev], axis=0)


def goal_image_obs(env: PointMass) -> np.ndarray:
    """The image observation the agent would see at rest on the goal.

    Rendered at the goal with zero velocity; stacked with itself, which
    is exact for a static goal (no apparent motion between frames).
    """
    g = env.goal
    goal_state = np.array([g[0], g[1], 0.0, 0.0], dtype=np.float64)
    frame = env.render(state=goal_state)
    return stacked_obs(frame, frame)


def run_image_episode(
    model: JEPA,
    seed: int,
    policy: str,
    env_config: PointMassConfig | None = None,
    max_steps: int = 60,
) -> tuple[bool, np.ndarray]:
    """Run one episode from pixels; return ``(reached, xy_trajectory)``.

    ``policy`` is ``"mpc"`` (latent MPC over the image world model) or
    ``"random"``. The agent only ever consumes rendered frames.
    """
    cfg = env_config or PointMassConfig.swift()
    env = PointMass(config=cfg, seed=seed)
    env.reset(seed=seed)
    planner = LatentMPCPlanner(
        model, PlannerConfig(seed=seed, max_accel=cfg.max_accel)
    )
    g_obs = goal_image_obs(env)
    rng = np.random.default_rng(seed * 1000 + 7)

    prev_frame = env.render()
    traj = [env.state[:2].copy()]
    reached = False
    for _ in range(max_steps):
        frame = env.render()
        if policy == "mpc":
            action = planner.plan(stacked_obs(frame, prev_frame), g_obs)
        else:
            action = rng.uniform(-cfg.max_accel, cfg.max_accel, size=2)
        _, _, _, info = env.step(action)
        prev_frame = frame
        traj.append(env.state[:2].copy())
        if info["dist"] <= REACH_TOL:
            reached = True
            break
    return reached, np.array(traj)


# -- privileged fitted-linear-dynamics baseline -----------------------------


def fit_linear_dynamics(
    env_config: PointMassConfig | None = None,
    n_episodes: int = 200,
    horizon: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of ``s_next = [s, a] @ W + b`` on random rollouts.

    Reads the TRUE state (privileged); same data budget as the world
    model's training set. Returns ``(W, b)`` with ``W`` shaped ``(6, 4)``.
    """
    cfg = env_config or PointMassConfig.swift()
    rng = np.random.default_rng(seed)
    env = PointMass(config=cfg, seed=seed)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep + 1)
        for _ in range(horizon):
            s = env.state.copy()
            a = rng.uniform(-cfg.max_accel, cfg.max_accel, size=2)
            s_next, _, _, _ = env.step(a)
            xs.append(np.concatenate([s, a]))
            ys.append(s_next)

    x = np.asarray(xs)
    y = np.asarray(ys)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    sol, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return sol[:-1], sol[-1]


def run_linear_mpc_episode(
    weights: tuple[np.ndarray, np.ndarray],
    seed: int,
    env_config: PointMassConfig | None = None,
    max_steps: int = 60,
    horizon: int = 12,
    n_samples: int = 256,
    cem_iters: int = 3,
) -> tuple[bool, np.ndarray]:
    """Receding-horizon CEM MPC over the fitted linear model (privileged).

    Mirrors the latent planner's search budget (horizon, samples, CEM
    iterations) so the comparison isolates the dynamics model, not the
    optimizer.
    """
    w, b = weights
    cfg = env_config or PointMassConfig.swift()
    env = PointMass(config=cfg, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    goal = env.goal.copy()
    n_elite = max(4, int(n_samples * 0.1))

    def rollout_cost(s0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        # actions: (n, H, 2); vectorized linear rollout in state space.
        n = actions.shape[0]
        s = np.tile(s0, (n, 1))
        running = np.zeros(n)
        for t in range(actions.shape[1]):
            s = np.concatenate([s, actions[:, t, :]], axis=1) @ w + b
            running += ((s[:, :2] - goal) ** 2).sum(axis=1)
        terminal = ((s[:, :2] - goal) ** 2).sum(axis=1)
        return terminal + 0.1 * running / actions.shape[1]

    traj = [env.state[:2].copy()]
    reached = False
    for _ in range(max_steps):
        mean = np.zeros((horizon, 2), dtype=np.float64)
        std = np.full((horizon, 2), cfg.max_accel * 0.7, dtype=np.float64)
        best_first = np.zeros(2)
        for _ in range(cem_iters):
            samples = rng.normal(mean[None], std[None], size=(n_samples, horizon, 2))
            samples = np.clip(samples, -cfg.max_accel, cfg.max_accel)
            costs = rollout_cost(env.state, samples)
            elite_idx = np.argsort(costs)[:n_elite]
            elites = samples[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
            best_first = elites[0, 0, :]
        _, _, _, info = env.step(best_first)
        traj.append(env.state[:2].copy())
        if info["dist"] <= REACH_TOL:
            reached = True
            break
    return reached, np.array(traj)


# -- episodic collection for the multi-step rollout gate --------------------


def collect_image_episode(
    seed: int,
    horizon: int,
    env_config: PointMassConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One random-policy episode of stacked image observations.

    Returns ``(obs, actions)`` where ``obs`` has shape ``(horizon + 1,
    2 * channels, H, W)`` (stacked frames at each step) and ``actions``
    has shape ``(horizon, 2)``. Used by the multi-step rollout gate,
    which needs whole trajectories rather than shuffled transitions.
    """
    cfg = env_config or PointMassConfig.swift()
    rng = np.random.default_rng(seed)
    env = PointMass(config=cfg, seed=seed)
    env.reset(seed=seed)

    prev_frame = env.render()
    obs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frame = env.render()
    obs.append(stacked_obs(frame, prev_frame))
    for _ in range(horizon):
        a = rng.uniform(-cfg.max_accel, cfg.max_accel, size=2)
        env.step(a)
        prev_frame = frame
        frame = env.render()
        obs.append(stacked_obs(frame, prev_frame))
        actions.append(a.astype(np.float32))

    return np.asarray(obs, dtype=np.float32), np.asarray(actions, dtype=np.float32)


@torch.no_grad()
def multistep_rollout_errors(
    model: JEPA,
    n_episodes: int,
    horizon: int,
    seed: int = 0,
    env_config: PointMassConfig | None = None,
) -> dict[str, float]:
    """Compounded H-step rollout error vs the H-step identity baseline.

    Encodes the first observation of each episode, rolls the predictor
    forward through the episode's true action sequence WITHOUT re-encoding
    (pure imagination, exactly what the planner relies on), and compares
    the final imagined latent to the target encoding of the true final
    observation. The identity baseline keeps the initial latent unchanged
    for H steps.
    """
    model.eval()
    pred_errs: list[float] = []
    ident_errs: list[float] = []
    for ep in range(n_episodes):
        obs, actions = collect_image_episode(
            seed=seed + ep + 1, horizon=horizon, env_config=env_config
        )
        o = torch.from_numpy(obs)
        a = torch.from_numpy(actions)
        z = model.encode(o[0].unsqueeze(0))
        z0 = z.clone()
        for t in range(horizon):
            z = model.predict(z, a[t].unsqueeze(0))
        z_true = model.encode_target(o[horizon].unsqueeze(0))
        pred_errs.append(float(((z - z_true) ** 2).mean()))
        ident_errs.append(float(((z0 - z_true) ** 2).mean()))

    pred = float(np.mean(pred_errs))
    ident = float(np.mean(ident_errs))
    return {
        "rollout_pred_mse": pred,
        "rollout_identity_mse": ident,
        "rollout_ratio": pred / max(ident, 1e-9),
    }
