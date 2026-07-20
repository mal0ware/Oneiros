"""MCP server exposing the Oneiros world model as agentic tools.

This is the differentiating piece of the project: instead of an LLM calling
hand-written functions, an agent plans by calling a *learned predictive world
model* as a tool over the Model Context Protocol. The tools let a client encode
observations into latents, roll the latent dynamics forward, plan a goal-reaching
action with latent-space MPC, and drive the underlying environment.

Run as a stdio server::

    python -m oneiros.mcp_server                    # vector model (default)
    python -m oneiros.mcp_server --obs-mode image   # committed image model
    python -m oneiros.mcp_server --checkpoint /path/to/any_checkpoint.pt

See the README for a ``claude_desktop_config.json`` snippet.
"""

from __future__ import annotations

import argparse

import numpy as np
from mcp.server.fastmcp import FastMCP

from .env import PointMass, PointMassConfig
from .planner import LatentMPCPlanner, PlannerConfig
from .train import CHECKPOINT_PATH, IMAGE_CHECKPOINT_PATH, load_model

mcp = FastMCP("oneiros")

# Lazily constructed singletons so importing this module is cheap and does not
# require a checkpoint to exist (the tool-list dump in CI must import cleanly).
# Environments are keyed by session id so independent agent sessions cannot
# clobber each other's episode state through the shared server process.
_STATE: dict[str, object] = {
    "checkpoint_path": CHECKPOINT_PATH,
    "model": None,
    "planner": None,
    "env_config": None,
}
_ENVS: dict[str, PointMass] = {}


def configure(obs_mode: str = "vector", checkpoint: str | None = None) -> str:
    """Select which trained world model the server exposes.

    ``obs_mode`` picks the committed checkpoint for that observation mode
    (``vector`` -> checkpoint.pt, ``image`` -> checkpoint_image.pt); an
    explicit ``checkpoint`` path overrides it. Cached singletons and all
    environment sessions are reset so the whole server (model, planner,
    environments) follows the newly selected checkpoint. Returns the
    selected path.
    """
    if obs_mode not in ("vector", "image"):
        raise ValueError(f"obs_mode must be 'vector' or 'image', got {obs_mode!r}")
    path = checkpoint or (
        IMAGE_CHECKPOINT_PATH if obs_mode == "image" else CHECKPOINT_PATH
    )
    _STATE["checkpoint_path"] = path
    _STATE["model"] = None
    _STATE["planner"] = None
    _STATE["env_config"] = None
    _ENVS.clear()
    return path


def _checkpoint_path() -> str:
    return _STATE["checkpoint_path"]  # type: ignore[return-value]


def _get_model():
    if _STATE["model"] is None:
        _STATE["model"] = load_model(_checkpoint_path())
    return _STATE["model"]


def _get_env_config() -> PointMassConfig:
    """The environment configuration the loaded checkpoint was trained on.

    The image model is trained on the swift preset; serving it against the
    default environment would silently mismatch the dynamics it learned.
    """
    if _STATE["env_config"] is None:
        import torch

        ckpt = torch.load(_checkpoint_path(), map_location="cpu", weights_only=False)
        raw = ckpt.get("env_config")
        _STATE["env_config"] = PointMassConfig(**raw) if raw else PointMassConfig()
    return _STATE["env_config"]  # type: ignore[return-value]


def _get_planner() -> LatentMPCPlanner:
    if _STATE["planner"] is None:
        cfg = PlannerConfig(max_accel=_get_env_config().max_accel)
        _STATE["planner"] = LatentMPCPlanner(_get_model(), cfg)
    return _STATE["planner"]  # type: ignore[return-value]


def _get_env(session: str = "default") -> PointMass:
    if session not in _ENVS:
        env = PointMass(config=_get_env_config(), seed=0)
        env.reset(seed=0)
        _ENVS[session] = env
    return _ENVS[session]


def _frame_stack() -> int:
    """Frames per image observation (from the checkpoint's channel count)."""
    return _get_model().cfg.img_channels // 2


def _image_obs(frame_t: np.ndarray, frame_prev: np.ndarray) -> list:
    """Stack rendered frames exactly the way the image model was trained."""
    if _frame_stack() == 2:
        return np.concatenate([frame_t, frame_prev], axis=0).tolist()
    return frame_t.tolist()


def _current_obs(env: PointMass, frame_prev: np.ndarray | None = None) -> list:
    """The observation for the env's current state, in the model's obs mode.

    For image mode, ``frame_prev`` is the frame rendered *before* the last
    step; at an episode start it defaults to the current frame (zero
    apparent velocity — exact, because episodes start at rest).
    """
    if _get_model().cfg.obs_mode == "image":
        frame = env.render()
        return _image_obs(frame, frame if frame_prev is None else frame_prev)
    return env.state_obs().tolist()


def _goal_obs(env: PointMass) -> list:
    """The observation the agent would see at rest on the goal.

    Image mode renders the goal state and stacks the frame with itself
    (exact for a static goal); vector mode is the goal position with zero
    velocity. This is the ``goal_obs`` the planning tools expect.
    """
    g = env.goal
    if _get_model().cfg.obs_mode == "image":
        goal_state = np.array([g[0], g[1], 0.0, 0.0], dtype=np.float64)
        frame = env.render(state=goal_state)
        return _image_obs(frame, frame)
    return [float(g[0]), float(g[1]), 0.0, 0.0, float(g[0]), float(g[1])]


@mcp.tool()
def encode_observation(observation: list) -> dict:
    """Encode an observation into a latent vector with the trained JEPA encoder.

    The observation format follows the loaded model (see model_info): a 6D
    vector [x, y, vx, vy, goal_x, goal_y] for the vector model, or a stacked
    rendered-frame array (channels x 32 x 32 nested lists, as returned by
    reset_env / step_env) for the image model. The latent is the space the
    world model predicts in; downstream tools operate on these latents rather
    than on raw observations.
    """
    model = _get_model()
    import torch

    with torch.no_grad():
        x = torch.tensor([observation], dtype=torch.float32)
        z = model.encode(x).squeeze(0).tolist()
    return {"latent": z, "latent_dim": len(z)}


@mcp.tool()
def predict_rollout(latent: list[float], actions: list[list[float]]) -> dict:
    """Roll the learned latent dynamics forward from a latent over an action sequence.

    Given a starting latent and a list of 2D acceleration actions, applies the
    predictor g step by step and returns the latent trajectory. This is the
    model imagining a future without touching the real environment.
    """
    model = _get_model()
    import torch

    with torch.no_grad():
        z = torch.tensor(latent, dtype=torch.float32)
        traj = []
        for a in actions:
            z = model.predict(z.unsqueeze(0), torch.tensor([a], dtype=torch.float32)).squeeze(0)
            traj.append(z.tolist())
    return {"latent_trajectory": traj, "n_steps": len(traj)}


@mcp.tool()
def plan_to_goal(current_obs: list, goal_obs: list) -> dict:
    """Plan the next action that drives the agent toward a goal, via latent MPC.

    Encodes the current and goal observations (in the loaded model's
    observation format; reset_env returns both, including a ready-made
    goal_observation), searches action sequences by rolling the world model
    forward in latent space (cross-entropy method), and returns the first
    action of the best plan. Call repeatedly (replanning each step) to follow
    a receding-horizon trajectory to the goal.
    """
    planner = _get_planner()
    action = planner.plan(np.asarray(current_obs, dtype=np.float32),
                          np.asarray(goal_obs, dtype=np.float32))
    return {"action": [float(a) for a in action]}


@mcp.tool()
def plan_trajectory(current_obs: list, goal_obs: list) -> dict:
    """Plan a full action sequence toward a goal and return the imagined plan.

    Unlike plan_to_goal (which returns only the next action for
    receding-horizon control), this returns the entire best action sequence
    the latent MPC search found, plus the imagined latent trajectory those
    actions produce under the world model. Useful when an agent wants to
    inspect or execute the model's whole plan open-loop.
    """
    planner = _get_planner()
    seq = planner.plan_sequence(
        np.asarray(current_obs, dtype=np.float32),
        np.asarray(goal_obs, dtype=np.float32),
    )
    model = _get_model()
    import torch

    with torch.no_grad():
        z = model.encode(
            torch.tensor(np.asarray([current_obs], dtype=np.float32))
        ).squeeze(0)
        latents = []
        for a in seq:
            z = model.predict(
                z.unsqueeze(0), torch.tensor(np.asarray([a], dtype=np.float32))
            ).squeeze(0)
            latents.append(z.tolist())
    return {
        "actions": [[float(x) for x in a] for a in seq],
        "imagined_latents": latents,
        "horizon": len(seq),
    }


@mcp.tool()
def model_info() -> dict:
    """Describe the loaded world model: observation mode, dims, and metrics.

    Surfaces the checkpoint's training metrics (including the honesty-gate
    numbers: predictor-vs-identity MSE ratio and latent std) so an agent can
    inspect what it is planning with.
    """
    model = _get_model()
    import torch

    ckpt = torch.load(_checkpoint_path(), map_location="cpu", weights_only=False)
    return {
        "obs_mode": model.cfg.obs_mode,
        "latent_dim": model.cfg.latent_dim,
        "action_dim": model.cfg.action_dim,
        "checkpoint_path": _checkpoint_path(),
        "metrics": {k: float(v) for k, v in ckpt.get("metrics", {}).items()},
        "env_config": ckpt.get("env_config"),
    }


@mcp.tool()
def reset_env(seed: int = 0, session: str = "default") -> dict:
    """Reset the point-mass environment to a deterministic start for the given seed.

    Returns the initial observation (in the loaded model's observation format),
    a ready-made goal_observation for the planning tools, the agent state, and
    the goal position. The environment uses the configuration the loaded
    checkpoint was trained on. Pass a distinct ``session`` id to hold an
    independent environment -- concurrent agent sessions do not share episode
    state.
    """
    env = _get_env(session)
    env.reset(seed=seed)
    return {
        "observation": _current_obs(env),
        "goal_observation": _goal_obs(env),
        "state": env.state.tolist(),
        "goal": env.goal.tolist(),
        "session": session,
    }


@mcp.tool()
def step_env(action: list[float], session: str = "default") -> dict:
    """Apply a 2D acceleration action to the environment and advance one timestep.

    Returns the new observation (in the loaded model's observation format),
    reward (negative distance to goal), a done flag (agent reached the goal),
    and the distance to the goal. ``session`` selects which independent
    environment to step (see reset_env).
    """
    env = _get_env(session)
    frame_prev = env.render() if _get_model().cfg.obs_mode == "image" else None
    state, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    return {
        "observation": _current_obs(env, frame_prev=frame_prev),
        "state": state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "dist": float(info["dist"]),
        "session": session,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Oneiros MCP server: a learned world model as agentic tools."
    )
    p.add_argument(
        "--obs-mode",
        choices=["vector", "image"],
        default="vector",
        help="which committed world model to serve "
        "(vector -> checkpoint.pt, image -> checkpoint_image.pt)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="explicit checkpoint path (overrides the --obs-mode default)",
    )
    args = p.parse_args()
    configure(obs_mode=args.obs_mode, checkpoint=args.checkpoint)
    mcp.run()


if __name__ == "__main__":
    main()
