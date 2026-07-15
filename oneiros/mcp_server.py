"""MCP server exposing the Oneiros world model as agentic tools.

This is the differentiating piece of the project: instead of an LLM calling
hand-written functions, an agent plans by calling a *learned predictive world
model* as a tool over the Model Context Protocol. The tools let a client encode
observations into latents, roll the latent dynamics forward, plan a goal-reaching
action with latent-space MPC, and drive the underlying environment.

Run as a stdio server::

    python -m oneiros.mcp_server

See the README for a ``claude_desktop_config.json`` snippet.
"""

from __future__ import annotations

import numpy as np
from mcp.server.fastmcp import FastMCP

from .env import PointMass
from .planner import LatentMPCPlanner, PlannerConfig
from .train import CHECKPOINT_PATH, load_model

mcp = FastMCP("oneiros")

# Lazily constructed singletons so importing this module is cheap and does not
# require a checkpoint to exist (the tool-list dump in CI must import cleanly).
# Environments are keyed by session id so independent agent sessions cannot
# clobber each other's episode state through the shared server process.
_STATE: dict[str, object] = {"model": None, "planner": None}
_ENVS: dict[str, PointMass] = {}


def _get_model():
    if _STATE["model"] is None:
        _STATE["model"] = load_model(CHECKPOINT_PATH)
    return _STATE["model"]


def _get_planner() -> LatentMPCPlanner:
    if _STATE["planner"] is None:
        _STATE["planner"] = LatentMPCPlanner(_get_model(), PlannerConfig())
    return _STATE["planner"]  # type: ignore[return-value]


def _get_env(session: str = "default") -> PointMass:
    if session not in _ENVS:
        env = PointMass(seed=0)
        env.reset(seed=0)
        _ENVS[session] = env
    return _ENVS[session]


@mcp.tool()
def encode_observation(observation: list[float]) -> dict:
    """Encode a 6D observation [x, y, vx, vy, goal_x, goal_y] into a latent vector.

    Returns the latent embedding produced by the trained JEPA encoder. The
    latent is the space the world model predicts in; downstream tools operate on
    these latents rather than on raw observations.
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
def plan_to_goal(current_obs: list[float], goal_obs: list[float]) -> dict:
    """Plan the next action that drives the agent toward a goal, via latent MPC.

    Encodes the current and goal observations, searches action sequences by
    rolling the world model forward in latent space (cross-entropy method), and
    returns the first action of the best plan. Call repeatedly (replanning each
    step) to follow a receding-horizon trajectory to the goal.
    """
    planner = _get_planner()
    action = planner.plan(np.asarray(current_obs, dtype=np.float32),
                          np.asarray(goal_obs, dtype=np.float32))
    return {"action": [float(a) for a in action]}


@mcp.tool()
def plan_trajectory(current_obs: list[float], goal_obs: list[float]) -> dict:
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
        z = model.encode(torch.tensor([current_obs], dtype=torch.float32)).squeeze(0)
        latents = []
        for a in seq:
            z = model.predict(
                z.unsqueeze(0), torch.tensor([a], dtype=torch.float32)
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

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return {
        "obs_mode": model.cfg.obs_mode,
        "latent_dim": model.cfg.latent_dim,
        "action_dim": model.cfg.action_dim,
        "metrics": {k: float(v) for k, v in ckpt.get("metrics", {}).items()},
        "env_config": ckpt.get("env_config"),
    }


@mcp.tool()
def reset_env(seed: int = 0, session: str = "default") -> dict:
    """Reset the point-mass environment to a deterministic start for the given seed.

    Returns the initial observation, the agent state, and the goal position.
    Pass a distinct ``session`` id to hold an independent environment --
    concurrent agent sessions do not share episode state.
    """
    env = _get_env(session)
    env.reset(seed=seed)
    return {
        "observation": env.state_obs().tolist(),
        "state": env.state.tolist(),
        "goal": env.goal.tolist(),
        "session": session,
    }


@mcp.tool()
def step_env(action: list[float], session: str = "default") -> dict:
    """Apply a 2D acceleration action to the environment and advance one timestep.

    Returns the new observation, reward (negative distance to goal), a done flag
    (agent reached the goal), and the distance to the goal. ``session`` selects
    which independent environment to step (see reset_env).
    """
    env = _get_env(session)
    state, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    return {
        "observation": env.state_obs().tolist(),
        "state": state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "dist": float(info["dist"]),
        "session": session,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
