"""Scripted agent demo: drive the point-mass to a goal using the world model.

The agent here is deliberately thin. It does not know the environment dynamics;
it only calls the planner/world-model interface (the same surface exposed over
MCP) to decide each action, then applies that action and observes the result.
This is the agent-calls-world-model-as-a-tool loop in miniature.

Outputs (written to ``assets/``):

* ``planned_trajectory.gif`` — the closed-loop MPC trajectory to the goal.
* ``latent_prediction.png`` — predicted-vs-true next-latent error vs the
  identity baseline.
* ``planning_vs_random.png`` — goal-reaching success rate, learned-model MPC
  vs random actions, over K seeds.

``matplotlib`` (plots) and ``imageio`` (gif assembly) are used purely for these
diagnostics; they are not part of the model, planner, or MCP interface.
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import PointMass  # noqa: E402
from .planner import LatentMPCPlanner, PlannerConfig  # noqa: E402
from .train import load_model  # noqa: E402

ASSETS = "assets"
REACH_TOL = 0.12  # distance (world units) counted as "reached"


def goal_observation(env: PointMass) -> np.ndarray:
    """Build the observation the agent would see standing on the goal."""
    g = env.goal
    goal_state = np.array([g[0], g[1], 0.0, 0.0], dtype=np.float64)
    return np.concatenate([goal_state, g]).astype(np.float32)


def run_episode(model, seed: int, policy: str, max_steps: int = 60):
    """Run one episode; return (reached, trajectory, goal)."""
    env = PointMass(seed=seed)
    env.reset(seed=seed)
    planner = LatentMPCPlanner(model, PlannerConfig(seed=seed))
    g_obs = goal_observation(env)
    rng = np.random.default_rng(seed * 1000 + 7)

    traj = [env.state[:2].copy()]
    reached = False
    for _ in range(max_steps):
        cur = env.state_obs()
        if policy == "mpc":
            action = planner.plan(cur, g_obs)
        else:
            action = rng.uniform(-1.0, 1.0, size=2)
        _, _, _, info = env.step(action)
        traj.append(env.state[:2].copy())
        if info["dist"] <= REACH_TOL:
            reached = True
            break
    return reached, np.array(traj), env.goal.copy()


def render_gif(model, seed: int, path: str) -> int:
    """Render the MPC trajectory to an animated GIF. Returns the file size."""
    env = PointMass(seed=seed)
    env.reset(seed=seed)
    planner = LatentMPCPlanner(model, PlannerConfig(seed=seed))
    g_obs = goal_observation(env)

    states = [env.state[:2].copy()]
    for _ in range(60):
        action = planner.plan(env.state_obs(), g_obs)
        _, _, _, info = env.step(action)
        states.append(env.state[:2].copy())
        if info["dist"] <= REACH_TOL:
            break

    frames = []
    box = env.config.box_size
    for i in range(1, len(states) + 1):
        fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=120)
        path_xy = np.array(states[:i])
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="#4c72b0", lw=1.8, alpha=0.9)
        ax.plot(path_xy[-1, 0], path_xy[-1, 1], "o", color="#dd8452", ms=11,
                label="agent")
        ax.plot(env.goal[0], env.goal[1], "*", color="#55a868", ms=18, label="goal")
        ax.add_patch(plt.Circle((env.goal[0], env.goal[1]), REACH_TOL,
                                color="#55a868", fill=False, ls="--", alpha=0.5))
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Latent-MPC plan to goal", fontsize=9)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

    # pad the final frame so the gif visibly "lands" on the goal.
    frames += [frames[-1]] * 6
    imageio.mimsave(path, frames, duration=0.12, loop=0)
    return os.path.getsize(path)


def plot_latent_prediction(model, path: str) -> dict:
    """Plot predicted-vs-true next-latent error vs the identity baseline."""
    import torch

    from .data import collect_rollouts

    data = collect_rollouts(n_episodes=40, horizon=50, seed=123)
    with torch.no_grad():
        z_t = model.encode(torch.from_numpy(data["obs_vec_t"]))
        z_next = model.encode_target(torch.from_numpy(data["obs_vec_next"]))
        z_hat = model.predict(z_t, torch.from_numpy(data["action"]))
    per_sample_pred = ((z_hat - z_next) ** 2).mean(dim=1).numpy()
    per_sample_iden = ((z_t - z_next) ** 2).mean(dim=1).numpy()

    fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=110)
    bins = np.linspace(0, np.percentile(per_sample_iden, 99), 40)
    ax.hist(per_sample_iden, bins=bins, alpha=0.6, label="identity baseline",
            color="#c44e52")
    ax.hist(per_sample_pred, bins=bins, alpha=0.7, label="learned predictor",
            color="#4c72b0")
    ax.axvline(per_sample_pred.mean(), color="#4c72b0", ls="--", lw=1.5)
    ax.axvline(per_sample_iden.mean(), color="#c44e52", ls="--", lw=1.5)
    ax.set_xlabel("next-latent MSE (per transition)")
    ax.set_ylabel("count")
    ax.set_title("Latent prediction error: learned vs identity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "pred_mse": float(per_sample_pred.mean()),
        "identity_mse": float(per_sample_iden.mean()),
    }


def plot_planning_vs_random(model, k: int, path: str) -> dict:
    """Bar chart of MPC vs random goal-reaching success rate over K seeds."""
    mpc = sum(run_episode(model, s, "mpc")[0] for s in range(k))
    rnd = sum(run_episode(model, s, "random")[0] for s in range(k))
    mpc_rate, rnd_rate = mpc / k, rnd / k

    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=110)
    bars = ax.bar(["learned-model\nMPC", "random\nactions"],
                  [mpc_rate, rnd_rate], color=["#4c72b0", "#c44e52"], width=0.6)
    for b, v in zip(bars, [mpc_rate, rnd_rate], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("goal-reaching success rate")
    ax.set_title(f"Planning success over {k} seeds")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"mpc_success": mpc_rate, "random_success": rnd_rate, "k": k}


def main() -> None:
    p = argparse.ArgumentParser(description="Oneiros demo agent.")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--checkpoint", default="oneiros/checkpoint.pt")
    args = p.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    model = load_model(args.checkpoint)

    print("Driving agent to goal via latent MPC (agent calls world model as a tool)...")
    reached, traj, goal = run_episode(model, args.seed, "mpc")
    print(f"  seed {args.seed}: reached={reached} in {len(traj)-1} steps; "
          f"final pos {traj[-1].round(3)}, goal {goal.round(3)}")

    gif_path = os.path.join(ASSETS, "planned_trajectory.gif")
    size = render_gif(model, args.seed, gif_path)
    print(f"  wrote {gif_path} ({size/1024:.1f} KB)")

    lp = plot_latent_prediction(model, os.path.join(ASSETS, "latent_prediction.png"))
    print(f"  latent prediction: pred_mse={lp['pred_mse']:.4f} "
          f"identity_mse={lp['identity_mse']:.4f}")

    pr = plot_planning_vs_random(model, args.k,
                                 os.path.join(ASSETS, "planning_vs_random.png"))
    print(f"  planning: MPC={pr['mpc_success']:.0%} "
          f"random={pr['random_success']:.0%} over {pr['k']} seeds")


if __name__ == "__main__":
    main()
