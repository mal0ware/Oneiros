# Oneiros

[![CI](https://github.com/mal0ware/Oneiros/actions/workflows/ci.yml/badge.svg)](https://github.com/mal0ware/Oneiros/actions/workflows/ci.yml)

**Status:** complete research artifact — results verified. v0.2 adds **image-based perception as a demonstrated result** (the four pixel gates below), closing v0.1's documented next step; the V-JEPA 2 scale-up remains future work (needs GPU + pretrained weights).

**A JEPA-style latent world model that an agent calls as a tool — over MCP — to plan.**

Oneiros is a small, CPU-only research artifact at the intersection of *agentic
systems* and *world models*. It trains a Joint Embedding Predictive Architecture
(JEPA) world model on a 2D point-mass environment, then exposes that model as a
set of [Model Context Protocol](https://modelcontextprotocol.io) tools. An agent
plans by **calling the learned predictive world model as a tool** — encoding
observations into latents, rolling latent dynamics forward, and running
model-predictive control entirely in latent space — rather than the usual
pattern of an LLM calling hand-written functions.

## Why predict in latent space

A JEPA predicts the future in a **learned latent space**, not in pixels or
tokens. Given an observation `o_t` and an action `a_t`, it learns an encoder
`f` and a predictor `g` such that `g(f(o_t), a_t)` matches `f(o_{t+1})` — there
is no decoder and no pixel-reconstruction loss. This matters because
reconstruction wastes capacity modeling perceptually salient but
control-irrelevant detail (texture, lighting, background), while a latent
predictor is free to discard everything that does not help it anticipate the
future. The cost is a well-known failure mode: latent prediction can collapse to
a constant (every observation maps to the same point, making prediction
trivially perfect). Oneiros defeats collapse with an EMA target encoder,
stop-gradients, and a VICReg-style variance + covariance penalty, and then uses
the resulting latent dynamics for planning.

## Architecture

```mermaid
flowchart LR
    subgraph Env["Point-mass environment (numpy)"]
        O["obs o_t"]
        ON["obs o_t+1"]
    end

    subgraph WM["JEPA world model (torch, CPU)"]
        F["encoder f"]
        FT["EMA target f_target<br/>(stop-grad)"]
        G["predictor g"]
        O --> F --> Z["latent z_t"]
        Z --> G
        A["action a_t"] --> G
        G --> ZH["z_hat_t+1"]
        ON --> FT --> ZT["z_t+1 (target)"]
        ZH -. "MSE + VICReg<br/>variance/covariance" .-> ZT
    end

    subgraph MCP["MCP server (agentic interface)"]
        T1["encode_observation"]
        T2["predict_rollout"]
        T3["plan_to_goal"]
        T4["reset_env / step_env"]
    end

    subgraph Agent["Agent loop"]
        P["latent MPC planner<br/>(CEM over g)"]
    end

    WM --> MCP
    MCP <--> Agent
    P -->|"first action"| Env
```

The agent never sees the environment's dynamics. It calls `plan_to_goal`, which
encodes the current and goal observations, searches action sequences by rolling
the predictor `g` forward `H` steps in latent space (cross-entropy method),
scores each candidate by predicted-latent distance to the goal, and returns the
first action. The planner replans every step (receding-horizon MPC).

## Verified results

Numbers below are from an actual run on this machine (CPU only, seed 0). Train
with `python -m oneiros.train` and reproduce the diagnostics with
`python -m oneiros.demo_agent`.

| Honesty gate | Metric | Result |
|---|---|---|
| **(a) Predictor beats no-op baseline** | next-latent MSE vs identity baseline | **0.0185 vs 0.1076** (ratio 0.17 — ~5.8x better) |
| **(b) Latent not collapsed** | per-dim latent std (mean / min) | **1.04 / 1.00** (threshold 0.1) |
| **(c) MPC beats random** | goal-reaching success over 20 seeds | **MPC 95% vs random 15-20%** |

Training takes about 21 seconds for 4000 steps. The checkpoint
(`oneiros/checkpoint.pt`, ~240 KB) is committed so the demo, MCP server, and
planning tests run without retraining.

### The pixel gates (v0.2): image-based perception, demonstrated

v0.1 documented why the image encoder could not beat the identity baseline.
The diagnosis had two parts, and each got a principled fix rather than a
knob-twiddle:

1. **Consecutive frames were nearly identical** (the blob moves ~a pixel per
   step), so "predict no change" was already an excellent predictor. Fix: the
   **swift environment preset** (`PointMassConfig.swift()`) — larger `dt` and
   acceleration so the agent moves several pixels per frame, plus
   **speed-proportional drag** so the dynamics are genuinely non-linear.
2. **A single frame hides velocity** — the dynamics are second-order, so no
   single-frame predictor can recover the next state, and the convolutional
   encoder exploited this by *temporal smoothing* (mapping consecutive frames
   to nearly identical latents; the dataset-wide variance penalty does not
   forbid it — more training made it worse, 0.89 → 0.94 MSE ratio). Fixes:
   **two-frame stacking** (velocity becomes observable from pixels, the same
   reason pixel world models from DQN to V-JEPA consume clips, not stills)
   and a **delta-variance penalty** (VICReg-style hinge on the std of
   `z_{t+1} - z_t`) that forbids the temporal collapse outright.

Results from the committed `checkpoint_image.pt` (~735 KB), heldout data,
enforced as tests in `tests/test_gates_image.py`:

| Pixel gate | Metric | Result |
|---|---|---|
| **(d) Image predictor beats no-op** | heldout next-latent MSE ratio vs identity | **0.11** (vector model: 0.17) |
| **(e) Latent not collapsed, incl. temporally** | per-dim std / one-step delta MSE | **1.10 / 1.08** (pre-fix delta was 0.012) |
| **(f) Pixel MPC beats random** | goal-reach rate + median steps over 20 seeds | **20/20, median 9.5 steps vs 27.5 random** |
| **(g) Imagination useful at horizon** | compounded H-step rollout MSE ratio | **0.45 at H=4, 0.86 at H=12** |

One honest negative, measured and deliberately not gated: a **privileged
linear-dynamics MPC** reading the true 4D state still reaches the goal about
twice as fast as the pixel planner (median ~5 vs ~9.5 steps). Planning from
pixels has not caught planning from privileged state, and gate (g) shows
open-loop imagination degrading by horizon 12 — which is exactly why the
planner replans every step. The baselines live in `oneiros/baselines.py`;
reproduce with the image-training command under [Run](#run).

![Latent-MPC plan to goal](assets/planned_trajectory.gif)

The agent drives the point-mass to the goal (green star) using only the
world-model planning interface.

| Latent prediction error | Planning success |
|---|---|
| ![latent prediction](assets/latent_prediction.png) | ![planning vs random](assets/planning_vs_random.png) |

## What this is — and isn't

This **is** a genuine, end-to-end demonstration that (1) a non-trivial latent
dynamics model can be learned without collapse and without reconstruction, and
(2) planning *in that latent space* solves a control task far better than chance,
all behind an agentic tool interface.

It **is not** at scale. The environments are toy 2D point-masses, the models
are tiny (a few hundred KB). The default committed model uses a
**vector-state observation** on the original environment; the committed
**image model** (v0.2, `checkpoint_image.pt`) perceives stacked 32x32 frames
on the swift environment and clears its own four gates above — v0.1's
"image-based perception is the documented next step" is now a demonstrated
result, with the diagnosis (frame similarity + hidden velocity + temporal
smoothing) and fixes documented rather than hand-waved. What remains future
work is the scale-up: a frozen pretrained perception encoder such as
[V-JEPA 2](https://ai.meta.com/vjepa/) with a learned latent dynamics head on
top, exactly the recipe this toy mirrors — that step needs a GPU and
pretrained weights, which this CPU-only artifact deliberately does not
assume.

## Install

Requires Python 3.12. A project-local virtual environment is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Unix: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

`torch` is installed from the CPU wheel index — no GPU is needed or used.

## Run

```bash
# Train the JEPA world model (writes oneiros/checkpoint.pt). ~21s on CPU.
python -m oneiros.train --obs-mode vector --steps 4000

# Train the image world model on the swift environment (~3 min on CPU).
# This is the exact recipe of the committed checkpoint: two-frame stacking,
# the delta-variance penalty, and an explicit --checkpoint so the vector
# model's default path is not clobbered.
python -m oneiros.train --obs-mode image --env swift --steps 6000 \
    --frame-stack 2 --delta-var-weight 25 --checkpoint oneiros/checkpoint_image.pt

# Run the scripted agent: drive to the goal via latent MPC, write the GIF +
# diagnostic plots to assets/.
python -m oneiros.demo_agent --seed 0 --k 20

# Tests, including the three honesty gates (uses the committed checkpoint).
pytest -q

# Lint.
ruff check oneiros tests
```

## MCP server

The world model is exposed over MCP as a stdio server:

```bash
# vector model (default)
python -m oneiros.mcp_server

# the committed image model: stacked-frame observations, swift environment
python -m oneiros.mcp_server --obs-mode image

# any other checkpoint
python -m oneiros.mcp_server --checkpoint /path/to/checkpoint.pt
```

`--obs-mode` selects which committed world model the server exposes. The
environment tools serve observations in the loaded model's format — 6D
vectors, or the stacked rendered frames the image model trains on — and
`reset_env` returns a ready-made `goal_observation` for the planning tools.
The served environment always uses the configuration the checkpoint was
trained on (the swift preset for the image model).

Tools:

| Tool | Purpose |
|---|---|
| `encode_observation` | observation -> latent `z` (the trained JEPA encoder) |
| `predict_rollout` | roll latent dynamics `g` forward over an action sequence |
| `plan_to_goal` | latent-space MPC; returns the next action toward a goal |
| `plan_trajectory` | full best action sequence + the imagined latent path |
| `model_info` | loaded model's obs mode, dims, and honesty-gate metrics |
| `reset_env` / `step_env` | drive the point-mass environment; a `session` id keeps concurrent agent sessions independent |

To register the server with Claude Desktop, add this to
`claude_desktop_config.json` (use absolute paths for your checkout):

```json
{
  "mcpServers": {
    "oneiros": {
      "command": "C:/path/to/Oneiros/.venv/Scripts/python.exe",
      "args": ["-m", "oneiros.mcp_server"],
      "cwd": "C:/path/to/Oneiros"
    }
  }
}
```

On Unix the `command` is `.venv/bin/python`. Append `"--obs-mode", "image"`
to `args` to serve the committed image world model instead of the vector one.

## Repository layout

```
oneiros/
  env.py          # deterministic 2D point-mass environment (numpy)
  model.py        # JEPA encoder + predictor + VICReg regularizers
  data.py         # random-policy rollout replay buffer
  train.py        # JEPA training loop, evaluation, checkpoint I/O
  planner.py      # latent-space MPC (CEM / random shooting)
  baselines.py    # pixel episode runners + privileged linear-MPC baseline
  mcp_server.py   # MCP tools exposing the world model
  demo_agent.py   # scripted agent + diagnostics (GIF, plots)
  checkpoint.pt        # committed vector model (~240 KB)
  checkpoint_image.pt  # committed image model, swift env (~735 KB)
tests/            # determinism, shapes, and the three honesty gates
assets/           # generated GIF and diagnostic figures
```

See [SYNERGY.md](SYNERGY.md) for how the same latent-dynamics idea connects to
regime-aware modeling in time series.

## License

MIT — see [LICENSE](LICENSE).
