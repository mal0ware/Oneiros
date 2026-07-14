# Oneiros v0.1.0

First release of Oneiros: a JEPA-style latent world model, trained on a 2D
point-mass environment, exposed to agents as MCP planning tools
(`encode_observation`, `predict_rollout`, `plan_to_goal`, `reset_env`,
`step_env`). CPU-only; the trained checkpoint (`oneiros/checkpoint.pt`, ~240 KB)
is committed so the demo, MCP server, and tests run without retraining.

## The three verified honesty gates

Numbers from an actual run (CPU only, seed 0), enforced as tests in
`tests/test_gates.py` and re-verified against the committed checkpoint at
release time (`pytest`: 18 passed).

| Gate | Metric | Result |
|---|---|---|
| **(a) Predictor beats no-op baseline** | next-latent MSE vs identity baseline | **0.0185 vs 0.1076** (ratio 0.17 — ~5.8x better) |
| **(b) Latent not collapsed** | per-dim latent std (mean / min) | **1.04 / 1.00** (threshold 0.1) |
| **(c) MPC beats random** | goal-reaching success over 20 seeds | **MPC 95% vs random 15-20%** |

Gate (a) shows the predictor learned real latent dynamics rather than nothing.
Gate (b) shows the EMA target encoder, stop-gradients, and VICReg-style
variance/covariance penalties prevented representation collapse. Gate (c) shows
planning in the learned latent space (CEM-based receding-horizon MPC) solves
the control task far better than chance.

## Scope and limitations

This is a toy-scale demonstration: the environment is a 2D point-mass, the
models are a few hundred KB, and the latent dynamics are nearly linear. The
default committed model uses vector-state observations. The image-based encoder
(`--obs-mode image`) trains without collapse but does not beat the identity
baseline on this environment (consecutive frames are nearly identical), so it
is documented as future work rather than claimed as a result.

## Verification at release

- `pytest` — 18 passed (includes the three honesty gates, run against the
  committed checkpoint)
- `ruff check oneiros tests` — all checks passed
- Python 3.12, CPU-only torch (2.13.0+cpu at verification time)
