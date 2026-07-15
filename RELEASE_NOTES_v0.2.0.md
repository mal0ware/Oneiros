# Oneiros v0.2.0

v0.1.0 shipped with one honest limitation: the image encoder could not beat
the identity baseline, so image-based perception was documented as future
work rather than claimed. v0.2.0 closes that gap as a demonstrated,
gate-enforced result, and hardens the MCP surface.

## Image-based perception: from documented limitation to result

The failure had two diagnosed causes and two principled fixes:

1. Consecutive frames were nearly identical (blob moves ~a pixel per step),
   so "predict no change" was near-optimal. Fix: the **swift environment
   preset** (`PointMassConfig.swift()`) -- larger per-step motion plus
   speed-proportional drag, making the dynamics genuinely non-linear.
2. A single frame hides velocity (second-order dynamics), and the CNN
   exploited it by **temporal smoothing** -- mapping consecutive frames to
   nearly identical latents (more training made the ratio worse, 0.89 to
   0.94). Fixes: **two-frame stacking** (velocity observable from pixels)
   and a **delta-variance penalty** hinging the std of `z_{t+1} - z_t`
   (temporal anti-collapse; the dataset-wide variance penalty cannot forbid
   this failure mode).

## The four pixel gates (tests/test_gates_image.py, heldout data)

| Gate | Metric | Result |
|---|---|---|
| (d) Image predictor beats no-op | heldout MSE ratio vs identity | 0.11 (vector model: 0.17) |
| (e) Latent not collapsed, incl. temporally | per-dim std / one-step delta MSE | 1.10 / 1.08 (pre-fix delta: 0.012) |
| (f) Pixel MPC beats random | reach rate + median steps, 20 seeds | 20/20, median 9.5 vs 27.5 |
| (g) Imagination useful at horizon | compounded rollout MSE ratio | 0.45 at H=4, 0.86 at H=12 |

Honest negative, measured and deliberately not gated: a privileged
linear-dynamics MPC reading the true 4D state still reaches the goal about
2x faster than the pixel planner (median ~5 vs ~9.5 steps). Baselines in
`oneiros/baselines.py`.

## MCP surface

- `plan_trajectory`: returns the full best action sequence plus the imagined
  latent path (verified by open-loop replay in tests).
- `model_info`: obs mode, dims, and the checkpoint's honesty-gate metrics.
- `reset_env`/`step_env` take a `session` id; concurrent agent sessions no
  longer share one global environment (isolation is tested).

## Fixes

- Checkpoint paths resolve relative to the package, so the console scripts
  and MCP server work from any working directory; checkpoints ship as
  package data in wheels.
- README install instructions match pyproject (the `mcp[cli]` drift is gone).
- The gate fixture's fallback now mirrors the committed checkpoint's actual
  training recipe instead of a weaker one.
- CI caches pip and runs the new pixel gates from the committed checkpoint.

## Verification at release

- `pytest` -- 31 passed (18 in v0.1.0), including all seven honesty gates,
  against the committed checkpoints
- `ruff check oneiros tests` -- clean
- Python 3.12, CPU-only torch (2.13.0+cpu at verification time)
