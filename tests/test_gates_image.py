"""The pixel gates: honesty gates for the image-based world model.

The README documented image-based perception as future work because on
the default environment consecutive frames are nearly identical and a
single frame hides velocity entirely (second-order dynamics). These
gates certify the two fixes that turned it into a result:

* the **swift environment** (``PointMassConfig.swift()``) gives frames
  real per-step motion and non-linear (speed-proportional) drag;
* **two-frame stacking** makes velocity observable from pixels, and a
  **delta-variance penalty** forbids the temporal-smoothing collapse a
  convolutional encoder otherwise exploits (mapping consecutive frames
  to nearly identical latents so "predict no change" looks great).

Gates (all against the committed ``checkpoint_image.pt``, heldout data):

  (d) the image predictor's next-latent MSE beats the identity baseline
      by a clear margin, from pixels alone;
  (e) the image latent is not collapsed -- neither across the dataset
      nor temporally (consecutive-frame deltas stay expressive);
  (f) latent MPC from pixels reaches the goal reliably and at least
      twice as fast (median steps) as random actions;
  (g) compounded multi-step imagination stays useful: clearly better
      than identity at the planner's short horizons and still better at
      the full horizon 12.

Honest negative, measured and deliberately NOT gated: a privileged
linear-dynamics MPC reading the true 4D state still reaches the goal
about twice as fast as the pixel planner (median ~5 vs ~9.5 steps).
Pixels have not caught privileged state; the gates only claim what the
numbers support.

Bars are set with margin below the measured values (e.g. measured
heldout mse_ratio 0.11 gated at < 0.30) so platform-level float noise
cannot flip them while genuine regressions still fail.
"""

import os

import numpy as np
import pytest

from oneiros import baselines as B
from oneiros.data import collect_rollouts
from oneiros.env import PointMassConfig
from oneiros.train import IMAGE_CHECKPOINT_PATH, evaluate, load_model, train

CKPT = IMAGE_CHECKPOINT_PATH


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(CKPT):
        # Fallback for a fresh checkout without the committed checkpoint.
        # Mirrors how the checkpoint was produced; ~3 min on CPU.
        train(
            obs_mode="image",
            steps=6000,
            env_config=PointMassConfig.swift(),
            frame_stack=2,
            delta_var_weight=25.0,
            checkpoint_path=CKPT,
            verbose=False,
        )
    return load_model(CKPT)


@pytest.fixture(scope="module")
def heldout(model):
    data = collect_rollouts(
        n_episodes=60,
        horizon=50,
        seed=999,
        env_config=PointMassConfig.swift(),
        frame_stack=2,
    )
    return evaluate(model, data, "image")


def test_gate_d_image_predictor_beats_identity(heldout):
    # measured 0.11 on heldout; same < 0.5 bar as the vector gate, and a
    # tighter 0.30 to hold the demonstrated margin.
    assert heldout["mse_ratio"] < 0.30, heldout
    assert heldout["pred_mse"] < 0.5 * heldout["identity_mse"], heldout


def test_gate_e_image_latent_not_collapsed(heldout):
    # dataset-wide spread (measured ~1.1 mean / ~1.1 min)...
    assert heldout["latent_mean_std"] > 0.5, heldout
    assert heldout["latent_min_std"] > 0.5, heldout
    # ...and temporal expressiveness: identity MSE is the mean squared
    # one-step latent delta, so a temporally smoothed encoder would show
    # a tiny value here (0.012 before the delta-variance fix; ~1.08 now).
    assert heldout["identity_mse"] > 0.3, heldout


def test_gate_f_image_mpc_reaches_goal_faster_than_random(model):
    k = 20
    max_steps = 60
    fail = max_steps + 1

    def steps_to_goal(policy: str) -> list[int]:
        out = []
        for s in range(k):
            reached, traj = B.run_image_episode(model, s, policy)
            out.append(len(traj) - 1 if reached else fail)
        return out

    mpc = steps_to_goal("mpc")
    rnd = steps_to_goal("random")
    mpc_rate = sum(s <= max_steps for s in mpc) / k
    # measured: 20/20 reached, median 9.5 vs random 27.5.
    assert mpc_rate >= 0.9, (mpc_rate, mpc)
    assert np.median(mpc) * 2 <= np.median(rnd), (np.median(mpc), np.median(rnd))


def test_gate_g_multistep_imagination_stays_useful(model):
    # measured: H=4 ratio 0.45, H=12 ratio 0.86. Imagination must stay
    # clearly better than identity at short planning horizons and must
    # not become WORSE than identity even at the full horizon.
    short = B.multistep_rollout_errors(model, n_episodes=40, horizon=4, seed=555)
    full = B.multistep_rollout_errors(model, n_episodes=40, horizon=12, seed=555)
    assert short["rollout_ratio"] < 0.65, short
    assert full["rollout_ratio"] < 1.0, full


def test_image_checkpoint_is_small_enough_to_commit():
    # same 2 MB budget as the vector checkpoint (measured ~735 KB).
    assert os.path.getsize(CKPT) < 2 * 1024 * 1024
