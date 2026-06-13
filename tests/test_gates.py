"""The three honesty gates.

These tests assert the artifact genuinely demonstrates what it claims:

  (a) the trained predictor's next-latent MSE beats a no-op/identity baseline by
      a clear margin (it learned real latent dynamics, not nothing);
  (b) the latent representation is NOT collapsed (per-dim std well above 0.1);
  (c) the latent-space MPC planner reaches the goal at a success rate clearly
      above a random-action baseline over K seeds.

They run against the committed checkpoint so CI needs no training. If the
checkpoint is missing they train a short model first (skipped in CI by the
presence of the committed checkpoint).
"""

import os

import pytest

from oneiros.data import collect_rollouts
from oneiros.demo_agent import run_episode
from oneiros.train import CHECKPOINT_PATH, evaluate, load_model, train

CKPT = CHECKPOINT_PATH


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(CKPT):
        train(obs_mode="vector", steps=2000, verbose=False)
    return load_model(CKPT)


def test_gate_a_predictor_beats_identity_baseline(model):
    data = collect_rollouts(n_episodes=60, horizon=50, seed=999)
    m = evaluate(model, data, "vector")
    # learned predictor MSE must be clearly below the identity (no-op) baseline.
    assert m["pred_mse"] < 0.5 * m["identity_mse"], m
    assert m["mse_ratio"] < 0.5, m


def test_gate_b_latent_not_collapsed(model):
    data = collect_rollouts(n_episodes=60, horizon=50, seed=888)
    m = evaluate(model, data, "vector")
    # mean and minimum per-dim std must be safely above the collapse threshold.
    assert m["latent_mean_std"] > 0.1, m
    assert m["latent_min_std"] > 0.1, m


def test_gate_c_mpc_beats_random(model):
    k = 20
    mpc = sum(run_episode(model, s, "mpc")[0] for s in range(k))
    rnd = sum(run_episode(model, s, "random")[0] for s in range(k))
    mpc_rate, rnd_rate = mpc / k, rnd / k
    # MPC must reach the goal far more often than random actions.
    assert mpc_rate >= 0.7, (mpc_rate, rnd_rate)
    assert mpc_rate >= rnd_rate + 0.4, (mpc_rate, rnd_rate)


def test_checkpoint_is_small_enough_to_commit():
    # committed checkpoint must stay under the 2 MB budget from the spec.
    assert os.path.getsize(CKPT) < 2 * 1024 * 1024
