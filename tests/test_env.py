"""Environment determinism and basic-physics tests."""

import numpy as np

from oneiros.env import PointMass, PointMassConfig


def test_reset_deterministic_given_seed():
    a = PointMass(seed=7)
    b = PointMass(seed=7)
    sa = a.reset(seed=7)
    sb = b.reset(seed=7)
    assert np.allclose(sa, sb)
    assert np.allclose(a.goal, b.goal)


def test_step_deterministic_given_seed():
    a = PointMass(seed=3)
    b = PointMass(seed=3)
    a.reset(seed=3)
    b.reset(seed=3)
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(20, 2))
    for act in actions:
        sa, _, _, _ = a.step(act)
        sb, _, _, _ = b.step(act)
        assert np.allclose(sa, sb)


def test_agent_stays_in_box():
    env = PointMass(seed=1)
    env.reset(seed=1)
    c = env.config
    rng = np.random.default_rng(2)
    for _ in range(200):
        env.step(rng.uniform(-c.max_accel, c.max_accel, size=2))
        x, y = env.state[0], env.state[1]
        assert c.radius - 1e-6 <= x <= c.box_size - c.radius + 1e-6
        assert c.radius - 1e-6 <= y <= c.box_size - c.radius + 1e-6


def test_render_shapes():
    env = PointMass(config=PointMassConfig(img_size=32), seed=0)
    env.reset(seed=0)
    img = env.render(with_goal=True)
    assert img.shape == (2, 32, 32)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    img1 = env.render(with_goal=False)
    assert img1.shape == (1, 32, 32)


def test_state_obs_shape():
    env = PointMass(seed=0)
    env.reset(seed=0)
    obs = env.state_obs()
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
