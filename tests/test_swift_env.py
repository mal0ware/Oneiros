"""Swift environment preset and frame-stacked data collection."""

import numpy as np

from oneiros.data import collect_rollouts
from oneiros.env import PointMass, PointMassConfig


def test_default_config_dynamics_unchanged_by_quad_drag_field():
    """quad_drag defaults to 0 and must reproduce the original dynamics
    exactly -- the committed vector checkpoint depends on it."""
    env = PointMass(seed=0)
    env.reset(seed=0)
    c = env.config
    s = env.state.copy()
    a = np.array([0.7, -0.3])
    env.step(a)
    # original update: v' = (v + a*dt) * (1 - friction), then position.
    vx = (s[2] + a[0] * c.dt) * (1.0 - c.friction)
    vy = (s[3] + a[1] * c.dt) * (1.0 - c.friction)
    assert np.isclose(env.state[2], vx)
    assert np.isclose(env.state[3], vy)


def test_swift_env_is_deterministic():
    e1 = PointMass(config=PointMassConfig.swift(), seed=7)
    e2 = PointMass(config=PointMassConfig.swift(), seed=7)
    e1.reset(seed=7)
    e2.reset(seed=7)
    for _ in range(20):
        a = np.array([0.9, 0.4])
        s1, *_ = e1.step(a)
        s2, *_ = e2.step(a)
    assert np.allclose(s1, s2)


def test_swift_env_drag_is_nonlinear_in_speed():
    """The effective damping factor must depend on speed, so a linear
    model in (state, action) cannot represent the dynamics exactly."""
    cfg = PointMassConfig.swift()
    slow = PointMass(config=cfg, seed=0)
    fast = PointMass(config=cfg, seed=0)
    slow.reset(state=np.array([0.5, 0.5, 0.1, 0.0]), goal=np.array([0.9, 0.9]))
    fast.reset(state=np.array([0.5, 0.5, 1.5, 0.0]), goal=np.array([0.9, 0.9]))
    a = np.zeros(2)
    s_slow, *_ = slow.step(a)
    s_fast, *_ = fast.step(a)
    damp_slow = s_slow[2] / 0.1
    damp_fast = s_fast[2] / 1.5
    assert damp_fast < damp_slow  # faster -> more heavily damped


def test_swift_env_moves_visibly_between_frames():
    """The swift preset exists to give image dynamics margin: a moving
    agent must displace by multiple pixels per step at cruise speed."""
    cfg = PointMassConfig.swift()
    env = PointMass(config=cfg, seed=0)
    env.reset(state=np.array([0.3, 0.5, 1.0, 0.0]), goal=np.array([0.9, 0.9]))
    x0 = env.state[0]
    env.step(np.zeros(2))
    pixels_moved = abs(env.state[0] - x0) / cfg.box_size * cfg.img_size
    assert pixels_moved >= 3.0, pixels_moved


def test_frame_stack_doubles_channels_and_encodes_motion():
    data = collect_rollouts(
        n_episodes=2, horizon=5, seed=3, env_config=PointMassConfig.swift(), frame_stack=2
    )
    assert data["obs_img_t"].shape[1] == 4  # 2 frames x 2 channels
    # the two stacked frames must differ once the agent is moving
    # (that difference is exactly the velocity signal).
    later = data["obs_img_t"][4]
    assert np.abs(later[0] - later[2]).max() > 0.05


def test_frame_stack_1_is_backward_compatible():
    d1 = collect_rollouts(n_episodes=1, horizon=3, seed=5)
    d2 = collect_rollouts(n_episodes=1, horizon=3, seed=5, frame_stack=1)
    assert d1["obs_img_t"].shape == d2["obs_img_t"].shape == (3, 2, 32, 32)
    assert np.allclose(d1["obs_img_t"], d2["obs_img_t"])
