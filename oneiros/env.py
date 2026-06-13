"""A 2D point-mass environment in a walled box.

State is (x, y, vx, vy). The action is a 2D acceleration vector, clipped to a
maximum magnitude per axis. A semi-implicit Euler integrator advances the
dynamics; walls reflect velocity with a restitution coefficient and friction
damps velocity each step. The environment is fully deterministic given a seed.

`render()` returns a small grayscale image (default 32x32) with the agent drawn
as a soft blob. An optional second channel marks the goal location, which lets
the world model perceive the goal directly from observations when desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointMassConfig:
    """Static configuration for the point-mass environment."""

    box_size: float = 1.0  # world spans [0, box_size] on each axis
    dt: float = 0.1
    max_accel: float = 1.0  # per-axis acceleration clip
    max_speed: float = 2.0  # per-axis speed clip (keeps dynamics bounded)
    friction: float = 0.10  # fraction of velocity removed per step
    restitution: float = 0.8  # wall bounce energy retention
    img_size: int = 32
    blob_radius: float = 0.09  # radius (world units) of the rendered agent blob
    goal_radius: float = 0.07  # radius (world units) of the rendered goal marker
    radius: float = 0.04  # collision radius of the point mass


@dataclass
class PointMass:
    """Deterministic 2D point-mass simulator with walls and friction."""

    config: PointMassConfig = field(default_factory=PointMassConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.state = np.zeros(4, dtype=np.float64)
        self.goal = np.array([0.8, 0.8], dtype=np.float64)

    # -- core API -----------------------------------------------------------
    def reset(
        self,
        state: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset the environment and return the initial state.

        If ``state``/``goal`` are omitted they are sampled deterministically
        from the (optionally re-seeded) internal RNG. Positions are sampled away
        from the walls; initial velocity is zero.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.config
        margin = c.radius + 0.05
        if state is None:
            pos = self._rng.uniform(margin, c.box_size - margin, size=2)
            self.state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        else:
            self.state = np.asarray(state, dtype=np.float64).copy()
        if goal is None:
            self.goal = self._rng.uniform(margin, c.box_size - margin, size=2)
        else:
            self.goal = np.asarray(goal, dtype=np.float64).copy()
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one timestep.

        Returns ``(state, reward, done, info)``. The reward is the negative
        distance to the goal (purely informational; the planner does not depend
        on it). ``done`` is True when the agent is within one collision radius of
        the goal.
        """
        c = self.config
        a = np.clip(np.asarray(action, dtype=np.float64), -c.max_accel, c.max_accel)

        x, y, vx, vy = self.state
        # semi-implicit Euler: integrate velocity first, then position.
        vx = (vx + a[0] * c.dt) * (1.0 - c.friction)
        vy = (vy + a[1] * c.dt) * (1.0 - c.friction)
        vx = float(np.clip(vx, -c.max_speed, c.max_speed))
        vy = float(np.clip(vy, -c.max_speed, c.max_speed))

        x = x + vx * c.dt
        y = y + vy * c.dt

        # wall collisions: reflect position and velocity about the boundary.
        lo, hi = c.radius, c.box_size - c.radius
        x, vx = self._bounce(x, vx, lo, hi, c.restitution)
        y, vy = self._bounce(y, vy, lo, hi, c.restitution)

        self.state = np.array([x, y, vx, vy], dtype=np.float64)

        dist = float(np.linalg.norm(self.state[:2] - self.goal))
        reward = -dist
        done = dist <= c.radius
        return self.state.copy(), reward, done, {"dist": dist}

    @staticmethod
    def _bounce(
        p: float, v: float, lo: float, hi: float, restitution: float
    ) -> tuple[float, float]:
        if p < lo:
            p = lo + (lo - p)
            v = -v * restitution
        elif p > hi:
            p = hi - (p - hi)
            v = -v * restitution
        # guard against a single overshoot landing outside after reflection.
        p = float(np.clip(p, lo, hi))
        return p, v

    # -- rendering ----------------------------------------------------------
    def render(
        self, state: np.ndarray | None = None, with_goal: bool = True
    ) -> np.ndarray:
        """Render an observation image.

        Returns a float32 array in ``[0, 1]``. When ``with_goal`` is True the
        shape is ``(2, H, W)`` (channel 0 = agent, channel 1 = goal marker);
        otherwise ``(1, H, W)``. This is the image observation consumed by the
        image-based encoder.
        """
        s = self.state if state is None else np.asarray(state, dtype=np.float64)
        c = self.config
        agent = self._blob(s[0], s[1], c.blob_radius)
        if not with_goal:
            return agent[None, :, :]
        goal = self._blob(self.goal[0], self.goal[1], c.goal_radius)
        return np.stack([agent, goal], axis=0)

    def _blob(self, wx: float, wy: float, radius: float) -> np.ndarray:
        c = self.config
        n = c.img_size
        # pixel centers mapped to world coordinates in [0, box_size].
        coords = (np.arange(n) + 0.5) / n * c.box_size
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        # row 0 is the top of the image; flip y so larger y renders higher up.
        gy = c.box_size - gy
        d2 = (gx - wx) ** 2 + (gy - wy) ** 2
        img = np.exp(-d2 / (2.0 * (radius / 2.0) ** 2))
        return img.astype(np.float32)

    # -- convenience --------------------------------------------------------
    def state_obs(self, state: np.ndarray | None = None) -> np.ndarray:
        """Return the 4D state vector concatenated with the 2D goal.

        This is the *vector* observation used by the fallback (state-based)
        encoder. It is deterministic and contains the same information the image
        observation encodes, in a form that trains reliably.
        """
        s = self.state if state is None else np.asarray(state, dtype=np.float64)
        return np.concatenate([s, self.goal]).astype(np.float32)
