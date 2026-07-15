"""MCP server smoke tests: the server imports and exposes the expected tools."""

import asyncio

EXPECTED_TOOLS = {
    "encode_observation",
    "predict_rollout",
    "plan_to_goal",
    "plan_trajectory",
    "model_info",
    "reset_env",
    "step_env",
}


def test_mcp_server_lists_expected_tools():
    from oneiros.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
    # every tool must carry a docstring-derived description.
    for t in tools:
        assert t.description and len(t.description) > 10


def test_mcp_tools_roundtrip():
    """Reset, encode, plan, and step through the in-process tool functions."""
    import oneiros.mcp_server as srv

    info = srv.reset_env(seed=0)
    assert len(info["observation"]) == 6
    enc = srv.encode_observation(info["observation"])
    assert enc["latent_dim"] >= 8
    roll = srv.predict_rollout(enc["latent"], [[0.5, 0.0], [0.5, 0.0]])
    assert roll["n_steps"] == 2
    goal_obs = info["observation"][:]
    goal_obs[0], goal_obs[1] = info["goal"][0], info["goal"][1]
    plan = srv.plan_to_goal(info["observation"], goal_obs)
    assert len(plan["action"]) == 2
    step = srv.step_env(plan["action"])
    assert "dist" in step

    meta = srv.model_info()
    assert meta["obs_mode"] == "vector"
    assert meta["latent_dim"] >= 8
    assert "mse_ratio" in meta["metrics"]


def test_mcp_sessions_are_isolated():
    """Two sessions hold independent environments: stepping one must not
    move the other (the old single-global-env server clobbered both)."""
    import oneiros.mcp_server as srv

    a = srv.reset_env(seed=1, session="a")
    b = srv.reset_env(seed=2, session="b")
    assert a["state"] != b["state"]

    before_b = srv.reset_env(seed=2, session="b")["state"]
    for _ in range(5):
        srv.step_env([1.0, 0.0], session="a")
    after_b = srv._get_env("b").state.tolist()
    assert after_b == before_b


def test_mcp_plan_trajectory_reaches_goal_when_replayed():
    """The full imagined plan, executed open-loop in the real environment,
    must move the agent materially closer to the goal -- the action
    sequence is a real plan, not decoration."""
    import numpy as np

    import oneiros.mcp_server as srv

    info = srv.reset_env(seed=3, session="traj")
    goal_obs = info["observation"][:]
    goal_obs[0], goal_obs[1] = info["goal"][0], info["goal"][1]
    goal_obs[2], goal_obs[3] = 0.0, 0.0

    start_dist = float(
        np.linalg.norm(np.asarray(info["state"][:2]) - np.asarray(info["goal"]))
    )
    plan = srv.plan_trajectory(info["observation"], goal_obs)
    assert plan["horizon"] == len(plan["actions"]) == len(plan["imagined_latents"])

    last = None
    for action in plan["actions"]:
        last = srv.step_env(action, session="traj")
    assert last is not None
    # A 12-step open-loop plan on the slow default env cannot close an
    # arbitrary gap (bounded speed), so the bar is material progress:
    # measured 0.70x of the start distance; gated at < 0.80x.
    assert last["dist"] < 0.8 * start_dist, (start_dist, last["dist"])
