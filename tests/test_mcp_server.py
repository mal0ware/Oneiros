"""MCP server smoke tests: the server imports and exposes the expected tools."""

import asyncio

EXPECTED_TOOLS = {
    "encode_observation",
    "predict_rollout",
    "plan_to_goal",
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
