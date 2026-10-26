"""MCP server behaviour when tool calls overlap (the mcp 2.x execution model).

mcp 2.x runs sync tool handlers on worker threads, so requests that are in
flight together run in parallel. The server shares one planner (and its numpy
RNG), lazily built singletons, and the session->environment map across all of
them. These tests send overlapping requests through a real mcp 2.x ``Client``
connected to the server, which is the worker-thread path the in-process
tool-function tests in test_mcp_server.py never reach.
"""

import asyncio
import json
import time

import numpy as np
from mcp import Client

import oneiros.mcp_server as srv
from oneiros.env import PointMass
from oneiros.planner import LatentMPCPlanner, PlannerConfig
from oneiros.train import CHECKPOINT_PATH, load_model


def _parse(result) -> dict:
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def _call_together(calls: list[tuple[str, dict]]) -> list[dict]:
    """Send every (tool, arguments) call at once over one client; results in order."""

    async def run():
        async with Client(srv.mcp) as client:
            results = await asyncio.gather(
                *(client.call_tool(name, args) for name, args in calls)
            )
        return [_parse(r) for r in results]

    return asyncio.run(run())


def test_concurrent_plans_match_serial_reference():
    """N identical plan_to_goal requests in flight at once must return exactly
    the N actions a fresh planner gives when called N times in a row.

    The planner draws its CEM samples from one shared RNG. If calls are
    serialized, each consumes one call's worth of draws, so the results are
    the serial ones in some order. Calls that interleave on worker threads
    take draws meant for each other and return actions the serial run never
    produces.
    """
    n = 8
    srv.configure()  # fresh planner: its RNG starts from the seed
    info = srv.reset_env(seed=0, session="conc-ref")
    args = {"current_obs": info["observation"], "goal_obs": info["goal_observation"]}

    got = [r["action"] for r in _call_together([("plan_to_goal", args)] * n)]

    reference = LatentMPCPlanner(
        load_model(CHECKPOINT_PATH),
        PlannerConfig(max_accel=srv._get_env_config().max_accel),
    )
    cur = np.asarray(args["current_obs"], dtype=np.float32)
    goal = np.asarray(args["goal_obs"], dtype=np.float32)
    want = [[float(a) for a in reference.plan(cur, goal)] for _ in range(n)]

    # every serial call differs (the RNG advances), so matching the multiset
    # really pins down which draws each call used.
    assert len({tuple(w) for w in want}) == n
    np.testing.assert_allclose(np.array(sorted(got)), np.array(sorted(want)), atol=1e-6)


def test_concurrent_first_calls_load_the_model_once(monkeypatch):
    """Requests that all arrive before the model is loaded must share one
    load. The model, env config and planner are built on first use; if that
    is not serialized, every request in flight sees an empty cache and loads
    its own copy (the slowed loader below holds that window open)."""
    loads = []
    real_load = srv.load_model

    def slow_load(path):
        loads.append(path)
        time.sleep(0.2)
        return real_load(path)

    monkeypatch.setattr(srv, "load_model", slow_load)
    srv.configure()
    try:
        results = _call_together([("model_info", {})] * 6)
    finally:
        srv.configure()

    assert loads == [CHECKPOINT_PATH]
    assert all(r == results[0] for r in results)


def test_concurrent_sessions_replay_exactly():
    """Agents driving separate sessions at the same time, each over its own
    client connection, must each get a self-consistent episode: replaying a
    session's own actions in a fresh environment reproduces every state and
    observation the server returned for that session."""
    seeds = {f"agent-{i}": i for i in range(4)}
    n_steps = 4

    async def agent(session: str, seed: int):
        async with Client(srv.mcp) as client:
            start = _parse(
                await client.call_tool("reset_env", {"seed": seed, "session": session})
            )
            obs, goal_obs = start["observation"], start["goal_observation"]
            trace = []
            for _ in range(n_steps):
                plan = _parse(
                    await client.call_tool(
                        "plan_to_goal", {"current_obs": obs, "goal_obs": goal_obs}
                    )
                )
                step = _parse(
                    await client.call_tool(
                        "step_env", {"action": plan["action"], "session": session}
                    )
                )
                assert step["session"] == session
                obs = step["observation"]
                trace.append((plan["action"], step["state"], step["observation"]))
            return start, trace

    async def run_all():
        return await asyncio.gather(*(agent(s, seed) for s, seed in seeds.items()))

    srv.configure()
    episodes = asyncio.run(run_all())

    cfg = srv._get_env_config()
    for (session, seed), (start, trace) in zip(seeds.items(), episodes, strict=True):
        env = PointMass(config=cfg, seed=0)
        env.reset(seed=seed)
        assert env.state.tolist() == start["state"], session
        for action, state, obs in trace:
            env.step(np.asarray(action, dtype=np.float32))
            assert env.state.tolist() == state, session
            assert env.state_obs().tolist() == obs, session
